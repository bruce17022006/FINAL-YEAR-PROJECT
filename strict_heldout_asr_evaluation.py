"""
Strict held-out evaluation for the current LibriSpeech student ASR checkpoint.

This script does NOT train or overwrite any checkpoint.
It excludes every speaker-chapter represented in the precomputed cache, so
the evaluation audio comes from chapters not used in the cached training set.
Run from the project root.
"""
import os
import glob
import random
import torch
import torchaudio
import sentencepiece as spm

from train_stage1_asr import (
    StudentAcousticModel,
    TOKENIZER_MODEL,
    CHECKPOINT_DIR,
)

DATA_DIR = os.path.join("Data", "train-clean-100", "LibriSpeech", "train-clean-100")
CACHE_DIR = "precomputed_pbkd_train_clean"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_student_asr_corrected.pt")
OUTPUT_CSV = "strict_heldout_asr_results.csv"
NUM_AUDIO = 30
SEED = 2026
BLANK_ID = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def find_reference(audio_path):
    file_id = os.path.splitext(os.path.basename(audio_path))[0]
    for trans_path in glob.glob(os.path.join(os.path.dirname(audio_path), "*.trans.txt")):
        try:
            with open(trans_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2 and parts[0] == file_id:
                        return parts[1]
        except OSError:
            pass
    return None


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def wer(reference, hypothesis):
    ref, hyp = reference.lower().split(), hypothesis.lower().split()
    return edit_distance(ref, hyp) / max(1, len(ref))


def cer(reference, hypothesis):
    ref = list("".join(reference.lower().split()))
    hyp = list("".join(hypothesis.lower().split()))
    return edit_distance(ref, hyp) / max(1, len(ref))


def greedy_decode(ids, sp):
    output, previous = [], None
    for token_id in ids:
        if token_id != previous and token_id != BLANK_ID:
            output.append(token_id)
        previous = token_id
    return sp.decode_ids(output)


def chapter_key(file_id):
    # LibriSpeech IDs look like speaker-chapter-utterance, e.g. 103-1240-0000.
    parts = str(file_id).strip().split("-")
    return "-".join(parts[:2]) if len(parts) >= 3 else None


def main():
    if not os.path.isfile(TOKENIZER_MODEL):
        raise FileNotFoundError(f"Tokenizer missing: {TOKENIZER_MODEL}")
    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint missing: {CHECKPOINT_PATH}")
    if not os.path.isdir(CACHE_DIR):
        raise FileNotFoundError(f"Cache directory missing: {CACHE_DIR}")

    # Read IDs from the exact cache used for training.
    try:
        from train_stage1_asr import FastPrecomputedDataset
        cache_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.pt")))
        if not cache_files:
            raise FileNotFoundError(f"No .pt files in {CACHE_DIR}")
        cached_chapters = set()
        cache_ids_missing = 0
        for path in cache_files:
            item = torch.load(path, map_location="cpu", weights_only=True)
            file_id = item.get("file_id")
            key = chapter_key(file_id) if file_id is not None else None
            if key:
                cached_chapters.add(key)
            else:
                cache_ids_missing += 1
        if not cached_chapters:
            raise RuntimeError(
                "No usable file_id values found in cache. Cannot guarantee chapter-level holdout."
            )
    except Exception as exc:
        raise RuntimeError(f"Could not inspect cache IDs safely: {exc}") from exc

    print(f"Device: {DEVICE}")
    print(f"Cached utterances: {len(cache_files)}")
    print(f"Training-cache chapters found: {len(cached_chapters)}")
    print(f"Cache entries without usable file_id: {cache_ids_missing}")

    all_audio = sorted(
        glob.glob(os.path.join(DATA_DIR, "**", "*.flac"), recursive=True)
        + glob.glob(os.path.join(DATA_DIR, "**", "*.wav"), recursive=True)
    )
    candidates = []
    for path in all_audio:
        key = chapter_key(os.path.splitext(os.path.basename(path))[0])
        if key and key not in cached_chapters:
            reference = find_reference(path)
            if reference:
                candidates.append((path, reference, key))

    if len(candidates) < NUM_AUDIO:
        raise RuntimeError(
            f"Only {len(candidates)} usable utterances from unseen chapters found; "
            f"reduce NUM_AUDIO (currently {NUM_AUDIO}) or check DATA_DIR."
        )

    rng = random.Random(SEED)
    selected = rng.sample(candidates, NUM_AUDIO)

    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL)
    model = StudentAcousticModel(
        input_dim=80, encoder_dim=256, num_encoder_layers=4,
        vocab_size=1024, teacher_dim=2048, blank_id=BLANK_ID
    ).to(DEVICE)
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=400, win_length=400,
        hop_length=160, n_mels=80, power=2.0
    )

    rows = []
    with torch.inference_mode():
        for i, (path, reference, key) in enumerate(selected, 1):
            waveform, sr = torchaudio.load(path)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            mel = mel_transform(waveform).squeeze(0)
            log_mel = torch.log(torch.clamp(mel, min=1e-6)).transpose(0, 1)
            log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
            logits, _ = model(log_mel.unsqueeze(0).to(DEVICE))
            probs = torch.softmax(logits.float(), dim=-1)
            ids = logits.argmax(dim=-1).squeeze(0).tolist()
            hypothesis = greedy_decode(ids, sp)
            item_wer = wer(reference, hypothesis)
            item_cer = cer(reference, hypothesis)
            blank_ratio = ids.count(BLANK_ID) / max(1, len(ids))

            rows.append({
                "file_id": os.path.splitext(os.path.basename(path))[0],
                "chapter": key,
                "wer": item_wer,
                "cer": item_cer,
                "blank_ratio": blank_ratio,
                "mean_blank_probability": probs[..., BLANK_ID].mean().item(),
                "reference": reference,
                "prediction": hypothesis,
            })
            print(
                f"[{i:02d}/{NUM_AUDIO}] {os.path.basename(path)} | "
                f"WER={item_wer*100:.2f}% CER={item_cer*100:.2f}% "
                f"blank={blank_ratio*100:.2f}%"
            )
            print("  REF :", reference[:160])
            print("  HYP :", hypothesis[:160])

    import csv
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== STRICT HELD-OUT SUMMARY =====")
    print(f"Utterances: {len(rows)}")
    print(f"Distinct unseen chapters evaluated: {len(set(r['chapter'] for r in rows))}")
    print(f"Mean WER: {sum(r['wer'] for r in rows)/len(rows)*100:.2f}%")
    print(f"Mean CER: {sum(r['cer'] for r in rows)/len(rows)*100:.2f}%")
    print(f"Mean blank ratio: {sum(r['blank_ratio'] for r in rows)/len(rows)*100:.2f}%")
    print(f"Mean blank probability: {sum(r['mean_blank_probability'] for r in rows)/len(rows):.4f}")
    print(f"Detailed results saved to: {OUTPUT_CSV}")
    print("No model weights were changed or saved.")


if __name__ == "__main__":
    main()
