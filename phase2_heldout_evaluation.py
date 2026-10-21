import os
import glob
import random
import torch
import torchaudio
import sentencepiece as spm
from train_stage1_asr import StudentAcousticModel, TOKENIZER_MODEL, CHECKPOINT_DIR

# Evaluate both checkpoints on held-out utterances from different chapters.
# No training occurs and no checkpoint is modified.
DATA_DIR = os.path.join("Data", "train-clean-100", "LibriSpeech", "train-clean-100")
CHECKPOINTS = {
    "baseline": os.path.join(CHECKPOINT_DIR, "best_student_asr_corrected.pt"),
    "overfit_test": os.path.join(CHECKPOINT_DIR, "phase2_overfit_test.pt"),
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2026
NUM_AUDIO = 12
BLANK_ID = 4

def find_reference(audio_path):
    file_id = os.path.splitext(os.path.basename(audio_path))[0]
    chapter_dir = os.path.dirname(audio_path)
    for trans_path in glob.glob(os.path.join(chapter_dir, "*.trans.txt")):
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
    dp = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        row = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            row[j] = min(dp[j] + 1, row[j-1] + 1, dp[j-1] + (x != y))
        dp = row
    return dp[-1]

def wer(ref, hyp):
    r, h = ref.lower().split(), hyp.lower().split()
    return edit_distance(r, h) / max(1, len(r))

def cer(ref, hyp):
    r = list(ref.lower().replace(" ", ""))
    h = list(hyp.lower().replace(" ", ""))
    return edit_distance(r, h) / max(1, len(r))

def greedy_decode(ids, sp):
    out, previous = [], None
    for token in ids:
        if token != previous and token != BLANK_ID:
            out.append(token)
        previous = token
    return sp.decode_ids(out)

def main():
    if not os.path.isfile(TOKENIZER_MODEL):
        raise FileNotFoundError(f"Tokenizer missing: {TOKENIZER_MODEL}")
    for path in CHECKPOINTS.values():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required checkpoint missing: {path}")

    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL)

    # The overfit script trained on the first eight sorted cache files,
    # which correspond to the first chapter's first eight utterances.
    all_audio = sorted(
        glob.glob(os.path.join(DATA_DIR, "**", "*.flac"), recursive=True)
        + glob.glob(os.path.join(DATA_DIR, "**", "*.wav"), recursive=True)
    )
    # Restrict to chapters other than 103-1240 to reduce overlap with the
    # small overfit set; prefer a deterministic sample across the corpus.
    candidates = []
    for p in all_audio:
        parts = os.path.normpath(p).split(os.sep)
        if len(parts) >= 3 and not (parts[-3] == "103" and parts[-2] == "1240"):
            ref = find_reference(p)
            if ref:
                candidates.append((p, ref))
    if len(candidates) < NUM_AUDIO:
        raise RuntimeError(f"Only found {len(candidates)} held-out candidates.")

    rng = random.Random(SEED)
    # Spread selection across the sorted candidate list rather than choosing
    # only the first chapter.
    if len(candidates) > NUM_AUDIO:
        indices = sorted(rng.sample(range(len(candidates)), NUM_AUDIO))
        selected = [candidates[i] for i in indices]
    else:
        selected = candidates

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=80
    )

    models = {}
    for name, path in CHECKPOINTS.items():
        model = StudentAcousticModel(
            input_dim=80, encoder_dim=256, num_encoder_layers=4,
            vocab_size=1024, teacher_dim=2048, blank_id=BLANK_ID
        ).to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True), strict=True)
        model.eval()
        models[name] = model

    totals = {name: {"wer": 0.0, "cer": 0.0, "blank": 0.0} for name in CHECKPOINTS}
    counts = {name: 0 for name in CHECKPOINTS}
    print(f"Device: {DEVICE}")
    print(f"Held-out utterances: {len(selected)}")
    print("Selection excludes chapter 103-1240 used by the overfit test.")
    print("Note: evaluation uses raw audio preprocessing; compare cautiously with cached-feature training.")
    for idx, (audio_path, reference) in enumerate(selected, 1):
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        mel = mel_transform(waveform).squeeze(0)
        log_mel = torch.log(torch.clamp(mel, min=1e-6)).transpose(0, 1)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
        log_mel = log_mel.unsqueeze(0).to(DEVICE)

        print(f"\n[{idx:02d}/{len(selected)}] {os.path.basename(audio_path)}")
        print("Reference:", reference[:220])
        for name, model in models.items():
            with torch.inference_mode():
                logits, _ = model(log_mel)
                probs = torch.softmax(logits.float(), dim=-1)
            ids = logits.argmax(-1).squeeze(0).tolist()
            prediction = greedy_decode(ids, sp)
            wr, cr = wer(reference, prediction), cer(reference, prediction)
            blank_ratio = 100.0 * ids.count(BLANK_ID) / max(1, len(ids))
            totals[name]["wer"] += wr
            totals[name]["cer"] += cr
            totals[name]["blank"] += blank_ratio
            counts[name] += 1
            print(f"  {name}: WER={wr*100:.2f}% CER={cr*100:.2f}% blank={blank_ratio:.2f}%")
            print(f"    Prediction: {prediction[:220]}")
        del log_mel

    print("\n========== HELD-OUT SUMMARY (UNWEIGHTED MEAN) ==========")
    for name in CHECKPOINTS:
        n = counts[name]
        print(f"{name}: mean WER={totals[name]['wer']/n*100:.2f}% | "
              f"mean CER={totals[name]['cer']/n*100:.2f}% | "
              f"mean blank ratio={totals[name]['blank']/n:.2f}%")
    print("Lower WER/CER is better. This small sample is diagnostic, not a final benchmark.")

if __name__ == "__main__":
    main()
