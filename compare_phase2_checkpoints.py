import os
import glob
import torch
import torchaudio
import sentencepiece as spm
from train_stage1_asr import StudentAcousticModel, TOKENIZER_MODEL, CHECKPOINT_DIR

DATA_DIR = os.path.join("Data", "train-clean-100", "LibriSpeech", "train-clean-100")
PREFERRED_AUDIO_NAME = "103-1240-0000.flac"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINTS = {
    "baseline_corrected": os.path.join(CHECKPOINT_DIR, "best_student_asr_corrected.pt"),
    "padding_finetuned": os.path.join(CHECKPOINT_DIR, "student_asr_padding_finetuned.pt"),
}

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

def edit_distance(ref_items, hyp_items):
    dp = list(range(len(hyp_items) + 1))
    for i, ri in enumerate(ref_items, start=1):
        row = [i] + [0] * len(hyp_items)
        for j, hi in enumerate(hyp_items, start=1):
            row[j] = min(dp[j] + 1, row[j - 1] + 1, dp[j - 1] + (ri != hi))
        dp = row
    return dp[-1]

def wer(reference, hypothesis):
    ref, hyp = reference.lower().split(), hypothesis.lower().split()
    return edit_distance(ref, hyp) / max(1, len(ref))

def cer(reference, hypothesis):
    ref = list(reference.lower().replace(" ", ""))
    hyp = list(hypothesis.lower().replace(" ", ""))
    return edit_distance(ref, hyp) / max(1, len(ref))

def greedy_decode(ids, sp, blank_id):
    output, previous = [], None
    for token_id in ids:
        if token_id != previous and token_id != blank_id:
            output.append(token_id)
        previous = token_id
    return sp.decode_ids(output)

def main():
    if not os.path.isfile(TOKENIZER_MODEL):
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_MODEL}")

    audio_files = sorted(
        glob.glob(os.path.join(DATA_DIR, "**", "*.flac"), recursive=True)
        + glob.glob(os.path.join(DATA_DIR, "**", "*.wav"), recursive=True)
    )
    if not audio_files:
        raise FileNotFoundError(f"No .flac/.wav files found under: {DATA_DIR}")
    matches = [p for p in audio_files if os.path.basename(p) == PREFERRED_AUDIO_NAME]
    audio_path = matches[0] if matches else audio_files[0]
    reference = find_reference(audio_path)

    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL)
    blank_id = sp.piece_to_id("<blank>")

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=80
    )
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    mel = mel_transform(waveform).squeeze(0)
    log_mel = torch.log(torch.clamp(mel, min=1e-6)).transpose(0, 1)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    log_mel = log_mel.unsqueeze(0).to(DEVICE)

    print("=" * 72)
    print("PHASE 2 CHECKPOINT COMPARISON")
    print("Device:", DEVICE)
    print("Audio:", audio_path)
    print("Reference:", reference if reference else "[not found]")
    print("=" * 72)

    results = {}
    for label, checkpoint_path in CHECKPOINTS.items():
        if not os.path.isfile(checkpoint_path):
            print(f"\n[{label}] SKIPPED — checkpoint not found: {checkpoint_path}")
            continue

        model = StudentAcousticModel(
            input_dim=80, encoder_dim=256, num_encoder_layers=4,
            vocab_size=1024, teacher_dim=2048, blank_id=blank_id
        ).to(DEVICE)
        state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()

        with torch.inference_mode():
            logits, _ = model(log_mel)
            probabilities = torch.softmax(logits.float(), dim=-1)

        raw_ids = torch.argmax(logits, dim=-1).squeeze(0).tolist()
        blank_count = raw_ids.count(blank_id)
        prediction = greedy_decode(raw_ids, sp, blank_id)
        blank_ratio = 100.0 * blank_count / max(1, len(raw_ids))
        mean_blank_prob = probabilities[..., blank_id].mean().item()
        results[label] = {
            "prediction": prediction,
            "wer": wer(reference, prediction) if reference else None,
            "cer": cer(reference, prediction) if reference else None,
            "blank_ratio": blank_ratio,
            "mean_blank_prob": mean_blank_prob,
        }

        print(f"\n--- {label} ---")
        print("Checkpoint:", checkpoint_path)
        print(f"Blank frames: {blank_count}/{len(raw_ids)} ({blank_ratio:.2f}%)")
        print(f"Mean blank probability: {mean_blank_prob:.4f}")
        print("Logits contain NaN:", torch.isnan(logits).any().item())
        print("Logits contain Inf:", torch.isinf(logits).any().item())
        print("Prediction:", prediction if prediction else "[EMPTY TRANSCRIPT]")
        if reference:
            print(f"WER: {results[label]['wer']:.4f} ({results[label]['wer'] * 100:.2f}%)")
            print(f"CER: {results[label]['cer']:.4f} ({results[label]['cer'] * 100:.2f}%)")

        del model, state, logits, probabilities
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    if "baseline_corrected" in results and "padding_finetuned" in results:
        base, new = results["baseline_corrected"], results["padding_finetuned"]
        print("\n" + "=" * 72)
        print("COMPARISON SUMMARY")
        if reference:
            print(f"WER: baseline={base['wer']*100:.2f}% | fine-tuned={new['wer']*100:.2f}%")
            print(f"CER: baseline={base['cer']*100:.2f}% | fine-tuned={new['cer']*100:.2f}%")
        print(f"Blank ratio: baseline={base['blank_ratio']:.2f}% | fine-tuned={new['blank_ratio']:.2f}%")
        print(f"Mean blank probability: baseline={base['mean_blank_prob']:.4f} | fine-tuned={new['mean_blank_prob']:.4f}")
        if reference:
            if new["wer"] < base["wer"]:
                print("Decision: fine-tuned checkpoint is better on this sample (validate on more samples before claiming general improvement).")
            elif new["wer"] > base["wer"]:
                print("Decision: fine-tuned checkpoint is worse on this sample; do not promote it based on this test.")
            else:
                print("Decision: WER is unchanged on this sample; compare additional held-out samples before choosing.")
        print("=" * 72)

if __name__ == "__main__":
    main()
