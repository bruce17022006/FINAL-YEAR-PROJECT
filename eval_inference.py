import os
import glob
import torch
import torchaudio
import sentencepiece as spm
from train_stage1_asr import StudentAcousticModel, TOKENIZER_MODEL, CHECKPOINT_DIR

# Evaluate the checkpoint produced by the corrected 20-epoch training run.
DATA_DIR = os.path.join("Data", "train-clean-100", "LibriSpeech", "train-clean-100")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_student_asr_corrected.pt")
PREFERRED_AUDIO_NAME = "103-1240-0000.flac"  # uses this sample if it exists; otherwise first audio found
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def find_reference(audio_path):
    """Find matching LibriSpeech *.trans.txt entry, if present."""
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


def word_error_rate(reference, hypothesis):
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if not ref:
        return None
    dp = list(range(len(hyp) + 1))
    for i, rw in enumerate(ref, start=1):
        new = [i] + [0] * len(hyp)
        for j, hw in enumerate(hyp, start=1):
            new[j] = min(
                dp[j] + 1,             # deletion
                new[j - 1] + 1,        # insertion
                dp[j - 1] + (rw != hw) # substitution
            )
        dp = new
    return dp[-1] / len(ref)


def greedy_decode(ids, sp, blank_id):
    output = []
    previous = None
    for token_id in ids:
        if token_id != previous and token_id != blank_id:
            output.append(token_id)
        previous = token_id
    return sp.decode_ids(output)


def main():
    if not os.path.isfile(TOKENIZER_MODEL):
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_MODEL}")
    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Corrected checkpoint not found: {CHECKPOINT_PATH}. "
            "Make sure training saved best_student_asr_corrected.pt."
        )

    audio_files = sorted(
        glob.glob(os.path.join(DATA_DIR, "**", "*.flac"), recursive=True)
        + glob.glob(os.path.join(DATA_DIR, "**", "*.wav"), recursive=True)
    )
    if not audio_files:
        raise FileNotFoundError(f"No .flac/.wav audio files found under: {DATA_DIR}")

    preferred = [p for p in audio_files if os.path.basename(p) == PREFERRED_AUDIO_NAME]
    test_audio = preferred[0] if preferred else audio_files[0]

    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL)
    blank_id = sp.piece_to_id("<blank>")

    model = StudentAcousticModel(
        input_dim=80,
        encoder_dim=256,
        num_encoder_layers=4,
        vocab_size=1024,
        teacher_dim=2048,
        blank_id=blank_id,
    ).to(DEVICE)
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=80
    )
    waveform, sample_rate = torchaudio.load(test_audio)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    mel = mel_transform(waveform).squeeze(0)
    log_mel = torch.log(torch.clamp(mel, min=1e-6)).transpose(0, 1)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    log_mel = log_mel.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits, _ = model(log_mel)
        probabilities = torch.softmax(logits.float(), dim=-1)

    raw_ids = torch.argmax(logits, dim=-1).squeeze(0).tolist()
    blank_count = raw_ids.count(blank_id)
    prediction = greedy_decode(raw_ids, sp, blank_id)
    reference = find_reference(test_audio)

    print("\n--- Corrected Checkpoint Inference Evaluation ---")
    print("Device:", DEVICE)
    print("Audio:", test_audio)
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Logits shape [B, T, V]:", tuple(logits.shape))
    print(f"Blank frames: {blank_count}/{len(raw_ids)} ({100 * blank_count / max(1, len(raw_ids)):.1f}%)")
    print(f"Mean blank probability: {probabilities[..., blank_id].mean().item():.4f}")
    print("Logits contain NaN:", torch.isnan(logits).any().item())
    print("Logits contain Inf:", torch.isinf(logits).any().item())

    print("\nREFERENCE:")
    print(reference if reference else "Not found (check the matching LibriSpeech *.trans.txt file).")
    print("\nPREDICTION:")
    print(prediction if prediction else "[EMPTY TRANSCRIPT]")

    if reference:
        wer = word_error_rate(reference, prediction)
        print(f"\nWER: {wer:.4f} ({wer * 100:.2f}%)")
    else:
        print("\nWER not calculated because the reference transcript was not found.")


if __name__ == "__main__":
    main()
