
import os
import glob
import torch
import torchaudio
import sentencepiece as spm

from train_stage1_asr import (
    StudentAcousticModel,
    TOKENIZER_MODEL,
)

DATA_DIR = os.path.join(
    "Data", "train-clean-100", "LibriSpeech", "train-clean-100"
)
CHECKPOINT_DIR = "checkpoints_stage1"
CHECKPOINTS = [
    os.path.join(CHECKPOINT_DIR, "best_student_asr.pt"),
    os.path.join(CHECKPOINT_DIR, "best_student_asr_corrected.pt"),
]
PREFERRED_AUDIO = "103-1240-0000.flac"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def normalize_text(text):
    return " ".join(text.lower().strip().split())


def find_reference(audio_path):
    file_id = os.path.splitext(os.path.basename(audio_path))[0]
    chapter_dir = os.path.dirname(audio_path)

    for path in glob.glob(os.path.join(chapter_dir, "*.trans.txt")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2 and parts[0] == file_id:
                    return normalize_text(parts[1])
    return None


def greedy_decode(ids, sp, blank_id):
    output = []
    previous = None

    for token_id in ids:
        if token_id != previous and token_id != blank_id:
            output.append(token_id)
        previous = token_id

    return sp.decode_ids(output)


def edit_distance(ref, hyp):
    previous = list(range(len(hyp) + 1))

    for i, r in enumerate(ref, start=1):
        current = [i] + [0] * len(hyp)

        for j, h in enumerate(hyp, start=1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (r != h),
            )

        previous = current

    return previous[-1]


def error_rates(reference, prediction):
    ref_words = reference.split()
    hyp_words = prediction.split()

    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(prediction.replace(" ", ""))

    wer = edit_distance(ref_words, hyp_words) / max(1, len(ref_words))
    cer = edit_distance(ref_chars, hyp_chars) / max(1, len(ref_chars))

    return wer, cer


def main():
    if not os.path.isfile(TOKENIZER_MODEL):
        raise FileNotFoundError(f"Tokenizer missing: {TOKENIZER_MODEL}")

    audio_files = sorted(
        glob.glob(os.path.join(DATA_DIR, "**", "*.flac"), recursive=True)
    )

    if not audio_files:
        raise FileNotFoundError(f"No FLAC files found in {DATA_DIR}")

    preferred = [
        p for p in audio_files
        if os.path.basename(p) == PREFERRED_AUDIO
    ]
    audio_path = preferred[0] if preferred else audio_files[0]

    reference = find_reference(audio_path)
    if reference is None:
        raise RuntimeError(f"Reference transcript not found for {audio_path}")

    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL)
    blank_id = sp.piece_to_id("<blank>")

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        win_length=400,
        hop_length=160,
        n_mels=80,
        power=2.0,
    )

    waveform, sr = torchaudio.load(audio_path)

    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    mel = mel_transform(waveform).squeeze(0)
    log_mel = torch.log(torch.clamp(mel, min=1e-6)).transpose(0, 1)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    features = log_mel.unsqueeze(0).to(DEVICE)

    print("Device:", DEVICE)
    print("Audio:", audio_path)
    print("Reference:", reference)

    for checkpoint_path in CHECKPOINTS:
        print("\n" + "=" * 60)
        print("Checkpoint:", checkpoint_path)

        if not os.path.isfile(checkpoint_path):
            print("Checkpoint not found; skipping.")
            continue

        model = StudentAcousticModel(
            input_dim=80,
            encoder_dim=256,
            num_encoder_layers=4,
            vocab_size=1024,
            teacher_dim=2048,
            blank_id=blank_id,
        ).to(DEVICE)

        state = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True,
        )
        model.load_state_dict(state)
        model.eval()

        with torch.no_grad():
            logits, _ = model(features)
            probabilities = torch.softmax(logits.float(), dim=-1)

        ids = logits.argmax(dim=-1).squeeze(0).tolist()
        prediction = greedy_decode(ids, sp, blank_id)

        blank_ratio = 100 * sum(
            token == blank_id for token in ids
        ) / max(1, len(ids))

        mean_blank_prob = probabilities[..., blank_id].mean().item()
        wer, cer = error_rates(reference, prediction)

        print("Prediction:", prediction)
        print(f"WER: {wer * 100:.2f}%")
        print(f"CER: {cer * 100:.2f}%")
        print(f"Blank frames: {blank_ratio:.2f}%")
        print(f"Mean blank probability: {mean_blank_prob:.4f}")

        del model, state, logits, probabilities
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
