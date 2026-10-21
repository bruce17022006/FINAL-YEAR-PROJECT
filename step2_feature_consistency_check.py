"""
Step 2 diagnostic: compare cached training features against features recomputed
from the original audio, then compare checkpoint logits on both inputs.
No training occurs. No checkpoint/cache files are modified.
Run from the project root.
"""
import os
import glob
import torch
import torchaudio

from train_stage1_asr import (
    StudentAcousticModel,
    TOKENIZER_MODEL,
    CHECKPOINT_DIR,
)

DATA_DIR = os.path.join("Data", "train-clean-100", "LibriSpeech", "train-clean-100")
CACHE_DIR = "precomputed_pbkd_train_clean"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_student_asr_corrected.pt")
TEST_FILE_ID = "103-1240-0000"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def find_audio(file_id):
    matches = (
        glob.glob(os.path.join(DATA_DIR, "**", file_id + ".flac"), recursive=True)
        + glob.glob(os.path.join(DATA_DIR, "**", file_id + ".wav"), recursive=True)
    )
    return matches[0] if matches else None


def recompute_features(audio_path):
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
    return log_mel.cpu()


def main():
    cache_path = os.path.join(CACHE_DIR, TEST_FILE_ID + ".pt")
    audio_path = find_audio(TEST_FILE_ID)

    if not os.path.isfile(cache_path):
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    if audio_path is None:
        raise FileNotFoundError(f"Original audio not found for {TEST_FILE_ID} under {DATA_DIR}")
    if not os.path.isfile(TOKENIZER_MODEL):
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_MODEL}")
    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    cached = cache["student_mel"].float().cpu()
    recomputed = recompute_features(audio_path).float().cpu()

    print("===== FEATURE PREPROCESSING COMPARISON =====")
    print("Audio:", audio_path)
    print("Cache:", cache_path)
    print("Cache file_id:", cache.get("file_id", "[missing]"))
    print("Cached shape:", tuple(cached.shape))
    print("Recomputed shape:", tuple(recomputed.shape))

    if cached.shape != recomputed.shape:
        print("FEATURE MATCH: FAIL (shape mismatch)")
        print("Cached frames:", cached.shape[0], "| Recomputed frames:", recomputed.shape[0])
        return

    diff = (cached - recomputed).abs()
    print(f"Max absolute feature difference : {diff.max().item():.10g}")
    print(f"Mean absolute feature difference: {diff.mean().item():.10g}")
    print(f"Cached mean/std: {cached.mean().item():.6f} / {cached.std().item():.6f}")
    print(f"Raw mean/std   : {recomputed.mean().item():.6f} / {recomputed.std().item():.6f}")
    feature_match = torch.allclose(cached, recomputed, rtol=1e-5, atol=1e-6)
    print("FEATURE MATCH:", "PASS" if feature_match else "FAIL")

    model = StudentAcousticModel(
        input_dim=80,
        encoder_dim=256,
        num_encoder_layers=4,
        vocab_size=1024,
        teacher_dim=2048,
        blank_id=4,
    ).to(DEVICE)
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    with torch.inference_mode():
        cached_logits, _ = model(cached.unsqueeze(0).to(DEVICE))
        raw_logits, _ = model(recomputed.unsqueeze(0).to(DEVICE))
        cached_ids = cached_logits.argmax(dim=-1)
        raw_ids = raw_logits.argmax(dim=-1)

    logit_diff = (cached_logits.float() - raw_logits.float()).abs()
    print("\n===== CHECKPOINT OUTPUT COMPARISON =====")
    print("Cached logits shape:", tuple(cached_logits.shape))
    print("Raw logits shape   :", tuple(raw_logits.shape))
    print(f"Max absolute logit difference : {logit_diff.max().item():.10g}")
    print(f"Mean absolute logit difference: {logit_diff.mean().item():.10g}")
    same_ids = torch.equal(cached_ids, raw_ids)
    changed = (cached_ids != raw_ids).sum().item()
    total = cached_ids.numel()
    print(f"Argmax frame IDs identical: {same_ids}")
    print(f"Different argmax frames   : {changed}/{total}")
    print("\nInterpretation:")
    if feature_match and same_ids:
        print("- Cached and raw-audio preprocessing match; inference feature mismatch is unlikely the main cause.")
    elif feature_match:
        print("- Features match but argmax differs; inspect model numerical/device behavior next.")
    else:
        print("- Feature mismatch detected; fix preprocessing/cache consistency before further training.")


if __name__ == "__main__":
    main()
