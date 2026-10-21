import os
import glob
import re
from pathlib import Path

import torch
import torchaudio
import sentencepiece as spm
from tqdm import tqdm


# ============================================================
# 1. Paths and configuration
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent

# Try Data folder inside Implementation first, then one level above.
DATA_RELATIVE = Path("Data") / "train-clean-100" / "LibriSpeech" / "train-clean-100"
DATA_CANDIDATES = [
    SCRIPT_DIR / DATA_RELATIVE,
    SCRIPT_DIR.parent / DATA_RELATIVE,
]

DATA_DIR = next((p for p in DATA_CANDIDATES if p.exists()), DATA_CANDIDATES[0])

TOKENIZER_MODEL = SCRIPT_DIR / "student_bpe_1024.model"
OUT_CACHE_DIR = SCRIPT_DIR / "precomputed_pbkd_train_clean"
MAX_SAMPLES = 6000
SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 400
WIN_LENGTH = 400
HOP_LENGTH = 160

os.makedirs(OUT_CACHE_DIR, exist_ok=True)

print(f"--> Script directory: {SCRIPT_DIR}")
print(f"--> Dataset directory: {DATA_DIR}")
print(f"--> Tokenizer model: {TOKENIZER_MODEL}")
print(f"--> Cache directory: {OUT_CACHE_DIR}")


# ============================================================
# 2. Validate paths and load tokenizer
# ============================================================
if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Dataset directory not found: {DATA_DIR}\n"
        "Check that Data/train-clean-100/LibriSpeech/train-clean-100 "
        "exists inside the project folder or its parent folder."
    )

if not TOKENIZER_MODEL.exists():
    raise FileNotFoundError(
        f"Tokenizer model not found: {TOKENIZER_MODEL}\n"
        "Place student_bpe_1024.model beside this Python script."
    )

sp = spm.SentencePieceProcessor()
sp.load(str(TOKENIZER_MODEL))

# Verify the special-token IDs expected by the project.
expected_ids = {
    "<pad>": 0,
    "<unk>": 1,
    "<s>": 2,
    "</s>": 3,
    "<blank>": 4,
}
for token, expected_id in expected_ids.items():
    actual_id = sp.piece_to_id(token)
    if actual_id != expected_id:
        raise ValueError(
            f"Tokenizer ID mismatch for {token}: expected {expected_id}, "
            f"got {actual_id}. Rebuild/check the tokenizer model."
        )

print("--> Tokenizer loaded and special-token IDs verified.")


# ============================================================
# 3. Read LibriSpeech transcript files
# ============================================================
def normalize_text(text):
    """Lowercase text and collapse repeated whitespace."""
    return re.sub(r"\s+", " ", text.lower()).strip()


transcripts = {}
transcript_files = sorted(DATA_DIR.rglob("*.trans.txt"))

for transcript_path in transcript_files:
    with open(transcript_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            # LibriSpeech line format: SPEAKER-CHAPTER-UTTERANCE transcript...
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue

            file_id, transcript = parts
            transcripts[file_id] = normalize_text(transcript)

print(f"--> Transcript files found: {len(transcript_files)}")
print(f"--> Transcript entries loaded: {len(transcripts)}")


# ============================================================
# 4. Discover audio files and select samples
# ============================================================
audio_files = sorted(str(p) for p in DATA_DIR.rglob("*.flac"))

print(f"--> Total audio files found: {len(audio_files)}")

selected_files = audio_files[:MAX_SAMPLES]
print(f"--> Processing {len(selected_files)} audio files")

if not selected_files:
    raise RuntimeError(
        f"No .flac audio files found under {DATA_DIR}. "
        "Check the dataset extraction and folder structure."
    )


# ============================================================
# 5. Set up log-Mel feature extraction
# ============================================================
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    win_length=WIN_LENGTH,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    power=2.0,
)


# ============================================================
# 6. Precompute features and save each cache
# ============================================================
saved_count = 0
skipped_count = 0
failed_count = 0
unknown_token_total = 0

for file_path in tqdm(selected_files, desc="Precomputing CTC cache"):
    file_id = Path(file_path).stem
    out_file = OUT_CACHE_DIR / f"{file_id}.pt"

    if file_id not in transcripts:
        print(f"Warning: transcript missing for {file_id}; skipping.")
        skipped_count += 1
        continue

    try:
        text = normalize_text(transcripts[file_id])

        # Load audio, resample to 16 kHz, and convert to mono.
        waveform, sr = torchaudio.load(file_path)

        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=SAMPLE_RATE
            )

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Log-Mel features: [time, mel_bins].
        mel = mel_transform(waveform).squeeze(0)
        log_mel = torch.log(torch.clamp(mel, min=1e-6)).transpose(0, 1)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)

        # Encode each transcript inside the per-audio loop.
        student_ids = sp.encode_as_ids(text)

        if not student_ids:
            print(f"Warning: empty token sequence for {file_id}; skipping.")
            skipped_count += 1
            continue

        unk_count = student_ids.count(sp.unk_id())
        unknown_token_total += unk_count

        if unk_count:
            print(
                f"Warning: {file_id} has {unk_count}/{len(student_ids)} "
                "unknown tokens."
            )

        # Save one cache file for each audio utterance.
        torch.save(
            {
                "student_mel": log_mel.cpu(),
                "student_target_ids": student_ids,
                "text": text,
                "file_id": file_id,
            },
            out_file,
        )
        saved_count += 1

    except Exception as exc:
        failed_count += 1
        print(f"\nError processing {file_id}: {exc}")


# ============================================================
# 7. Verify cache count and inspect one sample
# ============================================================
cache_files = sorted(OUT_CACHE_DIR.glob("*.pt"))

print("\n========== Cache Generation Summary ==========")
print(f"Audio files selected : {len(selected_files)}")
print(f"Cache files saved now: {saved_count}")
print(f"Skipped              : {skipped_count}")
print(f"Failed               : {failed_count}")
print(f"UNK tokens total     : {unknown_token_total}")
print(f"Total .pt files in cache directory: {len(cache_files)}")

if cache_files:
    try:
        sample = torch.load(
            cache_files[0], map_location="cpu", weights_only=True
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        sample = torch.load(cache_files[0], map_location="cpu")

    print(f"\nSample cache: {cache_files[0].name}")
    print(f"Mel shape: {tuple(sample['student_mel'].shape)}")
    print(f"Target token count: {len(sample['student_target_ids'])}")
    print(f"Normalized text: {sample['text'][:120]}")

if failed_count:
    print("\nWARNING: Some files failed. Review the errors above.")
elif saved_count == 0:
    print("\nWARNING: No new cache files were saved.")
else:
    print("\nCache generation finished. Check the counts above before training.")
    if unknown_token_total == 0:
        print("Tokenizer check: no unknown tokens were found in the saved samples.")
    else:
        print("Tokenizer check: unknown tokens were found; inspect warnings above.")
