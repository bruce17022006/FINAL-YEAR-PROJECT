
from pathlib import Path
import torch

CACHE_DIR = Path("precomputed_pbkd_train_clean")

# Student model has temporal downsampling by a factor of 2.
def output_length(input_length):
    return (input_length + 1) // 2

files = sorted(CACHE_DIR.glob("*.pt"))

if not files:
    raise FileNotFoundError(f"No cache files found: {CACHE_DIR}")

invalid = []
input_lengths = []
target_lengths = []

for path in files:
    item = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    mel = torch.as_tensor(item["student_mel"])
    target = torch.as_tensor(item["student_target_ids"]).reshape(-1)

    # Mel features should have shape [time, mel_bins].
    input_len = mel.shape[0]
    target_len = target.numel()
    output_len = output_length(input_len)

    input_lengths.append(input_len)
    target_lengths.append(target_len)

    # CTC needs enough output frames for target tokens
    # plus adjacent repeated tokens that need blank separation.
    repeats = (
        (target[1:] == target[:-1]).sum().item()
        if target_len > 1 else 0
    )
    required_frames = target_len + repeats

    if output_len < required_frames:
        invalid.append(
            (path.name, input_len, output_len,
             target_len, repeats, required_frames)
        )

print("Cache files checked:", len(files))
print("Input mel length min/max:",
      min(input_lengths), max(input_lengths))
print("Target length min/max:",
      min(target_lengths), max(target_lengths))
print("Invalid CTC alignments:", len(invalid))

if invalid:
    print("\nFirst invalid examples:")
    for row in invalid[:10]:
        print(row)
else:
    print("All cached samples satisfy the CTC minimum-length check.")
