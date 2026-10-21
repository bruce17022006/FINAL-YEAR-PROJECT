from pathlib import Path
import torch

CACHE_DIR = Path("precomputed_pbkd_train_clean")

files = sorted(CACHE_DIR.glob("*.pt"))
total_tokens = 0
unk_tokens = 0
samples_with_unk = 0

for path in files:
    item = torch.load(path, map_location="cpu", weights_only=False)
    ids = torch.as_tensor(item["student_target_ids"]).reshape(-1).long()

    count = (ids == 1).sum().item()
    total_tokens += ids.numel()
    unk_tokens += count

    if count > 0:
        samples_with_unk += 1

print("Cache files:", len(files))
print("Total target tokens:", total_tokens)
print("UNK tokens:", unk_tokens)
print("Samples containing UNK:", samples_with_unk)
print("UNK token percentage:",
      round(100 * unk_tokens / max(total_tokens, 1), 4), "%")