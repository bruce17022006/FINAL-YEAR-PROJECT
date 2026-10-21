
from pathlib import Path
import torch
import sentencepiece as spm

CACHE_DIR = Path("precomputed_pbkd_train_clean")
TOKENIZER_PATH = "student_bpe_1024.model"

sp = spm.SentencePieceProcessor()
if not sp.load(TOKENIZER_PATH):
    raise FileNotFoundError(TOKENIZER_PATH)

files = sorted(CACHE_DIR.glob("*.pt"))
mismatches = []
special_counts = {i: 0 for i in range(5)}

for path in files:
    item = torch.load(path, map_location="cpu", weights_only=False)

    ids = torch.as_tensor(item["student_target_ids"]).reshape(-1).long()
    saved_text = " ".join(item["text"].lower().split())
    decoded_text = " ".join(sp.decode_ids(ids.tolist()).lower().split())

    for token_id in range(5):
        special_counts[token_id] += (ids == token_id).sum().item()

    if saved_text != decoded_text:
        mismatches.append(
            (path.name, saved_text[:100], decoded_text[:100])
        )

print("Cache files checked:", len(files))
print("Special-token counts:")
for token_id, count in special_counts.items():
    print(f"  ID {token_id} ({sp.id_to_piece(token_id)}): {count}")

print("Transcript mismatches:", len(mismatches))
for row in mismatches[:10]:
    print("\nFile:", row[0])
    print("Saved:", row[1])
    print("Decoded:", row[2])
