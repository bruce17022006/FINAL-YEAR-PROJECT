
import os
import glob
import sentencepiece as spm

# ==========================================
# 1. Configuration
# ==========================================
DATA_DIR = os.path.join(
    "Data", "train-clean-100", "LibriSpeech", "train-clean-100"
)

CORPUS_FILE = "train_transcripts.txt"
MODEL_PREFIX = "student_bpe_1024"
VOCAB_SIZE = 1024


def normalize_text(text):
    """Use the same lowercase normalization for all transcripts."""
    return " ".join(text.lower().strip().split())


# ==========================================
# 2. Collect and Normalize Transcripts
# ==========================================
print("[1/3] Collecting training transcripts...")

trans_files = glob.glob(
    os.path.join(DATA_DIR, "**", "*.trans.txt"),
    recursive=True
)

if not trans_files:
    raise FileNotFoundError(
        f"No .trans.txt files found under: {DATA_DIR}\n"
        "Check your LibriSpeech dataset path."
    )

collected_count = 0

with open(CORPUS_FILE, "w", encoding="utf-8") as out_f:
    for trans_file in trans_files:
        with open(trans_file, "r", encoding="utf-8") as in_f:
            for line in in_f:
                parts = line.strip().split(maxsplit=1)

                if len(parts) == 2:
                    text = normalize_text(parts[1])

                    if text:
                        out_f.write(text + "\n")
                        collected_count += 1

if collected_count == 0:
    raise ValueError("No valid transcripts were collected.")

print(f"Transcript files: {len(trans_files)}")
print(f"Normalized transcripts: {collected_count}")


# ==========================================
# 3. Train SentencePiece Tokenizer
# ==========================================
print("\n[2/3] Training SentencePiece BPE tokenizer...")

spm.SentencePieceTrainer.train(
    input=CORPUS_FILE,
    model_prefix=MODEL_PREFIX,
    vocab_size=VOCAB_SIZE,
    model_type="bpe",
    character_coverage=1.0,

    # Preserve the project's special-token convention
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    user_defined_symbols=["<blank>"],

    hard_vocab_limit=False
)

MODEL_FILE = f"{MODEL_PREFIX}.model"

# ==========================================
# 4. Verify Special Token IDs
# ==========================================
print("\n[3/3] Verifying tokenizer...")

sp = spm.SentencePieceProcessor()
if not sp.load(MODEL_FILE):
    raise RuntimeError(f"Could not load tokenizer: {MODEL_FILE}")

expected_ids = {
    "<pad>": 0,
    "<unk>": 1,
    "<s>": 2,
    "</s>": 3,
    "<blank>": 4,
}

print(f"Actual vocabulary size: {sp.get_piece_size()}")

for token, expected_id in expected_ids.items():
    actual_id = sp.piece_to_id(token)
    print(f"{token:8s} expected={expected_id}, actual={actual_id}")

    if actual_id != expected_id:
        raise ValueError(
            f"Special-token mismatch for {token}: "
            f"expected {expected_id}, got {actual_id}"
        )

assert sp.pad_id() == 0
assert sp.unk_id() == 1
assert sp.bos_id() == 2
assert sp.eos_id() == 3
assert sp.piece_to_id("<blank>") == 4

# ==========================================
# 5. Normalization and Encoding Test
# ==========================================
test_texts = [
    "HELLO WORLD",
    "Hello World",
    "hello world",
]

for raw_text in test_texts:
    text = normalize_text(raw_text)
    ids = sp.encode_as_ids(text)
    decoded = sp.decode_ids(ids)
    unk_count = ids.count(sp.unk_id())

    print(f"\nInput      : {raw_text}")
    print(f"Normalized : {text}")
    print(f"Token IDs  : {ids}")
    print(f"Decoded    : {decoded}")
    print(f"UNK count  : {unk_count}")

    if unk_count > 0:
        print("WARNING: Unknown token detected.")

print("\nTokenizer training and verification completed.")
print(f"Model: {MODEL_FILE}")
print(f"Vocabulary: {MODEL_PREFIX}.vocab")
