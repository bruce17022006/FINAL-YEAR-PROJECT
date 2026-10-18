import os
import glob
import sentencepiece as spm

# ==========================================
# 1. Configurations & Paths
# ==========================================
# LibriSpeech trans.txt files irukra directory
DATA_DIR = os.path.join("Data", "dev-clean", "LibriSpeech", "dev-clean")
CORPUS_FILE = "train_transcripts.txt"
MODEL_PREFIX = "student_bpe_1024"
VOCAB_SIZE = 1024

# ==========================================
# 2. Collect Transcripts into Text File
# ==========================================
print("[1/2] Gathering all LibriSpeech transcripts...")

trans_files = glob.glob(os.path.join(DATA_DIR, "**", "*.trans.txt"), recursive=True)
collected_count = 0

with open(CORPUS_FILE, "w", encoding="utf-8") as out_f:
    for tf in trans_files:
        with open(tf, "r", encoding="utf-8") as in_f:
            for line in in_f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    text = " ".join(parts[1:]).lower()
                    out_f.write(text + "\n")
                    collected_count += 1

print(f"--> Saved {collected_count} normalized sentences into '{CORPUS_FILE}'")

# ==========================================
# 3. Train SentencePiece BPE Model
# ==========================================
print(f"\n[2/2] Training SentencePiece BPE Model (Vocab Size: {VOCAB_SIZE})...")

# CTC loss-ku pad token and blank token alignment-ku support tharum
spm.SentencePieceTrainer.train(
    input=CORPUS_FILE,
    model_prefix=MODEL_PREFIX,
    vocab_size=VOCAB_SIZE,
    model_type="bpe",
    character_coverage=1.0,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    user_defined_symbols=["<blank>"]  # Dedicated CTC blank index
)

print(f"\nTokenizer Training Finished!")
print(f"--> Generated files: '{MODEL_PREFIX}.model' and '{MODEL_PREFIX}.vocab'")

# Quick Test
sp = spm.SentencePieceProcessor()
sp.load(f"{MODEL_PREFIX}.model")
sample_text = "you did great today"
tokens = sp.encode_as_pieces(sample_text)
ids = sp.encode_as_ids(sample_text)
print(f"\nVerification Sample:")
print(f"Text: '{sample_text}'")
print(f"Subwords: {tokens}")
print(f"Token IDs: {ids}")