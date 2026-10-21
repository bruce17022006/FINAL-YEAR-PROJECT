import os
import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader

# Small diagnostic experiment only. Does NOT overwrite existing checkpoints.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "precomputed_pbkd_train_clean"
TOKENIZER_MODEL = "student_bpe_1024.model"
CHECKPOINT_DIR = "checkpoints_stage1"
BASE_CHECKPOINTS = [
    os.path.join(CHECKPOINT_DIR, "best_student_asr_corrected.pt"),
    os.path.join(CHECKPOINT_DIR, "best_student_asr.pt"),
]
OUTPUT_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "phase2_overfit_test.pt")

SEED = 1701
NUM_SAMPLES = 8
BATCH_SIZE = 4
STEPS = 120
LEARNING_RATE = 1e-4
VOCAB_SIZE = 1024
BLANK_ID = 4
PAD_ID = 0

torch.manual_seed(SEED)
random.seed(SEED)

class SmallCache(Dataset):
    def __init__(self, cache_dir, n):
        self.paths = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))[:n]
        if len(self.paths) < n:
            raise RuntimeError(f"Expected at least {n} cache files, found {len(self.paths)}")
        self.items = []
        for p in self.paths:
            d = torch.load(p, map_location="cpu", weights_only=True)
            mel = d["student_mel"].float()
            target = torch.as_tensor(d["student_target_ids"], dtype=torch.long)
            if target.numel() == 0 or (target == BLANK_ID).any() or (target == PAD_ID).any():
                raise ValueError(f"Invalid target in {p}")
            repeats = (target[1:] == target[:-1]).sum().item() if target.numel() > 1 else 0
            if (mel.shape[0] + 1) // 2 < target.numel() + repeats:
                raise ValueError(f"CTC alignment too short for {p}")
            self.items.append((p, mel, target, d.get("text", "")))
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def collate(batch):
    paths, mels, targets, texts = zip(*batch)
    original_lengths = torch.tensor([m.shape[0] for m in mels], dtype=torch.long)
    out_lengths = (original_lengths + 1) // 2
    target_lengths = torch.tensor([t.numel() for t in targets], dtype=torch.long)
    padded_mels = nn.utils.rnn.pad_sequence(mels, batch_first=True, padding_value=0.0)
    padded_targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=PAD_ID)
    return paths, padded_mels, original_lengths, out_lengths, padded_targets, target_lengths, texts

def greedy_decode(ids, sp):
    out, prev = [], None
    for token in ids:
        if token != prev and token != BLANK_ID:
            out.append(token)
        prev = token
    return sp.decode_ids(out)

def main():
    if not torch.cuda.is_available():
        print("WARNING: CUDA not found; running on CPU may be slow.")
    if not os.path.isdir(CACHE_DIR):
        raise FileNotFoundError(f"Cache directory missing: {CACHE_DIR}")
    if not os.path.isfile(TOKENIZER_MODEL):
        raise FileNotFoundError(f"Tokenizer missing: {TOKENIZER_MODEL}")

    from train_stage1_asr import StudentAcousticModel

    resume = next((p for p in BASE_CHECKPOINTS if os.path.isfile(p)), None)
    if resume is None:
        raise FileNotFoundError("No baseline checkpoint found; existing weights are required.")
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL)

    ds = SmallCache(CACHE_DIR, NUM_SAMPLES)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    model = StudentAcousticModel(
        input_dim=80, encoder_dim=256, num_encoder_layers=4,
        vocab_size=VOCAB_SIZE, teacher_dim=2048, blank_id=BLANK_ID
    ).to(DEVICE)
    model.load_state_dict(torch.load(resume, map_location=DEVICE, weights_only=True), strict=True)
    model.train()
    print(f"Device: {DEVICE}")
    print(f"Starting checkpoint: {resume}")
    print(f"Fixed samples: {len(ds)} | optimization steps: {STEPS}")
    print("This is a diagnostic only; no original checkpoint/cache is modified.")

    # Evaluate before updates, using deterministic unaugmented cached features.
    def evaluate():
        model.eval()
        rows, loss_sum = [], 0.0
        with torch.inference_mode():
            for paths, mels, original_lengths, out_lengths, targets, target_lengths, texts in loader:
                mels = mels.to(DEVICE)
                out_lengths = out_lengths.to(DEVICE)
                targets = targets.to(DEVICE)
                target_lengths = target_lengths.to(DEVICE)
                logits, _ = model(mels)
                lp = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)
                loss = nn.CTCLoss(blank=BLANK_ID, reduction="none", zero_infinity=False)(
                    lp, targets, torch.clamp(out_lengths, max=logits.size(1)), target_lengths
                )
                loss_sum += loss.mean().item()
                for i, path in enumerate(paths):
                    pred = greedy_decode(logits[i].argmax(-1).tolist(), sp)
                    rows.append((os.path.basename(path), texts[i], pred))
        model.train()
        return loss_sum / max(1, len(loader)), rows

    before_loss, before_rows = evaluate()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    ctc = nn.CTCLoss(blank=BLANK_ID, zero_infinity=False)
    iterator = iter(loader)
    running = []
    for step in range(1, STEPS + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        paths, mels, original_lengths, out_lengths, targets, target_lengths, texts = batch
        mels = mels.to(DEVICE)
        out_lengths = out_lengths.to(DEVICE)
        targets = targets.to(DEVICE)
        target_lengths = target_lengths.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(mels)
        log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)
        input_lengths = torch.clamp(out_lengths, max=logits.size(1))
        loss = ctc(log_probs, targets, input_lengths, target_lengths)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite CTC loss at step {step}: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running.append(loss.item())
        if step % 20 == 0:
            print(f"Step {step:03d}/{STEPS} | batch CTC loss: {loss.item():.4f} | recent mean: {sum(running[-20:])/len(running[-20:]):.4f}")

    after_loss, after_rows = evaluate()
    torch.save(model.state_dict(), OUTPUT_CHECKPOINT)
    print("\n========== OVERFIT TEST RESULT ==========")
    print(f"Fixed-set mean CTC loss before: {before_loss:.4f}")
    print(f"Fixed-set mean CTC loss after : {after_loss:.4f}")
    print(f"Relative loss reduction      : {(before_loss-after_loss)/max(abs(before_loss), 1e-8)*100:.2f}%")
    before_map = {r[0]: r for r in before_rows}
    for name, text, pred in after_rows:
        print(f"\nSample: {name}")
        print("Reference:", text)
        print("Before   :", before_map[name][2])
        print("After    :", pred)
    print(f"\nDiagnostic checkpoint saved to: {OUTPUT_CHECKPOINT}")
    print("Do not promote this checkpoint unless a separate held-out evaluation improves.")

if __name__ == "__main__":
    main()
