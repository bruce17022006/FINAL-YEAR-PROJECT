import os
import sys
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict

from iemocap_dataloader import IEMOCAPDataset, iemocap_collate_fn
from multitask_conformer_model import MultiTaskConformerASR_SER

sys.stdout.reconfigure(encoding='utf-8')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRETRAINED_CONFORMER_CKPT = os.path.join("checkpoints_conformer", "best_conformer_stage2_distilled.pt")
OUTPUT_CKPT_DIR = "checkpoints_multitask"
BEST_CKPT_PATH = os.path.join(OUTPUT_CKPT_DIR, "best_multitask_conformer_iemocap.pt")

# Hyperparameters
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-2
EMO_LOSS_WEIGHT = 0.5  # lambda_emo
BLANK_ID = 4
NUM_CLASSES = 4
LABEL_NAMES = ["neutral", "angry", "sad", "happy"]

def compute_wa_ua(all_preds, all_labels):
    # Weighted Accuracy (WA) = overall accuracy
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    wa = (correct / len(all_labels)) * 100.0 if len(all_labels) > 0 else 0.0

    # Unweighted Accuracy (UA) = average of recall per class
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    for p, l in zip(all_preds, all_labels):
        class_total[l] += 1
        if p == l:
            class_correct[l] += 1

    recalls = [class_correct[c] / class_total[c] for c in range(NUM_CLASSES) if class_total[c] > 0]
    ua = (sum(recalls) / len(recalls)) * 100.0 if len(recalls) > 0 else 0.0

    return wa, ua, class_correct, class_total

def evaluate(model, val_loader, ctc_loss_fn, emo_loss_fn):
    model.eval()
    total_loss, total_ctc, total_emo = 0.0, 0.0, 0.0
    all_preds = []
    all_labels = []
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue

            mels = batch["mels"].to(DEVICE)
            in_lens = batch["input_lengths"].to(DEVICE)
            targets = batch["targets"].to(DEVICE)
            target_lens = batch["target_lengths"].to(DEVICE)
            emo_labels = batch["emotion_labels"].to(DEVICE)

            ctc_logits, emo_logits, out_lens = model(mels, in_lens)

            # CTC Loss
            log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)  # [T', B, Vocab]
            l_ctc = ctc_loss_fn(log_probs, targets, out_lens, target_lens)

            # Emotion Cross-Entropy Loss
            l_emo = emo_loss_fn(emo_logits, emo_labels)

            loss = l_ctc + (EMO_LOSS_WEIGHT * l_emo)

            total_loss += loss.item()
            total_ctc += l_ctc.item()
            total_emo += l_emo.item()
            num_batches += 1

            preds = emo_logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(emo_labels.cpu().tolist())

    wa, ua, class_correct, class_total = compute_wa_ua(all_preds, all_labels)
    avg_loss = total_loss / max(1, num_batches)
    avg_ctc = total_ctc / max(1, num_batches)
    avg_emo = total_emo / max(1, num_batches)

    return avg_loss, avg_ctc, avg_emo, wa, ua

def train_multitask():
    print("=" * 80)
    print("STARTING MULTI-TASK CONFORMER TRAINING (ASR + SPEECH EMOTION RECOGNITION)")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Pretrained Conformer Checkpoint: {PRETRAINED_CONFORMER_CKPT}")
    print(f"Train Sessions: Session1, Session2, Session3, Session4")
    print(f"Validation Session: Session5 (LOSO Benchmark)")

    # 1. Datasets & Loaders
    train_dataset = IEMOCAPDataset(sessions=["Session1", "Session2", "Session3", "Session4"])
    val_dataset = IEMOCAPDataset(sessions=["Session5"])
    print(f"--> Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=iemocap_collate_fn, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=iemocap_collate_fn, num_workers=0, pin_memory=True
    )

    # 2. Model
    model = MultiTaskConformerASR_SER(
        input_dim=80,
        encoder_dim=256,
        num_layers=6,
        num_heads=4,
        vocab_size=1024,
        teacher_dim=2048,
        num_emotions=4,
        dropout=0.1
    ).to(DEVICE)

    # Load pretrained weights from Conformer Stage 2 Distillation
    if os.path.exists(PRETRAINED_CONFORMER_CKPT):
        model.load_pretrained_conformer(PRETRAINED_CONFORMER_CKPT, device=DEVICE)
    else:
        print("Warning: Pretrained checkpoint not found, training from scratch!")

    # 3. Loss Functions & Optimizer
    ctc_loss_fn = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
    emo_loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_val_ua = 0.0

    print("\n--> Starting Training Loop...")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss, epoch_ctc, epoch_emo = 0.0, 0.0, 0.0
        train_preds, train_labels = [], []
        step_count = 0
        start_t = time.time()

        for batch in train_loader:
            if batch is None:
                continue

            mels = batch["mels"].to(DEVICE)
            in_lens = batch["input_lengths"].to(DEVICE)
            targets = batch["targets"].to(DEVICE)
            target_lens = batch["target_lengths"].to(DEVICE)
            emo_labels = batch["emotion_labels"].to(DEVICE)

            optimizer.zero_grad()

            ctc_logits, emo_logits, out_lens = model(mels, in_lens)

            # CTC Loss
            log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
            l_ctc = ctc_loss_fn(log_probs, targets, out_lens, target_lens)

            # Emotion Cross-Entropy Loss
            l_emo = emo_loss_fn(emo_logits, emo_labels)

            total_loss = l_ctc + (EMO_LOSS_WEIGHT * l_emo)

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_ctc += l_ctc.item()
            epoch_emo += l_emo.item()
            step_count += 1

            train_preds.extend(emo_logits.argmax(dim=-1).cpu().tolist())
            train_labels.extend(emo_labels.cpu().tolist())

        scheduler.step()
        elapsed = time.time() - start_t

        train_wa, train_ua, _, _ = compute_wa_ua(train_preds, train_labels)
        avg_train_loss = epoch_loss / max(1, step_count)
        avg_train_ctc = epoch_ctc / max(1, step_count)
        avg_train_emo = epoch_emo / max(1, step_count)

        # Validation
        val_loss, val_ctc, val_emo, val_wa, val_ua = evaluate(model, val_loader, ctc_loss_fn, emo_loss_fn)

        print(f"Epoch {epoch:02d}/{NUM_EPOCHS:02d} [{elapsed:.1f}s] | "
              f"Train: Loss={avg_train_loss:.3f} (CTC={avg_train_ctc:.3f}, Emo={avg_train_emo:.3f}, WA={train_wa:.1f}%) | "
              f"Val: Loss={val_loss:.3f} (CTC={val_ctc:.3f}, Emo={val_emo:.3f}) | "
              f"Val WA={val_wa:.2f}%, Val UA={val_ua:.2f}%")

        if val_ua > best_val_ua:
            best_val_ua = val_ua
            torch.save(model.state_dict(), BEST_CKPT_PATH)
            print(f"  --> [NEW BEST CHECKPOINT] Saved to: {BEST_CKPT_PATH} (Val UA: {val_ua:.2f}%, WA: {val_wa:.2f}%)")

    print("\n" + "=" * 80)
    print(f"MULTI-TASK TRAINING COMPLETE! Best Val UA: {best_val_ua:.2f}%")
    print(f"Saved Checkpoint: {BEST_CKPT_PATH}")
    print("=" * 80)

if __name__ == "__main__":
    train_multitask()
