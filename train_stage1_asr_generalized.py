import os
import math
import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
ACCUMULATION_STEPS = 2  # Effective batch size: 16
NUM_EPOCHS = 15
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.1
VOCAB_SIZE = 1024
STUDENT_DIM = 256
TEACHER_DIM = 2048
VAL_SPLIT = 0.10  # 10% held-out validation set for true generalization tracking

CACHE_DIR = "precomputed_pbkd_train_clean"
TOKENIZER_MODEL = "student_bpe_1024.model"
CHECKPOINT_DIR = "checkpoints_stage1"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
OUTPUT_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_student_asr_generalized.pt")

print(f"--> Training Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"--> GPU: {torch.cuda.get_device_name(0)}")
print(f"--> Dataset Source: {CACHE_DIR}")
print(f"--> Checkpoint Destination: {OUTPUT_CHECKPOINT}")


# ============================================================
# 1. SpecAugment (Applied only to training data)
# ============================================================
class SpecAugment(nn.Module):
    def __init__(self, freq_mask_param=15, time_mask_param=35):
        super().__init__()
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)

    def forward(self, mel_spec):
        # mel_spec shape: [T, F] -> transpose to [1, F, T] for torchaudio transforms
        x = mel_spec.transpose(0, 1).unsqueeze(0)
        x = self.freq_mask(x)
        x = self.time_mask(x)
        return x.squeeze(0).transpose(0, 1)


# ============================================================
# 2. Positional Encoding
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# ============================================================
# 3. Generalized Student Acoustic Model
# ============================================================
class StudentAcousticModel(nn.Module):
    def __init__(
        self,
        input_dim=80,
        encoder_dim=256,
        num_encoder_layers=4,
        num_heads=4,
        vocab_size=1024,
        teacher_dim=2048,
        dropout=0.1,
    ):
        super().__init__()
        # Conv2D Subsampling: time downsampling factor = 2, freq downsampling factor = 4
        self.subsampling = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, encoder_dim, kernel_size=3, stride=(1, 2), padding=1),
            nn.ReLU(),
        )
        self.input_proj = nn.Linear(encoder_dim * (input_dim // 4), encoder_dim)
        self.pos_enc = PositionalEncoding(encoder_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=num_heads,
            dim_feedforward=encoder_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.ctc_head = nn.Linear(encoder_dim, vocab_size)
        self.semantic_proj = nn.Linear(encoder_dim, teacher_dim, bias=False)

    def forward(self, mel_inputs, input_lengths=None):
        # mel_inputs: [Batch, Time, Mel_Bins]
        x = self.subsampling(mel_inputs.unsqueeze(1))
        batch_size, channels, time_steps, freq_bins = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(
            batch_size, time_steps, channels * freq_bins
        )
        x = self.input_proj(x)
        x = self.pos_enc(x)

        # Dynamic Key Padding Mask: ensures valid audio frames never attend to zero padding
        src_key_padding_mask = None
        if input_lengths is not None:
            output_lengths = torch.div(input_lengths + 1, 2, rounding_mode="floor")
            positions = torch.arange(time_steps, device=x.device).unsqueeze(0)
            src_key_padding_mask = positions >= output_lengths.unsqueeze(1)

        acoustic_features = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        ctc_logits = self.ctc_head(acoustic_features)
        semantic_embeddings = self.semantic_proj(acoustic_features)
        return ctc_logits, semantic_embeddings


# ============================================================
# 4. Dataset & Splitting
# ============================================================
class PrecomputedASRDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = torch.load(self.file_paths[idx], map_location="cpu", weights_only=True)
        return (
            data["student_mel"].float(),
            torch.as_tensor(data["student_target_ids"], dtype=torch.long),
            data.get("text", "")
        )


def output_length(input_frames):
    return (input_frames + 1) // 2


def collate_fn(batch, spec_augment=None, blank_id=4, pad_id=0):
    valid_batch = []
    for mel, target, text in batch:
        if mel.ndim != 2 or mel.shape[1] != 80 or target.numel() == 0:
            continue
        # CTC constraint check
        repeats = (target[1:] == target[:-1]).sum().item() if target.numel() > 1 else 0
        if output_length(mel.shape[0]) < (target.numel() + repeats):
            continue
        valid_batch.append((mel, target, text))

    if not valid_batch:
        raise ValueError("Empty batch after length validation.")

    mels = [item[0] for item in valid_batch]
    targets = [item[1] for item in valid_batch]
    texts = [item[2] for item in valid_batch]

    orig_mel_lens = torch.tensor([m.shape[0] for m in mels], dtype=torch.long)
    out_lens = torch.tensor([output_length(m.shape[0]) for m in mels], dtype=torch.long)
    target_lens = torch.tensor([t.numel() for t in targets], dtype=torch.long)

    # SpecAugment applied per unpadded sample
    if spec_augment is not None:
        mels = [spec_augment(m) for m in mels]

    padded_mels = nn.utils.rnn.pad_sequence(mels, batch_first=True, padding_value=0.0)
    padded_targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=pad_id)

    return {
        "mels": padded_mels,
        "mel_lengths": out_lens,
        "orig_mel_lengths": orig_mel_lens,
        "tokens": padded_targets,
        "target_lengths": target_lens,
        "texts": texts,
    }


def ctc_decode(ids, blank_id, sp):
    collapsed = []
    prev = None
    for tid in ids:
        if tid != prev and tid != blank_id:
            collapsed.append(tid)
        prev = tid
    return sp.decode_ids(collapsed)


# ============================================================
# 5. Main Training & Validation Loop
# ============================================================
def main():
    sp = spm.SentencePieceProcessor()
    if not sp.load(TOKENIZER_MODEL):
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_MODEL}")
    blank_id = sp.piece_to_id("<blank>")
    pad_id = sp.piece_to_id("<pad>")

    all_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.pt")))
    if not all_files:
        raise FileNotFoundError(f"No cached .pt files found in {CACHE_DIR}")

    # Shuffle with fixed seed and split into Train / Val
    random.seed(SEED)
    random.shuffle(all_files)
    val_count = int(len(all_files) * VAL_SPLIT)
    val_files = all_files[:val_count]
    train_files = all_files[val_count:]

    print(f"--> Total utterances: {len(all_files)}")
    print(f"--> Training utterances: {len(train_files)}")
    print(f"--> Validation utterances (held-out for generalization): {len(val_files)}")

    spec_aug = SpecAugment()
    train_dataset = PrecomputedASRDataset(train_files)
    val_dataset = PrecomputedASRDataset(val_files)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, spec_augment=spec_aug, blank_id=blank_id, pad_id=pad_id),
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, spec_augment=None, blank_id=blank_id, pad_id=pad_id),
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )

    model = StudentAcousticModel(
        input_dim=80,
        encoder_dim=STUDENT_DIM,
        num_encoder_layers=4,
        num_heads=4,
        vocab_size=VOCAB_SIZE,
        teacher_dim=TEACHER_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    ctc_loss_fn = nn.CTCLoss(blank=blank_id, zero_infinity=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    best_val_loss = float("inf")
    print(f"\n--> Starting {NUM_EPOCHS} Training Epochs...")

    for epoch in range(1, NUM_EPOCHS + 1):
        # ------------------ TRAINING PHASE ------------------
        model.train()
        total_train_loss = 0.0
        train_batches = 0
        optimizer.zero_grad(set_to_none=True)
        pending_accumulation = 0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{NUM_EPOCHS}] (Train)")
        for step, batch in enumerate(pbar):
            mels = batch["mels"].to(DEVICE, non_blocking=True)
            mel_lengths = batch["mel_lengths"].to(DEVICE, non_blocking=True)
            orig_mel_lens = batch["orig_mel_lengths"].to(DEVICE, non_blocking=True)
            tokens = batch["tokens"].to(DEVICE, non_blocking=True)
            target_lengths = batch["target_lengths"].to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                ctc_logits, _ = model(mels, input_lengths=orig_mel_lens)
                log_probs = F.log_softmax(ctc_logits.float(), dim=-1).transpose(0, 1)
                loss_ctc = ctc_loss_fn(log_probs, tokens, mel_lengths, target_lengths)
                loss = loss_ctc / ACCUMULATION_STEPS

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                pending_accumulation = 0
                continue

            scaler.scale(loss).backward()
            pending_accumulation += 1

            if pending_accumulation == ACCUMULATION_STEPS or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                pending_accumulation = 0

            total_train_loss += loss_ctc.item()
            train_batches += 1
            pbar.set_postfix({"CTC": f"{loss_ctc.item():.3f}"})

        avg_train_loss = total_train_loss / max(1, train_batches)

        # ------------------ VALIDATION PHASE ------------------
        model.eval()
        total_val_loss = 0.0
        val_batches = 0
        sample_printed = False

        with torch.no_grad():
            for batch in val_loader:
                mels = batch["mels"].to(DEVICE, non_blocking=True)
                mel_lengths = batch["mel_lengths"].to(DEVICE, non_blocking=True)
                orig_mel_lens = batch["orig_mel_lengths"].to(DEVICE, non_blocking=True)
                tokens = batch["tokens"].to(DEVICE, non_blocking=True)
                target_lengths = batch["target_lengths"].to(DEVICE, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                    ctc_logits, _ = model(mels, input_lengths=orig_mel_lens)
                    log_probs = F.log_softmax(ctc_logits.float(), dim=-1).transpose(0, 1)
                    val_ctc = ctc_loss_fn(log_probs, tokens, mel_lengths, target_lengths)

                if torch.isfinite(val_ctc):
                    total_val_loss += val_ctc.item()
                    val_batches += 1

                # Live generalization check on first validation sample
                if not sample_printed and len(batch["texts"]) > 0:
                    raw_ids = ctc_logits[0][:mel_lengths[0]].argmax(dim=-1).cpu().tolist()
                    pred_text = ctc_decode(raw_ids, blank_id, sp)
                    ref_text = batch["texts"][0]
                    sample_printed = True

        avg_val_loss = total_val_loss / max(1, val_batches)
        scheduler.step()

        print(
            f"\n--> Epoch {epoch:02d} Summary | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}"
        )
        print(f"    Val REF : {ref_text[:110]}")
        print(f"    Val PRED: {pred_text[:110]}")

        # Checkpoint based on held-out validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), OUTPUT_CHECKPOINT)
            print(f"    --> Saved best generalized checkpoint (Val Loss: {best_val_loss:.4f}) -> {OUTPUT_CHECKPOINT}")

    print("\nTraining completed successfully!")
    print(f"Best generalized model saved at: {OUTPUT_CHECKPOINT}")


if __name__ == "__main__":
    main()
