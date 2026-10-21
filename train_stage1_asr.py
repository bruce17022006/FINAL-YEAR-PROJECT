import os
import math
import glob
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
ACCUMULATION_STEPS = 4
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
VOCAB_SIZE = 1024
STUDENT_DIM = 256
TEACHER_DIM = 2048

PBKD_DIR = "precomputed_pbkd_train_clean"
TOKENIZER_MODEL = "student_bpe_1024.model"
CHECKPOINT_DIR = "checkpoints_stage1"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"--> Training on Device: {DEVICE}")
print(f"--> Dataset Source: {PBKD_DIR}")


class SpecAugment(nn.Module):
    def __init__(self, freq_mask_param=15, time_mask_param=35):
        super().__init__()
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)

    def forward(self, mel_spec):
        x = mel_spec.transpose(1, 2)
        x = self.freq_mask(x)
        x = self.time_mask(x)
        return x.transpose(1, 2)


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
        if x.size(1) > self.pe.size(1):
            raise ValueError(
                f"{x.size(1)} frames exceed positional encoding limit "
                f"{self.pe.size(1)}; increase max_len."
            )
        return x + self.pe[:, :x.size(1), :]


class StudentAcousticModel(nn.Module):
    def __init__(
        self,
        input_dim=80,
        encoder_dim=256,
        num_encoder_layers=4,
        num_heads=4,
        vocab_size=1024,
        teacher_dim=2048,
        blank_id=4,
    ):
        super().__init__()
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
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.ctc_head = nn.Linear(encoder_dim, vocab_size)
        self.semantic_proj = nn.Linear(encoder_dim, teacher_dim, bias=False)
        with torch.no_grad():
            self.ctc_head.bias.fill_(0.0)
            self.ctc_head.bias[blank_id] = -5.0

    def forward(self, mel_inputs):
        x = self.subsampling(mel_inputs.unsqueeze(1))
        batch_size, channels, time_steps, freq_bins = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(
            batch_size, time_steps, channels * freq_bins
        )
        x = self.input_proj(x)
        x = self.pos_enc(x)
        acoustic_features = self.encoder(x)
        return self.ctc_head(acoustic_features), self.semantic_proj(acoustic_features)


class FastPrecomputedDataset(Dataset):
    def __init__(self, cache_dir):
        self.cache_files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
        if not self.cache_files:
            raise FileNotFoundError(
                f"No .pt cache files found in {cache_dir}. Check PBKD_DIR."
            )
        print(f"--> Loaded {len(self.cache_files)} cached tensors from {cache_dir}")

    def __len__(self):
        return len(self.cache_files)

    def __getitem__(self, idx):
        data = torch.load(self.cache_files[idx], map_location="cpu", weights_only=True)
        return (
            data["student_mel"].float(),
            torch.as_tensor(data["student_target_ids"], dtype=torch.long),
        )


def output_length(input_frames):
    # Time stride is 2 in Conv1 and 1 in Conv2: output length = ceil(T / 2).
    return (input_frames + 1) // 2


def collate_fn(batch, spec_augment=None, blank_id=4, pad_id=0):
    valid_batch = []
    for mel, target in batch:
        if mel.ndim != 2 or mel.shape[1] != 80:
            raise ValueError(f"Expected mel shape [T, 80], got {tuple(mel.shape)}")
        if target.numel() == 0:
            continue
        if (target == blank_id).any().item():
            raise ValueError("A cached CTC target contains the blank token ID.")
        if (target < 0).any().item() or (target >= VOCAB_SIZE).any().item():
            raise ValueError("A target ID is outside the vocabulary range.")

        # CTC requires an extra frame for adjacent repeated target tokens.
        repeats = (
            (target[1:] == target[:-1]).sum().item() if target.numel() > 1 else 0
        )
        min_frames = target.numel() + repeats
        if output_length(mel.shape[0]) < min_frames:
            continue
        valid_batch.append((mel, target))

    if not valid_batch:
        raise ValueError("No valid samples in batch; inspect input/target lengths.")

    mels = [item[0] for item in valid_batch]
    targets = [item[1] for item in valid_batch]
    mel_lengths = torch.tensor([output_length(m.shape[0]) for m in mels], dtype=torch.long)
    target_lengths = torch.tensor([t.numel() for t in targets], dtype=torch.long)
    padded_mels = nn.utils.rnn.pad_sequence(mels, batch_first=True, padding_value=0.0)

    if spec_augment is not None:
        padded_mels = spec_augment(padded_mels)

    padded_targets = nn.utils.rnn.pad_sequence(
        targets, batch_first=True, padding_value=pad_id
    )
    return {
        "mels": padded_mels,
        "mel_lengths": mel_lengths,
        "tokens": padded_targets,
        "target_lengths": target_lengths,
    }


def inspect_cached_targets(dataset, sp, blank_id, pad_id, count=5):
    print("\n--- Cached Target Diagnostic ---")
    for i in range(min(count, len(dataset))):
        mel, target = dataset[i]
        print(f"\nSample {i + 1}: {os.path.basename(dataset.cache_files[i])}")
        print("Mel shape:", tuple(mel.shape))
        print("Target length:", target.numel())
        print("Contains blank ID:", bool((target == blank_id).any().item()))
        print("Contains pad ID inside target:", bool((target == pad_id).any().item()))
        print("Decoded target:", sp.decode_ids(target.tolist())[:150])


def main():
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    if not os.path.isfile(TOKENIZER_MODEL):
        raise FileNotFoundError(f"Tokenizer model not found: {TOKENIZER_MODEL}")

    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL)
    blank_id = sp.piece_to_id("<blank>")
    pad_id = sp.piece_to_id("<pad>")

    if blank_id != 4:
        raise ValueError(f"Expected blank ID 4, got {blank_id}")
    if sp.get_piece_size() != VOCAB_SIZE:
        raise ValueError(
            f"Tokenizer has {sp.get_piece_size()} tokens; expected {VOCAB_SIZE}."
        )

    model = StudentAcousticModel(
        input_dim=80,
        encoder_dim=STUDENT_DIM,
        num_encoder_layers=4,
        vocab_size=VOCAB_SIZE,
        teacher_dim=TEACHER_DIM,
        blank_id=blank_id,
    ).to(DEVICE)

    dataset = FastPrecomputedDataset(PBKD_DIR)
    inspect_cached_targets(dataset, sp, blank_id, pad_id)
    

    spec_aug = SpecAugment()
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(
            b, spec_augment=spec_aug, blank_id=blank_id, pad_id=pad_id
        ),
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )

    ctc_loss_fn = nn.CTCLoss(blank=blank_id, zero_infinity=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    best_loss = float("inf")
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_student_asr_corrected.pt")
    print(f"\n--> Starting training for {NUM_EPOCHS} epochs...")
    print(f"--> SpecAugment: Active | AMP enabled: {DEVICE == 'cuda'}")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_epoch_loss = 0.0
        valid_batches = 0
        optimizer.zero_grad(set_to_none=True)
        pending_accumulation = 0

        pbar = tqdm(loader, desc=f"Epoch [{epoch:02d}/{NUM_EPOCHS}]")
        for step, batch in enumerate(pbar):
            mels = batch["mels"].to(DEVICE, non_blocking=True)
            mel_lengths = batch["mel_lengths"].to(DEVICE, non_blocking=True)
            tokens = batch["tokens"].to(DEVICE, non_blocking=True)
            target_lengths = batch["target_lengths"].to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                ctc_logits, _ = model(mels)
                input_lengths = torch.clamp(mel_lengths, max=ctc_logits.size(1))
                log_probs = F.log_softmax(ctc_logits.float(), dim=-1).transpose(0, 1)
                loss_ctc = ctc_loss_fn(
                    log_probs, tokens, input_lengths, target_lengths
                )
                loss = loss_ctc / ACCUMULATION_STEPS

            if not torch.isfinite(loss):
                print(f"\nWarning: non-finite loss at batch {step}; skipped.")
                optimizer.zero_grad(set_to_none=True)
                pending_accumulation = 0
                continue

            scaler.scale(loss).backward()
            pending_accumulation += 1

            if pending_accumulation == ACCUMULATION_STEPS or (step + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                pending_accumulation = 0

            total_epoch_loss += loss_ctc.item()
            valid_batches += 1
            pbar.set_postfix({"CTC_Loss": f"{loss_ctc.item():.3f}"})

        scheduler.step()
        if valid_batches:
            avg_loss = total_epoch_loss / valid_batches
            print(
                f"--> Epoch {epoch:02d} Finished | Avg CTC Loss: {avg_loss:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"--> Saved corrected checkpoint: {best_model_path}")


if __name__ == "__main__":
    main()
