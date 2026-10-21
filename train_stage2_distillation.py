import os
import math
import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from tqdm import tqdm

from Sinkorn_distillation import SinkhornWassersteinDistillationLoss
from train_stage1_asr_generalized import (
    StudentAcousticModel,
    PositionalEncoding,
    TOKENIZER_MODEL,
    output_length,
    ctc_decode,
    VOCAB_SIZE,
    STUDENT_DIM,
    TEACHER_DIM,
)

# ============================================================
# Configuration
# ============================================================
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
ACCUMULATION_STEPS = 4  # Effective batch size: 16
NUM_EPOCHS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
OT_LAMBDA = 0.1  # Weight for Sinkhorn Distillation Loss
EPSILON = 0.1    # Entropic regularization

STAGE1_CHECKPOINT = os.path.join("checkpoints_stage1", "best_student_asr_generalized.pt")
CHECKPOINT_DIR = "checkpoints_stage2"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
OUTPUT_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_student_asr_stage2.pt")

AUDIO_DIR = os.path.join("Data", "dev-clean", "LibriSpeech", "dev-clean")
PBKD_DIR = "precomputed_pbkd_dev_clean"
EMBEDDINGS_PATH = os.path.join(PBKD_DIR, "llm_embeddings.pt")

print(f"--> Stage 2 Distillation on Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"--> GPU: {torch.cuda.get_device_name(0)}")
print(f"--> Stage 1 Base Checkpoint: {STAGE1_CHECKPOINT}")
print(f"--> Output Checkpoint: {OUTPUT_CHECKPOINT}")


# ============================================================
# 1. Stage 2 Multi-Modal Dataset (Audio + Transcripts + PBKD)
# ============================================================
class Stage2DistillationDataset(Dataset):
    def __init__(self, audio_dir, pbkd_dir, embed_path, sp):
        self.sp = sp
        self.blank_id = sp.piece_to_id("<blank>")

        print("--> Loading static LLaMA Teacher Embeddings (1.05 GB)...")
        self.teacher_embeddings = torch.load(embed_path, map_location="cpu", weights_only=False)

        # 1. Map all audio files
        audio_files = {}
        for ext in ("*.flac", "*.wav"):
            for f in glob.glob(os.path.join(audio_dir, "**", ext), recursive=True):
                audio_files[os.path.splitext(os.path.basename(f))[0]] = f

        # 2. Map all transcripts
        transcripts = {}
        for tf in glob.glob(os.path.join(audio_dir, "**", "*.trans.txt"), recursive=True):
            with open(tf, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        transcripts[parts[0]] = parts[1].lower()

        # 3. Match with PBKD teacher targets
        matched_ids = []
        for utt_id, wav_path in audio_files.items():
            pbkd_file = os.path.join(pbkd_dir, f"{utt_id}.pt")
            if utt_id in transcripts and os.path.isfile(pbkd_file):
                matched_ids.append(utt_id)

        self.utt_ids = sorted(matched_ids)
        self.audio_files = {u: audio_files[u] for u in self.utt_ids}
        self.transcripts = {u: transcripts[u] for u in self.utt_ids}
        self.pbkd_dir = pbkd_dir

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=80
        )
        print(f"--> Successfully loaded {len(self.utt_ids)} paired multi-modal training samples.")

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, idx):
        utt_id = self.utt_ids[idx]
        wav_path = self.audio_files[utt_id]
        text = self.transcripts[utt_id]

        # A. Load audio & compute normalized log-mel
        waveform, sr = torchaudio.load(wav_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        mel = self.mel_transform(waveform).squeeze(0)
        log_mel = torch.log(torch.clamp(mel, min=1e-6)).transpose(0, 1) # [Time, 80]
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)

        # B. Encode target tokens
        student_tokens = torch.tensor(self.sp.encode_as_ids(text), dtype=torch.long)

        # C. Load teacher PBKD targets and compute weighted expected teacher embedding
        pbkd_file = os.path.join(self.pbkd_dir, f"{utt_id}.pt")
        pbkd = torch.load(pbkd_file, map_location="cpu", weights_only=False)
        topk_ids = pbkd["topk_indices"] # [M, 8]
        topk_probs = pbkd["topk_probs"] # [M, 8]

        # Weighted embedding vector: [M, 2048]
        teacher_vectors = (self.teacher_embeddings[topk_ids] * topk_probs.unsqueeze(-1)).sum(dim=1)

        return {
            "utt_id": utt_id,
            "mel": log_mel,
            "tokens": student_tokens,
            "text": text,
            "teacher_vectors": teacher_vectors,
            "teacher_probs": topk_probs.mean(dim=-1), # [M] marginal mass
        }


# ============================================================
# 2. Multi-Modal Batch Collate
# ============================================================
def stage2_collate_fn(batch, pad_id=0, blank_id=4):
    valid_batch = []
    for item in batch:
        mel = item["mel"]
        tokens = item["tokens"]
        repeats = (tokens[1:] == tokens[:-1]).sum().item() if tokens.numel() > 1 else 0
        if output_length(mel.shape[0]) < (tokens.numel() + repeats):
            continue
        valid_batch.append(item)

    if not valid_batch:
        raise ValueError("Empty batch in Stage 2 collate.")

    mels = [b["mel"] for b in valid_batch]
    tokens = [b["tokens"] for b in valid_batch]
    t_vectors = [b["teacher_vectors"] for b in valid_batch]
    t_probs = [b["teacher_probs"] for b in valid_batch]
    texts = [b["text"] for b in valid_batch]

    orig_mel_lens = torch.tensor([m.shape[0] for m in mels], dtype=torch.long)
    out_mel_lens = torch.tensor([output_length(m.shape[0]) for m in mels], dtype=torch.long)
    target_lens = torch.tensor([t.numel() for t in tokens], dtype=torch.long)
    t_lens = torch.tensor([v.shape[0] for v in t_vectors], dtype=torch.long)

    padded_mels = nn.utils.rnn.pad_sequence(mels, batch_first=True, padding_value=0.0)
    padded_tokens = nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=pad_id)
    padded_t_vectors = nn.utils.rnn.pad_sequence(t_vectors, batch_first=True, padding_value=0.0)
    padded_t_probs = nn.utils.rnn.pad_sequence(t_probs, batch_first=True, padding_value=0.0)

    return {
        "mels": padded_mels,
        "orig_mel_lens": orig_mel_lens,
        "out_mel_lens": out_mel_lens,
        "tokens": padded_tokens,
        "target_lens": target_lens,
        "teacher_vectors": padded_t_vectors,
        "teacher_probs": padded_t_probs,
        "teacher_lens": t_lens,
        "texts": texts,
    }


# ============================================================
# 3. Stage 2 Training Loop
# ============================================================
def main():
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL)
    blank_id = sp.piece_to_id("<blank>")
    pad_id = sp.piece_to_id("<pad>")

    dataset = Stage2DistillationDataset(AUDIO_DIR, PBKD_DIR, EMBEDDINGS_PATH, sp)

    # 90/10 Train/Val Split for Stage 2
    total_samples = len(dataset)
    val_size = int(total_samples * 0.1)
    train_size = total_samples - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: stage2_collate_fn(b, pad_id=pad_id, blank_id=blank_id),
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: stage2_collate_fn(b, pad_id=pad_id, blank_id=blank_id),
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )

    print(f"--> Stage 2 Train Samples: {len(train_ds)} | Val Samples: {len(val_ds)}")

    # Initialize model from best generalized Stage 1 checkpoint
    model = StudentAcousticModel(
        input_dim=80,
        encoder_dim=STUDENT_DIM,
        num_encoder_layers=4,
        num_heads=4,
        vocab_size=VOCAB_SIZE,
        teacher_dim=TEACHER_DIM,
        dropout=0.1,
    ).to(DEVICE)

    if os.path.isfile(STAGE1_CHECKPOINT):
        print(f"--> Initializing Student from Stage 1: {STAGE1_CHECKPOINT}")
        state = torch.load(STAGE1_CHECKPOINT, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state, strict=True)
    else:
        print("WARNING: Stage 1 checkpoint not found; starting from scratch.")

    ctc_loss_fn = nn.CTCLoss(blank=blank_id, zero_infinity=True)
    ot_loss_fn = SinkhornWassersteinDistillationLoss(
        student_embed_dim=STUDENT_DIM,
        teacher_embed_dim=TEACHER_DIM,
        epsilon=EPSILON,
        max_iters=10
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    best_val_loss = float("inf")
    print(f"\n--> Starting Stage 2 Multi-Modal Distillation ({NUM_EPOCHS} Epochs)...")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_epoch_loss = 0.0
        total_ctc_loss = 0.0
        total_ot_loss = 0.0
        train_batches = 0
        optimizer.zero_grad(set_to_none=True)
        pending_acc = 0

        pbar = tqdm(train_loader, desc=f"Stage 2 Epoch [{epoch:02d}/{NUM_EPOCHS}] (Train)")
        for step, batch in enumerate(pbar):
            mels = batch["mels"].to(DEVICE, non_blocking=True)
            orig_mel_lens = batch["orig_mel_lens"].to(DEVICE, non_blocking=True)
            out_mel_lens = batch["out_mel_lens"].to(DEVICE, non_blocking=True)
            tokens = batch["tokens"].to(DEVICE, non_blocking=True)
            target_lens = batch["target_lens"].to(DEVICE, non_blocking=True)
            teacher_vectors = batch["teacher_vectors"].to(DEVICE, non_blocking=True)
            teacher_probs = batch["teacher_probs"].to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                # Forward pass: student acoustic features + CTC logits
                x_sub = model.subsampling(mels.unsqueeze(1))
                b, c, t, f = x_sub.shape
                x_flat = x_sub.permute(0, 2, 1, 3).contiguous().view(b, t, c * f)
                x_proj = model.input_proj(x_flat)
                x_pos = model.pos_enc(x_proj)

                # Attention mask
                pos = torch.arange(t, device=DEVICE).unsqueeze(0)
                mask = pos >= out_mel_lens.unsqueeze(1)
                acoustic_feats = model.encoder(x_pos, src_key_padding_mask=mask)
                ctc_logits = model.ctc_head(acoustic_feats)

                # 1. CTC Loss
                log_probs = F.log_softmax(ctc_logits.float(), dim=-1).transpose(0, 1)
                loss_ctc = ctc_loss_fn(log_probs, tokens, out_mel_lens, target_lens)

                # 2. Sinkhorn-Wasserstein Optimal Transport Distillation Loss
                loss_ot = ot_loss_fn(
                    student_vectors=acoustic_feats,
                    teacher_vectors=teacher_vectors,
                    teacher_mass=teacher_probs
                )

                # Multi-task objective
                loss_total = loss_ctc + (OT_LAMBDA * loss_ot)
                loss_step = loss_total / ACCUMULATION_STEPS

            if not torch.isfinite(loss_step):
                optimizer.zero_grad(set_to_none=True)
                pending_acc = 0
                continue

            scaler.scale(loss_step).backward()
            pending_acc += 1

            if pending_acc == ACCUMULATION_STEPS or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                pending_acc = 0

            total_epoch_loss += loss_total.item()
            total_ctc_loss += loss_ctc.item()
            total_ot_loss += loss_ot.item()
            train_batches += 1
            pbar.set_postfix({"CTC": f"{loss_ctc.item():.2f}", "OT": f"{loss_ot.item():.3f}"})

        avg_train_loss = total_epoch_loss / max(1, train_batches)
        avg_train_ctc = total_ctc_loss / max(1, train_batches)
        avg_train_ot = total_ot_loss / max(1, train_batches)

        # ------------------ VALIDATION ------------------
        model.eval()
        val_ctc_sum = 0.0
        val_ot_sum = 0.0
        val_batches = 0
        val_pred = ""
        val_ref = ""

        with torch.no_grad():
            for v_idx, batch in enumerate(val_loader):
                mels = batch["mels"].to(DEVICE, non_blocking=True)
                orig_mel_lens = batch["orig_mel_lens"].to(DEVICE, non_blocking=True)
                out_mel_lens = batch["out_mel_lens"].to(DEVICE, non_blocking=True)
                tokens = batch["tokens"].to(DEVICE, non_blocking=True)
                target_lens = batch["target_lens"].to(DEVICE, non_blocking=True)
                teacher_vectors = batch["teacher_vectors"].to(DEVICE, non_blocking=True)
                teacher_probs = batch["teacher_probs"].to(DEVICE, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                    ctc_logits, sem_feats = model(mels, input_lengths=orig_mel_lens)
                    log_probs = F.log_softmax(ctc_logits.float(), dim=-1).transpose(0, 1)
                    v_ctc = ctc_loss_fn(log_probs, tokens, out_mel_lens, target_lens)
                    v_ot = ot_loss_fn(
                        student_vectors=model.encoder(
                            model.pos_enc(model.input_proj(model.subsampling(mels.unsqueeze(1)).permute(0, 2, 1, 3).contiguous().view(mels.shape[0], -1, STUDENT_DIM * 20))),
                            src_key_padding_mask=(torch.arange(out_mel_lens.max(), device=DEVICE).unsqueeze(0) >= out_mel_lens.unsqueeze(1))
                        ),
                        teacher_vectors=teacher_vectors,
                        teacher_mass=teacher_probs
                    )

                if torch.isfinite(v_ctc) and torch.isfinite(v_ot):
                    val_ctc_sum += v_ctc.item()
                    val_ot_sum += v_ot.item()
                    val_batches += 1

                if v_idx == 0:
                    raw_ids = ctc_logits[0][:out_mel_lens[0]].argmax(dim=-1).cpu().tolist()
                    val_pred = ctc_decode(raw_ids, blank_id, sp)
                    val_ref = batch["texts"][0]

        avg_val_ctc = val_ctc_sum / max(1, val_batches)
        avg_val_ot = val_ot_sum / max(1, val_batches)
        avg_val_loss = avg_val_ctc + (OT_LAMBDA * avg_val_ot)
        scheduler.step()

        print(
            f"\n--> Epoch {epoch:02d} Summary | Train Loss: {avg_train_loss:.4f} (CTC: {avg_train_ctc:.3f}, OT: {avg_train_ot:.3f}) | "
            f"Val Loss: {avg_val_loss:.4f} (CTC: {avg_val_ctc:.3f}, OT: {avg_val_ot:.3f})"
        )
        print(f"    Val REF : {val_ref[:100]}")
        print(f"    Val PRED: {val_pred[:100]}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), OUTPUT_CHECKPOINT)
            print(f"    --> Saved best Stage 2 Distilled Checkpoint: {OUTPUT_CHECKPOINT}")

    print("\nStage 2 Multi-Modal Distillation Training Complete!")
    print(f"Final Distilled Model Saved at: {OUTPUT_CHECKPOINT}")


if __name__ == "__main__":
    main()
