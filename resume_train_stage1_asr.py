import os
import math
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from tqdm import tqdm

from Sinkorn_distillation import SinkhornWassersteinDistillationLoss

# ==========================================
# 1. Configurations
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
START_EPOCH = 26
TOTAL_EPOCHS = 45          # Additional 20 epochs
LEARNING_RATE = 6e-5       # Gentle fine-tuning LR
ALPHA_OT = 0.3
VOCAB_SIZE = 1024
TEACHER_DIM = 2048
STUDENT_DIM = 256

DATA_DIR = os.path.join("Data", "dev-clean", "LibriSpeech", "dev-clean")
PBKD_DIR = "precomputed_pbkd_dev_clean"
TOKENIZER_MODEL = "student_bpe_1024.model"
CHECKPOINT_DIR = "checkpoints_stage1"
BASE_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_student_asr.pt")

print(f"--> Training on Device: {DEVICE}")

# ==========================================
# 2. Positional Encoding
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# ==========================================
# 3. Student Architecture
# ==========================================
class StudentAcousticModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 80,
        encoder_dim: int = 256,
        num_encoder_layers: int = 4,
        num_heads: int = 4,
        vocab_size: int = 1024,
        teacher_dim: int = 2048,
        blank_id: int = 4
    ):
        super().__init__()
        
        self.subsampling = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, encoder_dim, kernel_size=3, stride=(1, 2), padding=1),
            nn.ReLU()
        )
        self.input_proj = nn.Linear(encoder_dim * (input_dim // 4), encoder_dim)
        self.pos_enc = PositionalEncoding(encoder_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=num_heads,
            dim_feedforward=encoder_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.ctc_head = nn.Linear(encoder_dim, vocab_size)
        self.semantic_proj = nn.Linear(encoder_dim, teacher_dim, bias=False)

    def forward(self, mel_inputs):
        x = mel_inputs.unsqueeze(1)
        x = self.subsampling(x)
        
        B, C, T, F_dim = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, C * F_dim)
        x = self.input_proj(x)
        x = self.pos_enc(x)

        acoustic_features = self.encoder(x)
        ctc_logits = self.ctc_head(acoustic_features)
        semantic_embeddings = self.semantic_proj(acoustic_features)

        return ctc_logits, semantic_embeddings

# ==========================================
# 4. Dataset & Collate Function
# ==========================================
class ASRStage1Dataset(Dataset):
    def __init__(self, data_dir, pbkd_dir, tokenizer_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)
        self.blank_id = self.sp.piece_to_id("<blank>")

        embed_path = os.path.join(pbkd_dir, "llm_embeddings.pt")
        self.teacher_embeddings = torch.load(embed_path, map_location="cpu", weights_only=True)

        self.audio_files = {}
        for ext in ("*.flac", "*.wav"):
            for f in glob.glob(os.path.join(data_dir, "**", ext), recursive=True):
                utt_id = os.path.splitext(os.path.basename(f))[0]
                self.audio_files[utt_id] = f

        self.transcripts = {}
        trans_files = glob.glob(os.path.join(data_dir, "**", "*.trans.txt"), recursive=True)
        for tf in trans_files:
            with open(tf, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        self.transcripts[parts[0]] = " ".join(parts[1:]).lower()

        self.utt_ids = [
            utt for utt in self.audio_files
            if utt in self.transcripts and os.path.exists(os.path.join(pbkd_dir, f"{utt}.pt"))
        ]
        print(f"--> Dataset Loaded: {len(self.utt_ids)} cached pairs found.")

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=80
        )

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, idx):
        utt_id = self.utt_ids[idx]
        wav_path = self.audio_files[utt_id]

        waveform, sr = torchaudio.load(wav_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        mel = self.mel_transform(waveform).squeeze(0)
        log_mel = torch.log(torch.clamp(mel, min=1e-6)).transpose(0, 1)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)

        text = self.transcripts[utt_id]
        student_tokens = torch.tensor(self.sp.encode_as_ids(text), dtype=torch.long)

        cached = torch.load(os.path.join(PBKD_DIR, f"{utt_id}.pt"), map_location="cpu", weights_only=True)
        return {
            "log_mel": log_mel,
            "student_tokens": student_tokens,
            "teacher_vectors": self.teacher_embeddings[cached["topk_indices"]],
            "teacher_probs": cached["topk_probs"]
        }

def collate_fn(batch):
    valid_batch = [item for item in batch if (item["log_mel"].shape[0] // 2) > len(item["student_tokens"])]
    if not valid_batch:
        valid_batch = batch

    mels = [item["log_mel"] for item in valid_batch]
    tokens = [item["student_tokens"] for item in valid_batch]
    t_vecs = [item["teacher_vectors"] for item in valid_batch]
    t_probs = [item["teacher_probs"] for item in valid_batch]

    mel_lengths = torch.tensor([m.shape[0] // 2 for m in mels], dtype=torch.long)
    target_lengths = torch.tensor([len(t) for t in tokens], dtype=torch.long)

    return {
        "mels": nn.utils.rnn.pad_sequence(mels, batch_first=True),
        "mel_lengths": mel_lengths,
        "tokens": nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=0),
        "target_lengths": target_lengths,
        "teacher_vectors": torch.cat(t_vecs, dim=0),
        "teacher_probs": torch.cat(t_probs, dim=0)
    }

# ==========================================
# 5. Training Engine with Checkpoint Resume
# ==========================================
def main():
    dataset = ASRStage1Dataset(DATA_DIR, PBKD_DIR, TOKENIZER_MODEL)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)

    model = StudentAcousticModel(
        input_dim=80,
        encoder_dim=STUDENT_DIM,
        num_encoder_layers=4,
        vocab_size=VOCAB_SIZE,
        teacher_dim=TEACHER_DIM,
        blank_id=dataset.blank_id
    ).to(DEVICE)

    # Load Existing Checkpoint
    if os.path.exists(BASE_CHECKPOINT):
        print(f"--> [Resuming Training] Loading weights from {BASE_CHECKPOINT}")
        model.load_state_dict(torch.load(BASE_CHECKPOINT, map_location=DEVICE, weights_only=True))
    else:
        print(f"--> Error: Checkpoint not found at {BASE_CHECKPOINT}")
        return

    ctc_loss_fn = nn.CTCLoss(blank=dataset.blank_id, zero_infinity=True)
    ot_loss_fn = SinkhornWassersteinDistillationLoss(student_embed_dim=STUDENT_DIM, teacher_embed_dim=TEACHER_DIM).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(TOTAL_EPOCHS - START_EPOCH + 1), eta_min=5e-6)

    best_loss = 2.4261  # Previous best loss
    print(f"\n--> Resuming Training from Epoch {START_EPOCH} to {TOTAL_EPOCHS}...")

    for epoch in range(START_EPOCH, TOTAL_EPOCHS + 1):
        model.train()
        total_epoch_loss = 0.0
        total_ctc = 0.0
        total_ot = 0.0
        valid_batches = 0

        pbar = tqdm(loader, desc=f"Epoch [{epoch}/{TOTAL_EPOCHS}]")
        for batch in pbar:
            mels = batch["mels"].to(DEVICE)
            mel_lengths = batch["mel_lengths"].to(DEVICE)
            tokens = batch["tokens"].to(DEVICE)
            target_lengths = batch["target_lengths"].to(DEVICE)
            t_vecs = batch["teacher_vectors"].to(DEVICE)
            t_probs = batch["teacher_probs"].to(DEVICE)

            optimizer.zero_grad()
            ctc_logits, semantic_embeds = model(mels)

            log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
            loss_ctc = ctc_loss_fn(log_probs, tokens, mel_lengths, target_lengths)

            flat_student_vecs = semantic_embeds.view(-1, 1, STUDENT_DIM)
            min_tokens = min(flat_student_vecs.size(0), t_vecs.size(0))
            loss_ot = ot_loss_fn(flat_student_vecs[:min_tokens], t_vecs[:min_tokens], teacher_mass=t_probs[:min_tokens])

            total_loss = loss_ctc + (ALPHA_OT * loss_ot)

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_epoch_loss += total_loss.item()
            total_ctc += loss_ctc.item()
            total_ot += loss_ot.item()
            valid_batches += 1

            pbar.set_postfix({
                "Loss": f"{total_loss.item():.3f}",
                "CTC": f"{loss_ctc.item():.3f}",
                "OT": f"{loss_ot.item():.3f}"
            })

        scheduler.step()
        if valid_batches > 0:
            avg_loss = total_epoch_loss / valid_batches
            avg_ctc = total_ctc / valid_batches
            avg_ot = total_ot / valid_batches
            print(f"--> Epoch {epoch} Summary | Avg Loss: {avg_loss:.4f} | CTC: {avg_ctc:.4f} | OT: {avg_ot:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), BASE_CHECKPOINT)
                print(f"--> [New Best Model Saved! Loss: {best_loss:.4f}]")

if __name__ == "__main__":
    main()