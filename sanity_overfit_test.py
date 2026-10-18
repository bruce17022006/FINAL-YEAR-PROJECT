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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB_SIZE = 1024
TEACHER_DIM = 2048
STUDENT_DIM = 256
OVERFIT_SAMPLES = 4

DATA_DIR = os.path.join("Data", "dev-clean", "LibriSpeech", "dev-clean")
PBKD_DIR = "precomputed_pbkd_dev_clean"
TOKENIZER_MODEL = "student_bpe_1024.model"

# ==========================================
# 1. Positional Encoding
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
# 2. Fixed Student Model
# ==========================================
class RobustStudentAcousticModel(nn.Module):
    def __init__(self, input_dim=80, encoder_dim=256, num_layers=4, num_heads=4, vocab_size=1024, teacher_dim=2048, blank_id=4):
        super().__init__()
        self.subsampling = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, encoder_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.input_proj = nn.Linear(encoder_dim * (input_dim // 4), encoder_dim)
        self.pos_enc = PositionalEncoding(encoder_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=num_heads,
            dim_feedforward=encoder_dim * 4,
            dropout=0.0,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.ctc_head = nn.Linear(encoder_dim, vocab_size)
        self.semantic_proj = nn.Linear(encoder_dim, teacher_dim, bias=False)

        # Fix: Suppress blank bias initially so non-blank characters activate
        with torch.no_grad():
            self.ctc_head.bias.fill_(0.0)
            self.ctc_head.bias[blank_id] = -5.0

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
# 3. Small Subset Dataset
# ==========================================
class SubsetData(Dataset):
    def __init__(self, data_dir, pbkd_dir, sp_model, limit=4):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model)
        self.blank_id = self.sp.piece_to_id("<blank>")

        self.teacher_embeddings = torch.load(os.path.join(pbkd_dir, "llm_embeddings.pt"), map_location="cpu", weights_only=True)

        audio_files = {}
        for ext in ("*.flac", "*.wav"):
            for f in glob.glob(os.path.join(data_dir, "**", ext), recursive=True):
                audio_files[os.path.splitext(os.path.basename(f))[0]] = f

        transcripts = {}
        for tf in glob.glob(os.path.join(data_dir, "**", "*.trans.txt"), recursive=True):
            with open(tf, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        transcripts[parts[0]] = " ".join(parts[1:]).lower()

        valid_utts = [u for u in audio_files if u in transcripts and os.path.exists(os.path.join(pbkd_dir, f"{u}.pt"))]
        self.selected = valid_utts[:limit]
        self.audio_files = {u: audio_files[u] for u in self.selected}
        self.transcripts = {u: transcripts[u] for u in self.selected}

        self.mel_transform = torchaudio.transforms.MelSpectrogram(16000, n_fft=400, win_length=400, hop_length=160, n_mels=80)

    def __len__(self):
        return len(self.selected)

    def __getitem__(self, idx):
        utt = self.selected[idx]
        wav, sr = torchaudio.load(self.audio_files[utt])
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        mel = self.mel_transform(wav).squeeze(0)
        log_mel = torch.log(torch.clamp(mel, min=1e-5)).transpose(0, 1)

        tokens = torch.tensor(self.sp.encode_as_ids(self.transcripts[utt]), dtype=torch.long)
        cached = torch.load(os.path.join(PBKD_DIR, f"{utt}.pt"), map_location="cpu", weights_only=True)

        return {
            "utt": utt,
            "mel": log_mel,
            "tokens": tokens,
            "text": self.transcripts[utt],
            "t_vecs": self.teacher_embeddings[cached["topk_indices"]],
            "t_probs": cached["topk_probs"]
        }

def collate_fn(batch):
    mels = [b["mel"] for b in batch]
    tokens = [b["tokens"] for b in batch]
    mel_lengths = torch.tensor([m.shape[0] // 4 for m in mels], dtype=torch.long)
    target_lengths = torch.tensor([len(t) for t in tokens], dtype=torch.long)

    return {
        "mels": nn.utils.rnn.pad_sequence(mels, batch_first=True),
        "mel_lengths": mel_lengths,
        "tokens": nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=0),
        "target_lengths": target_lengths,
        "t_vecs": torch.cat([b["t_vecs"] for b in batch], dim=0),
        "t_probs": torch.cat([b["t_probs"] for b in batch], dim=0),
        "texts": [b["text"] for b in batch]
    }

# ==========================================
# 4. Overfit Run
# ==========================================
def run_test():
    ds = SubsetData(DATA_DIR, PBKD_DIR, TOKENIZER_MODEL, limit=OVERFIT_SAMPLES)
    loader = DataLoader(ds, batch_size=OVERFIT_SAMPLES, shuffle=False, collate_fn=collate_fn)

    model = RobustStudentAcousticModel(vocab_size=VOCAB_SIZE, blank_id=ds.blank_id).to(DEVICE)
    ctc_loss_fn = nn.CTCLoss(blank=ds.blank_id, zero_infinity=True)
    ot_loss_fn = SinkhornWassersteinDistillationLoss(student_embed_dim=STUDENT_DIM, teacher_embed_dim=TEACHER_DIM).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    print(f"--> Target Sentence to Learn: \"{ds.transcripts[ds.selected[0]]}\"")
    print("--> Starting 60-epoch Overfit Sanity Loop...\n")

    for epoch in range(1, 61):
        model.train()
        for batch in loader:
            mels = batch["mels"].to(DEVICE)
            mel_lens = batch["mel_lengths"].to(DEVICE)
            tokens = batch["tokens"].to(DEVICE)
            tgt_lens = batch["target_lengths"].to(DEVICE)
            t_vecs = batch["t_vecs"].to(DEVICE)
            t_probs = batch["t_probs"].to(DEVICE)

            optimizer.zero_grad()
            logits, sem = model(mels)

            loss_ctc = ctc_loss_fn(F.log_softmax(logits, dim=-1).transpose(0, 1), tokens, mel_lens, tgt_lens)
            flat_sem = sem.view(-1, 1, STUDENT_DIM)
            min_len = min(flat_sem.size(0), t_vecs.size(0))
            loss_ot = ot_loss_fn(flat_sem[:min_len], t_vecs[:min_len], teacher_mass=t_probs[:min_len])

            total_loss = loss_ctc + (0.1 * loss_ot)
            total_loss.backward()
            optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/60] | Loss: {total_loss.item():.4f} | CTC: {loss_ctc.item():.4f} | OT: {loss_ot.item():.4f}")

    # Final Decode Test
    model.eval()
    with torch.no_grad():
        logits, _ = model(mels)
        pred_ids = torch.argmax(logits[0], dim=-1).tolist()
        deduped = [p for i, p in enumerate(pred_ids) if p != ds.blank_id and (i == 0 or p != pred_ids[i-1])]
        decoded = ds.sp.decode_ids(deduped)

    print("\n--- Overfit Verification Result ---")
    print(f"Ground Truth: \"{batch['texts'][0]}\"")
    print(f"Decoded Text: \"{decoded}\"")

if __name__ == "__main__":
    run_test()