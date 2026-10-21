import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

from train_stage1_asr import (
    FastPrecomputedDataset,
    PBKD_DIR,
    TOKENIZER_MODEL,
    CHECKPOINT_DIR,
    VOCAB_SIZE,
    STUDENT_DIM,
    TEACHER_DIM,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_IDS = ["103-1240-0000", "103-1240-0001", "103-1240-0002", "103-1240-0003"]
CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_student_asr_corrected.pt")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            raise ValueError("Sequence exceeds positional encoding limit.")
        return x + self.pe[:, :x.size(1), :]

class MaskedStudentAcousticModel(nn.Module):
    def __init__(self, input_dim=80, encoder_dim=256, num_encoder_layers=4,
                 num_heads=4, vocab_size=1024, teacher_dim=2048, blank_id=4):
        super().__init__()
        self.subsampling = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, encoder_dim, kernel_size=3, stride=(1, 2), padding=1),
            nn.ReLU(),
        )
        self.input_proj = nn.Linear(encoder_dim * (input_dim // 4), encoder_dim)
        self.pos_enc = PositionalEncoding(encoder_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim, nhead=num_heads, dim_feedforward=encoder_dim * 4,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_encoder_layers)
        self.ctc_head = nn.Linear(encoder_dim, vocab_size)
        self.semantic_proj = nn.Linear(encoder_dim, teacher_dim, bias=False)
        with torch.no_grad():
            self.ctc_head.bias.fill_(0.0)
            self.ctc_head.bias[blank_id] = -5.0

    def forward(self, mel_inputs, input_lengths):
        x = self.subsampling(mel_inputs.unsqueeze(1))
        batch_size, channels, time_steps, freq_bins = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(
            batch_size, time_steps, channels * freq_bins
        )
        x = self.input_proj(x)
        x = self.pos_enc(x)

        # Conv subsampling's time stride is 2: output length = ceil(input length / 2).
        out_lengths = torch.div(input_lengths + 1, 2, rounding_mode="floor")
        positions = torch.arange(time_steps, device=x.device).unsqueeze(0)
        padding_mask = positions >= out_lengths.unsqueeze(1)  # True means ignore
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.ctc_head(encoded), self.semantic_proj(encoded), out_lengths, padding_mask

def main():
    print("=== Padding-mask forward/CTC diagnostic (no training) ===")
    print("Device:", DEVICE)
    if not os.path.isfile(CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    sp = spm.SentencePieceProcessor()
    if not sp.load(TOKENIZER_MODEL):
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_MODEL}")
    blank_id = sp.piece_to_id("<blank>")

    dataset = FastPrecomputedDataset(PBKD_DIR)
    index_by_id = {
        os.path.basename(path).removesuffix(".pt"): i
        for i, path in enumerate(dataset.cache_files)
    }
    missing = [sid for sid in SAMPLE_IDS if sid not in index_by_id]
    if missing:
        raise FileNotFoundError(f"Missing cached samples: {missing}")

    samples = [dataset[index_by_id[sid]] for sid in SAMPLE_IDS]
    mels = [x[0] for x in samples]
    targets = [x[1] for x in samples]
    true_lengths = torch.tensor([m.shape[0] for m in mels], dtype=torch.long, device=DEVICE)
    target_lengths = torch.tensor([t.numel() for t in targets], dtype=torch.long, device=DEVICE)
    padded_mels = nn.utils.rnn.pad_sequence(mels, batch_first=True, padding_value=0.0).to(DEVICE)
    padded_targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0).to(DEVICE)

    model = MaskedStudentAcousticModel(
        input_dim=80, encoder_dim=STUDENT_DIM, num_encoder_layers=4,
        vocab_size=VOCAB_SIZE, teacher_dim=TEACHER_DIM, blank_id=blank_id
    ).to(DEVICE)
    state = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    with torch.no_grad():
        logits, _, output_lengths, padding_mask = model(padded_mels, true_lengths)
        log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)
        losses = F.ctc_loss(
            log_probs, padded_targets, output_lengths, target_lengths,
            blank=blank_id, reduction="none", zero_infinity=False
        )

    print("Padded mel batch shape:", tuple(padded_mels.shape))
    print("True input lengths:", true_lengths.tolist())
    print("Expected output lengths:", [int((n + 1) // 2) for n in true_lengths.tolist()])
    print("Model output shape [B,T,V]:", tuple(logits.shape))
    print("Output lengths:", output_lengths.tolist())
    print("Padding mask shape:", tuple(padding_mask.shape))
    print("Masked positions per sample:", padding_mask.sum(dim=1).tolist())
    print("Finite logits:", bool(torch.isfinite(logits).all()))
    print("Finite CTC losses:", bool(torch.isfinite(losses).all()))
    print("Per-sample CTC losses:", [round(x, 4) for x in losses.tolist()])
    print("\nDiagnostic only: weights were loaded but not updated; no checkpoint was saved.")
    print("Note: this confirms the mask code runs, not that it fixes recognition accuracy.")

if __name__ == "__main__":
    main()
