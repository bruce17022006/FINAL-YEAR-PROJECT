import os
import sys
import math
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

sys.stdout.reconfigure(encoding='utf-8')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB_SIZE = 1024
STUDENT_DIM = 256
BLANK_ID = 4
PAD_ID = 0

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class FastStudentASR(nn.Module):
    def __init__(self, time_stride=4):
        super().__init__()
        self.time_stride = time_stride
        if time_stride == 4:
            self.subsampling = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=(2, 2), padding=1),
                nn.ReLU(),
                nn.Conv2d(32, STUDENT_DIM, kernel_size=3, stride=(2, 2), padding=1),
                nn.ReLU(),
            )
            # freq downsampled by 4: 80 // 4 = 20
            self.input_proj = nn.Linear(STUDENT_DIM * 20, STUDENT_DIM)
        else:
            self.subsampling = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=(2, 2), padding=1),
                nn.ReLU(),
                nn.Conv2d(32, STUDENT_DIM, kernel_size=3, stride=(1, 2), padding=1),
                nn.ReLU(),
            )
            self.input_proj = nn.Linear(STUDENT_DIM * 20, STUDENT_DIM)

        self.pos_enc = PositionalEncoding(STUDENT_DIM)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=STUDENT_DIM, nhead=4, dim_feedforward=STUDENT_DIM * 4,
            dropout=0.0, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.ctc_head = nn.Linear(STUDENT_DIM, VOCAB_SIZE)

    def forward(self, mel, lengths=None):
        # mel: [B, T, 80]
        x = self.subsampling(mel.unsqueeze(1)) # [B, C, T', F']
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, c * f)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        
        mask = None
        if lengths is not None:
            out_lens = torch.div(lengths + (self.time_stride - 1), self.time_stride, rounding_mode="floor")
            pos = torch.arange(t, device=x.device).unsqueeze(0)
            mask = pos >= out_lens.unsqueeze(1)
            
        feat = self.encoder(x, src_key_padding_mask=mask)
        logits = self.ctc_head(feat)
        return logits

def ctc_decode(ids, blank_id, sp):
    collapsed = []
    prev = None
    for tid in ids:
        if tid != prev and tid != blank_id:
            collapsed.append(tid)
        prev = tid
    return sp.decode_ids(collapsed)

def main():
    print("=" * 80)
    print("DEMONSTRATION: PROVING THE MODEL CAN RECOGNIZE & MATCH GROUND TRUTH")
    print("=" * 80)
    
    sp = spm.SentencePieceProcessor()
    sp.load("student_bpe_1024.model")

    # Pick 3 real samples from cache
    files = sorted(glob.glob("precomputed_pbkd_train_clean/*.pt"))[:3]
    samples = [torch.load(f, map_location="cpu", weights_only=True) for f in files]

    mels = [s["student_mel"].float() for s in samples]
    targets = [torch.as_tensor(s["student_target_ids"], dtype=torch.long) for s in samples]
    texts = [s["text"] for s in samples]

    mel_lens = torch.tensor([m.shape[0] for m in mels], dtype=torch.long, device=DEVICE)
    out_lens = torch.div(mel_lens + 3, 4, rounding_mode="floor") # 4x subsampling
    target_lens = torch.tensor([t.numel() for t in targets], dtype=torch.long, device=DEVICE)

    padded_mels = nn.utils.rnn.pad_sequence(mels, batch_first=True).to(DEVICE)
    padded_targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=PAD_ID).to(DEVICE)

    print(f"Device: {DEVICE}")
    print(f"Samples to fit: {len(samples)}")
    for i, txt in enumerate(texts):
        print(f"\n[Sample {i+1} Ground Truth]:\n  \"{txt.upper()}\"")

    model = FastStudentASR(time_stride=4).to(DEVICE)
    ctc_loss_fn = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print("\n" + "=" * 80)
    print("Beginning Training Steps to achieve exact Ground Truth Match...")
    print("=" * 80)

    for step in range(1, 101):
        model.train()
        optimizer.zero_grad()
        logits = model(padded_mels, lengths=mel_lens)
        log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)
        loss = ctc_loss_fn(log_probs, padded_targets, out_lens, target_lens)
        loss.backward()
        optimizer.step()

        if step in (1, 15, 30, 50, 75, 100):
            model.eval()
            with torch.no_grad():
                val_logits = model(padded_mels, lengths=mel_lens)
                pred_ids = val_logits.argmax(dim=-1).cpu().tolist()
                print(f"\n--- STEP {step:03d} | CTC Loss: {loss.item():.4f} ---")
                for s_idx in range(len(samples)):
                    pred_text = ctc_decode(pred_ids[s_idx][:out_lens[s_idx]], BLANK_ID, sp)
                    ref_text = texts[s_idx].upper()
                    match = " [PERFECT MATCH!]" if pred_text.upper() == ref_text else ""
                    print(f"Sample {s_idx+1} PRED: \"{pred_text.upper()}\"{match}")

if __name__ == "__main__":
    main()
