import os
import glob
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

# Step 3: Diagnose the EXISTING ASR checkpoint. This script does NOT train
# and does NOT save/overwrite any checkpoint.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "precomputed_pbkd_train_clean"
TOKENIZER_PATH = "student_bpe_1024.model"
CHECKPOINT_PATH = "checkpoints_stage1/best_student_asr_corrected.pt"
NUM_SAMPLES = 8
BLANK_ID = 4
VOCAB_SIZE = 1024
STUDENT_DIM = 256
TEACHER_DIM = 2048


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
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
            raise ValueError("Sequence exceeds positional encoding limit.")
        return x + self.pe[:, :x.size(1), :]


class StudentAcousticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.subsampling = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, STUDENT_DIM, kernel_size=3, stride=(1, 2), padding=1),
            nn.ReLU(),
        )
        self.input_proj = nn.Linear(STUDENT_DIM * (80 // 4), STUDENT_DIM)
        self.pos_enc = PositionalEncoding(STUDENT_DIM)
        layer = nn.TransformerEncoderLayer(
            d_model=STUDENT_DIM, nhead=4, dim_feedforward=STUDENT_DIM * 4,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=4)
        self.ctc_head = nn.Linear(STUDENT_DIM, VOCAB_SIZE)
        self.semantic_proj = nn.Linear(STUDENT_DIM, TEACHER_DIM, bias=False)

    def forward(self, mel):
        x = self.subsampling(mel.unsqueeze(1))
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, c * f)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return self.ctc_head(x), self.semantic_proj(x)


def output_length(frames):
    return (frames + 1) // 2


def ctc_greedy_decode(frame_ids, blank_id, sp):
    # CTC rule: collapse adjacent repeated IDs first, then remove blanks.
    collapsed = []
    previous = None
    for idx in frame_ids:
        idx = int(idx)
        if idx != previous and idx != blank_id:
            collapsed.append(idx)
        previous = idx
    return sp.decode_ids(collapsed), collapsed


def main():
    print(f"Device: {DEVICE}")
    for p in (TOKENIZER_PATH, CHECKPOINT_PATH):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Required file not found: {p}")
    files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.pt")))
    if not files:
        raise FileNotFoundError(f"No cache files in {CACHE_DIR}")

    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_PATH)
    if sp.get_piece_size() != VOCAB_SIZE:
        raise ValueError(f"Expected vocab {VOCAB_SIZE}, got {sp.get_piece_size()}")
    if sp.piece_to_id("<blank>") != BLANK_ID:
        raise ValueError(f"Expected <blank> ID {BLANK_ID}")

    model = StudentAcousticModel().to(DEVICE)
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    ctc_loss = nn.CTCLoss(blank=BLANK_ID, reduction="none", zero_infinity=False)

    # Fixed deterministic subset, no shuffle and no augmentation.
    indices = [round(i * (len(files) - 1) / max(1, NUM_SAMPLES - 1))
               for i in range(min(NUM_SAMPLES, len(files)))]
    results = []
    with torch.inference_mode():
        for n, file_index in enumerate(indices, start=1):
            path = files[file_index]
            data = torch.load(path, map_location="cpu", weights_only=True)
            mel = data["student_mel"].float()
            target = torch.as_tensor(data["student_target_ids"], dtype=torch.long)
            text = str(data.get("text", sp.decode_ids(target.tolist()))).strip()

            if mel.ndim != 2 or mel.shape[1] != 80:
                print(f"SKIP {os.path.basename(path)}: invalid mel shape {tuple(mel.shape)}")
                continue
            if target.numel() == 0 or (target == BLANK_ID).any():
                print(f"SKIP {os.path.basename(path)}: invalid/empty target")
                continue

            mel_len = output_length(mel.shape[0])
            repeats = int((target[1:] == target[:-1]).sum().item()) if target.numel() > 1 else 0
            if mel_len < target.numel() + repeats:
                print(f"SKIP {os.path.basename(path)}: CTC alignment impossible")
                continue

            logits, _ = model(mel.unsqueeze(0).to(DEVICE))
            logits = logits.float()
            if not torch.isfinite(logits).all():
                print(f"FAIL {os.path.basename(path)}: non-finite logits")
                continue
            valid_logits = logits[:, :mel_len, :]
            log_probs = F.log_softmax(valid_logits, dim=-1).transpose(0, 1)
            loss = ctc_loss(
                log_probs,
                target.to(DEVICE),
                torch.tensor([mel_len], dtype=torch.long, device=DEVICE),
                torch.tensor([target.numel()], dtype=torch.long, device=DEVICE),
            )[0].item()

            probs = torch.softmax(valid_logits, dim=-1)[0]
            frame_ids = probs.argmax(dim=-1)
            blank_ratio = (frame_ids == BLANK_ID).float().mean().item()
            mean_blank_prob = probs[:, BLANK_ID].mean().item()
            nonblank_ids = frame_ids[frame_ids != BLANK_ID]
            if nonblank_ids.numel():
                ids, counts = torch.unique(nonblank_ids, return_counts=True)
                top = sorted(zip(ids.tolist(), counts.tolist()), key=lambda x: x[1], reverse=True)[:5]
                top_tokens = ", ".join(
                    f"{sp.id_to_piece(int(tok))}:{count}" for tok, count in top
                )
            else:
                top_tokens = "(no nonblank argmax tokens)"

            prediction, decoded_ids = ctc_greedy_decode(
                frame_ids.cpu().tolist(), BLANK_ID, sp
            )
            target_decoded = sp.decode_ids(target.tolist())
            finite_loss = math.isfinite(loss)
            print(f"\n[{n}] {os.path.basename(path)}")
            print(f"  mel frames={mel.shape[0]} | CTC output frames={mel_len} | target tokens={target.numel()} | repeated adjacent target IDs={repeats}")
            print(f"  CTC loss sum={loss:.3f} | loss/target_token={loss / max(1, target.numel()):.3f} | finite={finite_loss}")
            print(f"  blank argmax ratio={blank_ratio:.2%} | mean blank probability={mean_blank_prob:.4f}")
            print(f"  top nonblank argmax pieces: {top_tokens}")
            print(f"  TARGET: {target_decoded[:220]}")
            print(f"  PRED  : {prediction[:220]}")
            results.append((loss, target.numel(), blank_ratio, mean_blank_prob, len(decoded_ids)))

    if results:
        import statistics
        print("\n========== STEP 3 SUMMARY ==========")
        print(f"Samples evaluated: {len(results)}")
        print(f"Mean CTC loss / target token: {statistics.mean(x[0] / max(1, x[1]) for x in results):.3f}")
        print(f"Mean blank argmax ratio: {statistics.mean(x[2] for x in results):.2%}")
        print(f"Mean blank probability: {statistics.mean(x[3] for x in results):.4f}")
        print("\nNo model weights or checkpoints were changed.")
    else:
        print("No valid samples evaluated. Check the reported skip/errors.")


if __name__ == "__main__":
    main()
