
import os
import glob
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

CACHE_DIR = "precomputed_pbkd_train_clean"
CHECKPOINT = os.path.join(
    "checkpoints_stage1", "best_student_asr_corrected.pt"
)
SAMPLE_ID = "103-1240-0000"
BLANK_ID = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class StudentAcousticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.subsampling = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 256, 3, stride=(1, 2), padding=1),
            nn.ReLU(),
        )
        self.input_proj = nn.Linear(256 * 20, 256)
        self.pos_enc = PositionalEncoding(256)

        layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=4)
        self.ctc_head = nn.Linear(256, 1024)
        self.semantic_proj = nn.Linear(256, 2048, bias=False)

    def forward(self, mel):
        x = self.subsampling(mel.unsqueeze(1))
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, c * f)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return self.ctc_head(x), self.semantic_proj(x)


def load_sample(sample_id):
    path = os.path.join(CACHE_DIR, sample_id + ".pt")
    item = torch.load(path, map_location="cpu", weights_only=False)
    return (
        item["student_mel"].float(),
        torch.as_tensor(item["student_target_ids"], dtype=torch.long),
        item.get("text", ""),
    )


def output_length(t):
    return (t + 1) // 2


def decode(logits, length):
    probs = logits[:length].softmax(-1)
    raw = logits[:length].argmax(-1).tolist()

    result = []
    previous = None
    for token in raw:
        if token != previous and token != BLANK_ID:
            result.append(token)
        previous = token

    blank_ratio = sum(x == BLANK_ID for x in raw) / len(raw)
    mean_blank_prob = probs[:, BLANK_ID].mean().item()
    return result, blank_ratio, mean_blank_prob


def main():
    print("Device:", DEVICE)

    if not os.path.isfile(CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.pt")))
    target_path = os.path.join(CACHE_DIR, SAMPLE_ID + ".pt")

    if not os.path.isfile(target_path):
        raise FileNotFoundError(f"Sample not found: {target_path}")

    other_files = [
        p for p in files if os.path.abspath(p) != os.path.abspath(target_path)
    ]
    if len(other_files) < 3:
        raise RuntimeError("Cache folder needs at least 4 .pt files.")

    paths = [target_path] + other_files[:3]
    sample_ids = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    data = [load_sample(sid) for sid in sample_ids]

    model = StudentAcousticModel().to(DEVICE)
    state = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)

    if "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=True)
    model.eval()
    print("Checkpoint loaded strictly: True")

    mel_one, target, reference = data[0]
    mels = [item[0] for item in data]
    lengths = [m.shape[0] for m in mels]

    batch_one = mel_one.unsqueeze(0).to(DEVICE)
    batch_four = pad_sequence(
        mels, batch_first=True, padding_value=0.0
    ).to(DEVICE)

    with torch.no_grad():
        logits_one, _ = model(batch_one)
        logits_four, _ = model(batch_four)

    n1 = output_length(lengths[0])
    n4 = output_length(lengths[0])
    a = logits_one[0].float()
    b = logits_four[0].float()

    target = target[target != 0].to(DEVICE)
    target_len = torch.tensor([target.numel()], device=DEVICE)
    ctc = nn.CTCLoss(blank=BLANK_ID, reduction="none", zero_infinity=True)

    loss_a = ctc(
        F.log_softmax(a[:n1], -1).unsqueeze(1),
        target, torch.tensor([n1], device=DEVICE), target_len
    ).item()

    loss_b = ctc(
        F.log_softmax(b[:n4], -1).unsqueeze(1),
        target, torch.tensor([n4], device=DEVICE), target_len
    ).item()

    ids_a, blank_a, prob_a = decode(a, n1)
    ids_b, blank_b, prob_b = decode(b, n4)

    print("\nSample:", sample_ids[0])
    print("Reference:", reference)
    print("Mel shape:", tuple(mel_one.shape))
    print("Batch IDs:", sample_ids)
    print("Padded batch shape:", tuple(batch_four.shape))

    print("\n--- Batch size 1 ---")
    print("CTC loss:", round(loss_a, 5))
    print("Blank frame ratio:", f"{blank_a:.2%}")
    print("Mean blank probability:", round(prob_a, 6))
    print("Collapsed token IDs:", ids_a)

    print("\n--- Batch size 4 ---")
    print("CTC loss:", round(loss_b, 5))
    print("Blank frame ratio:", f"{blank_b:.2%}")
    print("Mean blank probability:", round(prob_b, 6))
    print("Collapsed token IDs:", ids_b)

    print("\n--- Comparison ---")
    print("Absolute loss difference:", round(abs(loss_a - loss_b), 6))
    print("Same token sequence:", ids_a == ids_b)
    print(
        "Max logit difference:",
        round((a[:n1] - b[:n4]).abs().max().item(), 8)
    )


if __name__ == "__main__":
    main()
