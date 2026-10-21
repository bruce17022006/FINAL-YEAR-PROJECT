import os
import glob
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

CACHE_DIR = "precomputed_pbkd_train_clean"
CHECKPOINT = os.path.join("checkpoints_stage1", "best_student_asr_corrected.pt")
SAMPLE_ID = "103-1240-0000"
BLANK_ID = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
            raise ValueError("Sequence exceeds positional encoding length.")
        return x + self.pe[:, :x.size(1), :]


class StudentAcousticModel(nn.Module):
    """Checkpoint-compatible architecture with optional encoder padding mask."""
    def __init__(self):
        super().__init__()
        self.subsampling = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 256, kernel_size=3, stride=(1, 2), padding=1),
            nn.ReLU(),
        )
        self.input_proj = nn.Linear(256 * 20, 256)
        self.pos_enc = PositionalEncoding(256)
        layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=4, dim_feedforward=1024,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=4)
        self.ctc_head = nn.Linear(256, 1024)
        self.semantic_proj = nn.Linear(256, 2048, bias=False)

    def forward(self, mel, input_lengths=None, use_padding_mask=False):
        x = self.subsampling(mel.unsqueeze(1))
        bsz, channels, t_out, freq_bins = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(
            bsz, t_out, channels * freq_bins
        )
        x = self.input_proj(x)
        x = self.pos_enc(x)

        src_key_padding_mask = None
        if use_padding_mask:
            if input_lengths is None:
                raise ValueError("input_lengths required when use_padding_mask=True")
            output_lengths = torch.div(input_lengths + 1, 2, rounding_mode="floor")
            output_lengths = output_lengths.clamp(max=t_out)
            positions = torch.arange(t_out, device=x.device).unsqueeze(0)
            src_key_padding_mask = positions >= output_lengths.unsqueeze(1)

        features = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.ctc_head(features), self.semantic_proj(features)


def load_cache(sample_id):
    path = os.path.join(CACHE_DIR, sample_id + ".pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cache file not found: {path}")
    try:
        item = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        item = torch.load(path, map_location="cpu")
    return (
        item["student_mel"].float(),
        torch.as_tensor(item["student_target_ids"], dtype=torch.long),
        item.get("text", ""),
    )


def output_length(t):
    return (t + 1) // 2


def decode(logits, length):
    valid_logits = logits[:length]
    probs = valid_logits.softmax(-1)
    raw = valid_logits.argmax(-1).tolist()
    collapsed = []
    prev = None
    for token in raw:
        if token != prev and token != BLANK_ID:
            collapsed.append(token)
        prev = token
    blank_ratio = sum(token == BLANK_ID for token in raw) / max(1, len(raw))
    return collapsed, blank_ratio, probs[:, BLANK_ID].mean().item()


def calculate_ctc(logits, target, length):
    log_probs = F.log_softmax(logits[:length].float(), dim=-1).unsqueeze(1)
    target = target[target != 0].to(DEVICE)
    return nn.CTCLoss(blank=BLANK_ID, reduction="none", zero_infinity=True)(
        log_probs,
        target,
        torch.tensor([length], dtype=torch.long, device=DEVICE),
        torch.tensor([target.numel()], dtype=torch.long, device=DEVICE),
    ).item()


def print_metrics(name, logits, out_len, target):
    token_ids, blank_ratio, blank_prob = decode(logits, out_len)
    loss = calculate_ctc(logits, target, out_len)
    print(f"\n--- {name} ---")
    print("CTC loss:", round(loss, 5))
    print("Blank frame ratio:", f"{blank_ratio:.2%}")
    print("Mean blank probability:", round(blank_prob, 6))
    print("Collapsed token IDs:", token_ids)
    return token_ids, loss, blank_ratio, blank_prob


def main():
    print("Device:", DEVICE)
    print("Checkpoint:", CHECKPOINT)
    if not os.path.isfile(CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    cache_paths = sorted(glob.glob(os.path.join(CACHE_DIR, "*.pt")))
    target_path = os.path.join(CACHE_DIR, SAMPLE_ID + ".pt")
    if not os.path.isfile(target_path):
        raise FileNotFoundError(f"Sample cache not found: {target_path}")
    other_paths = [p for p in cache_paths if os.path.abspath(p) != os.path.abspath(target_path)]
    if len(other_paths) < 3:
        raise RuntimeError("Need at least 4 cached .pt files.")
    sample_ids = [SAMPLE_ID] + [
        os.path.splitext(os.path.basename(p))[0] for p in other_paths[:3]
    ]
    data = [load_cache(sid) for sid in sample_ids]

    # Keep the same four-item padded batch for both tests.
    mels = [item[0] for item in data]
    lengths = torch.tensor([m.shape[0] for m in mels], dtype=torch.long, device=DEVICE)
    batch = pad_sequence(mels, batch_first=True, padding_value=0.0).to(DEVICE)
    target = data[0][1]
    valid_out_len = output_length(int(lengths[0].item()))

    model = StudentAcousticModel().to(DEVICE)
    try:
        state = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    except TypeError:
        state = torch.load(CHECKPOINT, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    print("Checkpoint loaded strictly: True")
    print("Sample IDs:", sample_ids)
    print("Padded batch shape:", tuple(batch.shape))
    print("True input lengths:", lengths.tolist())
    print("Valid output frames for target:", valid_out_len)

    with torch.no_grad():
        # Same checkpoint and batch; only the attention padding mask changes.
        logits_unmasked, _ = model(batch, input_lengths=lengths, use_padding_mask=False)
        logits_masked, _ = model(batch, input_lengths=lengths, use_padding_mask=True)

    unmasked = logits_unmasked[0].float()
    masked = logits_masked[0].float()
    r1 = print_metrics("UNMASKED (original behaviour)", unmasked, valid_out_len, target)
    r2 = print_metrics("MASKED (src_key_padding_mask)", masked, valid_out_len, target)

    print("\n--- Direct comparison on valid frames ---")
    print("Max absolute logit difference:", round(
        (unmasked[:valid_out_len] - masked[:valid_out_len]).abs().max().item(), 8
    ))
    print("Mean absolute logit difference:", round(
        (unmasked[:valid_out_len] - masked[:valid_out_len]).abs().mean().item(), 8
    ))
    print("Same collapsed token sequence:", r1[0] == r2[0])
    print("CTC loss change (masked - unmasked):", round(r2[1] - r1[1], 6))
    print("\nInterpretation: this isolates the inference-time mask effect for this checkpoint.")
    print("The checkpoint was trained without this mask, so neither result proves masked training will improve WER.")


if __name__ == "__main__":
    main()
