import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

# Run from the project root. Adjust these paths only if your folders differ.
CACHE_DIR = r"precomputed_pbkd_train_clean"
CHECKPOINT = r"checkpoints_stage1\best_student_asr_corrected.pt"
SAMPLE_ID = "103-1240-0000"
BATCH_IDS = ["103-1240-0000", "103-1240-0001", "103-1240-0002", "103-1240-0003"]
BLANK_ID = 4
VOCAB_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class StudentAcousticModel(nn.Module):
    def __init__(self, vocab_size=1024, input_dim=80, d_model=256, nhead=4,
                 num_layers=4, semantic_dim=2048):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=(2, 2), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=(1, 2), padding=1)
        self.input_projection = nn.Linear(64 * 20, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=1024,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ctc_head = nn.Linear(d_model, vocab_size)
        self.semantic_proj = nn.Linear(d_model, semantic_dim)

    def forward(self, mel_inputs):
        x = mel_inputs.unsqueeze(1)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.flatten(2)
        x = self.input_projection(x)
        x = self.encoder(x)  # Intentionally matches the original unmasked checkpoint model.
        return self.ctc_head(x)


def load_cache(sample_id):
    path = os.path.join(CACHE_DIR, sample_id + ".pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing cache file: {path}\n"
            "Update BATCH_IDS to four IDs that exist in your cache folder."
        )
    try:
        item = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        item = torch.load(path, map_location="cpu")
    mel = item.get("student_mel", item.get("mel"))
    target = item.get("student_target_ids", item.get("target_ids"))
    text = item.get("text", "")
    if mel is None or target is None:
        raise KeyError(f"{path} does not contain expected mel/target keys. Found: {list(item.keys())}")
    return mel.float(), torch.as_tensor(target, dtype=torch.long), text


def output_length(length):
    return (length + 1) // 2


def greedy_decode(logits, out_len):
    # logits shape [T,V]
    ids = logits[:out_len].argmax(dim=-1).tolist()
    collapsed = []
    previous = None
    for token in ids:
        if token != previous and token != BLANK_ID:
            collapsed.append(token)
        previous = token
    return collapsed, ids.count(BLANK_ID) / max(1, out_len), logits[:out_len].softmax(-1)[:, BLANK_ID].mean().item()


def main():
    print("Device:", DEVICE)
    print("Checkpoint:", CHECKPOINT)
    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    data = {}
    for sid in dict.fromkeys([SAMPLE_ID] + BATCH_IDS):
        data[sid] = load_cache(sid)

    model = StudentAcousticModel(vocab_size=VOCAB_SIZE).to(DEVICE)
    try:
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    except TypeError:
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=True)
    model.eval()

    # Run A: the target sample alone, with no batch padding.
    mel_a, target_a, text_a = data[SAMPLE_ID]
    len_a = mel_a.shape[0]
    batch_a = mel_a.unsqueeze(0).to(DEVICE)

    # Run B: same sample in a four-item batch, padded to the longest item.
    missing = [sid for sid in BATCH_IDS if sid not in data]
    if missing:
        raise RuntimeError(f"Internal missing IDs: {missing}")
    mels = [data[sid][0] for sid in BATCH_IDS]
    lengths = [m.shape[0] for m in mels]
    batch_b = pad_sequence(mels, batch_first=True, padding_value=0.0).to(DEVICE)
    target_index = BATCH_IDS.index(SAMPLE_ID)
    len_b = lengths[target_index]

    with torch.no_grad():
        logits_a = model(batch_a)[0].float()
        logits_b = model(batch_b)[target_index].float()

    out_a = output_length(len_a)
    out_b = output_length(len_b)
    lp_a = F.log_softmax(logits_a[:out_a], dim=-1)
    lp_b = F.log_softmax(logits_b[:out_b], dim=-1)

    targets = target_a[target_a != 0].to(DEVICE)
    target_lens = torch.tensor([len(targets)], dtype=torch.long, device=DEVICE)
    input_lens_a = torch.tensor([out_a], dtype=torch.long, device=DEVICE)
    input_lens_b = torch.tensor([out_b], dtype=torch.long, device=DEVICE)
    ctc = nn.CTCLoss(blank=BLANK_ID, reduction="none", zero_infinity=True)
    loss_a = ctc(lp_a.unsqueeze(1), targets, input_lens_a, target_lens).item()
    loss_b = ctc(lp_b.unsqueeze(1), targets, input_lens_b, target_lens).item()

    ids_a, blank_ratio_a, blank_prob_a = greedy_decode(logits_a, out_a)
    ids_b, blank_ratio_b, blank_prob_b = greedy_decode(logits_b, out_b)

    print("\n=== TARGET SAMPLE ===")
    print("Sample ID:", SAMPLE_ID)
    print("Reference text:", text_a)
    print("Mel shape:", tuple(mel_a.shape), "| true input frames:", len_a, "| output frames:", out_a)
    print("\n=== RUN A: BATCH SIZE 1 ===")
    print("CTC loss (sum per sample):", round(loss_a, 4))
    print("Blank frame ratio:", f"{blank_ratio_a:.2%}")
    print("Mean blank probability:", round(blank_prob_a, 6))
    print("Collapsed token IDs:", ids_a)
    print("\n=== RUN B: BATCH SIZE 4 ===")
    print("Batch IDs:", BATCH_IDS)
    print("Batch mel shape:", tuple(batch_b.shape))
    print("Lengths:", lengths)
    print("CTC loss (sum per sample):", round(loss_b, 4))
    print("Blank frame ratio:", f"{blank_ratio_b:.2%}")
    print("Mean blank probability:", round(blank_prob_b, 6))
    print("Collapsed token IDs:", ids_b)
    print("\n=== COMPARISON ===")
    print("Absolute CTC loss difference:", round(abs(loss_a - loss_b), 6))
    print("Same collapsed token sequence:", ids_a == ids_b)
    print("Max absolute logit difference on valid frames:",
          round((logits_a[:out_a] - logits_b[:out_b]).abs().max().item(), 8))
    print("\nNote: this compares inference context only. It does not prove masking will improve WER.")
    print("If a batch ID is missing, update BATCH_IDS with four existing .pt filenames (without extension).")


if __name__ == "__main__":
    main()
