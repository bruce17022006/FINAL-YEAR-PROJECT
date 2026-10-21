import os
import glob
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

CACHE_DIR = "precomputed_pbkd_train_clean"
CHECKPOINT = os.path.join(
    "checkpoints_stage1", "best_student_asr_corrected.pt"
)
TOKENIZER_MODEL = "student_bpe_1024.model"

BLANK_ID = 4
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
VOCAB_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SAMPLES = 32


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(
            max_len, dtype=torch.float32
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class StudentAcousticModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.subsampling = nn.Sequential(
            nn.Conv2d(
                1, 32, 3, stride=(2, 2), padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                32, 256, 3, stride=(1, 2), padding=1
            ),
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

        self.encoder = nn.TransformerEncoder(
            layer, num_layers=4
        )

        self.ctc_head = nn.Linear(256, VOCAB_SIZE)
        self.semantic_proj = nn.Linear(
            256, 2048, bias=False
        )

    def forward(self, mel):
        x = self.subsampling(mel.unsqueeze(1))
        b, c, t, f = x.shape

        x = (
            x.permute(0, 2, 1, 3)
            .contiguous()
            .view(b, t, c * f)
        )

        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)

        return self.ctc_head(x)


def load_checkpoint(model):
    try:
        state = torch.load(
            CHECKPOINT,
            map_location=DEVICE,
            weights_only=False,
        )
    except TypeError:
        state = torch.load(
            CHECKPOINT, map_location=DEVICE
        )

    if isinstance(state, dict):
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]

    model.load_state_dict(state, strict=True)


def main():
    print("Device:", DEVICE)

    if not os.path.isdir(CACHE_DIR):
        raise FileNotFoundError(
            f"Cache folder not found: {CACHE_DIR}"
        )

    if not os.path.isfile(CHECKPOINT):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    files = sorted(
        glob.glob(os.path.join(CACHE_DIR, "*.pt"))
    )

    if not files:
        raise FileNotFoundError(
            f"No cache .pt files found in {CACHE_DIR}"
        )

    model = StudentAcousticModel().to(DEVICE)
    load_checkpoint(model)
    model.eval()

    print("Checkpoint loaded strictly: True")
    print("Cached files found:", len(files))
    print(
        f"Checking first {min(MAX_SAMPLES, len(files))} samples."
    )

    try:
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor()
        tokenizer_ok = sp.load(TOKENIZER_MODEL)

    except Exception as exc:
        sp = None
        tokenizer_ok = False
        print("Tokenizer could not load:", repr(exc))

    ctc = nn.CTCLoss(
        blank=BLANK_ID,
        reduction="none",
        zero_infinity=True,
    )

    total = 0
    finite_logits_all = True

    blank_frames = 0
    valid_frames = 0
    blank_prob_sum = 0.0
    ctc_losses = []

    target_lengths = []
    output_lengths = []
    infeasible = 0

    # Count targets containing special IDs.
    targets_with_bos = 0
    targets_with_eos = 0
    targets_with_pad = 0
    targets_with_blank = 0
    targets_with_unk = 0

    total_bos = 0
    total_eos = 0
    total_pad = 0
    total_blank = 0
    total_unk = 0

    rows = []

    with torch.no_grad():
        for index, path in enumerate(files[:MAX_SAMPLES]):
            filename = os.path.basename(path)

            try:
                try:
                    item = torch.load(
                        path,
                        map_location="cpu",
                        weights_only=False,
                    )
                except TypeError:
                    item = torch.load(
                        path, map_location="cpu"
                    )

                mel = item["student_mel"].float()
                target = torch.as_tensor(
                    item["student_target_ids"],
                    dtype=torch.long,
                )

                text = item.get("text", "")

                if mel.ndim != 2 or mel.shape[-1] != 80:
                    print(
                        "SKIP invalid mel shape:",
                        filename,
                        tuple(mel.shape),
                    )
                    continue

                if target.numel() == 0:
                    print("SKIP empty target:", filename)
                    continue

                if (
                    target.min().item() < 0
                    or target.max().item() >= VOCAB_SIZE
                ):
                    print(
                        "SKIP out-of-range target IDs:",
                        filename,
                    )
                    continue

                # Count occurrences of special IDs.
                bos_count = int((target == BOS_ID).sum().item())
                eos_count = int((target == EOS_ID).sum().item())
                pad_count = int((target == PAD_ID).sum().item())
                blank_count = int(
                    (target == BLANK_ID).sum().item()
                )
                unk_count = int((target == UNK_ID).sum().item())

                total_bos += bos_count
                total_eos += eos_count
                total_pad += pad_count
                total_blank += blank_count
                total_unk += unk_count

                targets_with_bos += int(bos_count > 0)
                targets_with_eos += int(eos_count > 0)
                targets_with_pad += int(pad_count > 0)
                targets_with_blank += int(blank_count > 0)
                targets_with_unk += int(unk_count > 0)

                # Print detailed target information for first 10 samples.
                if index < 10:
                    first_ids = target[:5].tolist()
                    last_ids = target[-5:].tolist()

                    print(f"\nTarget check: {filename}")
                    print("  Text:", repr(text[:100]))
                    print("  First IDs:", first_ids)
                    print("  Last IDs:", last_ids)
                    print(
                        "  BOS count:", bos_count,
                        "| EOS count:", eos_count,
                        "| PAD count:", pad_count,
                        "| BLANK count:", blank_count,
                        "| UNK count:", unk_count,
                    )

                t_in = mel.shape[0]
                t_out = (t_in + 1) // 2

                repeats = (
                    int(
                        (target[1:] == target[:-1])
                        .sum().item()
                    )
                    if target.numel() > 1
                    else 0
                )

                min_needed = target.numel() + repeats
                is_infeasible = t_out < min_needed
                infeasible += int(is_infeasible)

                target_lengths.append(target.numel())
                output_lengths.append(t_out)

                logits = model(
                    mel.unsqueeze(0).to(DEVICE)
                )[0].float()

                finite = bool(
                    torch.isfinite(logits).all().item()
                )
                finite_logits_all &= finite

                if not finite:
                    print(
                        "WARNING non-finite logits:",
                        filename,
                    )
                    continue

                lp = F.log_softmax(
                    logits[:t_out], dim=-1
                )

                pred = logits[:t_out].argmax(dim=-1)

                bcount = int(
                    (pred == BLANK_ID).sum().item()
                )

                bprob = (
                    logits[:t_out]
                    .softmax(dim=-1)[:, BLANK_ID]
                    .mean()
                    .item()
                )

                blank_frames += bcount
                valid_frames += t_out
                blank_prob_sum += bprob

                loss = ctc(
                    lp.unsqueeze(1),
                    target.to(DEVICE),
                    torch.tensor(
                        [t_out], device=DEVICE
                    ),
                    torch.tensor(
                        [target.numel()], device=DEVICE
                    ),
                ).item()

                ctc_losses.append(loss)

                # Greedy CTC decoding.
                collapsed = []
                prev = None

                for tok in pred.tolist():
                    if tok != prev and tok != BLANK_ID:
                        collapsed.append(tok)
                    prev = tok

                if tokenizer_ok:
                    pred_text = sp.decode_ids(collapsed)
                    ref_text = sp.decode_ids(
                        target.tolist()
                    )
                else:
                    pred_text = str(collapsed[:25])
                    ref_text = str(target.tolist()[:25])

                rows.append(
                    (
                        filename,
                        loss,
                        bcount / max(1, t_out),
                        bprob,
                        len(target),
                        t_out,
                        ref_text[:70],
                        pred_text[:70],
                    )
                )

                total += 1

            except Exception as exc:
                print("ERROR on", filename, ":", repr(exc))

    print("\n========== TARGET SPECIAL-ID SUMMARY ==========")
    print("Targets inspected:", len(target_lengths))
    print(
        f"BOS (ID {BOS_ID}): {targets_with_bos} targets, "
        f"{total_bos} total occurrences"
    )
    print(
        f"EOS (ID {EOS_ID}): {targets_with_eos} targets, "
        f"{total_eos} total occurrences"
    )
    print(
        f"PAD (ID {PAD_ID}): {targets_with_pad} targets, "
        f"{total_pad} total occurrences"
    )
    print(
        f"CTC blank (ID {BLANK_ID}): {targets_with_blank} targets, "
        f"{total_blank} total occurrences"
    )
    print(
        f"UNK (ID {UNK_ID}): {targets_with_unk} targets, "
        f"{total_unk} total occurrences"
    )

    print("\n========== MODEL DIAGNOSTIC SUMMARY ==========")
    print("Samples evaluated:", total)
    print("All evaluated logits finite:", finite_logits_all)

    if total:
        print(
            "Average per-sample CTC loss (sum reduction):",
            round(sum(ctc_losses) / len(ctc_losses), 4),
        )
        print(
            "Mean blank frame ratio:",
            f"{blank_frames / max(1, valid_frames):.2%}",
        )
        print(
            "Mean blank probability across samples:",
            round(blank_prob_sum / total, 6),
        )
        print(
            "Target length range:",
            min(target_lengths), "-", max(target_lengths),
        )
        print(
            "Output length range:",
            min(output_lengths), "-", max(output_lengths),
        )
        print(
            "Insufficient CTC alignment samples:",
            infeasible,
        )

        if rows:
            print("\nFirst sample details:")
            labels = [
                "File",
                "CTC loss",
                "Blank ratio",
                "Mean blank probability",
                "Target tokens",
                "Output frames",
                "Decoded reference",
                "Prediction",
            ]

            for label, value in zip(labels, rows[0]):
                print(f"{label}: {value}")

            print("\nHighest CTC-loss samples:")

            for row in sorted(
                rows, key=lambda r: r[1], reverse=True
            )[:5]:
                print(
                    f"{row[0]} | loss={row[1]:.2f} | "
                    f"blank={row[2]:.1%} | "
                    f"target={row[4]} | output={row[5]}"
                )

    print(
        "\nDiagnostic only: no training, checkpoint writing, "
        "or data modification was performed."
    )


if __name__ == "__main__":
    main()