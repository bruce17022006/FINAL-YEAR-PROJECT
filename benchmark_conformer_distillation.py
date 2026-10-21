import os
import sys
import torch
import torch.nn.functional as F
import sentencepiece as spm
import glob
import random

sys.stdout.reconfigure(encoding='utf-8')

from conformer_asr import StudentConformerASR
from train_stage1_asr_generalized import (
    StudentAcousticModel,
    TOKENIZER_MODEL,
    VOCAB_SIZE,
    STUDENT_DIM,
    TEACHER_DIM,
    ctc_decode,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLANK_ID = 4

def edit_distance(a, b):
    dp = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(dp[j] + 1, cur[j - 1] + 1, dp[j - 1] + (x != y))
        dp = cur
    return dp[-1]

def calc_wer(ref, hyp):
    r, h = ref.lower().split(), hyp.lower().split()
    return edit_distance(r, h) / max(1, len(r))

def calc_cer(ref, hyp):
    r, h = list(ref.lower().replace(" ", "")), list(hyp.lower().replace(" ", ""))
    return edit_distance(r, h) / max(1, len(r))

def eval_conformer(ckpt_path, test_samples, sp, ctc_loss_fn):
    model = StudentConformerASR(
        input_dim=80, encoder_dim=256, num_layers=6, num_heads=4,
        vocab_size=1024, teacher_dim=2048, conv_kernel_size=15, dropout=0.0
    ).to(DEVICE)
    
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    total_wer, total_cer, total_ctc = 0.0, 0.0, 0.0
    total_blank_ratio, total_blank_prob = 0.0, 0.0
    n = len(test_samples)

    with torch.no_grad():
        for sample in test_samples:
            mel = sample["student_mel"].float().unsqueeze(0).to(DEVICE)
            target = torch.tensor(sample["student_target_ids"], dtype=torch.long, device=DEVICE)
            ref = sample.get("text", "")

            logits, _ = model(mel)
            logits_log_softmax = F.log_softmax(logits, dim=-1)

            t_len = logits.shape[1]
            loss = ctc_loss_fn(
                logits_log_softmax.transpose(0, 1),
                target.unsqueeze(0),
                torch.tensor([t_len], dtype=torch.long),
                torch.tensor([len(target)], dtype=torch.long)
            )
            total_ctc += loss.item()

            probs = F.softmax(logits[0], dim=-1)
            blank_probs = probs[:, BLANK_ID]
            total_blank_prob += blank_probs.mean().item()

            preds = logits[0].argmax(dim=-1).cpu().tolist()
            blank_count = sum(1 for p in preds if p == BLANK_ID)
            total_blank_ratio += blank_count / len(preds)

            hyp = ctc_decode(preds, BLANK_ID, sp)
            total_wer += calc_wer(ref, hyp)
            total_cer += calc_cer(ref, hyp)

    return {
        "CTC Loss": total_ctc / n,
        "WER (%)": (total_wer / n) * 100,
        "CER (%)": (total_cer / n) * 100,
        "Blank Prob (%)": (total_blank_prob / n) * 100,
        "Blank Predictions (%)": (total_blank_ratio / n) * 100
    }

def main():
    print("=" * 70)
    print("COMPARING CONFORMER 4x BASELINE VS STAGE 2 SINKHORN DISTILLED MODEL")
    print("=" * 70)

    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL)
    ctc_loss_fn = torch.nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)

    cache_files = glob.glob(os.path.join("precomputed_pbkd_train_clean", "*.pt"))
    random.seed(42)
    eval_files = sorted(cache_files)[-100:]  # held-out 100 samples
    samples = [torch.load(f, map_location="cpu") for f in eval_files]

    checkpoints = {
        "Conformer 4x (Stage 1 CTC Only)": "checkpoints_conformer/best_student_conformer_4x.pt",
        "Conformer 4x (Stage 2 Sinkhorn Distilled)": "checkpoints_conformer/best_conformer_stage2_distilled.pt"
    }

    results = {}
    for name, path in checkpoints.items():
        if os.path.exists(path):
            print(f"--> Evaluating {name}...")
            res = eval_conformer(path, samples, sp, ctc_loss_fn)
            results[name] = res

    print("\n" + "=" * 80)
    print(f"{'Model Architecture':<42} | {'CTC Loss':<9} | {'WER (%)':<8} | {'CER (%)':<8} | {'Blank %':<8}")
    print("-" * 80)
    for name, m in results.items():
        print(f"{name:<42} | {m['CTC Loss']:<9.4f} | {m['WER (%)']:<8.2f} | {m['CER (%)']:<8.2f} | {m['Blank Predictions (%)']:<8.2f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
