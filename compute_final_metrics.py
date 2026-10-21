import os
import sys
import torch
import torch.nn.functional as F
import sentencepiece as spm
import glob
import random

sys.stdout.reconfigure(encoding='utf-8')

from train_stage1_asr_generalized import (
    StudentAcousticModel,
    TOKENIZER_MODEL,
    VOCAB_SIZE,
    STUDENT_DIM,
    TEACHER_DIM,
    ctc_decode,
    output_length,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLANK_ID = 4
PAD_ID = 0

CHECKPOINTS = {
    "Original Baseline (Before Fix)": "checkpoints_stage1/best_student_asr_corrected.pt",
    "Stage 1 Generalized (After Fix)": "checkpoints_stage1/best_student_asr_generalized.pt",
    "Stage 2 LLM Distilled (Final)": "checkpoints_stage2/best_student_asr_stage2.pt",
}

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

def evaluate_checkpoint(ckpt_path, test_samples, sp, ctc_loss_fn, blank_penalty=1.0):
    model = StudentAcousticModel(
        input_dim=80, encoder_dim=STUDENT_DIM, num_encoder_layers=4,
        num_heads=4, vocab_size=VOCAB_SIZE, teacher_dim=TEACHER_DIM, dropout=0.0
    ).to(DEVICE)
    
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    total_wer = 0.0
    total_cer = 0.0
    total_ctc = 0.0
    total_blank_ratio = 0.0
    total_blank_prob = 0.0
    n = len(test_samples)

    with torch.no_grad():
        for sample in test_samples:
            mel = sample["student_mel"].float().unsqueeze(0).to(DEVICE)
            target = torch.tensor(sample["student_target_ids"], dtype=torch.long, device=DEVICE)
            ref = sample.get("text", "")

            out_len = torch.tensor([output_length(mel.shape[1])], dtype=torch.long, device=DEVICE)
            tgt_len = torch.tensor([target.numel()], dtype=torch.long, device=DEVICE)

            logits, _ = model(mel)
            log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)
            loss = ctc_loss_fn(log_probs, target.unsqueeze(0), out_len, tgt_len)
            if torch.isfinite(loss):
                total_ctc += loss.item()

            probs = F.softmax(logits.float(), dim=-1)[0]
            blank_prob = probs[:, BLANK_ID].mean().item()
            total_blank_prob += blank_prob

            # Decoding
            dec_logits = logits.clone()
            if blank_penalty > 0.0:
                dec_logits[:, :, BLANK_ID] -= blank_penalty
            raw_ids = dec_logits.argmax(dim=-1)[0].cpu().tolist()

            blank_count = raw_ids.count(BLANK_ID)
            blank_ratio = blank_count / max(1, len(raw_ids))
            total_blank_ratio += blank_ratio

            pred = ctc_decode(raw_ids, BLANK_ID, sp)
            total_wer += calc_wer(ref, pred)
            total_cer += calc_cer(ref, pred)

    return {
        "ctc_loss": total_ctc / n,
        "blank_ratio": total_blank_ratio / n,
        "blank_prob": total_blank_prob / n,
        "wer": total_wer / n,
        "cer": total_cer / n,
        "word_acc": max(0.0, 1.0 - (total_wer / n)),
        "char_acc": max(0.0, 1.0 - (total_cer / n)),
    }

def main():
    print("=" * 80)
    print("COMPREHENSIVE FINAL BENCHMARK ACROSS 30 HELD-OUT TEST AUDIO UTTERANCES")
    print("=" * 80)

    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL)
    ctc_loss_fn = torch.nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)

    # Pick 30 deterministic held-out samples from precomputed cache
    all_files = sorted(glob.glob("precomputed_pbkd_train_clean/*.pt"))
    random.seed(42)
    random.shuffle(all_files)
    test_files = all_files[:30]

    test_samples = [torch.load(f, map_location="cpu", weights_only=True) for f in test_files]

    results = {}
    for name, ckpt in CHECKPOINTS.items():
        if os.path.isfile(ckpt):
            print(f"--> Evaluating: {name} ...")
            penalty = 0.0 if "Before Fix" in name else 1.0
            results[name] = evaluate_checkpoint(ckpt, test_samples, sp, ctc_loss_fn, blank_penalty=penalty)
        else:
            print(f"--> Skipping {name} (file not found: {ckpt})")

    print("\n" + "=" * 90)
    print(f"{'METRIC':<30} | {'ORIGINAL BASELINE':<20} | {'STAGE 1 GENERALIZED':<20} | {'STAGE 2 DISTILLED':<20}")
    print("=" * 90)

    row_keys = [
        ("CTC Loss", "ctc_loss", "{:.4f}"),
        ("Mean Blank Probability", "blank_prob", "{:.2%}"),
        ("Blank Prediction Ratio", "blank_ratio", "{:.2%}"),
        ("Character Error Rate (CER)", "cer", "{:.2%}"),
        ("Character Accuracy (1 - CER)", "char_acc", "{:.2%}"),
        ("Word Error Rate (WER)", "wer", "{:.2%}"),
        ("Word Accuracy (1 - WER)", "word_acc", "{:.2%}"),
    ]

    for label, key, fmt in row_keys:
        vals = []
        for name in ["Original Baseline (Before Fix)", "Stage 1 Generalized (After Fix)", "Stage 2 LLM Distilled (Final)"]:
            if name in results:
                vals.append(fmt.format(results[name][key]))
            else:
                vals.append("N/A")
        print(f"{label:<30} | {vals[0]:<20} | {vals[1]:<20} | {vals[2]:<20}")
    print("=" * 90)

if __name__ == "__main__":
    main()
