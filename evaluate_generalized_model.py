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
    CHECKPOINT_DIR,
    VOCAB_SIZE,
    STUDENT_DIM,
    TEACHER_DIM,
    ctc_decode,
    SEED,
    VAL_SPLIT,
    CACHE_DIR
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_student_asr_generalized.pt")

def edit_distance(a, b):
    dp = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(dp[j] + 1, cur[j-1] + 1, dp[j-1] + (x != y))
        dp = cur
    return dp[-1]

def wer(ref, hyp):
    r, h = ref.lower().split(), hyp.lower().split()
    return edit_distance(r, h) / max(1, len(r))

def cer(ref, hyp):
    r, h = list(ref.lower().replace(" ", "")), list(hyp.lower().replace(" ", ""))
    return edit_distance(r, h) / max(1, len(r))

def main():
    print("=" * 80)
    print("EVALUATING GENERALIZED ASR MODEL ON UNSEEN HELD-OUT AUDIO SAMPLES")
    print("=" * 80)

    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL)
    blank_id = sp.piece_to_id("<blank>")

    all_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.pt")))
    random.seed(SEED)
    random.shuffle(all_files)
    val_count = int(len(all_files) * VAL_SPLIT)
    val_files = all_files[:val_count]

    model = StudentAcousticModel(
        input_dim=80, encoder_dim=STUDENT_DIM, num_encoder_layers=4,
        num_heads=4, vocab_size=VOCAB_SIZE, teacher_dim=TEACHER_DIM, dropout=0.0
    ).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    eval_samples = val_files[:8]
    betas = [0.0, 1.0, 1.5, 2.0]

    for beta in betas:
        total_wer = 0.0
        total_cer = 0.0
        print(f"\n" + "-" * 80)
        print(f"--- DECODING RESULTS WITH BLANK PENALTY beta = {beta:.1f} ---")
        print("-" * 80)

        with torch.no_grad():
            for idx, fpath in enumerate(eval_samples, 1):
                data = torch.load(fpath, map_location="cpu", weights_only=True)
                mel = data["student_mel"].float().unsqueeze(0).to(DEVICE)
                ref = data.get("text", "")

                logits, _ = model(mel) # [1, T, V]
                mod_logits = logits.clone()
                mod_logits[:, :, blank_id] -= beta
                raw_ids = mod_logits.argmax(dim=-1)[0].cpu().tolist()
                pred = ctc_decode(raw_ids, blank_id, sp)

                sample_wer = wer(ref, pred)
                sample_cer = cer(ref, pred)
                total_wer += sample_wer
                total_cer += sample_cer

                if beta == 1.5 or beta == 0.0:
                    print(f"\n[Sample {idx}] File: {os.path.basename(fpath)}")
                    print(f"  REF : {ref.upper()}")
                    print(f"  PRED: {pred.upper()}")
                    print(f"  WER : {sample_wer*100:.1f}% | CER: {sample_cer*100:.1f}%")

        avg_wer = total_wer / len(eval_samples)
        avg_cer = total_cer / len(eval_samples)
        print(f"\n>> Beta={beta:.1f} Summary | Avg WER: {avg_wer*100:.2f}% | Avg CER: {avg_cer*100:.2f}%")

if __name__ == "__main__":
    main()
