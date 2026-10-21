import os
import sys
import torch
import torch.nn.functional as F
import sentencepiece as spm

# Force UTF-8 for console output on Windows
sys.stdout.reconfigure(encoding='utf-8')

from train_stage1_asr import (
    StudentAcousticModel,
    TOKENIZER_MODEL,
    CHECKPOINT_DIR,
    PBKD_DIR,
    VOCAB_SIZE,
    STUDENT_DIM,
    TEACHER_DIM,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_IDS = ["103-1240-0000", "103-1240-0001", "103-1240-0002"]
CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_student_asr_corrected.pt")

def decode_ctc(ids, blank_id, sp):
    collapsed = []
    prev = None
    for tid in ids:
        if tid != prev and tid != blank_id:
            collapsed.append(tid)
        prev = tid
    return sp.decode_ids(collapsed)

def main():
    print("=== Testing Blank Penalty & Temperature Decoding ===")
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL)
    blank_id = sp.piece_to_id("<blank>")

    model = StudentAcousticModel(
        input_dim=80,
        encoder_dim=STUDENT_DIM,
        num_encoder_layers=4,
        vocab_size=VOCAB_SIZE,
        teacher_dim=TEACHER_DIM,
        blank_id=blank_id,
    ).to(DEVICE)
    state = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    penalties = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    for sid in SAMPLE_IDS:
        cache_file = os.path.join(PBKD_DIR, f"{sid}.pt")
        if not os.path.isfile(cache_file):
            continue
        data = torch.load(cache_file, map_location="cpu", weights_only=True)
        mel = data["student_mel"].float().unsqueeze(0).to(DEVICE)
        target = data["student_target_ids"]
        ref = data.get("text", "")

        print(f"\n" + "=" * 80)
        print(f"Sample: {sid} | Mel frames: {mel.shape[1]} | Target tokens: {len(target)}")
        print(f"REF: {ref.upper()}")
        print("-" * 80)

        with torch.no_grad():
            logits, _ = model(mel) # [1, T, V]
            probs = F.softmax(logits.float(), dim=-1)[0]
            blank_prob = probs[:, blank_id].mean().item()
            print(f"Raw mean blank probability: {blank_prob:.4f}")

            for beta in penalties:
                mod_logits = logits.clone()
                mod_logits[:, :, blank_id] -= beta
                raw_ids = mod_logits.argmax(dim=-1)[0].cpu().tolist()
                pred = decode_ctc(raw_ids, blank_id, sp)
                non_blank_emissions = sum(x != blank_id for x in raw_ids)
                print(f"[Blank Penalty beta={beta:3.1f}] emissions={non_blank_emissions:3d}/{len(raw_ids)}: {pred.upper()}")

if __name__ == "__main__":
    main()
