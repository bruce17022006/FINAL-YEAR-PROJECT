import os
import torch
import torch.nn.functional as F
import sentencepiece as spm

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
SAMPLE_ID = "103-1240-0000"
CACHE_FILE = os.path.join(PBKD_DIR, SAMPLE_ID + ".pt")
CHECKPOINTS = [
    os.path.join(CHECKPOINT_DIR, "best_student_asr.pt"),
    os.path.join(CHECKPOINT_DIR, "best_student_asr_corrected.pt"),
]

def ctc_decode(ids, blank_id, sp):
    result = []
    previous = None
    for token_id in ids:
        if token_id != previous and token_id != blank_id:
            result.append(token_id)
        previous = token_id
    return sp.decode_ids(result)

def main():
    print("=== ASR Checkpoint Comparison ===")
    print("Device:", DEVICE)
    if not os.path.isfile(CACHE_FILE):
        raise FileNotFoundError(f"Cache not found: {CACHE_FILE}")
    if not os.path.isfile(TOKENIZER_MODEL):
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_MODEL}")

    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL)
    blank_id = sp.piece_to_id("<blank>")

    sample = torch.load(CACHE_FILE, map_location="cpu", weights_only=True)
    mel = sample["student_mel"].float()
    target = torch.as_tensor(sample["student_target_ids"], dtype=torch.long)
    reference = sample.get("text", "")

    print("Sample:", SAMPLE_ID)
    print("Reference:", reference.upper())
    print("Decoded target:", sp.decode_ids(target.tolist()).upper())
    print("Mel shape:", tuple(mel.shape), "| Target length:", target.numel())

    input_length = torch.tensor([(mel.shape[0] + 1) // 2], dtype=torch.long, device=DEVICE)
    target_length = torch.tensor([target.numel()], dtype=torch.long, device=DEVICE)

    for checkpoint_path in CHECKPOINTS:
        print("\n" + "=" * 65)
        print("Checkpoint:", checkpoint_path)
        if not os.path.isfile(checkpoint_path):
            print("SKIPPED: checkpoint file not found.")
            continue

        model = StudentAcousticModel(
            input_dim=80,
            encoder_dim=STUDENT_DIM,
            num_encoder_layers=4,
            vocab_size=VOCAB_SIZE,
            teacher_dim=TEACHER_DIM,
            blank_id=blank_id,
        ).to(DEVICE)

        state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()

        with torch.no_grad():
            logits, _ = model(mel.unsqueeze(0).to(DEVICE))
            log_probs = F.log_softmax(logits.float(), dim=-1)
            loss = F.ctc_loss(
                log_probs.transpose(0, 1),
                target.to(DEVICE),
                input_length,
                target_length,
                blank=blank_id,
                reduction="none",
                zero_infinity=False,
            )[0]

            probs = log_probs.exp()
            ids = logits.argmax(dim=-1)[0].cpu().tolist()
            blank_frames = sum(token == blank_id for token in ids)
            blank_probability = probs[0, :, blank_id].mean().item()
            prediction = ctc_decode(ids, blank_id, sp)

        print("Logits shape:", tuple(logits.shape))
        print(f"CTC loss: {loss.item():.4f}")
        print(f"Blank frames: {blank_frames}/{len(ids)} ({100 * blank_frames / len(ids):.2f}%)")
        print(f"Mean blank probability: {blank_probability:.6f}")
        print("Prediction:", prediction.upper())

    print("\nComparison complete. No training was run and no checkpoint was modified.")

if __name__ == "__main__":
    main()
