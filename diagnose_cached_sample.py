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
CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_student_asr_corrected.pt")
CACHE_FILE = os.path.join(PBKD_DIR, SAMPLE_ID + ".pt")

def decode_ctc(ids, blank_id, sp):
    collapsed = []
    previous = None
    for token_id in ids:
        if token_id != previous and token_id != blank_id:
            collapsed.append(token_id)
        previous = token_id
    return sp.decode_ids(collapsed), collapsed

def main():
    print("=== Cached Sample / Checkpoint Diagnostic ===")
    print("Device:", DEVICE)
    print("Cache:", CACHE_FILE)
    print("Checkpoint:", CHECKPOINT)

    if not os.path.isfile(CACHE_FILE):
        raise FileNotFoundError(f"Cache sample not found: {CACHE_FILE}")
    if not os.path.isfile(CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
    if not os.path.isfile(TOKENIZER_MODEL):
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_MODEL}")

    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL)
    blank_id = sp.piece_to_id("<blank>")

    item = torch.load(CACHE_FILE, map_location="cpu", weights_only=True)
    mel = item["student_mel"].float()
    target = torch.as_tensor(item["student_target_ids"], dtype=torch.long)
    reference = item.get("text", "")

    print("\n--- Cache checks ---")
    print("file_id:", item.get("file_id", SAMPLE_ID))
    print("mel shape:", tuple(mel.shape))
    print("mel finite:", bool(torch.isfinite(mel).all()))
    print("mel mean/std:", round(mel.mean().item(), 5), round(mel.std().item(), 5))
    print("target length:", target.numel())
    print("target ID min/max:", int(target.min()), int(target.max()))
    print("blank ID:", blank_id)
    print("target contains blank:", bool((target == blank_id).any()))
    print("reference:", reference.upper())
    print("decoded cached target:", sp.decode_ids(target.tolist()).upper())

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

    with torch.no_grad():
        logits, _ = model(mel.unsqueeze(0).to(DEVICE))
        log_probs = F.log_softmax(logits.float(), dim=-1)

        raw_ids = logits.argmax(dim=-1)[0].cpu().tolist()
        prediction, collapsed = decode_ctc(raw_ids, blank_id, sp)
        blank_frames = sum(x == blank_id for x in raw_ids)
        probs = log_probs.exp()
        mean_blank_prob = probs[0, :, blank_id].mean().item()

        input_length = torch.tensor([(mel.shape[0] + 1) // 2], dtype=torch.long, device=DEVICE)
        target_length = torch.tensor([target.numel()], dtype=torch.long, device=DEVICE)
        loss = F.ctc_loss(
            log_probs.transpose(0, 1),
            target.to(DEVICE),
            input_length,
            target_length,
            blank=blank_id,
            reduction="none",
            zero_infinity=False,
        )[0]

        mean_probs = probs.mean(dim=(0, 1))
        top_values, top_ids = torch.topk(mean_probs, k=10)

    print("\n--- Checkpoint prediction on cached features ---")
    print("logits shape [B,T,V]:", tuple(logits.shape))
    print("expected output frames ceil(T/2):", (mel.shape[0] + 1) // 2)
    print("actual output frames:", logits.shape[1])
    print("blank frames:", f"{blank_frames}/{len(raw_ids)} ({100 * blank_frames / len(raw_ids):.2f}%)")
    print("mean blank probability:", round(mean_blank_prob, 6))
    print("CTC loss on this cached sample:", float(loss.item()))
    print("logits finite:", bool(torch.isfinite(logits).all()))
    print("collapsed token count:", len(collapsed))
    print("prediction:", prediction.upper())
    print("\nTop 10 mean-probability tokens:")
    for token_id, value in zip(top_ids.tolist(), top_values.tolist()):
        print(f"  id={token_id:4d} piece={sp.id_to_piece(token_id)!r} mean_prob={value:.6f}")

    print("\nDiagnostic completed. No training was started and no checkpoint was modified.")

if __name__ == "__main__":
    main()
