import os
import sys
import argparse
import torch
import torchaudio
import sentencepiece as spm

from conformer_asr import StudentConformerASR
from train_stage1_asr_generalized import StudentAcousticModel, TOKENIZER_MODEL, ctc_decode
from rescore_pipeline import TwoPassRescorer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_CKPT = os.path.join("checkpoints_conformer", "best_conformer_stage2_distilled.pt")
FALLBACK_CKPTS = [
    os.path.join("checkpoints_conformer", "best_student_conformer_4x.pt"),
    os.path.join("checkpoints_stage2", "best_student_asr_stage2.pt"),
    os.path.join("checkpoints_stage1", "best_student_asr_generalized.pt"),
]

def resolve_checkpoint(path=None):
    if path and os.path.isfile(path):
        return path
    if os.path.isfile(DEFAULT_CKPT):
        return DEFAULT_CKPT
    for fb in FALLBACK_CKPTS:
        if os.path.isfile(fb):
            return fb
    raise FileNotFoundError("No ASR checkpoint found! Train or supply a valid checkpoint.")

def load_asr_model(checkpoint_path):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.isfile(TOKENIZER_MODEL):
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_MODEL}")

    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL)
    blank_id = sp.piece_to_id("<blank>")

    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    is_conformer = any("conformer_layers" in k for k in state.keys())

    if is_conformer:
        model = StudentConformerASR(
            input_dim=80,
            encoder_dim=256,
            num_layers=6,
            num_heads=4,
            vocab_size=1024,
            teacher_dim=2048,
            dropout=0.0
        ).to(DEVICE)
    else:
        model = StudentAcousticModel(
            input_dim=80,
            encoder_dim=256,
            num_encoder_layers=4,
            num_heads=4,
            vocab_size=1024,
            teacher_dim=2048,
            dropout=0.0
        ).to(DEVICE)

    model.load_state_dict(state, strict=True)
    model.eval()

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=80
    )
    return model, sp, blank_id, mel_transform, is_conformer

def transcribe_audio(audio_path, checkpoint_path=None, blank_penalty=0.5, two_pass=False):
    ckpt_path = resolve_checkpoint(checkpoint_path)
    print(f"--> Audio File : {audio_path}")
    print(f"--> Checkpoint : {ckpt_path}")

    if two_pass:
        print("--> Mode: Two-Pass Rescoring (Conformer ASR + Causal LM Rescorer)...")
        rescorer = TwoPassRescorer(
            asr_checkpoint_path=ckpt_path,
            sp_model_path=TOKENIZER_MODEL,
            lm_model_id="gpt2",
            beam_width=8,
            lm_weight=0.4
        )
        best_text, rescored = rescorer.transcribe(audio_path)
        return best_text, rescored

    model, sp, blank_id, mel_transform, is_conformer = load_asr_model(ckpt_path)

    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    mel = mel_transform(waveform).squeeze(0)
    log_mel = torch.log(torch.clamp(mel, min=1e-6)).transpose(0, 1)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    log_mel = log_mel.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits, _ = model(log_mel)
        if blank_penalty > 0.0:
            logits[:, :, blank_id] -= blank_penalty
        raw_ids = logits.argmax(dim=-1)[0].cpu().tolist()
        text = ctc_decode(raw_ids, blank_id, sp)

    return text, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe Any Audio using Trained ASR Model")
    parser.add_argument("audio", nargs="?", default="test_sample.wav", help="Path to audio file (.wav, .flac, .mp3)")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to model checkpoint (.pt)")
    parser.add_argument("--two_pass", action="store_true", help="Enable two-pass LM rescoring")
    parser.add_argument("--penalty", type=float, default=0.5, help="Blank penalty (default: 0.5)")

    args = parser.parse_args()

    if not os.path.isfile(args.audio):
        print(f"Error: Audio file not found at '{args.audio}'")
        sys.exit(1)

    result_text, extra = transcribe_audio(
        args.audio,
        checkpoint_path=args.ckpt,
        blank_penalty=args.penalty,
        two_pass=args.two_pass
    )

    if extra:
        print("\n--- Top Hypotheses from Two-Pass Rescorer ---")
        for i, c in enumerate(extra, 1):
            marker = " <-- [SELECTED]" if i == 1 else ""
            print(f"[{i}] \"{c['text'].upper()}\" | Total Score: {c['total_score']:.2f}{marker}")

    print("\n" + "=" * 60)
    print("FINAL TRANSCRIPTION RESULT:")
    print("=" * 60)
    print(result_text.upper() if result_text else "[No speech detected]")
