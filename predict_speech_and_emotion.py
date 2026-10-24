import os
import sys
import argparse
import torch
import torchaudio
import torch.nn.functional as F
import sentencepiece as spm

sys.stdout.reconfigure(encoding='utf-8')

from multitask_conformer_model import MultiTaskConformerASR_SER

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MULTITASK_CKPT = os.path.join("checkpoints_multitask", "best_multitask_conformer_iemocap.pt")
TOKENIZER_MODEL = "student_bpe_1024.model"
LABEL_NAMES = ["Neutral", "Angry", "Sad", "Happy"]
BLANK_ID = 4

def ctc_decode(raw_ids, blank_id, sp):
    dedup = []
    prev = None
    for idx in raw_ids:
        if idx != prev:
            if idx != blank_id:
                dedup.append(idx)
            prev = idx
    try:
        return sp.decode_ids(dedup)
    except:
        return ""

class JointSpeechEmotionPredictor:
    """
    Joint Inference Pipeline:
      Input: Raw Audio (.wav, .mp3, etc.)
      Output 1: Transcribed Speech Text (ASR)
      Output 2: Detected Emotion Class & Confidence Distribution (SER)
    """
    def __init__(self, checkpoint_path=MULTITASK_CKPT, tokenizer_path=TOKENIZER_MODEL):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if not os.path.isfile(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

        print(f"--> Loading Multi-Task Conformer from: {checkpoint_path}")
        self.device = DEVICE

        # 1. Load Tokenizer
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)

        # 2. Load Model
        self.model = MultiTaskConformerASR_SER(
            input_dim=80,
            encoder_dim=256,
            num_layers=6,
            num_heads=4,
            vocab_size=1024,
            teacher_dim=2048,
            num_emotions=4,
            dropout=0.0
        ).to(self.device)

        state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=80
        )
        print("--> Multi-Task Conformer Ready for Dual Inference!\n")

    def predict(self, audio_path, blank_penalty=0.5):
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        mel = self.mel_transform(waveform).squeeze(0)
        log_mel = torch.log(torch.clamp(mel, min=1e-6)).transpose(0, 1)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
        log_mel = log_mel.unsqueeze(0).to(self.device)
        in_lens = torch.tensor([log_mel.shape[1]], dtype=torch.long, device=self.device)

        with torch.no_grad():
            ctc_logits, emo_logits, out_lens = self.model(log_mel, in_lens)

            # 1. Decode Speech Text (ASR)
            if blank_penalty > 0.0:
                ctc_logits[:, :, BLANK_ID] -= blank_penalty
            raw_ids = ctc_logits.argmax(dim=-1)[0].cpu().tolist()
            text = ctc_decode(raw_ids, BLANK_ID, self.sp)

            # 2. Decode Emotion Probabilities (SER)
            emo_probs = F.softmax(emo_logits[0], dim=-1).cpu().tolist()
            pred_idx = int(torch.argmax(emo_logits[0]).item())
            pred_emotion = LABEL_NAMES[pred_idx]
            pred_confidence = emo_probs[pred_idx] * 100.0

            emotion_dist = {LABEL_NAMES[i]: emo_probs[i] * 100.0 for i in range(len(LABEL_NAMES))}

        return {
            "transcription": text.upper() if text else "[No Speech Detected]",
            "predicted_emotion": pred_emotion,
            "confidence": pred_confidence,
            "distribution": emotion_dist
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Joint Speech Recognition (ASR) and Emotion Recognition (SER)")
    parser.add_argument("audio", nargs="?", default="test_sample.wav", help="Path to input audio file (.wav)")
    parser.add_argument("--penalty", type=float, default=0.5, help="CTC blank penalty (default: 0.5)")
    args = parser.parse_args()

    predictor = JointSpeechEmotionPredictor()
    result = predictor.predict(args.audio, blank_penalty=args.penalty)

    print("=" * 65)
    print("JOINT MULTI-TASK PREDICTION RESULT:")
    print("=" * 65)
    print(f"File: {args.audio}")
    print(f"\n[1] RECOGNIZED SPEECH (TEXT):")
    print(f"    --> \"{result['transcription']}\"")
    print(f"\n[2] DETECTED EMOTION:")
    print(f"    --> {result['predicted_emotion'].upper()} ({result['confidence']:.2f}% confidence)")
    print("\n[3] EMOTION PROBABILITY DISTRIBUTION:")
    for emo, prob in result["distribution"].items():
        bar = "#" * int(prob / 5)
        print(f"    {emo:<8}: {prob:>5.1f}% [{bar:<20}]")
    print("=" * 65)
