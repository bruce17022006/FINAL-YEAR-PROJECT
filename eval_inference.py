import os
import glob
import torch
import torch.nn.functional as F
import torchaudio
import sentencepiece as spm
from train_stage1_asr import StudentAcousticModel, DATA_DIR, TOKENIZER_MODEL, CHECKPOINT_DIR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load Tokenizer & Model
sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_MODEL)
blank_id = sp.piece_to_id("<blank>")

model = StudentAcousticModel(
    input_dim=80,
    encoder_dim=256,
    num_encoder_layers=4,
    vocab_size=1024,
    teacher_dim=2048
).to(DEVICE)

ckpt_path = os.path.join(CHECKPOINT_DIR, "best_student_asr.pt")
model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
model.eval()

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=80
)

# 2. Pick sample audio
audio_files = glob.glob(os.path.join(DATA_DIR, "**", "*.flac"), recursive=True) or \
              glob.glob(os.path.join(DATA_DIR, "**", "*.wav"), recursive=True)

test_audio = audio_files[0]
waveform, sr = torchaudio.load(test_audio)
if sr != 16000:
    waveform = torchaudio.functional.resample(waveform, sr, 16000)
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

log_mel = torch.log(torch.clamp(mel_transform(waveform).squeeze(0), min=1e-5)).transpose(0, 1).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    logits, _ = model(log_mel) # (1, T_sub, Vocab)

# 3. Diagnostic check on raw token predictions
raw_preds = torch.argmax(logits, dim=-1).squeeze(0).tolist()
blank_count = raw_preds.count(blank_id)
print(f"Total Subsampled Frames: {len(raw_preds)}")
print(f"Blank Token Predictions: {blank_count}/{len(raw_preds)} ({(blank_count/len(raw_preds))*100:.1f}%)")

# 4. Standard Greedy Decode
def decode(token_list):
    deduped = []
    prev = None
    for p in token_list:
        if p != prev:
            if p != blank_id:
                deduped.append(p)
            prev = p
    return sp.decode_ids(deduped)

print(f"\nDefault Prediction: \"{decode(raw_preds)}\"")

# 5. Suppressed Blank Decode (Testing if non-blank tokens have learned ordering)
suppressed_logits = logits.clone()
suppressed_logits[:, :, blank_id] -= 3.0
suppressed_preds = torch.argmax(suppressed_logits, dim=-1).squeeze(0).tolist()
print(f"With Blank Penalty (-3.0): \"{decode(suppressed_preds)}\"")