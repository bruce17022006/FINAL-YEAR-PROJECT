import os
import math
import glob
import torch
import torchaudio
import sentencepiece as spm
from train_stage1_asr import (
    StudentAcousticModel, 
    DATA_DIR, 
    TOKENIZER_MODEL, 
    CHECKPOINT_DIR
)

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
    teacher_dim=2048,
    blank_id=blank_id
).to(DEVICE)

ckpt_path = os.path.join(CHECKPOINT_DIR, "best_student_asr.pt")
model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
model.eval()

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=80
)

# 2. Get Audio & Ground Truth
audio_files = glob.glob(os.path.join(DATA_DIR, "**", "*.flac"), recursive=True) or \
              glob.glob(os.path.join(DATA_DIR, "**", "*.wav"), recursive=True)

test_audio = audio_files[0]
utt_id = os.path.splitext(os.path.basename(test_audio))[0]

# Find Transcript
ground_truth = "Transcript not found"
for tf in glob.glob(os.path.join(DATA_DIR, "**", "*.trans.txt"), recursive=True):
    with open(tf, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] == utt_id:
                ground_truth = " ".join(parts[1:]).lower()
                break

# 3. Audio Preprocessing (Matching Training Exactly)
waveform, sr = torchaudio.load(test_audio)
if sr != 16000:
    waveform = torchaudio.functional.resample(waveform, sr, 16000)
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

mel = mel_transform(waveform).squeeze(0)
log_mel = torch.log(torch.clamp(mel, min=1e-6)).transpose(0, 1)
log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
log_mel = log_mel.unsqueeze(0).to(DEVICE)

# 4. Greedy CTC Decoding
with torch.no_grad():
    logits, _ = model(log_mel)
    preds = torch.argmax(logits, dim=-1).squeeze(0).tolist()

deduped = []
prev = None
for p in preds:
    if p != prev:
        if p != blank_id:
            deduped.append(p)
        prev = p

decoded_text = sp.decode_ids(deduped)

print(f"\n================ ASR Evaluation ================")
print(f"Audio File     : {os.path.basename(test_audio)}")
print(f"Ground Truth   : \"{ground_truth}\"")
print(f"ASR Prediction : \"{decoded_text}\"")