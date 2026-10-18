import os
import glob
import torch
import torch.nn.functional as F
import torchaudio
import sentencepiece as spm
from train_stage1_asr import StudentAcousticModel, DATA_DIR, TOKENIZER_MODEL, CHECKPOINT_DIR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load Tokenizer
sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_MODEL)
blank_id = sp.piece_to_id("<blank>")

# 2. Load Model Checkpoint
model = StudentAcousticModel(
    input_dim=80,
    encoder_dim=256,
    num_encoder_layers=4,
    vocab_size=1024,
    teacher_dim=2048
).to(DEVICE)

ckpt_path = os.path.join(CHECKPOINT_DIR, "student_asr_epoch_5.pt")
model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
model.eval()
print(f"--> Loaded Checkpoint: {ckpt_path}")

# 3. Audio Transform
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    win_length=400,
    hop_length=160,
    n_mels=80
)

def greedy_ctc_decode(logits, blank_idx=blank_id):
    # logits: (1, T_sub, Vocab)
    preds = torch.argmax(logits, dim=-1).squeeze(0).tolist()
    deduped = []
    prev = None
    for p in preds:
        if p != prev:
            if p != blank_idx:
                deduped.append(p)
            prev = p
    return sp.decode_ids(deduped)

# 4. Run on a sample audio
audio_files = glob.glob(os.path.join(DATA_DIR, "**", "*.flac"), recursive=True)
if not audio_files:
    audio_files = glob.glob(os.path.join(DATA_DIR, "**", "*.wav"), recursive=True)

test_audio_path = audio_files[0]
waveform, sr = torchaudio.load(test_audio_path)
if sr != 16000:
    waveform = torchaudio.functional.resample(waveform, sr, 16000)
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

log_mel = torch.log(torch.clamp(mel_transform(waveform).squeeze(0), min=1e-5)).transpose(0, 1).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    ctc_logits, _ = model(log_mel)
    predicted_text = greedy_ctc_decode(ctc_logits)

print(f"\nSample Audio: {os.path.basename(test_audio_path)}")
print(f"Decoded Prediction: \"{predicted_text}\"")