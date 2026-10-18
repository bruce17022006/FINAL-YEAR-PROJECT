import os
import torch
import torchaudio
import sentencepiece as spm
import whisper
import textwrap
from jiwer import wer
from train_stage1_asr import (
    StudentAcousticModel, 
    TOKENIZER_MODEL, 
    CHECKPOINT_DIR
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load Model & Tokenizer
sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_MODEL)
blank_id = sp.piece_to_id("<blank>")

model = StudentAcousticModel(
    input_dim=80, encoder_dim=256, num_encoder_layers=4,
    vocab_size=1024, teacher_dim=2048, blank_id=blank_id
).to(DEVICE)

ckpt_path = os.path.join(CHECKPOINT_DIR, "best_student_asr.pt")
model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
model.eval()

print("DEBUG: Loading Whisper 'base'...")
whisper_model = whisper.load_model("base")

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=80
)

def print_boxed_result(audio_path, reference, prediction, error_rate):
    width = 80
    wrapper = textwrap.TextWrapper(width=width-4)
    
    # Header
    print("\n" + "╔" + "═"*(width-2) + "╗")
    print(f"║ {'ASR EVALUATION REPORT':^{width-4}} ║")
    print("╠" + "═"*(width-2) + "╣")
    
    # File Info
    print(f"║ File: {os.path.basename(audio_path):<{width-10}} ║")
    print("╟" + "─"*(width-2) + "╢")
    
    # Reference (Whisper)
    print(f"║ REFERENCE (WHISPER): {' '*(width-23)} ║")
    for line in wrapper.wrap(reference):
        print(f"║ {line:<{width-4}} ║")
    
    print("╟" + "─"*(width-2) + "╢")
    
    # Prediction (Yours)
    print(f"║ PREDICTION (STUDENT): {' '*(width-24)} ║")
    for line in wrapper.wrap(prediction):
        print(f"║ {line:<{width-4}} ║")
    
    print("╠" + "═"*(width-2) + "╣")
    
    # Metrics
    acc = (1 - error_rate) * 100
    metrics_str = f"WER: {error_rate:.2%}  |  ACCURACY: {acc:.2f}%"
    print(f"║ {metrics_str:^{width-4}} ║")
    
    # Status Indicator
    status = "SUCCESS" if error_rate < 0.3 else "POOR PERFORMANCE"
    print(f"║ Status: {status:^{width-12}} ║")
    
    print("╚" + "═"*(width-2) + "╝\n")

def evaluate_unseen_mp3(audio_path):
    # A. Whisper Reference
    whisper_result = whisper_model.transcribe(audio_path)
    reference_text = whisper_result["text"].lower().strip()

    # B. Audio Loading
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    mel = mel_transform(waveform).squeeze(0)
    log_mel = torch.log(torch.clamp(mel, min=1e-6)).transpose(0, 1)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    log_mel = log_mel.unsqueeze(0).to(DEVICE)

    # C. Prediction
    with torch.no_grad():
        logits, _ = model(log_mel)
        preds = torch.argmax(logits, dim=-1).squeeze(0).tolist()

    deduped = []
    prev = None
    for p in preds:
        if p != prev and p != blank_id:
            deduped.append(p)
        prev = p
    
    predicted_text = sp.decode_ids(deduped).lower().strip()

    # D. Calculation
    error_rate = wer(reference_text, predicted_text)

    # E. Neat Print
    print_boxed_result(audio_path, reference_text, predicted_text, error_rate)

# --- EXECUTION ---
my_mp3_file = r"E:\FINAL YEAR PROJECT\Implementation\sample-speech-1m.mp3"

if os.path.exists(my_mp3_file):
    evaluate_unseen_mp3(my_mp3_file)
else:
    print(f"Error: File not found at {my_mp3_file}")