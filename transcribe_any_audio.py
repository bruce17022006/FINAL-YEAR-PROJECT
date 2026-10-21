import os
import torch
import torchaudio
import sentencepiece as spm
from train_stage1_asr import StudentAcousticModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER_PATH = "student_bpe_1024.model"
CHECKPOINT_PATH = os.path.join("checkpoints_stage1", "best_student_asr.pt")

# 1. Load Tokenizer & IDs
sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_PATH)
blank_id = sp.piece_to_id("<blank>")
unk_id = sp.unk_id()

print(f"--> Blank ID: {blank_id} | UNK ID: {unk_id}")

model = StudentAcousticModel(
    input_dim=80,
    encoder_dim=256,
    num_encoder_layers=4,
    vocab_size=1024,
    teacher_dim=2048,
    blank_id=blank_id
).to(DEVICE)

model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
model.eval()

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=80
)

# 2. Decode Single Chunk
def decode_chunk(waveform_chunk):
    mel = mel_transform(waveform_chunk).squeeze(0)
    log_mel = torch.log(torch.clamp(mel, min=1e-6)).transpose(0, 1)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    log_mel = log_mel.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits, _ = model(log_mel)
        preds = torch.argmax(logits, dim=-1).squeeze(0).tolist()

    # Deduplicate consecutive tokens & remove blank and UNK tokens
    deduped = []
    prev = None
    for p in preds:
        if p != prev:
            if p != blank_id and p != unk_id:
                deduped.append(p)
            prev = p
            
    # Fallback: Top non-blank IDs
    text = sp.decode_ids(deduped)
    return text, deduped[:15]

# 3. Process Long Audio File
def transcribe_audio_file(audio_path, chunk_sec=6.0):
    waveform, sr = torchaudio.load(audio_path)

    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    total_samples = waveform.shape[1]
    chunk_samples = int(chunk_sec * 16000)
    transcription_list = []

    print(f"\n--> Audio Path: {audio_path}")
    print(f"--> Total Duration: {total_samples / 16000:.2f} seconds\n")

    chunk_idx = 1
    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        chunk = waveform[:, start:end]

        if chunk.shape[1] < int(1.5 * 16000):
            continue

        chunk_text, token_sample = decode_chunk(chunk)
        print(f"Chunk {chunk_idx} raw tokens: {token_sample}")
        if chunk_text.strip():
            print(f"Chunk {chunk_idx} ({start/16000:.1f}s - {end/16000:.1f}s): {chunk_text}")
            transcription_list.append(chunk_text.strip())
        chunk_idx += 1

    return " ".join(transcription_list)

if __name__ == "__main__":
    target_audio = "speech.wav"
    if os.path.exists(target_audio):
        final_text = transcribe_audio_file(target_audio, chunk_sec=6.0)
        print("\n" + "="*50)
        print("FINAL RECOGNIZED TRANSCRIPT:")
        print("="*50)
        print(final_text if final_text else "[Silence / Unknown Tokens filtered]")
        print("="*50 + "\n")