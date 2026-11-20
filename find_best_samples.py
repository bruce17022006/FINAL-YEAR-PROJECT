import os
import json
import torch
import torchaudio
import torch.nn.functional as F
import sentencepiece as spm
from multitask_conformer_model import MultiTaskConformerASR_SER

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MULTITASK_CKPT = os.path.join("checkpoints_multitask", "best_multitask_conformer_iemocap.pt")
TOKENIZER_MODEL = "student_bpe_1024.model"
LABEL_NAMES = ["Neutral", "Angry", "Sad", "Happy"]
BLANK_ID = 4

sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_MODEL)

model = MultiTaskConformerASR_SER(
    input_dim=80, encoder_dim=256, num_layers=6, num_heads=4,
    vocab_size=1024, teacher_dim=2048, num_emotions=4, dropout=0.0
).to(DEVICE)
state = torch.load(MULTITASK_CKPT, map_location=DEVICE, weights_only=True)
model.load_state_dict(state, strict=True)
model.eval()

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=80
)

manifest = json.load(open("iemocap_4class_manifest.json", encoding="utf-8"))
xxx_manifest = json.load(open("iemocap_xxx_disambiguated.json", encoding="utf-8"))

def eval_wav(wav_path):
    waveform, sr = torchaudio.load(wav_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    mel = mel_transform(waveform).squeeze(0)
    log_mel = torch.log(torch.clamp(mel, min=1e-6)).transpose(0, 1)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    log_mel = log_mel.unsqueeze(0).to(DEVICE)
    in_lens = torch.tensor([log_mel.shape[1]], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        ctc_logits, emo_logits, out_lens = model(log_mel, in_lens)
        probs = F.softmax(emo_logits[0], dim=-1).cpu().tolist()
        pred_idx = int(torch.argmax(emo_logits[0]).item())
    return LABEL_NAMES[pred_idx], probs[pred_idx], probs

# Find top matches for each emotion
best_samples = {}
target_emotions = ["Neutral", "Angry", "Sad", "Happy"]

for item in manifest:
    if item["session"] == "Session5" and os.path.exists(item["wav_path"]):
        lbl = item["emotion_label"].capitalize()
        # only consider meaningful sentences (length > 10 chars)
        if len(item["text"].strip()) < 12:
            continue
        pred_lbl, conf, probs = eval_wav(item["wav_path"])
        if pred_lbl == lbl:
            if lbl not in best_samples or conf > best_samples[lbl]["confidence"]:
                best_samples[lbl] = {
                    "item": item,
                    "confidence": conf,
                    "probs": probs
                }
        if len(best_samples) == 4 and all(best_samples[k]["confidence"] > 0.70 for k in target_emotions):
            break

print("Best 4-class matches found:")
for k, v in best_samples.items():
    print(f"{k}: {v['item']['utt_id']} (Conf: {v['confidence']*100:.1f}%) | Text: {v['item']['text']}")
