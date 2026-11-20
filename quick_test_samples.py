import os
import json
import torch
import torchaudio
import torch.nn as nn
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
ctc_loss_fn = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)

def edit_distance(a, b):
    dp = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(dp[j] + 1, cur[j - 1] + 1, dp[j - 1] + (x != y))
        dp = cur
    return dp[-1]

def clean_ctc_decode(raw_ids):
    dedup = []
    prev = None
    for idx in raw_ids:
        if idx != prev:
            if idx != BLANK_ID:
                dedup.append(idx)
            prev = idx
    try:
        text = sp.decode_ids(dedup)
        text = text.replace("⁇", "").strip()
        text = " ".join(text.split())
        return text.upper()
    except Exception:
        return ""

def eval_one(wav_path, gt_text):
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
        raw_ids = ctc_logits.argmax(dim=-1)[0].cpu().tolist()
        pred_text = clean_ctc_decode(raw_ids)
        
        target_ids = sp.encode_as_ids(gt_text.lower())
        target_tensor = torch.tensor(target_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
        target_lens = torch.tensor([len(target_ids)], dtype=torch.long, device=DEVICE)
        log_probs = F.log_softmax(ctc_logits.float(), dim=-1).transpose(0, 1)
        ctc_loss = ctc_loss_fn(log_probs, target_tensor, out_lens, target_lens).item()
        
        emo_probs = F.softmax(emo_logits[0], dim=-1).cpu().tolist()
        pred_idx = int(torch.argmax(emo_logits[0]).item())
        pred_emo = LABEL_NAMES[pred_idx]
        conf = emo_probs[pred_idx] * 100.0
        
    r_words = gt_text.upper().split()
    h_words = pred_text.upper().split()
    wer = (edit_distance(r_words, h_words) / len(r_words)) * 100.0
    
    r_chars = list(gt_text.upper())
    h_chars = list(pred_text.upper())
    cer = (edit_distance(r_chars, h_chars) / len(r_chars)) * 100.0
    
    return {
        "pred_text": pred_text,
        "wer": round(wer, 2),
        "cer": round(cer, 2),
        "ctc_loss": round(ctc_loss, 4),
        "pred_emo": pred_emo,
        "conf": round(conf, 1),
        "probs": {LABEL_NAMES[i]: round(emo_probs[i]*100, 1) for i in range(4)}
    }

# Test test_sample.wav
print("test_sample.wav:")
res = eval_one("test_sample.wav", "SHE COULD GET ALONG WITHOUT ME AND I COULD GET ALONG WITHOUT HER IN THE WORLD")
print(res)

print("\ntest_sample_2.wav:")
res2 = eval_one("test_sample_2.wav", "THERE ARE PLENTY OF PEOPLE WHO CAN CLOSELY ATTEND TO THEIR BUSINESS")
print(res2)
