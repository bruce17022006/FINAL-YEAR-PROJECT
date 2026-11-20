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

manifest = json.load(open("iemocap_4class_manifest.json", encoding="utf-8"))
xxx_manifest = json.load(open("iemocap_xxx_disambiguated.json", encoding="utf-8"))

best_by_emotion = {"Happy": [], "Sad": [], "Angry": [], "Neutral": []}

print("Searching for low WER / low CER / low CTC loss samples across IEMOCAP...")

count = 0
for item in manifest:
    wav_path = item["wav_path"]
    if not os.path.exists(wav_path):
        continue
    gt_text = item["text"].strip()
    if len(gt_text.split()) < 3 or len(gt_text.split()) > 12 or "[" in gt_text:
        continue
    
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
        
        # CTC Loss
        target_ids = sp.encode_as_ids(gt_text.lower())
        target_tensor = torch.tensor(target_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
        target_lens = torch.tensor([len(target_ids)], dtype=torch.long, device=DEVICE)
        log_probs = F.log_softmax(ctc_logits.float(), dim=-1).transpose(0, 1)
        ctc_loss = ctc_loss_fn(log_probs, target_tensor, out_lens, target_lens).item()
        
        # Emotion
        emo_probs = F.softmax(emo_logits[0], dim=-1).cpu().tolist()
        pred_idx = int(torch.argmax(emo_logits[0]).item())
        pred_emo = LABEL_NAMES[pred_idx]
        conf = emo_probs[pred_idx]
        
    # WER & CER
    r_words = gt_text.upper().split()
    h_words = pred_text.upper().split()
    w_dist = edit_distance(r_words, h_words)
    wer = (w_dist / len(r_words)) * 100.0
    
    r_chars = list(gt_text.upper())
    h_chars = list(pred_text.upper())
    c_dist = edit_distance(r_chars, h_chars)
    cer = (c_dist / len(r_chars)) * 100.0
    
    target_emo = item["emotion_label"].capitalize()
    if pred_emo == target_emo and wer <= 50.0 and ctc_loss < 3.0:
        best_by_emotion[target_emo].append({
            "utt_id": item["utt_id"],
            "wav_path": wav_path,
            "session": item["session"],
            "gt_text": gt_text,
            "pred_text": pred_text,
            "ctc_loss": round(ctc_loss, 4),
            "wer": round(wer, 2),
            "cer": round(cer, 2),
            "conf": round(conf * 100, 1),
            "distribution": {LABEL_NAMES[i]: round(emo_probs[i]*100, 1) for i in range(4)}
        })
        
    count += 1
    if count % 200 == 0:
        print(f"Scanned {count} samples... Found: { {k: len(v) for k, v in best_by_emotion.items()} }")
        if all(len(v) >= 3 for v in best_by_emotion.values()):
            break

# Sort each by lowest WER then lowest CTC loss
print("\n=== TOP LOW-ERROR SAMPLES PER EMOTION ===")
for emo in ["Happy", "Sad", "Angry", "Neutral"]:
    best_by_emotion[emo].sort(key=lambda x: (x["wer"], x["cer"], x["ctc_loss"]))
    print(f"\n--- Best for {emo} ---")
    for s in best_by_emotion[emo][:3]:
        print(f"  {s['utt_id']} | WER: {s['wer']}% | CER: {s['cer']}% | CTC Loss: {s['ctc_loss']} | Conf: {s['conf']}%")
        print(f"    GT:   \"{s['gt_text']}\"")
        print(f"    Pred: \"{s['pred_text']}\"")

# Search xxx samples for lowest WER
print("\nSearching for low-error Ambiguous (xxx) samples...")
xxx_best = []
for item in xxx_manifest:
    utt = item["utt_id"]
    sess = item["session"]
    dialogue = "_".join(utt.split("_")[:-1])
    wav_path = os.path.join(
        "IEMOCAP_full_release_withoutVideos", "IEMOCAP_full_release",
        sess, "sentences", "wav", dialogue, utt + ".wav"
    )
    if not os.path.exists(wav_path):
        continue
    gt_text = item["text"].strip()
    if len(gt_text.split()) < 3 or len(gt_text.split()) > 10 or "[" in gt_text:
        continue
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
        conf = emo_probs[pred_idx]
    r_words = gt_text.upper().split()
    h_words = pred_text.upper().split()
    wer = (edit_distance(r_words, h_words) / len(r_words)) * 100.0
    cer = (edit_distance(list(gt_text.upper()), list(pred_text.upper())) / len(gt_text)) * 100.0
    if wer <= 50.0 and ctc_loss < 3.0:
        xxx_best.append({
            "utt_id": utt,
            "wav_path": wav_path,
            "session": sess,
            "gt_text": gt_text,
            "pred_text": pred_text,
            "ctc_loss": round(ctc_loss, 4),
            "wer": round(wer, 2),
            "cer": round(cer, 2),
            "conf": round(conf * 100, 1),
            "pred_emo": pred_emo,
            "votes": item["human_evaluator_votes"],
            "distribution": {LABEL_NAMES[i]: round(emo_probs[i]*100, 1) for i in range(4)}
        })
        if len(xxx_best) >= 5:
            break

xxx_best.sort(key=lambda x: (x["wer"], x["cer"], x["ctc_loss"]))
print("\n--- Best for Ambiguous (xxx) ---")
for s in xxx_best[:3]:
    print(f"  {s['utt_id']} | WER: {s['wer']}% | CER: {s['cer']}% | CTC Loss: {s['ctc_loss']} | Pred: {s['pred_emo']} ({s['conf']}%)")
    print(f"    GT:   \"{s['gt_text']}\"")
    print(f"    Pred: \"{s['pred_text']}\"")
    print(f"    Votes: {s['votes']}")

with open("best_low_error_samples.json", "w", encoding="utf-8") as f:
    json.dump({"4class": best_by_emotion, "xxx": xxx_best}, f, indent=2)
print("Saved to best_low_error_samples.json")
