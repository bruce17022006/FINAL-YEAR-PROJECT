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

best_candidates = {emo: [] for emo in LABEL_NAMES}

# Scan up to 500 candidate utterances with 3 to 8 words
for item in manifest:
    wav_path = item["wav_path"]
    if not os.path.exists(wav_path):
        continue
    words = item["text"].strip().split()
    if len(words) < 3 or len(words) > 7 or any(c in item["text"] for c in "[]()<>"):
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
        
        target_ids = sp.encode_as_ids(item["text"].lower())
        target_tensor = torch.tensor(target_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
        target_lens = torch.tensor([len(target_ids)], dtype=torch.long, device=DEVICE)
        log_probs = F.log_softmax(ctc_logits.float(), dim=-1).transpose(0, 1)
        ctc_loss = ctc_loss_fn(log_probs, target_tensor, out_lens, target_lens).item()
        
        emo_probs = F.softmax(emo_logits[0], dim=-1).cpu().tolist()
        pred_idx = int(torch.argmax(emo_logits[0]).item())
        pred_emo = LABEL_NAMES[pred_idx]
        conf = emo_probs[pred_idx] * 100.0

    target_emo = item["emotion_label"].capitalize()
    if pred_emo == target_emo and conf > 50.0:
        r_words = item["text"].upper().split()
        h_words = pred_text.upper().split()
        wer = (edit_distance(r_words, h_words) / len(r_words)) * 100.0
        
        r_chars = list(item["text"].upper())
        h_chars = list(pred_text.upper())
        cer = (edit_distance(r_chars, h_chars) / len(r_chars)) * 100.0
        
        best_candidates[target_emo].append({
            "utt_id": item["utt_id"],
            "wav_path": wav_path,
            "session": item["session"],
            "gt_text": item["text"],
            "pred_text": pred_text,
            "ctc_loss": round(ctc_loss, 4),
            "wer": round(wer, 2),
            "cer": round(cer, 2),
            "conf": round(conf, 1),
            "probs": {LABEL_NAMES[i]: round(emo_probs[i]*100, 1) for i in range(4)}
        })

print("\n--- RESULTS ---")
for emo, cands in best_candidates.items():
    cands.sort(key=lambda x: (x["wer"], x["cer"], x["ctc_loss"]))
    print(f"\nTarget Emotion: {emo} (Total candidates matching: {len(cands)})")
    for c in cands[:3]:
        print(f"  {c['utt_id']} | WER: {c['wer']}% | CER: {c['cer']}% | CTC Loss: {c['ctc_loss']} | Conf: {c['conf']}%")
        print(f"    GT:   \"{c['gt_text']}\"")
        print(f"    Pred: \"{c['pred_text']}\"")

# Ambiguous xxx
print("\n--- AMBIGUOUS (xxx) CANDIDATES ---")
xxx_cands = []
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
    words = item["text"].strip().split()
    if len(words) < 3 or len(words) > 7 or any(c in item["text"] for c in "[]()<>"):
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
        target_ids = sp.encode_as_ids(item["text"].lower())
        target_tensor = torch.tensor(target_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
        target_lens = torch.tensor([len(target_ids)], dtype=torch.long, device=DEVICE)
        log_probs = F.log_softmax(ctc_logits.float(), dim=-1).transpose(0, 1)
        ctc_loss = ctc_loss_fn(log_probs, target_tensor, out_lens, target_lens).item()
        emo_probs = F.softmax(emo_logits[0], dim=-1).cpu().tolist()
        pred_idx = int(torch.argmax(emo_logits[0]).item())
        pred_emo = LABEL_NAMES[pred_idx]
        conf = emo_probs[pred_idx] * 100.0
    r_words = item["text"].upper().split()
    h_words = pred_text.upper().split()
    wer = (edit_distance(r_words, h_words) / len(r_words)) * 100.0
    cer = (edit_distance(list(item["text"].upper()), list(pred_text.upper())) / len(item["text"])) * 100.0
    xxx_cands.append({
        "utt_id": utt,
        "wav_path": wav_path,
        "gt_text": item["text"],
        "pred_text": pred_text,
        "ctc_loss": round(ctc_loss, 4),
        "wer": round(wer, 2),
        "cer": round(cer, 2),
        "pred_emo": pred_emo,
        "conf": round(conf, 1),
        "votes": item["human_evaluator_votes"],
        "probs": {LABEL_NAMES[i]: round(emo_probs[i]*100, 1) for i in range(4)}
    })
    if len(xxx_cands) >= 100:
        break

xxx_cands.sort(key=lambda x: (x["wer"], x["cer"], x["ctc_loss"]))
for c in xxx_cands[:3]:
    print(f"  {c['utt_id']} | WER: {c['wer']}% | CER: {c['cer']}% | CTC Loss: {c['ctc_loss']} | Pred: {c['pred_emo']} ({c['conf']}%)")
    print(f"    GT:   \"{c['gt_text']}\"")
    print(f"    Pred: \"{c['pred_text']}\"")
    print(f"    Votes: {c['votes']}")

with open("found_low_error.json", "w", encoding="utf-8") as f:
    json.dump({"best": best_candidates, "xxx": xxx_cands}, f, indent=2)
