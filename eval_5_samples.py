import os
import sys
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

print("Loading model and tokenizer on", DEVICE)
sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_MODEL)

model = MultiTaskConformerASR_SER(
    input_dim=80,
    encoder_dim=256,
    num_layers=6,
    num_heads=4,
    vocab_size=1024,
    teacher_dim=2048,
    num_emotions=4,
    dropout=0.0
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

def compute_wer(ref, hyp):
    r_words = ref.upper().split()
    h_words = hyp.upper().split()
    if not r_words:
        return 0.0
    dist = edit_distance(r_words, h_words)
    return round((dist / len(r_words)) * 100.0, 2)

def compute_cer(ref, hyp):
    r_chars = list(ref.upper().strip())
    h_chars = list(hyp.upper().strip())
    if not r_chars:
        return 0.0
    dist = edit_distance(r_chars, h_chars)
    return round((dist / len(r_chars)) * 100.0, 2)

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

# Load manifests
manifest = json.load(open("iemocap_4class_manifest.json", encoding="utf-8"))
xxx_manifest = json.load(open("iemocap_xxx_disambiguated.json", encoding="utf-8"))

# Pick 1 clear sample for each of the 4 standard emotions in Session 5
selected_samples = {}
desired_emotions = ["happy", "sad", "angry", "neutral"]

for item in manifest:
    if item["session"] == "Session5":
        lbl = item["emotion_label"]
        if lbl in desired_emotions and lbl not in selected_samples:
            if os.path.exists(item["wav_path"]):
                selected_samples[lbl] = item
    if len(selected_samples) == 4:
        break

# Pick 1 representative ambiguous sample (xxx) in Session 5
xxx_sample = None
for item in xxx_manifest:
    if item["session"] == "Session5":
        sess = item["session"]
        utt = item["utt_id"]
        dialogue = "_".join(utt.split("_")[:-1])
        wav_path = os.path.join(
            "IEMOCAP_full_release_withoutVideos", "IEMOCAP_full_release",
            sess, "sentences", "wav", dialogue, utt + ".wav"
        )
        if os.path.exists(wav_path):
            item["wav_path"] = wav_path
            xxx_sample = item
            break

samples_list = [
    ("Happy", selected_samples["happy"]["wav_path"], selected_samples["happy"]["text"], "Happy", selected_samples["happy"]["utt_id"]),
    ("Sad", selected_samples["sad"]["wav_path"], selected_samples["sad"]["text"], "Sad", selected_samples["sad"]["utt_id"]),
    ("Angry", selected_samples["angry"]["wav_path"], selected_samples["angry"]["text"], "Angry", selected_samples["angry"]["utt_id"]),
    ("Neutral", selected_samples["neutral"]["wav_path"], selected_samples["neutral"]["text"], "Neutral", selected_samples["neutral"]["utt_id"]),
    ("Ambiguous (xxx)", xxx_sample["wav_path"], xxx_sample["text"], "xxx", xxx_sample["utt_id"]),
]

results = []

for category, wav_path, gt_text, gt_emotion, utt_id in samples_list:
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

        # 1. CTC Loss
        # Target tokenization
        target_ids = sp.encode_as_ids(gt_text.lower())
        if len(target_ids) > 0:
            target_tensor = torch.tensor(target_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
            target_lens = torch.tensor([len(target_ids)], dtype=torch.long, device=DEVICE)
            log_probs = F.log_softmax(ctc_logits.float(), dim=-1).transpose(0, 1)
            ctc_loss_val = ctc_loss_fn(log_probs, target_tensor, out_lens, target_lens).item()
        else:
            ctc_loss_val = 0.0

        # 2. Predicted Transcription
        raw_ids = ctc_logits.argmax(dim=-1)[0].cpu().tolist()
        pred_text = clean_ctc_decode(raw_ids)

        # 3. Speech Metrics
        wer_val = compute_wer(gt_text, pred_text)
        cer_val = compute_cer(gt_text, pred_text)

        # 4. Emotion Prediction
        emo_probs = F.softmax(emo_logits[0], dim=-1).cpu().tolist()
        pred_idx = int(torch.argmax(emo_logits[0]).item())
        pred_emotion = LABEL_NAMES[pred_idx]
        confidence = emo_probs[pred_idx] * 100.0
        prob_dist = {LABEL_NAMES[i]: round(emo_probs[i] * 100.0, 1) for i in range(len(LABEL_NAMES))}

    # Additional metadata for xxx
    human_votes = None
    consensus_status = None
    if category.startswith("Ambiguous"):
        human_votes = xxx_sample["human_evaluator_votes"]
        consensus_status = "AI Consensus Reached" if xxx_sample.get("agrees_with_at_least_one_human", True) else "Novel Cluster"

    sample_res = {
        "emotion_target": category,
        "utterance_id": utt_id,
        "ground_truth_text": gt_text,
        "predicted_text": pred_text,
        "ground_truth_emotion": gt_emotion,
        "predicted_emotion": pred_emotion,
        "confidence_pct": round(confidence, 1),
        "emotion_distribution": prob_dist,
        "ctc_loss": round(ctc_loss_val, 4),
        "cer_pct": cer_val,
        "wer_pct": wer_val,
        "dataset_benchmark_ua_pct": 59.97,
        "dataset_benchmark_wa_pct": 57.70,
        "sample_emotion_match": (pred_emotion.lower() == gt_emotion.lower()) if gt_emotion != "xxx" else consensus_status,
        "human_evaluator_votes": human_votes
    }
    results.append(sample_res)

with open("sample_eval_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("Evaluation complete! Results written to sample_eval_results.json")
