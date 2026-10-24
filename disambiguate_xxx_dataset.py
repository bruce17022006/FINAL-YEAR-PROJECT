import os
import sys
import glob
import re
import json
import time
import torch
import torchaudio
import torch.nn.functional as F
from collections import Counter, defaultdict

from multitask_conformer_model import MultiTaskConformerASR_SER
from predict_speech_and_emotion import LABEL_NAMES, ctc_decode, BLANK_ID

sys.stdout.reconfigure(encoding='utf-8')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IEMOCAP_ROOT = os.path.join("IEMOCAP_full_release_withoutVideos", "IEMOCAP_full_release")
MULTITASK_CKPT = os.path.join("checkpoints_multitask", "best_multitask_conformer_iemocap.pt")
TOKENIZER_MODEL = "student_bpe_1024.model"

# Normalized map for human annotator votes
VOTE_NORMALIZER = {
    "neutral": "Neutral",
    "anger": "Angry",
    "ang": "Angry",
    "sadness": "Sad",
    "sad": "Sad",
    "happiness": "Happy",
    "hap": "Happy",
    "excited": "Happy",
    "exc": "Happy",
    "frustration": "Angry",  # Frustration is high-arousal negative affect, closely paired with Anger in 4-class mapping
    "fru": "Angry",
    "fear": "Sad",
    "disgust": "Angry",
    "surprise": "Happy"
}

def parse_all_xxx_utterances(root_dir=IEMOCAP_ROOT):
    print("--> Scanning and parsing all 'xxx' (No-Consensus) utterances across Sessions 1-5...")
    xxx_records = []

    for session in [f"Session{i}" for i in range(1, 6)]:
        session_dir = os.path.join(root_dir, session)
        if not os.path.isdir(session_dir):
            continue

        emo_dir = os.path.join(session_dir, "dialog", "EmoEvaluation")
        wav_base_dir = os.path.join(session_dir, "sentences", "wav")
        trans_dir = os.path.join(session_dir, "dialog", "transcriptions")

        trans_dict = {}
        for trans_file in glob.glob(os.path.join(trans_dir, "*.txt")):
            with open(trans_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    match = re.match(r"^(\w+)\s+\[.*?\]:\s+(.*)$", line.strip())
                    if match:
                        trans_dict[match.group(1)] = match.group(2).strip()

        for emo_file in glob.glob(os.path.join(emo_dir, "*.txt")):
            with open(emo_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            blocks = content.split("\n\n")
            for block in blocks:
                lines = [l.strip() for l in block.split("\n") if l.strip()]
                if not lines:
                    continue
                header = lines[0]
                if header.startswith("[") and "\txxx\t" in header:
                    parts = header.split("\t")
                    utt_id = parts[1].strip()

                    # Extract all raw human evaluator votes
                    raw_votes = []
                    for line in lines[1:]:
                        if line.startswith("C-") and ":" in line:
                            val = line.split(":")[1].split(";")[0].strip().lower()
                            if val:
                                raw_votes.append(val)

                    dialog_id = "_".join(utt_id.split("_")[:-1])
                    wav_path = os.path.join(wav_base_dir, dialog_id, f"{utt_id}.wav")

                    if os.path.isfile(wav_path):
                        xxx_records.append({
                            "utt_id": utt_id,
                            "session": session,
                            "wav_path": wav_path.replace("\\", "/"),
                            "raw_votes": raw_votes,
                            "normalized_votes": list(set([VOTE_NORMALIZER.get(v, v.capitalize()) for v in raw_votes])),
                            "text": trans_dict.get(utt_id, "")
                        })

    print(f"--> Total 'xxx' audio utterances collected: {len(xxx_records)}")
    return xxx_records

def disambiguate_with_conformer(records, batch_size=16):
    print("=" * 80)
    print("RUNNING AI DISAMBIGUATION ON NO-CONSENSUS (xxx) AUDIO UTTERANCES")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Loading Model: {MULTITASK_CKPT}")

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

    results = []
    matches_any_human = 0
    resolved_emotions_counter = Counter()
    confidence_tiers = {"High (>70%)": 0, "Moderate (50-70%)": 0, "Low (<50%)": 0}

    start_time = time.time()

    with torch.no_grad():
        for i, item in enumerate(records):
            waveform, sr = torchaudio.load(item["wav_path"])
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            mel = mel_transform(waveform).squeeze(0)
            log_mel = torch.log(torch.clamp(mel, min=1e-6)).transpose(0, 1)
            log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
            log_mel = log_mel.unsqueeze(0).to(DEVICE)
            in_lens = torch.tensor([log_mel.shape[1]], dtype=torch.long, device=DEVICE)

            ctc_logits, emo_logits, _ = model(log_mel, in_lens)

            # Emotion probabilities
            probs = F.softmax(emo_logits[0], dim=-1).cpu().tolist()
            pred_idx = int(torch.argmax(emo_logits[0]).item())
            pred_emotion = LABEL_NAMES[pred_idx]
            confidence = probs[pred_idx] * 100.0

            # Does AI agree with at least one human evaluator?
            human_agrees = pred_emotion in item["normalized_votes"]
            if human_agrees:
                matches_any_human += 1

            resolved_emotions_counter[pred_emotion] += 1

            if confidence >= 70.0:
                confidence_tiers["High (>70%)"] += 1
            elif confidence >= 50.0:
                confidence_tiers["Moderate (50-70%)"] += 1
            else:
                confidence_tiers["Low (<50%)"] += 1

            results.append({
                "utt_id": item["utt_id"],
                "session": item["session"],
                "text": item["text"],
                "human_evaluator_votes": item["raw_votes"],
                "human_normalized_options": item["normalized_votes"],
                "ai_predicted_emotion": pred_emotion,
                "ai_confidence_pct": round(confidence, 2),
                "ai_emotion_distribution": {LABEL_NAMES[k]: round(probs[k] * 100.0, 2) for k in range(4)},
                "agrees_with_at_least_one_human": human_agrees
            })

            if (i + 1) % 500 == 0 or (i + 1) == len(records):
                print(f"Processed {i + 1}/{len(records)} utterances ({((i + 1)/len(records))*100:.1f}%)...")

    elapsed = time.time() - start_time
    agreement_rate = (matches_any_human / len(records)) * 100.0

    print("\n" + "=" * 80)
    print("AI DISAMBIGUATION RESULTS & BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Total 'xxx' Utterances Analyzed       : {len(records)}")
    print(f"Total Processing Time                 : {elapsed:.2f} seconds ({len(records)/elapsed:.1f} utt/s)")
    print(f"Human-AI Consensus Rate              : {agreement_rate:.2f}% ({matches_any_human}/{len(records)})")
    print(f"(AI's prediction matches at least one human evaluator who was in disagreement)")

    print("\nEmotion Distribution of Resolved Ambiguous Utterances:")
    for emo, count in resolved_emotions_counter.most_common():
        pct = (count / len(records)) * 100.0
        print(f"  {emo:<10} : {count:>5} utterances ({pct:.2f}%)")

    print("\nPrediction Confidence Levels:")
    for tier, count in confidence_tiers.items():
        pct = (count / len(records)) * 100.0
        print(f"  {tier:<20} : {count:>5} utterances ({pct:.2f}%)")

    # Save disambiguated manifest
    out_file = "iemocap_xxx_disambiguated.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n--> Successfully saved full disambiguated manifest to: {out_file}")

    return results, agreement_rate, resolved_emotions_counter, confidence_tiers

if __name__ == "__main__":
    records = parse_all_xxx_utterances()
    disambiguate_with_conformer(records)
