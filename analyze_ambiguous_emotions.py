import os
import glob
import re
import json
import torch
import torchaudio
import torch.nn.functional as F

from predict_speech_and_emotion import JointSpeechEmotionPredictor, LABEL_NAMES

IEMOCAP_ROOT = os.path.join("IEMOCAP_full_release_withoutVideos", "IEMOCAP_full_release")

def parse_ambiguous_samples(max_samples=5):
    """Finds samples marked as 'xxx' and extracts the individual human annotator votes."""
    ambiguous_records = []
    
    for session in ["Session1"]:
        emo_dir = os.path.join(IEMOCAP_ROOT, session, "dialog", "EmoEvaluation")
        wav_base_dir = os.path.join(IEMOCAP_ROOT, session, "sentences", "wav")
        trans_dir = os.path.join(IEMOCAP_ROOT, session, "dialog", "transcriptions")

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

                    # Extract annotator votes
                    votes = []
                    for line in lines[1:]:
                        if line.startswith("C-") and ":" in line:
                            # e.g., C-E2: Neutral; ()
                            vote_content = line.split(":")[1].split(";")[0].strip()
                            votes.append(vote_content)

                    dialog_id = "_".join(utt_id.split("_")[:-1])
                    wav_path = os.path.join(wav_base_dir, dialog_id, f"{utt_id}.wav")

                    if os.path.isfile(wav_path):
                        ambiguous_records.append({
                            "utt_id": utt_id,
                            "wav_path": wav_path.replace("\\", "/"),
                            "annotator_votes": votes,
                            "text": trans_dict.get(utt_id, "")
                        })

                    if len(ambiguous_records) >= max_samples:
                        return ambiguous_records

    return ambiguous_records

def demonstrate_ai_disambiguation():
    print("=" * 75)
    print("DEMONSTRATION: RESOLVING 'NO CONSENSUS' (xxx) AMBIGUOUS EMOTIONS USING AI")
    print("=" * 75)

    samples = parse_ambiguous_samples(max_samples=4)
    predictor = JointSpeechEmotionPredictor()

    for i, s in enumerate(samples, 1):
        print(f"\n--- Case Study {i}: Utterance {s['utt_id']} ---")
        print(f"Transcribed Text : \"{s['text']}\"")
        print(f"Human Annotators : {', '.join(s['annotator_votes'])} (NO CONSENSUS -> 'xxx')")

        pred = predictor.predict(s["wav_path"])
        print(f"\nAI Model Disambiguation Decision:")
        print(f"  --> Predicted Emotion : {pred['predicted_emotion'].upper()} ({pred['confidence']:.2f}% Confidence)")
        print(f"  --> Soft Probability Distribution:")
        for emo, prob in pred["distribution"].items():
            bar = "#" * int(prob / 5)
            print(f"      {emo:<8}: {prob:>5.1f}% [{bar:<15}]")

if __name__ == "__main__":
    demonstrate_ai_disambiguation()
