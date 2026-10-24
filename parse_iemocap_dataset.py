import os
import glob
import re
import json

IEMOCAP_ROOT = os.path.join("IEMOCAP_full_release_withoutVideos", "IEMOCAP_full_release")

# Standard 4-class emotion mapping (IEEE & SER benchmark standard):
# 'neu' -> Neutral (0)
# 'ang' -> Angry (1)
# 'sad' -> Sad (2)
# 'hap' & 'exc' -> Happy/Excited (3)
EMOTION_MAP_4CLASS = {
    "neu": 0,
    "ang": 1,
    "sad": 2,
    "hap": 3,
    "exc": 3,  # 'exc' (excited) merged into 'hap' as is standard in IEMOCAP literature
}

LABEL_NAMES = ["neutral", "angry", "sad", "happy"]

def parse_iemocap(root_dir=IEMOCAP_ROOT):
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"IEMOCAP root directory not found at: {root_dir}")

    sessions = [f"Session{i}" for i in range(1, 6)]
    dataset_records = []
    skipped_count = 0
    missing_audio_count = 0

    print("--> Scanning IEMOCAP dataset sessions...")

    for session in sessions:
        session_dir = os.path.join(root_dir, session)
        if not os.path.isdir(session_dir):
            print(f"Warning: {session_dir} not found, skipping...")
            continue

        emo_dir = os.path.join(session_dir, "dialog", "EmoEvaluation")
        trans_dir = os.path.join(session_dir, "dialog", "transcriptions")
        wav_base_dir = os.path.join(session_dir, "sentences", "wav")

        # 1. Parse all transcriptions for this session into memory
        trans_dict = {}
        for trans_file in glob.glob(os.path.join(trans_dir, "*.txt")):
            with open(trans_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    # Example line: Ses01F_impro01_F000 [006.2901-008.2357]: Excuse me.
                    match = re.match(r"^(\w+)\s+\[.*?\]:\s+(.*)$", line)
                    if match:
                        utt_id, text = match.group(1), match.group(2)
                        trans_dict[utt_id] = text.strip()

        # 2. Parse emotion evaluations
        emo_files = glob.glob(os.path.join(emo_dir, "*.txt"))
        for emo_file in emo_files:
            with open(emo_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    # Example line: [6.2901 - 8.2357]\tSes01F_impro01_F000\tneu\t[2.5000, 2.5000, 2.5000]
                    if line.startswith("[") and "]" in line:
                        parts = line.split("\t")
                        if len(parts) >= 3:
                            utt_id = parts[1].strip()
                            raw_emo = parts[2].strip().lower()

                            # Only keep the 4 core classes
                            if raw_emo in EMOTION_MAP_4CLASS:
                                label_idx = EMOTION_MAP_4CLASS[raw_emo]
                                dialog_id = "_".join(utt_id.split("_")[:-1])
                                wav_path = os.path.join(wav_base_dir, dialog_id, f"{utt_id}.wav")

                                if os.path.isfile(wav_path):
                                    text = trans_dict.get(utt_id, "")
                                    dataset_records.append({
                                        "utt_id": utt_id,
                                        "session": session,
                                        "wav_path": wav_path.replace("\\", "/"),
                                        "emotion_raw": raw_emo,
                                        "emotion_class": label_idx,
                                        "emotion_label": LABEL_NAMES[label_idx],
                                        "text": text
                                    })
                                else:
                                    missing_audio_count += 1
                            else:
                                skipped_count += 1

    return dataset_records, skipped_count, missing_audio_count

if __name__ == "__main__":
    records, skipped, missing = parse_iemocap()
    print("=" * 70)
    print("IEMOCAP DATASET PARSER SUMMARY (4-CLASS BENCHMARK)")
    print("=" * 70)
    print(f"Total Valid Speech Samples Filtered : {len(records)}")
    print(f"Other Emotions Filtered Out (xxx/fru/sur/etc) : {skipped}")
    print(f"Missing Audio Files : {missing}")

    # Class distribution
    from collections import Counter
    class_counts = Counter(r["emotion_label"] for r in records)
    print("\nClass Distribution:")
    for label, count in class_counts.items():
        pct = (count / len(records)) * 100
        print(f"  {label:<10} : {count:>5} utterances ({pct:.2f}%)")

    # Session distribution
    session_counts = Counter(r["session"] for r in records)
    print("\nSession Distribution (for LOSO Cross-Validation):")
    for sess, count in sorted(session_counts.items()):
        print(f"  {sess:<10} : {count:>5} utterances")

    # Save manifest as JSON
    out_manifest = "iemocap_4class_manifest.json"
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"\n--> Successfully saved dataset manifest to: {out_manifest}")
