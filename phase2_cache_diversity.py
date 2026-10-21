import os
import glob
from collections import Counter
import torch

CACHE_DIR = "precomputed_pbkd_train_clean"
MAX_CHECK = None  # None = inspect every cached file


def main():
    files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.pt")))
    if not files:
        raise FileNotFoundError(f"No .pt files found in {CACHE_DIR}")

    if MAX_CHECK is not None:
        files = files[:MAX_CHECK]

    speakers = Counter()
    chapters = Counter()
    token_lengths = []
    mel_lengths = []
    token_ids = Counter()
    special_first_last = Counter()
    bad = []

    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        parts = stem.split("-")
        # LibriSpeech-style IDs: speaker-chapter-utterance
        if len(parts) >= 3:
            speakers[parts[0]] += 1
            chapters["-".join(parts[:2])] += 1
        else:
            bad.append(stem)

        try:
            try:
                item = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                item = torch.load(path, map_location="cpu")
            mel = item["student_mel"]
            target = torch.as_tensor(item["student_target_ids"]).long()
            mel_lengths.append(int(mel.shape[0]))
            token_lengths.append(int(target.numel()))
            token_ids.update(target.tolist())
            if target.numel():
                special_first_last[f"first_id={int(target[0])}"] += 1
                special_first_last[f"last_id={int(target[-1])}"] += 1
        except Exception as e:
            bad.append(f"{stem}: {e}")

    print("========== CACHE DIVERSITY CHECK ==========")
    print("Cache files inspected:", len(files))
    print("Unique speakers:", len(speakers))
    print("Unique speaker-chapters:", len(chapters))
    print("Top 10 speakers by cached utterances:")
    for name, count in speakers.most_common(10):
        print(f"  speaker {name}: {count}")
    print("\nTop 10 chapters by cached utterances:")
    for name, count in chapters.most_common(10):
        print(f"  {name}: {count}")
    print("\nFirst 15 cache filenames:")
    for p in files[:15]:
        print(" ", os.path.basename(p))
    print("Last 15 cache filenames:")
    for p in files[-15:]:
        print(" ", os.path.basename(p))

    if token_lengths:
        print("\nMel length min/median/max:",
              min(mel_lengths), sorted(mel_lengths)[len(mel_lengths)//2], max(mel_lengths))
        print("Target token length min/median/max:",
              min(token_lengths), sorted(token_lengths)[len(token_lengths)//2], max(token_lengths))
        print("Most common token IDs:", token_ids.most_common(12))
        print("First/last target ID counts:")
        for k, v in special_first_last.most_common():
            print(f"  {k}: {v}")

    print("\nMalformed/unreadable entries:", len(bad))
    for entry in bad[:10]:
        print(" ", entry)
    print("\nThis script only reads cache files; it does not train or modify anything.")


if __name__ == "__main__":
    main()
