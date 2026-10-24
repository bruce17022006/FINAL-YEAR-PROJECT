import os
import json
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm

TOKENIZER_MODEL = "student_bpe_1024.model"

class IEMOCAPDataset(Dataset):
    """
    IEMOCAP Dataset for Joint Speech Recognition (ASR) and Speech Emotion Recognition (SER).
    Returns:
      - log_mel: [T, 80] normalized log-mel spectrogram
      - input_len: int (number of frames before 4x subsampling)
      - target_ids: list[int] BPE subword token IDs for CTC
      - emotion_class: int (0: neutral, 1: angry, 2: sad, 3: happy)
      - text: str
      - utt_id: str
    """
    def __init__(self, manifest_path="iemocap_4class_manifest.json", sessions=None, sp_model_path=TOKENIZER_MODEL):
        with open(manifest_path, "r", encoding="utf-8") as f:
            all_records = json.load(f)

        if sessions is not None:
            self.records = [r for r in all_records if r["session"] in sessions]
        else:
            self.records = all_records

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=80
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        wav_path = item["wav_path"]
        waveform, sr = torchaudio.load(wav_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        mel = self.mel_transform(waveform).squeeze(0)
        log_mel = torch.log(torch.clamp(mel, min=1e-6)).transpose(0, 1)  # [T, 80]
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)

        text = item.get("text", "")
        target_ids = self.sp.encode_as_ids(text) if text else []
        emotion_class = item["emotion_class"]

        return {
            "log_mel": log_mel,
            "input_len": log_mel.shape[0],
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "target_len": len(target_ids),
            "emotion_class": torch.tensor(emotion_class, dtype=torch.long),
            "text": text,
            "utt_id": item["utt_id"]
        }


def iemocap_collate_fn(batch):
    # Filter out empty audio or empty targets
    batch = [b for b in batch if b["input_len"] > 8 and b["target_len"] > 0]
    if not batch:
        return None

    # Pad mel spectrograms
    max_t = max(b["input_len"] for b in batch)
    feature_dim = batch[0]["log_mel"].shape[1]
    padded_mels = torch.zeros(len(batch), max_t, feature_dim, dtype=torch.float32)

    input_lens = []
    target_list = []
    target_lens = []
    emotion_classes = []
    texts = []
    utt_ids = []

    for i, b in enumerate(batch):
        t = b["input_len"]
        padded_mels[i, :t, :] = b["log_mel"]
        input_lens.append(t)
        target_list.append(b["target_ids"])
        target_lens.append(b["target_len"])
        emotion_classes.append(b["emotion_class"])
        texts.append(b["text"])
        utt_ids.append(b["utt_id"])

    # Concatenate targets for CTC loss
    targets_concat = torch.cat(target_list)

    return {
        "mels": padded_mels,                               # [B, T_max, 80]
        "input_lengths": torch.tensor(input_lens, dtype=torch.long), # [B]
        "targets": targets_concat,                         # [Total_tokens]
        "target_lengths": torch.tensor(target_lens, dtype=torch.long), # [B]
        "emotion_labels": torch.stack(emotion_classes),    # [B]
        "texts": texts,
        "utt_ids": utt_ids
    }

if __name__ == "__main__":
    print("Testing IEMOCAP Dataset and Collate...")
    ds = IEMOCAPDataset(sessions=["Session1"])
    print(f"Session 1 size: {len(ds)}")
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=iemocap_collate_fn)
    batch = next(iter(loader))
    print("Batch loaded successfully:")
    print("  Mels shape:", batch["mels"].shape)
    print("  Input lengths:", batch["input_lengths"])
    print("  Targets shape:", batch["targets"].shape)
    print("  Target lengths:", batch["target_lengths"])
    print("  Emotion labels:", batch["emotion_labels"])
    print("  Sample text:", batch["texts"][0])
