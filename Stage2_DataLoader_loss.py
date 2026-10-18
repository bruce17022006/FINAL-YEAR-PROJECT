import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. Sinkhorn-Wasserstein Loss Module
# ==========================================
class SinkhornWassersteinDistillationLoss(nn.Module):
    def __init__(
        self, 
        student_embed_dim: int = 256, 
        teacher_embed_dim: int = 2048, 
        epsilon: float = 0.1, 
        max_iters: int = 10
    ):
        super().__init__()
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.proj = nn.Linear(student_embed_dim, teacher_embed_dim, bias=False)

    def forward(
        self, 
        student_vectors: torch.Tensor,     # (B, N, student_dim)
        teacher_vectors: torch.Tensor,     # (B, M, teacher_dim)
        student_mass: torch.Tensor = None, # (B, N)
        teacher_mass: torch.Tensor = None  # (B, M)
    ) -> torch.Tensor:
        B, N, _ = student_vectors.shape
        _, M, _ = teacher_vectors.shape
        device = student_vectors.device

        # Project & Normalize for Cosine Similarity
        proj_student = F.normalize(self.proj(student_vectors), p=2, dim=-1)
        teacher_norm = F.normalize(teacher_vectors, p=2, dim=-1)

        # Ground Cost Matrix C = 1 - CosineSimilarity
        sim_matrix = torch.bmm(proj_student, teacher_norm.transpose(1, 2))
        C = torch.clamp(1.0 - sim_matrix, min=0.0)

        # Marginal Distributions
        if student_mass is None:
            r = torch.full((B, N), 1.0 / N, device=device, dtype=torch.float32)
        else:
            r = student_mass / (student_mass.sum(dim=-1, keepdim=True) + 1e-12)

        if teacher_mass is None:
            c = torch.full((B, M), 1.0 / M, device=device, dtype=torch.float32)
        else:
            c = teacher_mass / (teacher_mass.sum(dim=-1, keepdim=True) + 1e-12)

        # Gibbs Kernel K = exp(-C / epsilon)
        K = torch.exp(-C / self.epsilon) + 1e-12

        # Sinkhorn Iterations
        v = torch.ones((B, M), device=device, dtype=torch.float32)
        u = torch.ones((B, N), device=device, dtype=torch.float32)

        for _ in range(self.max_iters):
            Kv = torch.bmm(K, v.unsqueeze(-1)).squeeze(-1)
            u = r / (Kv + 1e-12)

            Ktu = torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1)
            v = c / (Ktu + 1e-12)

        P_star = u.unsqueeze(-1) * K * v.unsqueeze(1)
        loss = torch.sum(P_star * C, dim=(-2, -1))
        return loss.mean()

# ==========================================
# 2. Stage 2 Multi-Modal Dataset
# ==========================================
class Stage2SpeechDataset(Dataset):
    def __init__(self, pbkd_dir: str, audio_map: dict, emotion_map: dict = None):
        """
        pbkd_dir: Stage 1 output folder containing {utt_id}.pt and llm_embeddings.pt
        audio_map: dict mapping utt_id -> wav_path
        emotion_map: optional dict mapping utt_id -> integer emotion class (0 to 4)
        """
        self.pbkd_dir = pbkd_dir
        self.audio_map = audio_map
        self.emotion_map = emotion_map or {}

        # Available precomputed files
        self.utt_ids = [
            f[:-3] for f in os.listdir(pbkd_dir) 
            if f.endswith(".pt") and f != "llm_embeddings.pt" and f[:-3] in audio_map
        ]
        print(f"Loaded Dataset: {len(self.utt_ids)} valid utterance pairs found.")

        # Load static LLM teacher embedding table to RAM
        embed_path = os.path.join(pbkd_dir, "llm_embeddings.pt")
        self.teacher_embeddings = torch.load(embed_path, map_location="cpu") # (128000, 2048)

        # 80-channel Log-Mel Spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=80
        )

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, idx):
        utt_id = self.utt_ids[idx]
        wav_path = self.audio_map[utt_id]

        # 1. Load Audio and compute log-mel spectrogram
        waveform, sr = torchaudio.load(wav_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        mel_spec = self.mel_transform(waveform).squeeze(0) # (80, T_frames)
        log_mel = torch.log(torch.clamp(mel_spec, min=1e-5)).transpose(0, 1) # (T_frames, 80)

        # 2. Load cached PBKD Stage 1 target file
        pt_path = os.path.join(self.pbkd_dir, f"{utt_id}.pt")
        cached_data = torch.load(pt_path, map_location="cpu")

        # Extract top-8 teacher token IDs and probabilities
        topk_indices = cached_data["topk_indices"]  # (Seq_Len, 8)
        topk_probs = cached_data["topk_probs"]      # (Seq_Len, 8)

        # 3. Gather actual teacher vectors from static embedding table
        # Shape: (Seq_Len, 8, 2048)
        teacher_vectors = self.teacher_embeddings[topk_indices]

        # 4. Optional Tone/Emotion Target
        emotion_label = self.emotion_map.get(utt_id, -1)

        return {
            "utt_id": utt_id,
            "log_mel": log_mel,
            "teacher_vectors": teacher_vectors,
            "teacher_probs": topk_probs,
            "emotion_label": torch.tensor(emotion_label, dtype=torch.long)
        }

# ==========================================
# 3. Custom Collate Function for Padding
# ==========================================
def pad_collate_fn(batch):
    mels = [item["log_mel"] for item in batch]
    t_vectors = [item["teacher_vectors"] for item in batch]
    t_probs = [item["teacher_probs"] for item in batch]
    emotions = torch.stack([item["emotion_label"] for item in batch])
    utt_ids = [item["utt_id"] for item in batch]

    mel_lengths = torch.tensor([m.shape[0] for m in mels], dtype=torch.long)
    padded_mels = nn.utils.rnn.pad_sequence(mels, batch_first=True)

    # Flatten Seq_Len and top-k tokens for loss alignment: (Total_Tokens, 8, 2048)
    flat_t_vectors = torch.cat(t_vectors, dim=0)
    flat_t_probs = torch.cat(t_probs, dim=0)

    return {
        "utt_ids": utt_ids,
        "padded_mels": padded_mels,
        "mel_lengths": mel_lengths,
        "teacher_vectors": flat_t_vectors,
        "teacher_probs": flat_t_probs,
        "emotions": emotions
    }

# ==========================================
# 4. Pipeline Smoke Test
# ==========================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--> Initializing Pipeline Test on: {device}")

    PBKD_DIR = "precomputed_pbkd_dev_clean"
    
    # Check if Stage 1 directory exists
    if os.path.exists(PBKD_DIR) and os.path.exists(os.path.join(PBKD_DIR, "llm_embeddings.pt")):
        print(f"--> Found Stage 1 directory: {PBKD_DIR}")
        loss_fn = SinkhornWassersteinDistillationLoss(
            student_embed_dim=256,
            teacher_embed_dim=2048,
            epsilon=0.1
        ).to(device)
        print("--> Sinkhorn Loss module integrated and ready for training loop.")
    else:
        print(f"--> Directory '{PBKD_DIR}' not found. Verify your Stage 1 precomputed folder path.")