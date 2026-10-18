import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        # Step 1: Projection layer (Student Space -> Teacher 2048 Space)
        self.proj = nn.Linear(student_embed_dim, teacher_embed_dim, bias=False)

    def forward(
        self, 
        student_vectors: torch.Tensor,     # Shape: (B, N, student_dim) e.g., N=4 student tokens
        teacher_vectors: torch.Tensor,     # Shape: (B, M, teacher_dim) e.g., M=8 teacher tokens
        student_mass: torch.Tensor = None, # Shape: (B, N) distribution 'r'
        teacher_mass: torch.Tensor = None  # Shape: (B, M) distribution 'c'
    ) -> torch.Tensor:
        """
        Calculates Entropic Optimal Transport Loss between student and teacher sets.
        """
        B, N, _ = student_vectors.shape
        _, M, _ = teacher_vectors.shape
        device = student_vectors.device

        # 1. Project student vectors to teacher embedding dimension
        proj_student = self.proj(student_vectors)  # (B, N, teacher_dim)

        # 2. Normalize for Cosine Similarity
        proj_student_norm = F.normalize(proj_student, p=2, dim=-1)
        teacher_norm = F.normalize(teacher_vectors, p=2, dim=-1)

        # 3. Ground Cost Matrix C (Step 2 in hand derivation)
        # Cosine Distance: C_ij = 1 - CosineSimilarity
        sim_matrix = torch.bmm(proj_student_norm, teacher_norm.transpose(1, 2)) # (B, N, M)
        C = 1.0 - sim_matrix
        C = torch.clamp(C, min=0.0)

        # 4. Probability distributions r and c (Step 1 in hand derivation)
        if student_mass is None:
            r = torch.full((B, N), 1.0 / N, device=device, dtype=torch.float32)
        else:
            r = student_mass / (student_mass.sum(dim=-1, keepdim=True) + 1e-12)

        if teacher_mass is None:
            c = torch.full((B, M), 1.0 / M, device=device, dtype=torch.float32)
        else:
            c = teacher_mass / (teacher_mass.sum(dim=-1, keepdim=True) + 1e-12)

        # 5. Gibbs Kernel K = exp(-C / epsilon) (Step 3 in hand derivation)
        K = torch.exp(-C / self.epsilon) + 1e-12  # (B, N, M)

        # 6. Sinkhorn Iterations for u and v (Step 4 in hand derivation)
        v = torch.ones((B, M), device=device, dtype=torch.float32)
        u = torch.ones((B, N), device=device, dtype=torch.float32)

        for _ in range(self.max_iters):
            # u = r / (K * v)
            Kv = torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) # (B, N)
            u = r / (Kv + 1e-12)

            # v = c / (K^T * u)
            Ktu = torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) # (B, M)
            v = c / (Ktu + 1e-12)

        # 7. Optimal Transport Plan P* = diag(u) * K * diag(v) (Step 5 in derivation)
        P_star = u.unsqueeze(-1) * K * v.unsqueeze(1) # (B, N, M)

        # 8. Final Wasserstein Distance Loss = sum(P* * C) (Step 6 in derivation)
        loss = torch.sum(P_star * C, dim=(-2, -1)) # (B,)
        return loss.mean()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing Sinkhorn OT Module on: {device}")

    # Dummy test dimensions
    B = 2
    N = 4   # 4 student tokens ("you", "did", "great", "today")
    M = 8   # Top-8 PBKD teacher tokens
    student_dim = 256
    teacher_dim = 2048

    ot_loss_fn = SinkhornWassersteinDistillationLoss(
        student_embed_dim=student_dim,
        teacher_embed_dim=teacher_dim,
        epsilon=0.1,
        max_iters=10
    ).to(device)

    dummy_student_feats = torch.randn(B, N, student_dim, device=device)
    dummy_teacher_feats = torch.randn(B, M, teacher_dim, device=device)

    # Optional: teacher probs from Stage 1 (.pt files)
    dummy_teacher_probs = torch.softmax(torch.randn(B, M, device=device), dim=-1)

    loss = ot_loss_fn(
        dummy_student_feats, 
        dummy_teacher_feats, 
        teacher_mass=dummy_teacher_probs
    )

    print(f"Calculated Wasserstein Distillation Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss contains NaN!"
    print("Module Passed Verification!")