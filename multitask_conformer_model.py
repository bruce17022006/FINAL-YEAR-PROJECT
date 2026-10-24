import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from conformer_asr import StudentConformerASR

class MultiTaskConformerASR_SER(nn.Module):
    """
    Unified Multi-Task Conformer Network:
      - Shared Acoustic Backbone: 6-Layer Conformer (4x Subsampling, D=256, K=15)
      - Head 1: ASR Linear CTC Projection Head (Vocab 1024)
      - Head 2: Speech Emotion Recognition (SER) Head (Temporal Pooling + MLP Classifier -> 4 Classes)
    """
    def __init__(
        self,
        input_dim=80,
        encoder_dim=256,
        num_layers=6,
        num_heads=4,
        vocab_size=1024,
        teacher_dim=2048,
        num_emotions=4,
        dropout=0.1,
    ):
        super().__init__()
        # 1. Base Conformer Backbone
        self.encoder = StudentConformerASR(
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            vocab_size=vocab_size,
            teacher_dim=teacher_dim,
            dropout=dropout,
        )

        # 2. Speech Emotion Recognition (SER) Classifier Head
        self.emotion_head = nn.Sequential(
            nn.Linear(encoder_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_emotions)
        )

    def load_pretrained_conformer(self, checkpoint_path, device="cpu"):
        print(f"--> Loading Pretrained Conformer Weights from: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # Load into encoder
        missing, unexpected = self.encoder.load_state_dict(state, strict=False)
        print(f"--> Pretrained Backbone Loaded. Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")

    def forward(self, mel_inputs, input_lengths=None):
        # 1. Forward through Conformer Backbone
        # x shape: [B, T', 80] -> subsampling -> conformer layers
        # StudentConformerASR forward returns (ctc_logits, semantic_embeddings)
        # But we also need the intermediate latent representations for emotion pooling
        
        # Subsampling
        x = self.encoder.subsampling(mel_inputs.unsqueeze(1))
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, c * f)
        x = self.encoder.input_proj(x)
        x = self.encoder.pos_enc(x)

        # 4x subsampled attention mask
        mask = None
        out_lens = None
        if input_lengths is not None:
            out_lens = torch.div(input_lengths + 3, 4, rounding_mode="floor")
            pos = torch.arange(t, device=x.device).unsqueeze(0)
            mask = pos >= out_lens.unsqueeze(1)

        # Conformer Blocks
        for layer in self.encoder.conformer_layers:
            x = layer(x, key_padding_mask=mask)

        # Head 1: CTC Logits [B, T', Vocab]
        ctc_logits = self.encoder.ctc_head(x)

        # Head 2: Temporal Pooling for Emotion Classification
        if mask is not None:
            # Mask out padding frames with zeros before summing
            mask_expanded = mask.unsqueeze(-1).expand_as(x) # [B, T', D]
            x_masked = x.masked_fill(mask_expanded, 0.0)
            pooled = x_masked.sum(dim=1) / out_lens.unsqueeze(-1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)

        # Emotion Logits [B, Num_Emotions]
        emotion_logits = self.emotion_head(pooled)

        return ctc_logits, emotion_logits, out_lens

if __name__ == "__main__":
    model = MultiTaskConformerASR_SER()
    dummy_mels = torch.randn(2, 400, 80)
    dummy_lens = torch.tensor([400, 320])
    ctc_out, emo_out, out_lens = model(dummy_mels, dummy_lens)
    print("Multi-task Conformer forward pass test:")
    print("  CTC Logits shape:", ctc_out.shape)
    print("  Emotion Logits shape:", emo_out.shape)
    print("  Subsampled lengths:", out_lens)
