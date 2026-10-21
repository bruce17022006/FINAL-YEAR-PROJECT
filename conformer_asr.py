import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class DepthwiseConvModule(nn.Module):
    """
    Conformer Depthwise Convolution Module for local acoustic formant modeling.
    LayerNorm -> Pointwise Conv1D -> GLU -> Depthwise Conv1D -> BatchNorm -> Swish -> Pointwise Conv1D
    """
    def __init__(self, dim, kernel_size=15, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.pointwise_conv1 = nn.Conv1d(dim, 2 * dim, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size,
            stride=1, padding=(kernel_size - 1) // 2, groups=dim
        )
        self.batch_norm = nn.BatchNorm1d(dim)
        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [B, T, D]
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.pointwise_conv1(x)  # [B, 2*D, T]
        x = F.glu(x, dim=1)  # [B, D, T]
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # [B, T, D]
        return residual + x


class FeedForwardModule(nn.Module):
    """Macaron-style half-step Feed Forward Module."""
    def __init__(self, dim, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * expansion_factor)
        self.activation = Swish()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim * expansion_factor, dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return residual + 0.5 * x


class ConformerBlock(nn.Module):
    """
    Standard Conformer Block:
    FFN (0.5x) -> Multi-Head Self Attention -> Conv Module -> FFN (0.5x) -> LayerNorm
    """
    def __init__(self, dim=256, num_heads=4, conv_kernel_size=15, dropout=0.1):
        super().__init__()
        self.ffn1 = FeedForwardModule(dim, dropout=dropout)
        self.norm_attn = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout_attn = nn.Dropout(dropout)
        self.conv_module = DepthwiseConvModule(dim, kernel_size=conv_kernel_size, dropout=dropout)
        self.ffn2 = FeedForwardModule(dim, dropout=dropout)
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x, key_padding_mask=None):
        # 1. First FFN (Macaron half-step)
        x = self.ffn1(x)

        # 2. Multi-Head Self Attention
        norm_x = self.norm_attn(x)
        attn_out, _ = self.self_attn(norm_x, norm_x, norm_x, key_padding_mask=key_padding_mask)
        x = x + self.dropout_attn(attn_out)

        # 3. Convolution Module
        x = self.conv_module(x)

        # 4. Second FFN (Macaron half-step)
        x = self.ffn2(x)
        return self.final_norm(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class StudentConformerASR(nn.Module):
    """
    Conformer ASR Student with 4x Time Subsampling.
    Time downsampling = 4x (25 fps).
    Freq downsampling = 4x (80 -> 20 bins).
    """
    def __init__(
        self,
        input_dim=80,
        encoder_dim=256,
        num_layers=6,
        num_heads=4,
        vocab_size=1024,
        teacher_dim=2048,
        conv_kernel_size=15,
        dropout=0.1,
    ):
        super().__init__()
        # 4x 2D Conv Subsampling (stride=(2,2) in both layers)
        self.subsampling = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, encoder_dim, kernel_size=3, stride=(2, 2), padding=1),
            nn.ReLU(),
        )
        self.input_proj = nn.Linear(encoder_dim * (input_dim // 4), encoder_dim)
        self.pos_enc = PositionalEncoding(encoder_dim)

        self.conformer_layers = nn.ModuleList([
            ConformerBlock(
                dim=encoder_dim,
                num_heads=num_heads,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.ctc_head = nn.Linear(encoder_dim, vocab_size)
        self.semantic_proj = nn.Linear(encoder_dim, teacher_dim, bias=False)

    def forward(self, mel_inputs, input_lengths=None):
        # mel_inputs: [B, T, 80]
        x = self.subsampling(mel_inputs.unsqueeze(1))
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, c * f)
        x = self.input_proj(x)
        x = self.pos_enc(x)

        # 4x subsampled attention mask: out_len = (input_len + 3) // 4
        mask = None
        if input_lengths is not None:
            out_lens = torch.div(input_lengths + 3, 4, rounding_mode="floor")
            pos = torch.arange(t, device=x.device).unsqueeze(0)
            mask = pos >= out_lens.unsqueeze(1)

        for layer in self.conformer_layers:
            x = layer(x, key_padding_mask=mask)

        ctc_logits = self.ctc_head(x)
        semantic_embeddings = self.semantic_proj(x)
        return ctc_logits, semantic_embeddings
