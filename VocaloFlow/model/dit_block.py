"""Diffusion Transformer (DiT) block with Adaptive Layer Normalization (AdaLN-Zero)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .rope import precompute_freqs_cis, apply_rotary_emb


class AdaLNZero(nn.Module):
    """Adaptive Layer Normalization with zero initialization.

    Projects a conditioning vector into scale, shift, and gate parameters
    for two sub-layers (attention and FFN). Zero-initialized so that each
    DiT block starts as an identity function.

    Args:
        hidden_dim: Model hidden dimension.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(hidden_dim, 6 * hidden_dim)
        # Zero-initialize so blocks start as identity
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, c: Tensor) -> tuple:
        """Compute AdaLN parameters from conditioning vector.

        Args:
            c: (B, hidden_dim) conditioning vector (timestep embedding).

        Returns:
            Tuple of 6 tensors, each (B, 1, hidden_dim):
            (gamma1, beta1, alpha1, gamma2, beta2, alpha2)
        """
        params = self.linear(self.silu(c)).unsqueeze(1)  # (B, 1, 6*D)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = params.chunk(6, dim=-1)
        return gamma1, beta1, alpha1, gamma2, beta2, alpha2


class DiTBlock(nn.Module):
    """Transformer block with RoPE self-attention, GELU FFN, AdaLN-Zero, and dropout.

    Args:
        hidden_dim: Model hidden dimension (default 512).
        num_heads: Number of attention heads (default 8).
        ffn_dim: Feed-forward intermediate dimension (default 2048).
        max_len: Maximum sequence length for RoPE precomputation (default 512).
        dropout: Dropout rate for attention and FFN outputs (default 0.1).
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        max_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Layer norms (pre-norm architecture)
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        # Self-attention projections
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.attn_out = nn.Linear(hidden_dim, hidden_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )

        # Dropout (regularization)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)

        # AdaLN-Zero conditioning
        self.adaln = AdaLNZero(hidden_dim)

        # Precompute RoPE frequencies
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(self.head_dim, max_len),
            persistent=False,
        )

    def forward(
        self,
        x: Tensor,
        c: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through the DiT block.

        Args:
            x: (B, T, hidden_dim) input hidden states.
            c: (B, hidden_dim) timestep conditioning vector.
            padding_mask: (B, T) bool, True = valid frame.

        Returns:
            (B, T, hidden_dim) output hidden states.
        """
        B, T, D = x.shape
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaln(c)

        # ── Self-Attention with AdaLN + RoPE ──────────────────────────────
        h = self.norm1(x)
        h = (1 + gamma1) * h + beta1

        qkv = self.qkv(h).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, H, D_h)
        q = q.transpose(1, 2)  # (B, H, T, D_h)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE to Q and K
        q, k = apply_rotary_emb(q, k, self.freqs_cis)

        # Build attention mask from padding mask
        attn_mask = None
        if padding_mask is not None:
            # (B, T) -> (B, 1, 1, T) for broadcasting with (B, H, T, T)
            attn_mask = padding_mask[:, None, None, :].expand(B, 1, T, T)
            # Also mask query positions
            query_mask = padding_mask[:, None, :, None].expand(B, 1, T, T)
            attn_mask = attn_mask & query_mask

        # Scaled dot-product attention (uses FlashAttention when available)
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
        attn_out = self.attn_out(attn_out)
        attn_out = self.attn_dropout(attn_out)

        # Gated residual
        x = x + alpha1 * attn_out

        # ── Feed-Forward with AdaLN ───────────────────────────────────────
        h = self.norm2(x)
        h = (1 + gamma2) * h + beta2
        ffn_out = self.ffn(h)
        ffn_out = self.ffn_dropout(ffn_out)

        # Gated residual
        x = x + alpha2 * ffn_out

        return x
