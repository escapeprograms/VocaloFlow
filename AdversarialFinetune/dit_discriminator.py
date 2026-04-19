"""DiT-based discriminator for mel-spectrogram adversarial training.

Operates on full mel sequences ``(B, T, mel_dim)`` using transformer blocks
with a learnable [CLS] token for classification.  Returns a scalar logit
per sample and intermediate feature maps for feature matching loss.

Follows the Adversarial Flow Models paper (Lin et al., 2024): the
discriminator mirrors the generator's transformer architecture but uses
standard pre-norm (no AdaLN conditioning).

Architecture:
  1. Linear input projection: mel_dim -> hidden_dim
  2. Prepend learnable [CLS] token
  3. N x TransformerBlock (pre-norm, RoPE, GELU FFN)
  4. LayerNorm -> Linear on [CLS] position -> scalar logit
"""

from __future__ import annotations

import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "VocaloFlow")))
from model.rope import precompute_freqs_cis, apply_rotary_emb


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with RoPE attention and GELU FFN.

    Unlike the generator's DiTBlock, this uses standard learnable LayerNorm
    (no AdaLN-Zero conditioning) and plain residual connections (no gating).

    Args:
        hidden_dim: Model hidden dimension.
        num_heads: Number of attention heads.
        ffn_dim: Feed-forward intermediate dimension.
        max_len: Maximum sequence length for RoPE precomputation.
        dropout: Dropout rate (default 0.0 for discriminators).
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        max_len: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.attn_out = nn.Linear(hidden_dim, hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(self.head_dim, max_len),
            persistent=False,
        )

    def forward(self, x: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        """Forward pass.

        Args:
            x: (B, T, hidden_dim) input hidden states.
            padding_mask: (B, T) bool, True = valid position.

        Returns:
            (B, T, hidden_dim) output hidden states.
        """
        B, T, D = x.shape

        # ── Self-Attention ────────────────────────────────────────────────
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, H, T, D_h)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = apply_rotary_emb(q, k, self.freqs_cis[:T])

        attn_mask = None
        if padding_mask is not None:
            attn_mask = padding_mask[:, None, None, :].expand(B, 1, T, T)

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
        attn_out = self.attn_out(attn_out)
        attn_out = self.attn_dropout(attn_out)

        x = x + attn_out

        # ── Feed-Forward ──────────────────────────────────────────────────
        h = self.norm2(x)
        ffn_out = self.ffn(h)
        ffn_out = self.ffn_dropout(ffn_out)

        x = x + ffn_out
        return x


class DiTDiscriminator(nn.Module):
    """Transformer discriminator with [CLS] token classification head.

    Args:
        mel_dim: Input mel-spectrogram dimension (default 128).
        hidden_dim: Transformer hidden dimension (default 512).
        num_blocks: Number of transformer blocks (default 4).
        num_heads: Number of attention heads (default 8).
        ffn_dim: Feed-forward intermediate dimension (default 2048).
        max_len: Maximum sequence length including [CLS] (default 512).
        dropout: Dropout rate (default 0.0).
        feature_block_indices: Which block outputs to return for feature
            matching loss (default [1, 3]).
    """

    def __init__(
        self,
        mel_dim: int = 128,
        hidden_dim: int = 512,
        num_blocks: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        max_len: int = 512,
        dropout: float = 0.0,
        feature_block_indices: list[int] | None = None,
    ) -> None:
        super().__init__()
        if feature_block_indices is None:
            feature_block_indices = [1, 3]
        self.feature_block_indices = set(feature_block_indices)

        self.input_proj = nn.Linear(mel_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ffn_dim, max_len, dropout)
            for _ in range(num_blocks)
        ])

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, mel: Tensor, padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        """Classify mel sequences as real or fake.

        Args:
            mel: (B, T, mel_dim) mel-spectrogram sequence.
            padding_mask: (B, T) bool, True = valid frame.

        Returns:
            logits: (B, 1) per-sample real/fake score.
            features: List of intermediate block outputs (post-block hidden
                states) at ``feature_block_indices``, each (B, T+1, hidden_dim).
        """
        B, T, _ = mel.shape

        h = self.input_proj(mel)  # (B, T, D)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        h = torch.cat([cls, h], dim=1)  # (B, T+1, D)

        # Extend padding mask to include [CLS] (always valid)
        if padding_mask is not None:
            cls_mask = torch.ones(B, 1, device=mel.device, dtype=torch.bool)
            mask = torch.cat([cls_mask, padding_mask], dim=1)  # (B, T+1)
        else:
            mask = None

        features: list[Tensor] = []
        for i, block in enumerate(self.blocks):
            h = block(h, mask)
            if i in self.feature_block_indices:
                features.append(h)

        # Classification from [CLS] position
        cls_out = self.output_norm(h[:, 0])  # (B, D)
        logits = self.output_head(cls_out)  # (B, 1)

        return logits, features
