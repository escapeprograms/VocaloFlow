"""VocaloFlow: Conditional Flow Matching model for singing voice enhancement."""

import sys
import os

import torch
import torch.nn as nn
from torch import Tensor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.default import VocaloFlowConfig
from model.dit_block import DiTBlock
from model.convnext import ConvNeXtStack
from model.embeddings import TimestepMLP, PhonemeEmbedding, BlurredPhonemeEmbedding, F0Embedding


class VocaloFlow(nn.Module):
    """Conditional flow matching model mapping Vocaloid mel -> high-quality mel.

    Architecture (v3 — ConvNeXt pre-processing + blurred phoneme boundaries):
      1. PhonemeEmbedding: (B, T) int IDs -> (B, T, 64) [optionally blurred]
      2. F0Embedding: (B, T) Hz -> (B, T, 64)
      3. Per-stream LayerNorm on x_t (128), prior_mel (128), f0_emb (64), ph_emb (64)
      4. Concatenate + vuv: (B, T, 385)
      5. Input projection: Linear(385, 512)
      6. ConvNeXt pre-processing: 4x ConvNeXtV2 blocks (512 -> 512)
      7. Timestep conditioning: sinusoidal -> MLP -> (B, 512)
      8. DiT Block x6: Transformer with AdaLN-Zero, RoPE, GELU FFN, dropout
      9. Output projection: LayerNorm -> Linear(512, 128) -> velocity

    The model predicts the velocity field v_theta(x_t, t, cond) for OT-CFM.

    Args:
        config: VocaloFlowConfig with all hyperparameters.
    """

    def __init__(self, config: VocaloFlowConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = VocaloFlowConfig()
        self.config = config

        # Phoneme embedding table (blurred or hard lookup)
        if config.phoneme_blur_enabled:
            self.phoneme_embed = BlurredPhonemeEmbedding(
                vocab_size=config.phoneme_vocab_size,
                embed_dim=config.phoneme_embed_dim,
                blend_fraction=config.phoneme_blend_fraction,
            )
        else:
            self.phoneme_embed = PhonemeEmbedding(
                vocab_size=config.phoneme_vocab_size,
                embed_dim=config.phoneme_embed_dim,
            )

        # Learned F0 embedding
        self.f0_embed = F0Embedding(embed_dim=config.f0_embed_dim)

        # Per-stream normalization (before concatenation)
        self.norm_xt = nn.LayerNorm(config.mel_channels)
        self.norm_prior = nn.LayerNorm(config.mel_channels)
        self.norm_f0 = nn.LayerNorm(config.f0_embed_dim)
        self.norm_ph = nn.LayerNorm(config.phoneme_embed_dim)

        # Input projection: 385 -> 512
        self.input_proj = nn.Linear(config.input_channels, config.hidden_dim)

        # ConvNeXt pre-processing blocks (local temporal feature extraction)
        if config.num_convnext_blocks > 0:
            self.convnext_stack = ConvNeXtStack(
                dim=config.hidden_dim,
                num_blocks=config.num_convnext_blocks,
                kernel_size=config.convnext_kernel_size,
                dropout=config.dropout,
            )
        else:
            self.convnext_stack = None

        # Timestep conditioning MLP
        self.timestep_mlp = TimestepMLP(hidden_dim=config.hidden_dim)

        # DiT blocks
        self.dit_blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                ffn_dim=config.ffn_dim,
                max_len=config.max_seq_len * 2,  # headroom for longer sequences
                dropout=config.dropout,
            )
            for _ in range(config.num_dit_blocks)
        ])

        # Output projection
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.output_proj = nn.Linear(config.hidden_dim, config.mel_channels)

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        prior_mel: Tensor,
        f0: Tensor,
        voicing: Tensor,
        phoneme_ids: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Predict the velocity field v_theta.

        Args:
            x_t:         (B, T, 128) interpolated mel state.
            t:           (B,) flow timestep in [0, 1].
            prior_mel:   (B, T, 128) Vocaloid prior mel (x_0 conditioning).
            f0:          (B, T) F0 contour (0 for unvoiced).
            voicing:     (B, T) voiced/unvoiced binary flag.
            phoneme_ids: (B, T) resolved phoneme token IDs.
            padding_mask: (B, T) bool, True = valid frame.

        Returns:
            (B, T, 128) predicted velocity vector.
        """
        # 1. Embed phonemes: (B, T) -> (B, T, 64)
        ph_emb = self.phoneme_embed(phoneme_ids)

        # 2. Embed F0: (B, T) -> (B, T, 64)
        f0_emb = self.f0_embed(f0)

        # 3. Per-stream normalization
        x_t_normed = self.norm_xt(x_t)
        prior_normed = self.norm_prior(prior_mel)
        f0_normed = self.norm_f0(f0_emb)
        ph_normed = self.norm_ph(ph_emb)

        # 4. Concatenate all inputs: (B, T, 385)
        cond = torch.cat([
            x_t_normed,                         # (B, T, 128)
            prior_normed,                        # (B, T, 128)
            f0_normed,                           # (B, T, 64)
            ph_normed,                           # (B, T, 64)
            voicing.unsqueeze(-1),               # (B, T, 1)
        ], dim=-1)

        # 5. Input projection: (B, T, 385) -> (B, T, 512)
        h = self.input_proj(cond)

        # 6. ConvNeXt pre-processing (local temporal features)
        if self.convnext_stack is not None:
            h = self.convnext_stack(h)

        # 7. Timestep conditioning: (B,) -> (B, 512)
        c = self.timestep_mlp(t)

        # 8. DiT blocks
        for block in self.dit_blocks:
            h = block(h, c, padding_mask)

        # 9. Output projection: LayerNorm -> Linear -> (B, T, 128)
        v = self.output_proj(self.output_norm(h))

        return v
