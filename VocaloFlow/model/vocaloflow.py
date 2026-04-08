"""VocaloFlow: Conditional Flow Matching model for singing voice enhancement."""

import sys
import os

import torch
import torch.nn as nn
from torch import Tensor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.default import VocaloFlowConfig
from model.dit_block import DiTBlock
from model.embeddings import TimestepMLP, PhonemeEmbedding


class VocaloFlow(nn.Module):
    """Conditional flow matching model mapping Vocaloid mel -> high-quality mel.

    Architecture:
      1. PhonemeEmbedding: (B, T) int IDs -> (B, T, 256)
      2. Input projection: Conv1d(514, 1024, kernel_size=1)
      3. Timestep conditioning: sinusoidal -> MLP -> (B, 1024)
      4. DiT Block x2: Transformer with AdaLN-Zero, RoPE, GELU FFN
      5. Output projection: LayerNorm -> Linear(1024, 128) -> velocity

    The model predicts the velocity field v_theta(x_t, t, cond) for OT-CFM.

    Args:
        config: VocaloFlowConfig with all hyperparameters.
    """

    def __init__(self, config: VocaloFlowConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = VocaloFlowConfig()
        self.config = config

        # Phoneme embedding table
        self.phoneme_embed = PhonemeEmbedding(
            vocab_size=config.phoneme_vocab_size,
            embed_dim=config.phoneme_embed_dim,
        )

        # Input projection: 514 -> 1024 via Conv1d(kernel=1)
        self.input_proj = nn.Conv1d(config.input_channels, config.hidden_dim, kernel_size=1)

        # Timestep conditioning MLP
        self.timestep_mlp = TimestepMLP(hidden_dim=config.hidden_dim)

        # DiT blocks
        self.dit_blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                ffn_dim=config.ffn_dim,
                max_len=config.max_seq_len * 2,  # headroom for longer sequences
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
        # 1. Embed phonemes: (B, T) -> (B, T, 256)
        ph_emb = self.phoneme_embed(phoneme_ids)

        # 2. Concatenate all inputs along channel dim: (B, T, 514)
        cond = torch.cat([
            x_t,                                # (B, T, 128)
            prior_mel,                          # (B, T, 128)
            f0.unsqueeze(-1),                   # (B, T, 1)
            voicing.unsqueeze(-1),              # (B, T, 1)
            ph_emb,                             # (B, T, 256)
        ], dim=-1)

        # 3. Input projection: (B, T, 514) -> (B, 514, T) -> Conv1d -> (B, 1024, T) -> (B, T, 1024)
        h = self.input_proj(cond.transpose(1, 2)).transpose(1, 2)

        # 4. Timestep conditioning: (B,) -> (B, 1024)
        c = self.timestep_mlp(t)

        # 5. DiT blocks
        for block in self.dit_blocks:
            h = block(h, c, padding_mask)

        # 6. Output projection: LayerNorm -> Linear -> (B, T, 128)
        v = self.output_proj(self.output_norm(h))

        return v
