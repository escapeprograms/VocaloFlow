"""VocaloFlowHybrid: DiT backbone with optional ConvNeXt / WaveNet pre-processing.

Architecture:
  1-7.  Shared conditioning encoder (see ``VocaloFlow`` base class)
  8a.   ConvNeXt pre-processing (optional): Nx ConvNeXtV2 blocks
  8b.   WaveNet pre-processing (optional): Nx WaveNet residual blocks,
        wrapped in an outer residual ``h = h + wavenet_stack(h, c)``
  9.    DiT Block xN: Transformer with AdaLN-Zero, RoPE, GELU FFN, dropout
  10.   Output projection: LayerNorm -> Linear(hidden_dim, 128) -> velocity

Selected via ``config.architecture = "hybrid"`` (default).
"""

import sys
import os

import torch.nn as nn
from torch import Tensor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.default import VocaloFlowConfig
from model.vocaloflow import VocaloFlow
from model.dit_block import DiTBlock
from model.convnext import ConvNeXtStack
from model.wavenet import WaveNetStack


class VocaloFlowHybrid(VocaloFlow):
    """Hybrid conditional flow matching model (DiT backbone).

    Uses DiT transformer blocks as the primary backbone, with optional
    ConvNeXt and/or WaveNet pre-processing stacks.

    Args:
        config: VocaloFlowConfig with all hyperparameters.
    """

    def __init__(self, config: VocaloFlowConfig | None = None) -> None:
        if config is None:
            config = VocaloFlowConfig()
        super().__init__(config, hidden_dim=config.hidden_dim)

        # ── ConvNeXt pre-processing (optional) ───────────────────────────
        if config.num_convnext_blocks > 0:
            self.convnext_stack = ConvNeXtStack(
                dim=config.hidden_dim,
                num_blocks=config.num_convnext_blocks,
                kernel_size=config.convnext_kernel_size,
                dropout=config.dropout,
            )
        else:
            self.convnext_stack = None

        # ── WaveNet pre-processing (optional) ────────────────────────────
        if config.num_wavenet_blocks > 0:
            self.wavenet_stack = WaveNetStack(
                hidden_channels=config.hidden_dim,
                cond_channels=config.hidden_dim,
                skip_channels=config.wavenet_skip_channels,
                kernel_size=config.wavenet_kernel_size,
                n_layers=config.num_wavenet_blocks,
                dilation_cycle=config.wavenet_dilation_cycle,
                dropout=config.wavenet_dropout,
            )
        else:
            self.wavenet_stack = None

        # ── DiT blocks ───────────────────────────────────────────────────
        self.dit_blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                ffn_dim=config.ffn_dim,
                max_len=config.max_seq_len * 2,
                dropout=config.dropout,
            )
            for _ in range(config.num_dit_blocks)
        ])

        # ── Output projection ────────────────────────────────────────────
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.output_proj = nn.Linear(config.hidden_dim, config.mel_channels)

    def _backbone_forward(
        self,
        h: Tensor,
        c: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        # ConvNeXt pre-processing (optional)
        if self.convnext_stack is not None:
            h = self.convnext_stack(h)

        # WaveNet pre-processing (optional, outer residual)
        if self.wavenet_stack is not None:
            h = h + self.wavenet_stack(h, c)

        # DiT blocks
        for block in self.dit_blocks:
            h = block(h, c, padding_mask)

        # Output projection
        return self.output_proj(self.output_norm(h))
