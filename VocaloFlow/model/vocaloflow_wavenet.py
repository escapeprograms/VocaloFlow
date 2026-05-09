"""VocaloFlowWaveNet: WaveNet backbone with optional DiT refinement.

Architecture:
  1-7.  Shared conditioning encoder (see ``VocaloFlow`` base class)
  8.    WaveNetDenoiser: residual blocks with skip-sum
  9.    Optional DiT blocks: transformer with AdaLN-Zero, RoPE (if num_dit > 0)
  10.   Output projection: LayerNorm -> Linear(rc, 128) (if DiT) or direct (if pure)

Selected via ``config.architecture = "wavenet_pure"``.  DiT refinement blocks
are controlled by ``config.wavenet_pure_num_dit_blocks`` (0 = pure WaveNet).
"""

import sys
import os

import torch.nn as nn
from torch import Tensor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.default import VocaloFlowConfig
from model.vocaloflow import VocaloFlow
from model.dit_block import DiTBlock
from model.wavenet import WaveNetDenoiser


class VocaloFlowWaveNet(VocaloFlow):
    """WaveNet conditional flow matching model with optional DiT refinement.

    Uses a WaveNetDenoiser as the primary backbone, optionally followed by
    DiT transformer blocks for global attention refinement.

    Args:
        config: VocaloFlowConfig with architecture="wavenet_pure".
    """

    def __init__(self, config: VocaloFlowConfig | None = None) -> None:
        if config is None:
            config = VocaloFlowConfig()

        rc = config.wavenet_pure_residual_channels
        super().__init__(config, hidden_dim=rc)

        # ── WaveNet backbone ─────────────────────────────────────────────
        self.has_dit = config.wavenet_pure_num_dit_blocks > 0
        wavenet_output = rc if self.has_dit else config.mel_channels

        self.denoiser = WaveNetDenoiser(
            residual_channels=rc,
            cond_channels=rc,
            skip_channels=config.wavenet_pure_skip_channels,
            mel_channels=config.mel_channels,
            output_channels=wavenet_output,
            kernel_size=config.wavenet_pure_kernel_size,
            n_layers=config.wavenet_pure_num_layers,
            dilation_cycle=config.wavenet_pure_dilation_cycle,
            dropout=config.wavenet_pure_dropout,
        )

        # ── DiT refinement blocks (optional) ────────────────────────────
        if self.has_dit:
            self.dit_blocks = nn.ModuleList([
                DiTBlock(
                    hidden_dim=rc,
                    num_heads=config.num_heads,
                    ffn_dim=config.ffn_dim,
                    max_len=config.max_seq_len * 2,
                    dropout=config.dropout,
                )
                for _ in range(config.wavenet_pure_num_dit_blocks)
            ])
            self.output_norm = nn.LayerNorm(rc)
            self.output_proj = nn.Linear(rc, config.mel_channels)
        else:
            self.dit_blocks = None
            self.output_norm = None
            self.output_proj = None

    def _backbone_forward(
        self,
        h: Tensor,
        c: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        # WaveNet backbone
        h = self.denoiser(h, c, padding_mask)

        # DiT refinement (optional)
        if self.dit_blocks is not None:
            for block in self.dit_blocks:
                h = block(h, c, padding_mask)
            return self.output_proj(self.output_norm(h))

        return h
