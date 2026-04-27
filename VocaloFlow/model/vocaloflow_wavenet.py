"""VocaloFlow Pure WaveNet: Flow matching denoiser without self-attention.

Replaces the hybrid WaveNet+DiT backbone with a single WaveNetDenoiser.
Same forward signature as VocaloFlow for drop-in compatibility with the
training loop, loss computation, and inference pipeline.

Selected via ``config.architecture = "wavenet_pure"``.
"""

import sys
import os

import torch
import torch.nn as nn
from torch import Tensor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.default import VocaloFlowConfig
from model.wavenet import WaveNetDenoiser
from model.embeddings import TimestepMLP, ConditioningEncoder


class VocaloFlowPureWaveNet(nn.Module):
    """Pure WaveNet conditional flow matching model.

    Same forward signature as VocaloFlow.  Replaces DiT blocks + optional
    ConvNeXt/WaveNet pre-processing with a single WaveNetDenoiser backbone.

    Architecture:
      1. ConditioningEncoder: embed + normalize + concat -> (B, T, input_dim)
      2. Input projection: Linear(input_dim, residual_channels)
      3. Timestep conditioning: sinusoidal -> MLP -> (B, residual_channels)
      4. Optional speaker embedding: additive to conditioning vector
      5. WaveNetDenoiser: skip-sum -> (B, T, mel_channels) velocity

    Args:
        config: VocaloFlowConfig with architecture="wavenet_pure".
    """

    def __init__(self, config: VocaloFlowConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = VocaloFlowConfig()
        self.config = config

        rc = config.wavenet_pure_residual_channels

        self.cond_encoder = ConditioningEncoder(config)
        self.input_proj = nn.Linear(self.cond_encoder.output_dim, rc)
        self.timestep_mlp = TimestepMLP(hidden_dim=rc, sinusoidal_dim=rc // 2)

        self.use_speaker_embedding = config.use_speaker_embedding
        if config.use_speaker_embedding:
            self.speaker_proj = nn.Linear(config.speaker_embedding_dim, rc)
            nn.init.zeros_(self.speaker_proj.weight)
            nn.init.zeros_(self.speaker_proj.bias)

        self.denoiser = WaveNetDenoiser(
            residual_channels=rc,
            cond_channels=rc,
            skip_channels=config.wavenet_pure_skip_channels,
            mel_channels=config.mel_channels,
            kernel_size=config.wavenet_pure_kernel_size,
            n_layers=config.wavenet_pure_num_layers,
            dilation_cycle=config.wavenet_pure_dilation_cycle,
            dropout=config.wavenet_pure_dropout,
        )

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        prior_mel: Tensor,
        f0: Tensor,
        voicing: Tensor,
        phoneme_ids: Tensor,
        padding_mask: Tensor | None = None,
        plbert_features: Tensor | None = None,
        speaker_embedding: Tensor | None = None,
    ) -> Tensor:
        """Predict the velocity field v_theta.

        Args:
            x_t:               (B, T, 128) interpolated mel state.
            t:                 (B,) flow timestep in [0, 1].
            prior_mel:         (B, T, 128) Vocaloid prior mel (x_0 conditioning).
            f0:                (B, T) F0 contour (0 for unvoiced).
            voicing:           (B, T) voiced/unvoiced binary flag.
            phoneme_ids:       (B, T) resolved phoneme token IDs.
            padding_mask:      (B, T) bool, True = valid frame.
            plbert_features:   (B, T, 768) frozen PL-BERT embeddings (optional).
            speaker_embedding: (B, 192) ECAPA-TDNN speaker embedding (optional).

        Returns:
            (B, T, 128) predicted velocity vector.
        """
        cond_features = self.cond_encoder(
            x_t, prior_mel, f0, voicing, phoneme_ids, plbert_features,
        )
        h = self.input_proj(cond_features)

        c = self.timestep_mlp(t)
        if self.use_speaker_embedding and speaker_embedding is not None:
            c = c + self.speaker_proj(speaker_embedding)

        return self.denoiser(h, c, padding_mask)
