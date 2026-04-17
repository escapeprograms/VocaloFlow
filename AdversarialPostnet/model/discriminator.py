"""2D patch discriminator for mel-spectrogram adversarial training.

Operates on random rectangular crops of mel spectrograms treated as
single-channel 2D images ``(B, 1, T_crop, F_crop)``.  Returns per-patch
real/fake logits and intermediate feature maps for feature matching loss.

Follows the WeSinger 2 / HiFi-GAN convention: strided Conv2d layers with
weight normalization (not batch norm, not spectral norm).

~1.2M parameters with the default [32, 64, 128, 256] channel configuration.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor
from torch.nn.utils.parametrizations import weight_norm


class PatchDiscriminator(nn.Module):
    """Strided 2D convolutional patch discriminator.

    Architecture: 4 downsampling layers (kernel=4, stride=2) with
    weight normalization and LeakyReLU, followed by 1 output layer
    (kernel=3, stride=1) producing raw logits.

    The forward pass returns both the final logits and a list of
    intermediate activations (post-LeakyReLU) for feature matching loss.

    Args:
        channels: Per-layer channel counts for layers 1-4.
            Defaults to [32, 64, 128, 256].
    """

    def __init__(self, channels: list[int] | None = None) -> None:
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256]

        # Build conv layers
        layers: list[nn.Module] = []
        in_ch = 1
        for out_ch in channels:
            layers.append(
                weight_norm(nn.Conv2d(
                    in_ch, out_ch,
                    kernel_size=4, stride=2, padding=1,
                ))
            )
            layers.append(nn.LeakyReLU(0.2))
            in_ch = out_ch

        # Store as individual modules so we can extract intermediate features
        self.layers = nn.ModuleList(layers)

        # Output layer (no weight norm, no activation — raw logits)
        self.output_conv = nn.Conv2d(
            channels[-1], 1,
            kernel_size=3, stride=1, padding=1,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Classify mel patches as real or fake.

        Args:
            x: (B, 1, T, F) mel-spectrogram crop.

        Returns:
            logits: (B, 1, H', W') per-patch real/fake scores.
            features: List of 4 intermediate feature maps (post-LeakyReLU
                outputs from each downsampling stage) for feature matching.
        """
        features: list[Tensor] = []
        h = x

        # Each stage is (Conv2d, LeakyReLU) — collect post-activation features
        for i in range(0, len(self.layers), 2):
            h = self.layers[i](h)       # Conv2d
            h = self.layers[i + 1](h)   # LeakyReLU
            features.append(h)

        logits = self.output_conv(h)
        return logits, features
