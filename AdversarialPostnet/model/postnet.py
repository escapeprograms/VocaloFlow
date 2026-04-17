"""Lightweight residual post-network for mel-spectrogram sharpening.

A small Conv1d residual stack that refines VocaloFlow's predicted mel
by learning a residual correction.  Operates in channels-last (B, T, 128)
externally; transposes to (B, 128, T) for the internal conv stack.

~260K parameters with the default 4-block / 128-channel configuration.
"""

import torch.nn as nn
from torch import Tensor


def _kaiming_init_conv(conv: nn.Module) -> None:
    """Kaiming-normal init with zero bias (codebase convention)."""
    nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0.0)


class ResidualConv1dBlock(nn.Module):
    """Single residual block: two Conv1d layers with a skip connection.

    Conv1d -> LeakyReLU -> Conv1d, then add the input (residual).

    Args:
        channels: Number of input/output channels.
        kernel_size: Kernel size for both convolutions (odd, same-padded).
    """

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.act = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)

        _kaiming_init_conv(self.conv1)
        _kaiming_init_conv(self.conv2)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with residual connection.

        Args:
            x: (B, C, T) channels-first input.

        Returns:
            (B, C, T) output = x + conv_path(x).
        """
        h = self.conv1(x)
        h = self.act(h)
        h = self.conv2(h)
        return x + h


class PostNet(nn.Module):
    """Residual post-network for mel-spectrogram refinement.

    Stacks several ``ResidualConv1dBlock`` modules followed by a 1x1 conv,
    with a global residual connection so the network starts as near-identity.

    Args:
        mel_channels: Number of mel frequency bins (default 128).
        num_blocks: Number of residual conv blocks (default 4).
        kernel_size: Kernel size for the conv blocks (default 3).
    """

    def __init__(
        self,
        mel_channels: int = 128,
        num_blocks: int = 4,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualConv1dBlock(mel_channels, kernel_size)
            for _ in range(num_blocks)
        ])
        self.final_conv = nn.Conv1d(mel_channels, mel_channels, kernel_size=1)
        _kaiming_init_conv(self.final_conv)

    def forward(self, predicted_mel: Tensor) -> Tensor:
        """Refine a predicted mel-spectrogram.

        Args:
            predicted_mel: (B, T, 128) VocaloFlow output mel.

        Returns:
            (B, T, 128) sharpened mel = predicted_mel + learned residual.
        """
        # Channels-last -> channels-first for Conv1d
        h = predicted_mel.transpose(1, 2)  # (B, 128, T)

        for block in self.blocks:
            h = block(h)
        h = self.final_conv(h)

        # Channels-first -> channels-last
        h = h.transpose(1, 2)  # (B, T, 128)

        # Global residual: network learns a correction on top of the input
        return predicted_mel + h
