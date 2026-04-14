"""ConvNeXtV2 pre-processing blocks for VocaloFlow.

Inserts between input_proj and DiT blocks to capture local temporal patterns
(consonants, fast spectral transitions) before the transformer processes them.
Design follows ConvNeXtV2 (depthwise-separable conv with inverted bottleneck
and Global Response Normalization).
"""

import torch
import torch.nn as nn
from torch import Tensor


class GlobalResponseNorm(nn.Module):
    """Global Response Normalization (ConvNeXtV2).

    Adds inter-channel competition by normalizing each channel's activation
    relative to the global (across-time) L2 norm of all channels.

    Args:
        dim: Feature dimension (typically the expanded FFN width).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Apply GRN.

        Args:
            x: (B, T, dim) input tensor.

        Returns:
            (B, T, dim) normalized tensor.
        """
        # Per-channel L2 norm across time: (B, 1, dim)
        norm = x.norm(p=2, dim=1, keepdim=True)
        # Normalize by mean norm across channels
        x_norm = x / (norm.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * x_norm + self.beta + x


class ConvNeXtV2Block(nn.Module):
    """Single ConvNeXtV2 block: depthwise conv -> LN -> FFN w/ GRN -> residual.

    Args:
        dim: Hidden dimension (input and output).
        kernel_size: Depthwise convolution kernel size.
        expansion: FFN expansion ratio (default 4x).
        dropout: Dropout rate after projection.
    """

    def __init__(
        self,
        dim: int = 512,
        kernel_size: int = 7,
        expansion: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        ffn_dim = dim * expansion

        # Depthwise convolution (channels-first)
        self.dwconv = nn.Conv1d(
            dim, dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )

        # Channels-last processing
        self.norm = nn.LayerNorm(dim)
        self.pw_expand = nn.Linear(dim, ffn_dim)
        self.act = nn.GELU()
        self.grn = GlobalResponseNorm(ffn_dim)
        self.pw_project = nn.Linear(ffn_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with residual connection.

        Args:
            x: (B, T, dim) input tensor.

        Returns:
            (B, T, dim) output tensor.
        """
        residual = x

        # Depthwise conv in channels-first layout
        h = x.transpose(1, 2)          # (B, dim, T)
        h = self.dwconv(h)             # (B, dim, T)
        h = h.transpose(1, 2)          # (B, T, dim)

        # Pointwise FFN with GRN
        h = self.norm(h)
        h = self.pw_expand(h)          # (B, T, ffn_dim)
        h = self.act(h)
        h = self.grn(h)
        h = self.pw_project(h)         # (B, T, dim)
        h = self.drop(h)

        return residual + h


class ConvNeXtStack(nn.Module):
    """Stack of ConvNeXtV2 blocks for local temporal feature extraction.

    Args:
        dim: Hidden dimension.
        num_blocks: Number of ConvNeXtV2 blocks.
        kernel_size: Depthwise convolution kernel size.
        expansion: FFN expansion ratio.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        dim: int = 512,
        num_blocks: int = 4,
        kernel_size: int = 7,
        expansion: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            ConvNeXtV2Block(dim, kernel_size, expansion, dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, x: Tensor) -> Tensor:
        """Pass through all ConvNeXt blocks sequentially.

        Args:
            x: (B, T, dim) input tensor.

        Returns:
            (B, T, dim) output tensor enriched with local temporal features.
        """
        for block in self.blocks:
            x = block(x)
        return x
