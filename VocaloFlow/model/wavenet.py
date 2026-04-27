"""WaveNet-style residual pre-processing blocks for VocaloFlow.

Optional alternative to the ConvNeXt pre-processing stack. Dilated non-causal
convolutions with gated activations and per-layer timestep conditioning, after
the non-causal WaveNet denoiser of Parallel WaveGAN / DiffSinger.

Toggled via ``config.num_wavenet_blocks`` (0 = disabled, default).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _kaiming_init_conv(conv: nn.Conv1d) -> None:
    """Kaiming-normal init (relu nonlinearity) with zero bias."""
    nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0.0)


class WaveNetResidualBlock(nn.Module):
    """Single WaveNet residual block.

    Non-causal dilated conv → gated (tanh * sigmoid) activation with per-layer
    timestep conditioning injected before the gate → 1x1 out + 1x1 skip
    projections → residual connection scaled by sqrt(0.5).

    Args:
        channels: Residual channel width (C).
        cond_channels: Conditioning vector width (C_cond).
        skip_channels: Skip connection width (C_skip).
        kernel_size: Dilated conv kernel (odd, typically 3).
        dilation: Dilation rate for this layer.
        dropout: Dropout rate applied at the block input.
    """

    def __init__(
        self,
        channels: int,
        cond_channels: int,
        skip_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd for same-padding"
        padding = (kernel_size - 1) // 2 * dilation

        self.dropout = nn.Dropout(dropout)
        # Dilated conv producing 2C for the gated split
        self.dilated_conv = nn.Conv1d(
            channels, 2 * channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        # Per-block conditioning projection (NOT shared across blocks)
        self.cond_proj = nn.Conv1d(cond_channels, 2 * channels, kernel_size=1)
        # Post-gate projections
        self.out_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(channels, skip_channels, kernel_size=1)

        for m in (self.dilated_conv, self.cond_proj, self.out_conv, self.skip_conv):
            _kaiming_init_conv(m)

    def forward(self, x: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x:    (B, C, T) channels-first hidden state.
            cond: (B, C_cond, T) channels-first conditioning, broadcast in T.

        Returns:
            (x_out, skip) — x_out is (B, C, T) for the next block,
            skip is (B, C_skip, T) accumulated outside the block.
        """
        residual = x
        h = self.dropout(x)
        h = self.dilated_conv(h)                 # (B, 2C, T)
        h = h + self.cond_proj(cond)             # gate-dim conditioning add
        xa, xb = h.chunk(2, dim=1)               # each (B, C, T)
        h = torch.tanh(xa) * torch.sigmoid(xb)   # gated activation
        skip = self.skip_conv(h)                 # (B, C_skip, T)
        out = self.out_conv(h)                   # (B, C, T)
        out = (out + residual) * math.sqrt(0.5)  # PWG residual scaling
        return out, skip


class WaveNetStack(nn.Module):
    """Stack of WaveNet residual blocks with accumulated skip connections.

    Takes channels-last input ``(B, T, C)`` and conditioning vector
    ``(B, C_cond)`` and returns channels-last output ``(B, T, C)``. Internally
    operates channels-first; transposes happen only at the outer boundary.

    The caller typically wraps this with an outer residual:
        ``h = h + wavenet_stack(h, c)``
    so the DiT blocks still see the raw projected input at init time.

    Args:
        hidden_channels: Residual channel width (C).
        cond_channels: Conditioning vector width.
        skip_channels: Width of accumulated skip connections.
        kernel_size: Dilated conv kernel (odd).
        n_layers: Number of residual blocks.
        dilation_cycle: Dilation repeats every ``dilation_cycle`` layers
            (dilation_i = 2 ** (i % dilation_cycle)).
        dropout: Dropout rate inside each residual block.
    """

    def __init__(
        self,
        hidden_channels: int = 512,
        cond_channels: int = 512,
        skip_channels: int = 512,
        kernel_size: int = 3,
        n_layers: int = 8,
        dilation_cycle: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_conv = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1)

        self.residual_blocks = nn.ModuleList([
            WaveNetResidualBlock(
                channels=hidden_channels,
                cond_channels=cond_channels,
                skip_channels=skip_channels,
                kernel_size=kernel_size,
                dilation=2 ** (i % dilation_cycle),
                dropout=dropout,
            )
            for i in range(n_layers)
        ])

        self.output_conv1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.output_conv2 = nn.Conv1d(skip_channels, hidden_channels, kernel_size=1)

        for m in (self.input_conv, self.output_conv1, self.output_conv2):
            _kaiming_init_conv(m)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Run the WaveNet stack.

        Args:
            x:    (B, T, hidden_channels) channels-last hidden state.
            cond: (B, cond_channels) timestep conditioning vector.

        Returns:
            (B, T, hidden_channels) channels-last output.
        """
        # Channels-last → channels-first
        x = x.transpose(1, 2)                             # (B, C, T)

        # Broadcast conditioning across time
        cond = cond.unsqueeze(-1).expand(-1, -1, x.size(-1))  # (B, C_cond, T)

        h = self.input_conv(x)

        skip_sum: Tensor | int = 0
        for block in self.residual_blocks:
            h, skip = block(h, cond)
            skip_sum = skip_sum + skip

        out = F.relu(skip_sum)
        out = F.relu(self.output_conv1(out))
        out = self.output_conv2(out)

        # Channels-first → channels-last
        return out.transpose(1, 2)


class WaveNetDenoiser(nn.Module):
    """Pure WaveNet denoiser backbone for flow matching velocity prediction.

    Unlike WaveNetStack (which projects skip-sum back to hidden_channels for
    use as a pre-processing residual), this outputs directly to mel_channels
    via the skip-sum path.  Designed to be the sole backbone (no DiT blocks).

    Args:
        residual_channels: Width of the residual stream (C).
        cond_channels: Width of the timestep conditioning vector.
        skip_channels: Width of accumulated skip connections.
        mel_channels: Output dimension (128 for mel-spectrogram velocity).
        kernel_size: Dilated conv kernel (odd).
        n_layers: Number of residual blocks.
        dilation_cycle: Dilation repeats every this many layers.
        dropout: Dropout rate inside each residual block.
    """

    def __init__(
        self,
        residual_channels: int = 256,
        cond_channels: int = 256,
        skip_channels: int = 256,
        mel_channels: int = 128,
        kernel_size: int = 3,
        n_layers: int = 20,
        dilation_cycle: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.residual_blocks = nn.ModuleList([
            WaveNetResidualBlock(
                channels=residual_channels,
                cond_channels=cond_channels,
                skip_channels=skip_channels,
                kernel_size=kernel_size,
                dilation=2 ** (i % dilation_cycle),
                dropout=dropout,
            )
            for i in range(n_layers)
        ])

        self.output_conv1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.output_conv2 = nn.Conv1d(skip_channels, mel_channels, kernel_size=1)

        _kaiming_init_conv(self.output_conv1)
        # Zero-init final conv so the denoiser starts predicting near-zero
        # velocity (identity behaviour), analogous to AdaLN-Zero in DiT.
        nn.init.zeros_(self.output_conv2.weight)
        nn.init.zeros_(self.output_conv2.bias)

    def forward(
        self,
        x: Tensor,
        cond: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Run the WaveNet denoiser.

        Args:
            x:    (B, T, residual_channels) channels-last input.
            cond: (B, cond_channels) timestep conditioning vector.
            padding_mask: (B, T) bool, True = valid frame.

        Returns:
            (B, T, mel_channels) channels-last velocity prediction.
        """
        h = x.transpose(1, 2)                                    # (B, C, T)
        cond = cond.unsqueeze(-1).expand(-1, -1, h.size(-1))     # (B, C_cond, T)

        mask_cf = None
        if padding_mask is not None:
            mask_cf = padding_mask.unsqueeze(1).float()           # (B, 1, T)

        skip_sum: Tensor | int = 0
        for block in self.residual_blocks:
            h, skip = block(h, cond)
            if mask_cf is not None:
                h = h * mask_cf
                skip = skip * mask_cf
            skip_sum = skip_sum + skip

        out = F.relu(skip_sum)
        out = F.relu(self.output_conv1(out))
        out = self.output_conv2(out)

        return out.transpose(1, 2)
