"""Multi-resolution STFT auxiliary loss on mel-spectrogram time axis.

Operates on (B, T, 128) mel tensors: treats each of the 128 mel channels as an
independent 1-D time series, computes an STFT along the time axis at multiple
(n_fft, hop, win) resolutions, and combines spectral-convergence + log-magnitude
L1 losses. Intended as an auxiliary signal against temporal-modulation blur in
flow-matching velocity predictions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(
        self,
        resolutions: tuple[tuple[int, int, int], ...] = (
            (16, 4, 16),
            (32, 8, 32),
            (64, 16, 64),
        ),
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.resolutions = tuple((int(n), int(h), int(w)) for n, h, w in resolutions)
        self.eps = eps
        for i, (_, _, win_size) in enumerate(self.resolutions):
            self.register_buffer(f"window_{i}", torch.hann_window(win_size), persistent=False)

    def _stft_loss_single(
        self, x_hat_flat: Tensor, x_tgt_flat: Tensor, n_fft: int, hop: int, window: Tensor
    ) -> Tensor:
        stft_hat = torch.stft(
            x_hat_flat, n_fft=n_fft, hop_length=hop, win_length=window.shape[0],
            window=window, return_complex=True, center=True,
        )
        stft_tgt = torch.stft(
            x_tgt_flat, n_fft=n_fft, hop_length=hop, win_length=window.shape[0],
            window=window, return_complex=True, center=True,
        )
        mag_hat = stft_hat.abs()
        mag_tgt = stft_tgt.abs()

        sc = torch.norm(mag_hat - mag_tgt, p="fro") / (torch.norm(mag_tgt, p="fro") + self.eps)
        log_mag = torch.mean(torch.abs(
            torch.log(mag_hat + self.eps) - torch.log(mag_tgt + self.eps)
        ))
        return sc + log_mag

    def forward(
        self, x_hat: Tensor, x_target: Tensor, padding_mask: Tensor | None = None
    ) -> Tensor:
        """
        Args:
            x_hat, x_target: (B, T, M) mel tensors.
            padding_mask: (B, T) bool, True = valid. Padded frames zeroed in both
                tensors before STFT (errors cancel at the boundary — see plan).
        Returns:
            Scalar tensor, mean of (spectral convergence + log-mag L1) across
            resolutions whose win_size <= T. Zero tensor if no resolutions apply.
        """
        if padding_mask is not None:
            m = padding_mask.unsqueeze(-1).to(x_hat.dtype)
            x_hat = x_hat * m
            x_target = x_target * m

        B, T, M = x_hat.shape
        x_hat_flat = x_hat.permute(0, 2, 1).reshape(B * M, T)
        x_tgt_flat = x_target.permute(0, 2, 1).reshape(B * M, T)

        total = torch.zeros((), device=x_hat.device, dtype=x_hat.dtype)
        n_valid = 0
        for i, (n_fft, hop, win_size) in enumerate(self.resolutions):
            if T < win_size:
                continue
            window = getattr(self, f"window_{i}").to(dtype=x_hat.dtype)
            total = total + self._stft_loss_single(x_hat_flat, x_tgt_flat, n_fft, hop, window)
            n_valid += 1

        if n_valid > 0:
            total = total / n_valid
        return total
