"""Energy-balanced weighting for flow matching velocity MSE loss."""

import torch
from torch import Tensor


class EnergyBalancedWeight:
    """Inverse-energy weighting to upweight low-energy mel regions.

    Computes per-element importance weights inversely proportional to local
    energy along frequency and/or time axes. Weights are mean-normalized to 1.0
    over valid (non-padded) elements so overall loss scale is preserved.

    References:
        RFWave (Peng et al., ICLR 2025)
        YingMusic-SVC (Chen et al., arXiv:2512.04793)
    """

    def __init__(self, epsilon: float = 1e-4, mode: str = "both") -> None:
        assert mode in ("freq", "time", "both"), f"Invalid mode: {mode}"
        self.epsilon = epsilon
        self.mode = mode

    def __call__(self, x_1: Tensor, padding_mask: Tensor | None) -> Tensor:
        """Compute energy-balanced weights.

        Args:
            x_1: Target mel spectrogram, shape (B, T, M).
            padding_mask: Boolean mask (B, T), True = valid frame. None = all valid.

        Returns:
            Weights tensor (B, T, M), mean ~1.0 over valid elements.
        """
        B, T, M = x_1.shape
        device = x_1.device

        if padding_mask is not None:
            mask_f = padding_mask.float()                                    # (B, T)
            valid_frames = mask_f.sum(dim=1, keepdim=True).clamp(min=1)     # (B, 1)
            x_masked = x_1 * mask_f.unsqueeze(-1)                           # (B, T, M)
        else:
            mask_f = torch.ones(B, T, device=device, dtype=x_1.dtype)
            valid_frames = torch.full((B, 1), T, device=device, dtype=x_1.dtype)
            x_masked = x_1

        weight = torch.ones(B, T, M, device=device, dtype=x_1.dtype)

        if self.mode in ("freq", "both"):
            bin_energy = x_masked.abs().sum(dim=1) / valid_frames           # (B, M)
            w_freq = 1.0 / (bin_energy + self.epsilon)                      # (B, M)
            w_freq = w_freq * M / w_freq.sum(dim=1, keepdim=True)           # normalize mean=1
            weight = w_freq.unsqueeze(1)                                    # (B, 1, M)

        if self.mode in ("time", "both"):
            frame_energy = x_masked.abs().sum(dim=2)                        # (B, T)
            w_time = 1.0 / (frame_energy + self.epsilon)                    # (B, T)
            w_time_valid_sum = (w_time * mask_f).sum(dim=1, keepdim=True).clamp(min=1e-8)
            w_time = w_time * valid_frames / w_time_valid_sum               # normalize mean=1
            weight = weight * w_time.unsqueeze(2)                           # (B, T, M)

        if self.mode == "both":
            valid_mask = mask_f.unsqueeze(-1).expand_as(weight)             # (B, T, M)
            weight_sum = (weight * valid_mask).sum(dim=(1, 2), keepdim=True)
            weight_count = valid_mask.sum(dim=(1, 2), keepdim=True).clamp(min=1)
            weight = weight * weight_count / weight_sum.clamp(min=1e-8)

        return weight
