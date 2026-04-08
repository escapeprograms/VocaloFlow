"""Optimal Transport Conditional Flow Matching (OT-CFM) loss."""

import torch
import torch.nn as nn
from torch import Tensor


class FlowMatchingLoss(nn.Module):
    """OT-CFM training objective for VocaloFlow.

    Interpolation path:  x_t = (1 - (1-sigma)*t) * x_0  +  t * x_1
    Target velocity:     v   = x_1  -  (1-sigma) * x_0

    Loss = E_t[ || v_theta(x_t, t, cond) - v ||^2 ],  masked for padding.

    Args:
        sigma_min: Small noise floor for numerical stability (default 1e-4).
    """

    def __init__(self, sigma_min: float = 1e-4) -> None:
        super().__init__()
        self.sigma_min = sigma_min

    def forward(
        self,
        model: nn.Module,
        x_0: Tensor,
        x_1: Tensor,
        f0: Tensor,
        voicing: Tensor,
        phoneme_ids: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute the flow matching loss for one batch.

        Args:
            model: VocaloFlow model.
            x_0: (B, T, 128) prior mel-spectrogram.
            x_1: (B, T, 128) target mel-spectrogram.
            f0: (B, T) F0 contour.
            voicing: (B, T) V/UV flag.
            phoneme_ids: (B, T) resolved phoneme token IDs.
            padding_mask: (B, T) bool, True = valid frame.

        Returns:
            Scalar loss tensor.
        """
        B = x_0.shape[0]
        device = x_0.device

        # Sample timestep t ~ Uniform(0, 1)
        t = torch.rand(B, device=device)

        # Construct interpolated state x_t
        s = 1.0 - self.sigma_min
        t_expand = t[:, None, None]  # (B, 1, 1) for broadcasting with (B, T, 128)
        x_t = (1.0 - s * t_expand) * x_0 + t_expand * x_1

        # Target velocity (constant along the OT path)
        v_target = x_1 - s * x_0

        # Predict velocity
        v_pred = model(x_t, t, x_0, f0, voicing, phoneme_ids, padding_mask)

        # Compute MSE loss, masked for padding
        diff = (v_pred - v_target) ** 2  # (B, T, 128)

        if padding_mask is not None:
            # (B, T) -> (B, T, 1) for broadcasting
            mask = padding_mask.unsqueeze(-1).float()
            loss = (diff * mask).sum() / (mask.sum() * diff.shape[-1])
        else:
            loss = diff.mean()

        return loss
