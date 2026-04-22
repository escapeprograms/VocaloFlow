"""Optimal Transport Conditional Flow Matching (OT-CFM) loss with CFG dropout."""

import torch
import torch.nn as nn
from torch import Tensor


class FlowMatchingLoss(nn.Module):
    """OT-CFM training objective for VocaloFlow.

    Interpolation path:  x_t = (1 - (1-sigma)*t) * x_0  +  t * x_1
    Target velocity:     v   = x_1  -  (1-sigma) * x_0

    Velocity MSE = E_t[ || v_theta(x_t, t, cond) - v ||^2 ],  masked for padding.

    Optional auxiliary: multi-resolution STFT loss on the one-step output estimate
    x_1_hat = x_t.detach() + (1 - t) * v_pred, summed as
        total = velocity + stft_lambda * stft.

    Returns a dict {total, velocity, stft} so both components can be logged.

    Supports classifier-free guidance (CFG) dropout: during training,
    conditioning signals (f0, voicing, phoneme_ids) are randomly replaced
    with zeros to train the unconditional path.

    Args:
        sigma_min: Small noise floor for numerical stability (default 1e-4).
        cfg_dropout_prob: Probability of dropping conditioning per sample (default 0.0).
        stft_loss: Optional MultiResolutionSTFTLoss module. When None, STFT term is skipped.
        stft_lambda: Weight applied to the STFT loss in the total.
    """

    def __init__(
        self,
        sigma_min: float = 1e-4,
        cfg_dropout_prob: float = 0.0,
        stft_loss: nn.Module | None = None,
        stft_lambda: float = 0.0,
        energy_balance: object | None = None,
        energy_balance_lambda: float = 0.4,
    ) -> None:
        super().__init__()
        self.sigma_min = sigma_min
        self.cfg_dropout_prob = cfg_dropout_prob
        self.stft_loss = stft_loss
        self.stft_lambda = stft_lambda
        self.energy_balance = energy_balance
        self.energy_balance_lambda = energy_balance_lambda

    def forward(
        self,
        model: nn.Module,
        x_0: Tensor,
        x_1: Tensor,
        f0: Tensor,
        voicing: Tensor,
        phoneme_ids: Tensor,
        padding_mask: Tensor | None = None,
        plbert_features: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute the flow matching loss (+ optional STFT aux) for one batch.

        Args:
            model: VocaloFlow model.
            x_0: (B, T, 128) prior mel-spectrogram.
            x_1: (B, T, 128) target mel-spectrogram.
            f0: (B, T) F0 contour.
            voicing: (B, T) V/UV flag.
            phoneme_ids: (B, T) resolved phoneme token IDs.
            padding_mask: (B, T) bool, True = valid frame.
            plbert_features: (B, T, 768) frozen PL-BERT embeddings (optional).

        Returns:
            Dict with scalar tensors: {"total", "velocity", "stft"}.
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

        # CFG dropout: zero out conditioning for random samples
        if self.training and self.cfg_dropout_prob > 0:
            drop_mask = torch.rand(B, device=device) < self.cfg_dropout_prob  # (B,)
            if drop_mask.any():
                f0 = f0.clone()
                voicing = voicing.clone()
                phoneme_ids = phoneme_ids.clone()
                f0[drop_mask] = 0.0
                voicing[drop_mask] = 0.0
                phoneme_ids[drop_mask] = 0
                if plbert_features is not None:
                    plbert_features = plbert_features.clone()
                    plbert_features[drop_mask] = 0.0

        # Predict velocity
        v_pred = model(x_t, t, x_0, f0, voicing, phoneme_ids, padding_mask,
                       plbert_features=plbert_features)

        # Velocity MSE, optionally energy-balanced, masked for padding
        diff = (v_pred - v_target) ** 2  # (B, T, 128)

        if self.training and self.energy_balance is not None:
            eb_weights = self.energy_balance(x_1, padding_mask)  # (B, T, 128)
            blend = (1.0 - self.energy_balance_lambda) + self.energy_balance_lambda * eb_weights
            diff = diff * blend

        if padding_mask is not None:
            mask = padding_mask.unsqueeze(-1).float()
            velocity = (diff * mask).sum() / (mask.sum() * diff.shape[-1])
        else:
            velocity = diff.mean()

        # EB diagnostic for logging
        if self.training and self.energy_balance is not None:
            eb_std = eb_weights[padding_mask].std() if padding_mask is not None else eb_weights.std()
        else:
            eb_std = torch.zeros((), device=device, dtype=velocity.dtype)

        # Optional STFT aux loss on one-step output estimate.
        # x_t.detach() is essential: v_pred already carries the parameter gradient;
        # letting grad flow back through x_t would add a redundant path via the
        # (frozen) data tensors x_0/x_1.
        if self.stft_loss is not None:
            x_1_hat = x_t.detach() + (1.0 - t_expand) * v_pred
            stft = self.stft_loss(x_1_hat, x_1, padding_mask)
        else:
            stft = torch.zeros((), device=device, dtype=velocity.dtype)

        total = velocity + self.stft_lambda * stft
        return {"total": total, "velocity": velocity, "stft": stft, "eb_std": eb_std}
