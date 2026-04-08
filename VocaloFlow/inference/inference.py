"""ODE integration for sampling from the learned flow."""

import torch
from torch import Tensor
import torch.nn as nn


@torch.no_grad()
def sample_ode(
    model: nn.Module,
    x_0: Tensor,
    f0: Tensor,
    voicing: Tensor,
    phoneme_ids: Tensor,
    num_steps: int = 32,
    method: str = "midpoint",
    padding_mask: Tensor | None = None,
    diagnostics: bool = True,
) -> Tensor:
    """Integrate the learned ODE from t=0 (prior) to t=1 (target).

    Args:
        model: Trained VocaloFlow model (should be in eval mode).
        x_0: (B, T, 128) prior mel-spectrogram.
        f0: (B, T) F0 contour.
        voicing: (B, T) V/UV flag.
        phoneme_ids: (B, T) resolved phoneme token IDs.
        num_steps: Number of ODE integration steps.
        method: "euler" or "midpoint".
        padding_mask: (B, T) bool, True = valid frame.
        diagnostics: If True, print velocity stats at key steps.

    Returns:
        (B, T, 128) predicted high-quality mel-spectrogram.
    """
    model.eval()
    dt = 1.0 / num_steps
    x_t = x_0.clone()

    # Steps to log diagnostics (first, middle, last)
    _diag_steps = {0, num_steps // 4, num_steps // 2, num_steps - 1}

    for i in range(num_steps):
        t_val = i * dt
        t = torch.full((x_0.shape[0],), t_val, device=x_0.device)

        if method == "euler":
            v = model(x_t, t, x_0, f0, voicing, phoneme_ids, padding_mask)
            x_t = x_t + dt * v
            _v_diag = v

        elif method == "midpoint":
            # Half step
            v1 = model(x_t, t, x_0, f0, voicing, phoneme_ids, padding_mask)
            x_mid = x_t + 0.5 * dt * v1

            # Full step using midpoint velocity
            t_mid = torch.full((x_0.shape[0],), t_val + 0.5 * dt, device=x_0.device)
            v2 = model(x_mid, t_mid, x_0, f0, voicing, phoneme_ids, padding_mask)
            x_t = x_t + dt * v2
            _v_diag = v2

        else:
            raise ValueError(f"Unknown ODE method: {method}")

        if diagnostics and i in _diag_steps:
            _vabs = _v_diag.abs()
            print(f"  [ode] step {i:3d}/{num_steps}  t={t_val:.4f}  "
                  f"|v| mean={_vabs.mean():.6f}  max={_vabs.max():.6f}  "
                  f"|x_t| mean={x_t.abs().mean():.4f}")

    return x_t
