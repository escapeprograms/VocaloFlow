"""ODE integration for sampling from the learned flow, with optional CFG."""

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
    cfg_scale: float = 1.0,
) -> Tensor:
    """Integrate the learned ODE from t=0 (prior) to t=1 (target).

    When cfg_scale > 1.0, uses classifier-free guidance: runs both a
    conditional and unconditional forward pass per step, then interpolates.

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
        cfg_scale: Classifier-free guidance scale (1.0 = no guidance).

    Returns:
        (B, T, 128) predicted high-quality mel-spectrogram.
    """
    model.eval()
    use_cfg = cfg_scale > 1.0
    dt = 1.0 / num_steps
    x_t = x_0.clone()

    # Pre-build unconditional inputs (zeros) if using CFG
    if use_cfg:
        zeros_f0 = torch.zeros_like(f0)
        zeros_voicing = torch.zeros_like(voicing)
        zeros_ph = torch.zeros_like(phoneme_ids)

    # Steps to log diagnostics (first, quarter, middle, last)
    _diag_steps = {0, num_steps // 4, num_steps // 2, num_steps - 1}

    def _get_velocity(x: Tensor, t_tensor: Tensor) -> Tensor:
        """Get velocity, applying CFG if enabled."""
        v_cond = model(x, t_tensor, x_0, f0, voicing, phoneme_ids, padding_mask)
        if not use_cfg:
            return v_cond
        v_uncond = model(x, t_tensor, x_0, zeros_f0, zeros_voicing, zeros_ph, padding_mask)
        return v_uncond + cfg_scale * (v_cond - v_uncond)

    for i in range(num_steps):
        t_val = i * dt
        t = torch.full((x_0.shape[0],), t_val, device=x_0.device)

        if method == "euler":
            v = _get_velocity(x_t, t)
            x_t = x_t + dt * v

        elif method == "midpoint":
            # Half step
            v1 = _get_velocity(x_t, t)
            x_mid = x_t + 0.5 * dt * v1

            # Full step using midpoint velocity
            t_mid = torch.full((x_0.shape[0],), t_val + 0.5 * dt, device=x_0.device)
            v2 = _get_velocity(x_mid, t_mid)
            x_t = x_t + dt * v2
            v = v2  # for diagnostics

        else:
            raise ValueError(f"Unknown ODE method: {method}")

        if diagnostics and i in _diag_steps:
            _vabs = v.abs()
            print(f"  [ode] step {i:3d}/{num_steps}  t={t_val:.4f}  "
                  f"|v| mean={_vabs.mean():.6f}  max={_vabs.max():.6f}  "
                  f"|x_t| mean={x_t.abs().mean():.4f}")

    return x_t
