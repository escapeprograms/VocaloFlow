"""ODE integration for sampling from the learned flow, with optional CFG."""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor
import torch.nn as nn

VelocityFn = Callable[[Tensor, Tensor], Tensor]


# ── Time schedule ─────────────────────────────────────────────────────────

def _build_time_schedule(
    num_steps: int, schedule: str = "sway", sway_coeff: float = -1.0,
) -> list[float]:
    """Build a list of num_steps+1 time values from t=0 to t=1.

    Args:
        num_steps: Number of ODE integration steps.
        schedule: "uniform" for evenly spaced steps, "sway" for cosine-based
            sway sampling that concentrates steps near the endpoints.
        sway_coeff: Sway coefficient *s* (only used when schedule="sway").
            Negative values push more steps toward t=0 and t=1.

    Returns:
        List of num_steps+1 floats in [0, 1], starting at 0 and ending at 1.
    """
    ts_uniform = [i / num_steps for i in range(num_steps + 1)]
    if schedule == "uniform":
        return ts_uniform
    if schedule == "sway":
        return [
            t + sway_coeff * (math.cos(math.pi / 2 * t) - 1 + t)
            for t in ts_uniform
        ]
    raise ValueError(f"Unknown time schedule: {schedule}")


# ── Fixed-step stepper functions ──────────────────────────────────────────
# Each takes (x_t, t_val, dt, get_velocity, device, batch_size)
# and returns (x_new, v_last).

def _step_euler(
    x_t: Tensor, t_val: float, dt: float,
    get_velocity: VelocityFn, device: torch.device, batch_size: int,
) -> tuple[Tensor, Tensor]:
    t = torch.full((batch_size,), t_val, device=device)
    v = get_velocity(x_t, t)
    return x_t + dt * v, v


def _step_midpoint(
    x_t: Tensor, t_val: float, dt: float,
    get_velocity: VelocityFn, device: torch.device, batch_size: int,
) -> tuple[Tensor, Tensor]:
    t = torch.full((batch_size,), t_val, device=device)
    v1 = get_velocity(x_t, t)
    x_mid = x_t + 0.5 * dt * v1
    t_mid = torch.full((batch_size,), t_val + 0.5 * dt, device=device)
    v2 = get_velocity(x_mid, t_mid)
    return x_t + dt * v2, v2


def _step_rk4(
    x_t: Tensor, t_val: float, dt: float,
    get_velocity: VelocityFn, device: torch.device, batch_size: int,
) -> tuple[Tensor, Tensor]:
    t0 = torch.full((batch_size,), t_val, device=device)
    t_half = torch.full((batch_size,), t_val + 0.5 * dt, device=device)
    t1 = torch.full((batch_size,), t_val + dt, device=device)

    k1 = get_velocity(x_t, t0)
    k2 = get_velocity(x_t + 0.5 * dt * k1, t_half)
    k3 = get_velocity(x_t + 0.5 * dt * k2, t_half)
    k4 = get_velocity(x_t + dt * k3, t1)

    x_new = x_t + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x_new, k4


_FIXED_STEP_METHODS: dict[str, Callable] = {
    "euler": _step_euler,
    "midpoint": _step_midpoint,
    "rk4": _step_rk4,
}


# ── Adaptive Dormand-Prince 4(5) ─────────────────────────────────────────

# Butcher tableau coefficients for DOPRI5
_A = [
    [],
    [1.0 / 5.0],
    [3.0 / 40.0, 9.0 / 40.0],
    [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0],
    [19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0],
    [9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0],
    [35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0],
]
_C = [0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0]
# 5th-order weights (same as row 6 of A, used for the step)
_B5 = [35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0]
# 4th-order weights (for error estimation)
_B4 = [5179.0 / 57600.0, 0.0, 7571.0 / 16695.0, 393.0 / 640.0,
       -92097.0 / 339200.0, 187.0 / 2100.0, 1.0 / 40.0]
# Error coefficients: E_i = B5_i - B4_i
_E = [b5 - b4 for b5, b4 in zip(_B5, _B4)]

_SAFETY = 0.9
_MIN_FACTOR = 0.2
_MAX_FACTOR = 5.0


def _integrate_dopri5(
    x_0: Tensor,
    get_velocity: VelocityFn,
    device: torch.device,
    batch_size: int,
    h_init: float = 0.03125,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    max_steps: int = 1000,
    diagnostics: bool = True,
) -> Tensor:
    """Adaptive Dormand-Prince 4(5) ODE integration from t=0 to t=1."""
    t_cur = 0.0
    h = min(h_init, 1.0)
    x_t = x_0.clone()

    n_accept = 0
    n_reject = 0
    diag_thresholds = [0.25, 0.5, 0.75]
    diag_idx = 0

    # FSAL: first evaluation
    t_tensor = torch.full((batch_size,), t_cur, device=device)
    k_prev = get_velocity(x_t, t_tensor)

    while t_cur < 1.0:
        if n_accept + n_reject >= max_steps:
            print(f"  [dopri5] WARNING: hit max_steps={max_steps} at t={t_cur:.6f}")
            break

        h = min(h, 1.0 - t_cur)

        # 7 stages (FSAL: k1 = k_prev from last accepted step)
        k = [k_prev]
        for s in range(1, 7):
            t_s = t_cur + _C[s] * h
            t_tensor = torch.full((batch_size,), t_s, device=device)
            dx = sum(a * ki for a, ki in zip(_A[s], k))
            k.append(get_velocity(x_t + h * dx, t_tensor))

        # 5th-order solution (the step we'll take if accepted)
        x_new = x_t + h * sum(b * ki for b, ki in zip(_B5, k))

        # Error estimate
        err_vec = h * sum(e * ki for e, ki in zip(_E, k))
        err_norm = torch.max(
            (err_vec / (atol + rtol * torch.max(x_t.abs(), x_new.abs()))).abs()
        ).item()
        err_norm = max(err_norm, 1e-10)

        if err_norm <= 1.0:
            # Accept step
            t_cur += h
            x_t = x_new
            k_prev = k[6]  # FSAL
            n_accept += 1

            if diagnostics:
                while diag_idx < len(diag_thresholds) and t_cur >= diag_thresholds[diag_idx]:
                    _vabs = k_prev.abs()
                    print(f"  [dopri5] t={t_cur:.4f}  h={h:.6f}  "
                          f"|v| mean={_vabs.mean():.6f}  max={_vabs.max():.6f}  "
                          f"|x_t| mean={x_t.abs().mean():.4f}")
                    diag_idx += 1
        else:
            n_reject += 1

        # Step size update (applied for both accept and reject)
        h = h * min(_MAX_FACTOR, max(_MIN_FACTOR, _SAFETY * err_norm ** (-0.2)))
        h = max(h, 1e-8)

        if err_norm > 1.0:
            # Re-evaluate k1 at same t_cur since we rejected
            t_tensor = torch.full((batch_size,), t_cur, device=device)
            k_prev = get_velocity(x_t, t_tensor)

    if diagnostics:
        print(f"  [dopri5] done: {n_accept} accepted, {n_reject} rejected steps")

    return x_t


# ── Main entry point ──────────────────────────────────────────────────────

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
    plbert_features: Tensor | None = None,
    speaker_embedding: Tensor | None = None,
    time_schedule: str = "sway",
    sway_coeff: float = -1.0,
    atol: float = 1e-5,
    rtol: float = 1e-5,
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
        num_steps: Number of ODE integration steps (fixed-step methods) or
            initial step size hint (dopri5 uses h_init = 1/num_steps).
        method: "euler", "midpoint", "rk4", or "dopri5".
        padding_mask: (B, T) bool, True = valid frame.
        diagnostics: If True, print velocity stats at key steps.
        cfg_scale: Classifier-free guidance scale (1.0 = no guidance).
        time_schedule: "uniform" or "sway" timestep distribution
            (ignored by dopri5, which adapts its own steps).
        sway_coeff: Sway coefficient for "sway" schedule (default -1.0).
        atol: Absolute tolerance for adaptive methods (dopri5).
        rtol: Relative tolerance for adaptive methods (dopri5).

    Returns:
        (B, T, 128) predicted high-quality mel-spectrogram.
    """
    model.eval()
    use_cfg = cfg_scale > 1.0
    B = x_0.shape[0]
    x_t = x_0.clone()

    # Pre-build unconditional inputs (zeros) if using CFG
    if use_cfg:
        zeros_f0 = torch.zeros_like(f0)
        zeros_voicing = torch.zeros_like(voicing)
        zeros_ph = torch.zeros_like(phoneme_ids)
        zeros_plbert = (torch.zeros_like(plbert_features)
                        if plbert_features is not None else None)

    def _get_velocity(x: Tensor, t_tensor: Tensor) -> Tensor:
        v_cond = model(x, t_tensor, x_0, f0, voicing, phoneme_ids, padding_mask,
                       plbert_features=plbert_features,
                       speaker_embedding=speaker_embedding)
        if not use_cfg:
            return v_cond
        v_uncond = model(x, t_tensor, x_0, zeros_f0, zeros_voicing, zeros_ph, padding_mask,
                         plbert_features=zeros_plbert,
                         speaker_embedding=speaker_embedding)
        return v_uncond + cfg_scale * (v_cond - v_uncond)

    # ── Adaptive method ───────────────────────────────────────────────
    if method == "dopri5":
        return _integrate_dopri5(
            x_t, _get_velocity, x_0.device, B,
            h_init=1.0 / num_steps, atol=atol, rtol=rtol,
            diagnostics=diagnostics,
        )

    # ── Fixed-step methods ────────────────────────────────────────────
    if method not in _FIXED_STEP_METHODS:
        raise ValueError(
            f"Unknown ODE method: {method!r}. "
            f"Choose from {list(_FIXED_STEP_METHODS) + ['dopri5']}"
        )
    stepper = _FIXED_STEP_METHODS[method]
    ts = _build_time_schedule(num_steps, time_schedule, sway_coeff)
    _diag_steps = {0, num_steps // 4, num_steps // 2, num_steps - 1}

    for i in range(num_steps):
        t_val = ts[i]
        dt = ts[i + 1] - ts[i]
        x_t, v = stepper(x_t, t_val, dt, _get_velocity, x_0.device, B)

        if diagnostics and i in _diag_steps:
            _vabs = v.abs()
            print(f"  [ode] step {i:3d}/{num_steps}  t={t_val:.4f}  "
                  f"|v| mean={_vabs.mean():.6f}  max={_vabs.max():.6f}  "
                  f"|x_t| mean={x_t.abs().mean():.4f}")

    return x_t
