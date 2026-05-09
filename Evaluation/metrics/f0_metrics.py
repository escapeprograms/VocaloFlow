"""F0 evaluation metrics: RMSE, Pearson correlation, GPE/VDE/FFE."""

from __future__ import annotations

import numpy as np


def _align_and_mask(
    pred_f0: np.ndarray, ref_f0: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Truncate to common length and compute voicing masks.

    Returns:
        (pred, ref, mutual_voiced_mask, either_voiced_mask)
    """
    T = min(len(pred_f0), len(ref_f0))
    pred = pred_f0[:T].astype(np.float64)
    ref = ref_f0[:T].astype(np.float64)
    pred_voiced = pred > 0
    ref_voiced = ref > 0
    mutual = pred_voiced & ref_voiced
    either = pred_voiced | ref_voiced
    return pred, ref, mutual, either


def f0_rmse(pred_f0: np.ndarray, ref_f0: np.ndarray) -> float:
    """F0 RMSE in Hz over mutually voiced frames.

    Returns NaN if no mutually voiced frames exist.
    """
    pred, ref, mutual, _ = _align_and_mask(pred_f0, ref_f0)
    if not mutual.any():
        return float("nan")
    diff = pred[mutual] - ref[mutual]
    return float(np.sqrt(np.mean(diff ** 2)))


def f0_pearson(pred_f0: np.ndarray, ref_f0: np.ndarray) -> float:
    """Pearson correlation of F0 contours over mutually voiced frames.

    Returns NaN if fewer than 2 mutually voiced frames.
    """
    from scipy.stats import pearsonr

    pred, ref, mutual, _ = _align_and_mask(pred_f0, ref_f0)
    if mutual.sum() < 2:
        return float("nan")
    r, _ = pearsonr(pred[mutual], ref[mutual])
    return float(r)


def compute_gpe_vde_ffe(pred_f0: np.ndarray, ref_f0: np.ndarray) -> dict:
    """Gross Pitch Error, Voicing Decision Error, F0 Frame Error.

    Definitions (standard):
      - VDE: % of ALL frames where voicing status disagrees.
      - GPE: % of MUTUALLY VOICED frames where |pred - ref| / ref > 0.20.
      - FFE: % of ALL frames with either a GPE or VDE error.

    Returns:
        {"gpe": float, "vde": float, "ffe": float} as percentages [0-100].
    """
    pred, ref, mutual, _ = _align_and_mask(pred_f0, ref_f0)
    T = len(pred)
    if T == 0:
        return {"gpe": float("nan"), "vde": float("nan"), "ffe": float("nan")}

    pred_voiced = pred > 0
    ref_voiced = ref > 0

    # VDE: voicing disagreement on ALL frames
    vde_mask = pred_voiced != ref_voiced
    vde = 100.0 * vde_mask.sum() / T

    # GPE: gross pitch error on mutually voiced frames
    gpe_frame_mask = np.zeros(T, dtype=bool)
    if mutual.any():
        rel_error = np.abs(pred[mutual] - ref[mutual]) / ref[mutual]
        gpe_on_mutual = rel_error > 0.20
        gpe_pct = 100.0 * gpe_on_mutual.sum() / mutual.sum()
        mutual_indices = np.where(mutual)[0]
        gpe_frame_mask[mutual_indices[gpe_on_mutual]] = True
    else:
        gpe_pct = float("nan")

    # FFE: union of GPE errors and VDE errors across ALL frames
    ffe_mask = gpe_frame_mask | vde_mask
    ffe = 100.0 * ffe_mask.sum() / T

    return {"gpe": float(gpe_pct), "vde": float(vde), "ffe": float(ffe)}
