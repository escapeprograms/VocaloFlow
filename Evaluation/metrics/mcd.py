"""Mel Cepstral Distortion computed directly from log-mel spectrograms.

Avoids the audio round-trip by applying DCT to the existing 128-bin
log-mel spectrograms (Kubichek 1993).
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct

from eval_utils.audio import denormalize_mel


def compute_mcd(
    pred_mel: np.ndarray,
    target_mel: np.ndarray,
    n_mfcc: int = 13,
) -> float:
    """Mel Cepstral Distortion in dB.

    Algorithm:
      1. Denormalize both mels (undo z-score → raw log-mel).
      2. DCT Type-II along frequency axis → MFCCs (T, 128).
      3. Keep coefficients 1..n_mfcc (skip 0th / energy).
      4. Per-frame Euclidean distance.
      5. Scale: MCD = (10 * sqrt(2) / ln(10)) * mean(distances).

    Args:
        pred_mel:   (T, 128) normalized log-mel.
        target_mel: (T, 128) normalized log-mel.
        n_mfcc:     Number of cepstral coefficients (default 13).

    Returns:
        MCD in dB.  Lower is better.  Typical singing: 4–8 dB.
    """
    T = min(pred_mel.shape[0], target_mel.shape[0])
    pred = denormalize_mel(pred_mel[:T]).astype(np.float64)
    target = denormalize_mel(target_mel[:T]).astype(np.float64)

    pred_mfcc = dct(pred, type=2, axis=1, norm="ortho")[:, 1 : n_mfcc + 1]
    target_mfcc = dct(target, type=2, axis=1, norm="ortho")[:, 1 : n_mfcc + 1]

    diff = pred_mfcc - target_mfcc
    frame_dist = np.sqrt(np.sum(diff ** 2, axis=1))

    scale = 10.0 * np.sqrt(2.0) / np.log(10.0)
    return float(scale * np.mean(frame_dist))
