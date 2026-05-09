"""Mean Opinion Score prediction via UTMOS."""

from __future__ import annotations

import numpy as np

_UTMOS_MODEL = None


def _load_utmos(device: str):
    global _UTMOS_MODEL
    if _UTMOS_MODEL is not None:
        return _UTMOS_MODEL

    import torch

    _UTMOS_MODEL = torch.hub.load(
        "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
    )
    _UTMOS_MODEL = _UTMOS_MODEL.to(device).eval()
    return _UTMOS_MODEL


def unload():
    """Free UTMOS model from GPU."""
    global _UTMOS_MODEL
    _UTMOS_MODEL = None


def compute_mos(
    audio: np.ndarray,
    sr: int = 24000,
    device: str = "cuda",
) -> float:
    """Predict MOS using UTMOS.

    Resamples to 16 kHz if needed (UTMOS expectation).

    Returns:
        Predicted MOS score in [1, 5].
    """
    import torch
    import librosa

    model = _load_utmos(device)

    if sr != 16000:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)

    wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        score = model(wav, sr=16000)

    return float(score.item())
