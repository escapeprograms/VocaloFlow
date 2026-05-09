"""ContentVec embedding analysis for SVC baseline motivation."""

from __future__ import annotations

import numpy as np

_CV_MODEL = None
_CV_PROCESSOR = None


def _load_contentvec(device: str = "cuda"):
    global _CV_MODEL, _CV_PROCESSOR
    if _CV_MODEL is not None:
        return _CV_MODEL, _CV_PROCESSOR

    import torch
    from transformers import AutoModel, AutoFeatureExtractor

    model_name = "facebook/hubert-base-ls960"
    _CV_PROCESSOR = AutoFeatureExtractor.from_pretrained(model_name)
    _CV_MODEL = AutoModel.from_pretrained(model_name).to(device).eval()
    return _CV_MODEL, _CV_PROCESSOR


def extract_features(
    audio: np.ndarray,
    sr: int = 24000,
    device: str = "cuda",
) -> np.ndarray:
    """Extract HuBERT/ContentVec frame-level features.

    Resamples to 16 kHz.

    Returns:
        (T_cv, 768) float32 array.
    """
    import torch
    import librosa

    model, processor = _load_contentvec(device)

    if sr != 16000:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs)

    return out.last_hidden_state.squeeze(0).cpu().numpy().astype(np.float32)


def compute_cosine_similarity(
    feats_a: np.ndarray,
    feats_b: np.ndarray,
) -> dict:
    """Cosine similarity statistics between two feature sequences.

    Truncates to min length.

    Returns:
        {"cosine_mean": float, "cosine_std": float}
    """
    T = min(len(feats_a), len(feats_b))
    a = feats_a[:T].astype(np.float64)
    b = feats_b[:T].astype(np.float64)

    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    cos_sim = np.sum(a_norm * b_norm, axis=1)

    return {
        "cosine_mean": float(np.mean(cos_sim)),
        "cosine_std": float(np.std(cos_sim)),
    }
