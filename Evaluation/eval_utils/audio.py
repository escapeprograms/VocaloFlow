"""Audio utilities: vocoding, F0 extraction, mel denormalization."""

from __future__ import annotations

import os
import tempfile

import numpy as np

from eval_utils.imports import (
    DATASYNTHESIZER_DIR,
    SOULX_DIR,
    import_from_path,
)

# ---------------------------------------------------------------------------
# Lazy-loaded cross-module references
# ---------------------------------------------------------------------------
_vocoders_mod = None
_f0_extractor_cls = None
_f0_extractor_instance = None


def _get_vocoders():
    global _vocoders_mod
    if _vocoders_mod is None:
        _vocoders_mod = import_from_path(
            "eval_vocoders",
            os.path.join(DATASYNTHESIZER_DIR, "utils", "vocoders.py"),
        )
    return _vocoders_mod


def _get_f0_extractor_cls():
    global _f0_extractor_cls
    if _f0_extractor_cls is None:
        mod = import_from_path(
            "eval_f0_extraction",
            os.path.join(SOULX_DIR, "preprocess", "tools", "f0_extraction.py"),
        )
        _f0_extractor_cls = mod.F0Extractor
    return _f0_extractor_cls


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

SOULX_MEL_MEAN = -4.92
SOULX_MEL_VAR = 8.14


def unload_f0_extractor():
    """Free RMVPE F0 extractor from GPU."""
    global _f0_extractor_instance
    _f0_extractor_instance = None


def free_vram():
    """Release cached PyTorch CUDA memory."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def denormalize_mel(mel: np.ndarray) -> np.ndarray:
    """Undo SoulX z-score normalization → raw log-mel.

    Args:
        mel: (T, 128) normalized log-mel.

    Returns:
        (T, 128) log-mel (log-compressed, not z-scored).
    """
    return mel * np.sqrt(SOULX_MEL_VAR) + SOULX_MEL_MEAN


def vocode_mel(mel: np.ndarray) -> np.ndarray:
    """Convert (T, 128) normalized log-mel → 1D audio at 24 kHz via SoulX Vocos.

    Matches the transpose convention used in pipeline.mel_to_wav().
    """
    voc = _get_vocoders()
    mel_transposed = mel.T.astype(np.float32)  # (128, T)
    audio = voc.invert_mel_to_audio_soulxsinger(mel_transposed, config={})
    return audio


def extract_f0(
    audio_or_path,
    rmvpe_model_path: str,
    device: str,
    sr: int = 24000,
    hop: int = 480,
) -> np.ndarray:
    """Extract F0 via RMVPE.

    Args:
        audio_or_path: 1D numpy array (float32) or path to WAV file.
        rmvpe_model_path: Path to rmvpe.pt weights.
        device: "cuda" or "cpu".

    Returns:
        (T,) float32 array of F0 in Hz (0 = unvoiced).
    """
    global _f0_extractor_instance
    cls = _get_f0_extractor_cls()

    if _f0_extractor_instance is None:
        _f0_extractor_instance = cls(
            model_path=rmvpe_model_path,
            device=device,
            target_sr=sr,
            hop_size=hop,
        )

    if isinstance(audio_or_path, (str, os.PathLike)):
        wav_path = str(audio_or_path)
    else:
        import soundfile as sf

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            sf.write(tmp.name, audio_or_path, sr)
            wav_path = tmp.name
        finally:
            tmp.close()

    f0 = _f0_extractor_instance.process(wav_path)

    if not isinstance(audio_or_path, (str, os.PathLike)):
        os.unlink(wav_path)

    return f0.astype(np.float32)


def audio_to_mel(audio: np.ndarray, sr: int = 24000) -> np.ndarray:
    """Convert 1D audio → (T, 128) normalized log-mel matching SoulX format.

    Uses the same mel computation as the data pipeline (n_fft=1920, hop=480,
    128 mels, log-compressed + z-score normalized).
    """
    voc = _get_vocoders()
    mel = voc.mel_to_soulx_mel(audio, sr)  # (128, T)
    return mel.T.astype(np.float32)         # (T, 128)
