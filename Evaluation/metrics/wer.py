"""Word Error Rate via Whisper transcription."""

from __future__ import annotations

import re

import numpy as np

_WHISPER_MODEL = None
_WHISPER_NAME = None


def _load_whisper(model_name: str, device: str):
    global _WHISPER_MODEL, _WHISPER_NAME
    if _WHISPER_MODEL is not None and _WHISPER_NAME == model_name:
        return _WHISPER_MODEL
    import whisper

    _WHISPER_MODEL = whisper.load_model(model_name, device=device)
    _WHISPER_NAME = model_name
    return _WHISPER_MODEL


def unload():
    """Free Whisper model from GPU."""
    global _WHISPER_MODEL, _WHISPER_NAME
    _WHISPER_MODEL = None
    _WHISPER_NAME = None


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def transcribe(
    audio: np.ndarray,
    sr: int = 24000,
    model_name: str = "large-v3",
    device: str = "cuda",
) -> str:
    """Transcribe audio array using Whisper.

    Resamples to 16 kHz internally (Whisper requirement).
    Returns normalized lowercase text.
    """
    import librosa

    model = _load_whisper(model_name, device)
    if sr != 16000:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
    audio = audio.astype(np.float32)
    result = model.transcribe(audio, language="en")
    return _normalize(result["text"])


def compute_wer(hypothesis: str, reference: str) -> float:
    """Word Error Rate between hypothesis and reference.

    Uses jiwer library.  Returns WER as float in [0, inf).
    """
    import jiwer

    hypothesis = _normalize(hypothesis)
    reference = _normalize(reference)
    if not reference:
        return float("nan")
    return float(jiwer.wer(reference, hypothesis))
