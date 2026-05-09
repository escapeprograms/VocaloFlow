"""DALI ground-truth extraction for Context 2 (controllability) evaluation.

DALI annotations store note frequencies directly in Hz as
``note['freq'] = (start_Hz, end_Hz)`` tuples — no MIDI-to-Hz conversion needed.
"""

from __future__ import annotations

import gzip
import os
import pickle
import re

import numpy as np


_DALI_CACHE: dict[str, object] = {}


def load_dali_entry(dali_annot_dir: str, dali_id: str):
    """Load and cache a DALI annotation entry.

    Returns the DALI Annotations object with horizontal layout.
    """
    if dali_id in _DALI_CACHE:
        return _DALI_CACHE[dali_id]

    gz_path = os.path.join(dali_annot_dir, f"{dali_id}.gz")
    with gzip.open(gz_path, "rb") as f:
        entry = pickle.load(f)

    try:
        entry.vertical2horizontal()
    except Exception:
        pass

    _DALI_CACHE[dali_id] = entry
    return entry


def get_chunk_time_range(
    dali_entry, chunk_name: str
) -> tuple[float, float]:
    """Map chunk_name (e.g. ``'line_3'``) to absolute time in seconds.

    Uses the DALI line annotations: ``chunk_name='line_N'`` maps to
    ``dali_entry.annotations['annot']['lines'][N]['time']``.
    """
    m = re.match(r"line_(\d+)", chunk_name)
    if m is None:
        raise ValueError(f"Cannot parse chunk_name={chunk_name!r} as 'line_N'")
    idx = int(m.group(1))

    lines = dali_entry.annotations["annot"]["lines"]
    if idx >= len(lines):
        raise IndexError(
            f"chunk_name={chunk_name!r} requests line index {idx} "
            f"but only {len(lines)} lines exist"
        )

    start_s, end_s = lines[idx]["time"]
    return float(start_s), float(end_s)


def extract_dali_f0_reference(
    dali_entry,
    chunk_start_s: float,
    chunk_end_s: float,
    total_frames: int,
    sr: int = 24000,
    hop: int = 480,
) -> np.ndarray:
    """Build a frame-level step-function F0 from DALI note annotations.

    For each note overlapping the chunk window, fills the corresponding
    frames with the note's frequency (average of start and end Hz).
    Frames outside any note are 0 (unvoiced).

    Returns:
        (total_frames,) float32 array of F0 in Hz.
    """
    notes = dali_entry.annotations["annot"]["notes"]
    f0 = np.zeros(total_frames, dtype=np.float32)
    frame_dur_s = hop / sr

    for note in notes:
        note_start, note_end = note["time"]
        if note_end <= chunk_start_s or note_start >= chunk_end_s:
            continue

        freq = float(np.mean(note["freq"]))
        rel_start = max(0.0, note_start - chunk_start_s)
        rel_end = min(chunk_end_s - chunk_start_s, note_end - chunk_start_s)

        start_frame = int(rel_start / frame_dur_s)
        end_frame = int(rel_end / frame_dur_s)
        start_frame = max(0, min(start_frame, total_frames))
        end_frame = max(0, min(end_frame, total_frames))

        f0[start_frame:end_frame] = freq

    return f0


def extract_dali_lyrics(
    dali_entry, chunk_start_s: float, chunk_end_s: float
) -> str:
    """Extract word-level lyrics from DALI for WER reference.

    Filters words whose midpoint falls within the chunk window.
    Returns space-joined lowercase string.
    """
    words = dali_entry.annotations["annot"]["words"]
    texts = []
    for w in words:
        mid = (w["time"][0] + w["time"][1]) / 2.0
        if chunk_start_s <= mid <= chunk_end_s:
            texts.append(w["text"].strip())
    return " ".join(texts).lower()


def extract_dali_note_timings(
    dali_entry,
    chunk_start_s: float,
    chunk_end_s: float,
) -> list[dict]:
    """Extract note onset/offset times from DALI (chunk-local coordinates).

    Returns list of dicts with keys: start_s, end_s, duration_s, freq_hz, text.
    """
    notes = dali_entry.annotations["annot"]["notes"]
    result = []
    for note in notes:
        note_start, note_end = note["time"]
        if note_end <= chunk_start_s or note_start >= chunk_end_s:
            continue

        rel_start = max(0.0, note_start - chunk_start_s)
        rel_end = min(chunk_end_s - chunk_start_s, note_end - chunk_start_s)

        result.append({
            "start_s": rel_start,
            "end_s": rel_end,
            "duration_s": rel_end - rel_start,
            "freq_hz": float(np.mean(note["freq"])),
            "text": note.get("text", ""),
        })
    return result
