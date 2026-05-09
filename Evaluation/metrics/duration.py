"""Onset MAE: note onset detection from F0 contours and timing comparison."""

from __future__ import annotations

import numpy as np


def detect_note_onsets(
    f0: np.ndarray,
    sr: int = 24000,
    hop: int = 480,
    semitone_threshold: float = 1.0,
) -> np.ndarray:
    """Detect note onsets from an F0 contour.

    An onset is a frame where either:
      - A voicing transition occurs (previous frame unvoiced, current voiced)
      - A pitch jump exceeds *semitone_threshold* between consecutive voiced frames

    Args:
        f0: (T,) F0 in Hz. 0 = unvoiced.
        sr: Sample rate.
        hop: Hop size in samples.
        semitone_threshold: Minimum pitch jump (in semitones) to trigger an onset.

    Returns:
        1-D float32 array of onset times in seconds (sorted).
    """
    if len(f0) == 0:
        return np.array([], dtype=np.float32)

    voiced = f0 > 0
    onset_frames: list[int] = []

    if voiced[0]:
        onset_frames.append(0)

    for i in range(1, len(f0)):
        if not voiced[i]:
            continue
        if not voiced[i - 1]:
            onset_frames.append(i)
            continue
        semitones = abs(12.0 * np.log2(max(f0[i], 1e-6) / max(f0[i - 1], 1e-6)))
        if semitones > semitone_threshold:
            onset_frames.append(i)

    frame_dur_s = hop / sr
    return np.array(onset_frames, dtype=np.float32) * frame_dur_s


def compute_onset_mae(
    pred_onsets: np.ndarray,
    ref_onsets: np.ndarray,
    max_dist_s: float = 0.200,
) -> float:
    """Greedy nearest-neighbor onset MAE in milliseconds.

    For each predicted onset (in sorted order), match it to the closest
    unmatched reference onset within *max_dist_s*.  Returns the mean
    absolute error of matched pairs, in milliseconds.

    Returns NaN if no matches are found.
    """
    if len(pred_onsets) == 0 or len(ref_onsets) == 0:
        return float("nan")

    ref = np.sort(ref_onsets)
    pred = np.sort(pred_onsets)
    used = set()
    diffs: list[float] = []

    for p in pred:
        distances = np.abs(ref - p)
        for idx in np.argsort(distances):
            if idx in used:
                continue
            if distances[idx] > max_dist_s:
                break
            diffs.append(distances[idx])
            used.add(idx)
            break

    if not diffs:
        return float("nan")
    return float(np.mean(diffs) * 1000.0)


def extract_dali_onset_times(dali_notes: list[dict]) -> np.ndarray:
    """Extract onset times from DALI note timing dicts.

    Args:
        dali_notes: List of dicts with ``start_s`` key, as returned by
            ``eval_utils.dali_ref.extract_dali_note_timings()``.

    Returns:
        1-D float32 array of onset times in seconds (sorted).
    """
    if not dali_notes:
        return np.array([], dtype=np.float32)
    times = np.array([n["start_s"] for n in dali_notes], dtype=np.float32)
    times.sort()
    return times
