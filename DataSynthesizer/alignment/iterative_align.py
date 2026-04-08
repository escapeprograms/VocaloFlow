"""
Iterative alignment: repeatedly adjust UTAU note durations until the prior
timing converges to match the target timing.

Instead of warping audio after the fact, this module modifies the *source*
(note durations) so that OpenUtau naturally renders audio whose timing
matches SoulX-Singer's output.

Uses DTW (not MFA) to measure timing discrepancies — DTW directly compares
the mel-spectrograms frame-by-frame, giving a warp path that reveals exactly
where the prior runs ahead of or behind the target.

Algorithm:
    1. Generate prior.wav from current notes via OpenUtau.
    2. Compute mel-spectrograms for prior and target audio.
    3. Run DTW to get the warp path mapping prior frames → target frames.
    4. For each note, measure how many target frames its prior frames map to.
       The ratio (mapped_target_frames / prior_frames) is the local time scale.
    5. Adjust each note's duration by that ratio (with damping).
    6. Regenerate and re-evaluate until the DTW cost stabilises or all
       per-note ratios are within threshold.
"""

import copy
import json
import os
import sys
import time as _time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add DataSynthesizer root to path for cross-package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import librosa
import numpy as np
from utils.grab_midi import recompute_note_pitches
from utils.vocoders import mel_to_soulx_mel, SOULX_MEL_CONFIG


# ---------------------------------------------------------------------------
# DTW analysis
# ---------------------------------------------------------------------------

def _compute_dtw_warp_path(
    prior_audio_path: str,
    target_audio_path: str,
    sr: int = 24000,
    cached_target_mel: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """Compute DTW between prior and target mel-spectrograms.

    Args:
        cached_target_mel: Pre-computed target mel (n_mels, T_target), already
            sanitised.  When provided, ``target_audio_path`` is not loaded,
            saving ~0.1-0.2 s per call.

    Returns:
        prior_mel: (n_mels, T_prior)
        warp_path: (N, 2) array of (prior_frame, target_frame) pairs, chronological.
        dtw_cost: Total DTW cost.
        hop_length: Hop length used for frame↔sample conversion.
    """
    y_prior, _ = librosa.load(prior_audio_path, sr=sr)
    prior_mel = mel_to_soulx_mel(y_prior, sr=sr)
    prior_mel = np.nan_to_num(prior_mel, nan=0.0) + 1e-8

    if cached_target_mel is not None:
        target_mel = cached_target_mel
    else:
        y_target, _ = librosa.load(target_audio_path, sr=sr)
        target_mel = mel_to_soulx_mel(y_target, sr=sr)
        target_mel = np.nan_to_num(target_mel, nan=0.0) + 1e-8

    D, wp = librosa.sequence.dtw(X=prior_mel, Y=target_mel, metric="cosine")
    cost = float(D[-1, -1])
    wp = wp[::-1]  # chronological

    hop_length = SOULX_MEL_CONFIG["hop_length"]
    return prior_mel, wp, cost, hop_length


def _compute_per_note_ratios(
    notes: List[Dict],
    warp_path: np.ndarray,
    hop_length: int,
    sr: int = 24000,
) -> List[Dict]:
    """For each note, compute the local time-scale ratio from the DTW warp path.

    For a note spanning prior frames [f_start, f_end), find all warp-path entries
    whose prior index falls in that range.  The corresponding target indices span
    some range [p_min, p_max).  The ratio = target_span / prior_span tells us
    whether the prior is too fast (ratio > 1) or too slow (ratio < 1) for that note.

    Returns a list of dicts with note_index, ratio, prior_frames, target_frames.
    """
    prior_indices = warp_path[:, 0]
    target_indices = warp_path[:, 1]

    results = []
    for i, note in enumerate(notes):
        # Convert note timing to frames
        start_frame = int(note["start_s"] * sr / hop_length)
        end_frame = int((note["start_s"] + note["note_dur"]) * sr / hop_length)
        prior_span = max(end_frame - start_frame, 1)

        # Find warp-path entries in this frame range
        mask = (prior_indices >= start_frame) & (prior_indices < end_frame)
        if not np.any(mask):
            results.append({
                "note_index": i,
                "ratio": 1.0,
                "prior_frames": prior_span,
                "target_frames": prior_span,
            })
            continue

        matched_target = target_indices[mask]
        target_span = max(int(matched_target.max()) - int(matched_target.min()) + 1, 1)

        ratio = target_span / prior_span
        results.append({
            "note_index": i,
            "ratio": round(ratio, 4),
            "prior_frames": int(prior_span),
            "target_frames": int(target_span),
        })

    return results


def _save_prior_mel(prior_path: str, chunk_dir: str, sr: int = 24000) -> None:
    """Extract and save the mel-spectrogram of the final prior.wav."""
    mel_path = os.path.join(chunk_dir, "prior_mel.npy")
    y_prior, _ = librosa.load(prior_path, sr=sr)
    prior_mel = mel_to_soulx_mel(y_prior, sr=sr)  # (128, T)
    prior_mel = prior_mel.T  # (T, 128) to match target_mel convention
    np.save(mel_path, prior_mel)
    print(f"    Prior mel saved to {mel_path} (shape {prior_mel.shape})")


# ---------------------------------------------------------------------------
# Note adjustment
# ---------------------------------------------------------------------------

def _adjust_note_durations(
    notes: List[Dict],
    per_note_ratios: List[Dict],
    damping: float = 0.5,
) -> List[Dict]:
    """Scale each note's duration by its DTW-derived ratio (with damping).

    Args:
        damping: Fraction of correction to apply (0..1).  Lower values
                 converge more slowly but avoid oscillation.
    """
    adjusted = copy.deepcopy(notes)

    for info in per_note_ratios:
        i = info["note_index"]
        raw_ratio = info["ratio"]
        # Damped: move only a fraction toward the target
        ratio = 1.0 + damping * (raw_ratio - 1.0)
        # Clamp to prevent extreme adjustments
        ratio = max(0.3, min(3.0, ratio))
        adjusted[i]["note_dur"] *= ratio

    # Recompute start_s by accumulating durations, preserving original gaps
    if adjusted:
        t = adjusted[0]["start_s"]
        for i, note in enumerate(adjusted):
            note["start_s"] = round(t, 6)
            note["note_dur"] = round(note["note_dur"], 6)
            if i + 1 < len(adjusted):
                original_gap = notes[i + 1]["start_s"] - (notes[i]["start_s"] + notes[i]["note_dur"])
                t += note["note_dur"] + max(0.0, original_gap)
            else:
                t += note["note_dur"]

    return adjusted


# ---------------------------------------------------------------------------
# Main iterative alignment function
# ---------------------------------------------------------------------------

def iterative_align(
    chunk_dir: str,
    notes: List[Dict],
    target_audio_path: str,
    lyrics_text: str,
    player,
    *,
    use_phonemes: bool = True,
    max_iterations: int = 5,
    duration_threshold: float = 0.15,
    damping: float = 0.5,
    sr: int = 24000,
) -> Tuple[List[Dict], Dict]:
    """Iteratively adjust note durations until prior timing matches target.

    Uses DTW to compare mel-spectrograms frame-by-frame and derive per-note
    time-scale ratios.  Each iteration adjusts note durations (with damping),
    regenerates the prior via OpenUtau, and re-evaluates.

    Args:
        chunk_dir:           Working directory for this chunk.
        notes:               Initial extracted_notes list (from ROSVOT).
        target_audio_path:     Path to target.wav (SoulX-Singer output).
        lyrics_text:         Full text (unused by DTW, kept for API compat).
        player:              Pre-initialised OpenUtau Player instance.
        use_phonemes:        Use g2p_en ARPAbet phoneticHints.
        max_iterations:      Maximum alignment iterations.
        duration_threshold:  Per-note ratio tolerance (e.g. 0.15 = 15%).
        damping:             Fraction of correction applied per iteration.
        sr:                  Sample rate for mel extraction.

    Returns:
        (adjusted_notes, metrics_dict)
        adjusted_notes:  Note list with corrected durations/start times.
        metrics_dict:    Final alignment metrics.
    """
    from stages.synthesizePrior import generate_prior_from_notes, rerender_prior_with_adjusted_durations

    ts = lambda: datetime.now().strftime("%H:%M:%S")

    current_notes = copy.deepcopy(notes)
    best_notes = current_notes
    best_max_deviation = float("inf")
    best_cost = float("inf")

    # Pre-compute target mel once (it never changes across iterations)
    y_target, _ = librosa.load(target_audio_path, sr=sr)
    cached_target_mel = mel_to_soulx_mel(y_target, sr=sr)
    cached_target_mel = np.nan_to_num(cached_target_mel, nan=0.0) + 1e-8

    # Pre-load target F0 for pitch recalculation after duration adjustments
    f0_path = os.path.join(chunk_dir, "target_f0.npy")
    target_f0 = np.load(f0_path) if os.path.exists(f0_path) else None

    for iteration in range(max_iterations):
        print(f"  [{ts()}] Iteration {iteration + 1}/{max_iterations}")

        # 1. Generate prior from current notes
        prior_path = os.path.join(chunk_dir, "prior.wav")

        if iteration == 0:
            # First iteration: full path (phonemizer setup + note creation)
            iter_notes_path = os.path.join(chunk_dir, "iter_notes.json")
            with open(iter_notes_path, "w", encoding="utf-8") as f:
                json.dump({"notes": current_notes, "source": "iterative"}, f, indent=2)

            for cleanup in [prior_path, os.path.join(chunk_dir, "prior.ustx")]:
                if os.path.exists(cleanup):
                    os.remove(cleanup)

            ok = generate_prior_from_notes(chunk_dir, iter_notes_path, player,
                                            use_phonemes=use_phonemes)
        else:
            # Iterations 2+: fast path (update durations only, skip phonemizer)
            ok = rerender_prior_with_adjusted_durations(chunk_dir, current_notes, player)

        if not ok:
            print(f"    Prior generation failed.")
            break

        # 2. DTW between prior and target
        prior_mel, warp_path, dtw_cost, hop_length = _compute_dtw_warp_path(
            prior_path, target_audio_path, sr=sr,
            cached_target_mel=cached_target_mel,
        )

        # 3. Compute per-note ratios from warp path
        per_note = _compute_per_note_ratios(current_notes, warp_path, hop_length, sr=sr)

        # Evaluate: deviation = |ratio - 1|
        deviations = [abs(p["ratio"] - 1.0) for p in per_note]
        max_dev = max(deviations) if deviations else 0.0
        mean_dev = sum(deviations) / len(deviations) if deviations else 0.0
        n_over = sum(1 for d in deviations if d > duration_threshold)

        print(f"    DTW cost: {dtw_cost:.2f}, "
              f"mean note deviation: {mean_dev:.1%}, max: {max_dev:.1%}, "
              f"over threshold: {n_over}/{len(deviations)}")

        # Track best
        if max_dev < best_max_deviation:
            best_max_deviation = max_dev
            best_cost = dtw_cost
            best_notes = copy.deepcopy(current_notes)

        # 4. Check convergence
        if max_dev <= duration_threshold:
            print(f"    Converged! All notes within {duration_threshold:.0%} tolerance.")
            if os.path.exists(iter_notes_path):
                os.remove(iter_notes_path)
            _save_prior_mel(prior_path, chunk_dir, sr)
            return current_notes, {
                "converged": True,
                "iterations": iteration + 1,
                "dtw_cost": round(dtw_cost, 4),
                "mean_deviation": round(mean_dev, 4),
                "max_deviation": round(max_dev, 4),
                "per_note": [{"note": notes[p["note_index"]]["note_text"],
                              "ratio": p["ratio"],
                              "deviation": round(abs(p["ratio"] - 1.0), 4)}
                             for p in per_note],
            }

        # 5. Adjust note durations for next iteration
        current_notes = _adjust_note_durations(current_notes, per_note, damping=damping)

        # 6. Recalculate pitches for the new time windows
        if target_f0 is not None:
            recompute_note_pitches(current_notes, target_f0, sr=sr)

    # Did not fully converge — regenerate from best notes
    print(f"    Max iterations reached. Best max deviation: {best_max_deviation:.1%}, "
          f"best DTW cost: {best_cost:.2f}")

    # Regenerate prior from best notes
    iter_notes_path = os.path.join(chunk_dir, "iter_notes.json")
    with open(iter_notes_path, "w", encoding="utf-8") as f:
        json.dump({"notes": best_notes, "source": "iterative"}, f, indent=2)

    prior_path = os.path.join(chunk_dir, "prior.wav")
    for cleanup in [prior_path, os.path.join(chunk_dir, "prior.ustx")]:
        if os.path.exists(cleanup):
            os.remove(cleanup)
    generate_prior_from_notes(chunk_dir, iter_notes_path, player, use_phonemes=use_phonemes)

    if os.path.exists(iter_notes_path):
        os.remove(iter_notes_path)

    _save_prior_mel(prior_path, chunk_dir, sr)

    return best_notes, {
        "converged": False,
        "iterations": max_iterations,
        "dtw_cost": round(best_cost, 4),
        "best_max_deviation": round(best_max_deviation, 4),
    }
