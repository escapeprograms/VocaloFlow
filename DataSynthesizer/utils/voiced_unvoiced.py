"""
Voiced/unvoiced extraction utilities for the v2 data synthesis pipeline.

Supplements F0 extraction (target_f0.npy) with frame-level voicing masks,
energy-validated voicing, per-note voicing statistics, and gap detection.

All frame-level defaults (sr=24000, hop=480) match SoulX-Singer config.
"""

import argparse
import os

import numpy as np


# ── Core masks ────────────────────────────────────────────────────────────────

def get_voiced_mask(f0):
    """Extract a frame-level voiced/unvoiced boolean mask from an F0 array.

    Args:
        f0: 1D array of F0 values in Hz (0 = unvoiced), as in target_f0.npy.

    Returns:
        1D boolean array, same length as f0. True = voiced, False = unvoiced.
    """
    return f0 > 0


def get_energy_mask(audio, sr=24000, hop=480, threshold_db=-40.0,
                    frame_length=2048):
    """Compute a frame-level energy mask from audio.

    Frames whose RMS energy (in dB relative to peak) is at or above
    threshold_db are marked True (energetic / non-silent).

    Args:
        audio: 1D audio waveform.
        sr: Sample rate (default 24000).
        hop: Hop size in samples (default 480).
        threshold_db: RMS threshold in dB relative to peak (default -40.0).
        frame_length: RMS analysis window in samples (default 2048).

    Returns:
        1D boolean array. True = above energy threshold.
    """
    import librosa
    rms = librosa.feature.rms(y=audio, frame_length=frame_length,
                              hop_length=hop)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    return rms_db >= threshold_db


def get_validated_voiced_mask(f0, audio, sr=24000, hop=480,
                              threshold_db=-40.0):
    """Voiced mask validated by both F0 and energy.

    A frame is marked voiced only when F0 > 0 AND RMS energy >= threshold.
    The two masks are aligned to the shorter length before combining.

    Args:
        f0: Frame-level F0 in Hz (0 = unvoiced).
        audio: 1D audio waveform.
        sr: Sample rate (default 24000).
        hop: Hop size in samples (default 480).
        threshold_db: Energy silence threshold in dB (default -40.0).

    Returns:
        1D boolean array. True = confidently voiced.
    """
    f0_mask = get_voiced_mask(f0)
    energy_mask = get_energy_mask(audio, sr=sr, hop=hop,
                                 threshold_db=threshold_db)
    min_len = min(len(f0_mask), len(energy_mask))
    return f0_mask[:min_len] & energy_mask[:min_len]


# ── Per-note statistics ───────────────────────────────────────────────────────

def compute_note_voicing_stats(f0, notes, sr=24000, hop=480):
    """Compute per-note voicing statistics from F0 and note boundaries.

    For each note, computes:
      - voiced_ratio: fraction of frames with F0 > 0 (0.0 to 1.0)
      - voiced_frames: count of voiced frames
      - total_frames: total frames spanned by this note
      - onset_frame: first voiced frame index relative to note start, or -1
      - offset_frame: last voiced frame index relative to note start, or -1
      - mean_f0_hz: mean F0 over voiced frames, or 0.0 if fully unvoiced

    Args:
        f0: Frame-level F0 in Hz (0 = unvoiced).
        notes: List of note dicts with 'start_s' and 'note_dur' keys
               (matching extracted_notes.json format).
        sr: Sample rate (default 24000).
        hop: Hop size in samples (default 480).

    Returns:
        List of dicts (one per note) with the statistics above.
    """
    hop_s = hop / sr
    stats = []

    for note in notes:
        start_s = note["start_s"]
        dur_s = note["note_dur"]
        start_frame = int(start_s / hop_s)
        end_frame = int((start_s + dur_s) / hop_s)

        if len(f0) > 0 and start_frame < len(f0):
            f0_slice = f0[start_frame:min(end_frame, len(f0))]
        else:
            f0_slice = np.array([])

        total = len(f0_slice)
        voiced_mask = f0_slice > 0
        voiced_count = int(np.sum(voiced_mask))

        if voiced_count > 0:
            voiced_indices = np.where(voiced_mask)[0]
            onset = int(voiced_indices[0])
            offset = int(voiced_indices[-1])
            mean_hz = float(np.mean(f0_slice[voiced_mask]))
        else:
            onset = -1
            offset = -1
            mean_hz = 0.0

        stats.append({
            "voiced_ratio": round(voiced_count / total, 4) if total > 0 else 0.0,
            "voiced_frames": voiced_count,
            "total_frames": total,
            "onset_frame": onset,
            "offset_frame": offset,
            "mean_f0_hz": round(mean_hz, 2),
        })

    return stats


# ── Gap detection ─────────────────────────────────────────────────────────────

def find_unvoiced_gaps(f0, min_gap_frames=3, margin_frames=3,
                       hop=480, sr=24000):
    """Find contiguous unvoiced gaps in an F0 array, ignoring edges.

    Scans for runs of F0 == 0 in the interior of the array (excluding
    margin_frames from each end). Returns gaps of at least min_gap_frames.

    Args:
        f0: Frame-level F0 in Hz (0 = unvoiced).
        min_gap_frames: Minimum gap length in frames (default 3 = 60ms).
        margin_frames: Frames to ignore at start/end (default 3 = 60ms).
        hop: Hop size in samples (default 480).
        sr: Sample rate (default 24000).

    Returns:
        List of dicts, each with:
          - start_frame, end_frame: frame indices in the original array
          - duration_frames: gap length
          - start_sec, end_sec, duration_sec: times in seconds
    """
    frame_sec = hop / sr
    mask = (f0 == 0)
    n = len(mask)

    if n <= 2 * margin_frames:
        return []

    interior = mask[margin_frames: n - margin_frames].copy()
    padded = np.concatenate([[False], interior, [False]])
    diff = np.diff(padded.astype(np.int8))

    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    gaps = []
    for s, e in zip(starts, ends):
        length = e - s
        if length >= min_gap_frames:
            abs_start = s + margin_frames
            abs_end = e + margin_frames
            gaps.append({
                "start_frame": int(abs_start),
                "end_frame": int(abs_end),
                "duration_frames": int(length),
                "start_sec": round(abs_start * frame_sec, 3),
                "end_sec": round(abs_end * frame_sec, 3),
                "duration_sec": round(length * frame_sec, 3),
            })
    return gaps


# ── Save / load ───────────────────────────────────────────────────────────────

def save_voicing(path, voiced_mask):
    """Save a voiced/unvoiced boolean mask as a .npy file.

    Args:
        path: Output file path (e.g., '<chunk_dir>/target_voicing.npy').
        voiced_mask: 1D boolean array from get_voiced_mask or
                     get_validated_voiced_mask.
    """
    np.save(path, voiced_mask.astype(np.bool_))


def load_voicing(path):
    """Load a voiced/unvoiced boolean mask from a .npy file.

    Args:
        path: Path to a .npy file saved by save_voicing.

    Returns:
        1D boolean array.
    """
    return np.load(path).astype(np.bool_)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Print voiced/unvoiced summary from a target_f0.npy file")
    parser.add_argument("f0_path", help="Path to target_f0.npy")
    parser.add_argument("--save", default=None,
                        help="Save voicing mask to this path")
    args = parser.parse_args()

    f0 = np.load(args.f0_path)
    mask = get_voiced_mask(f0)

    total = len(f0)
    voiced = int(np.sum(mask))
    unvoiced = total - voiced
    frame_sec = 480 / 24000

    print(f"F0 file:        {args.f0_path}")
    print(f"Total frames:   {total} ({total * frame_sec:.2f}s)")
    print(f"Voiced frames:  {voiced} ({voiced * frame_sec:.2f}s, "
          f"{100 * voiced / total:.1f}%)")
    print(f"Unvoiced frames:{unvoiced} ({unvoiced * frame_sec:.2f}s, "
          f"{100 * unvoiced / total:.1f}%)")

    if voiced > 0:
        voiced_f0 = f0[mask]
        print(f"F0 range:       {voiced_f0.min():.1f} - {voiced_f0.max():.1f} Hz")
        print(f"F0 mean:        {voiced_f0.mean():.1f} Hz")

    gaps = find_unvoiced_gaps(f0)
    if gaps:
        print(f"\nUnvoiced gaps (interior, >=60ms): {len(gaps)}")
        for i, g in enumerate(gaps):
            print(f"  Gap {i+1}: {g['start_sec']:.3f}s - {g['end_sec']:.3f}s "
                  f"({g['duration_sec'] * 1000:.0f}ms)")
    else:
        print("\nNo interior unvoiced gaps (>=60ms).")

    if args.save:
        save_voicing(args.save, mask)
        print(f"\nVoicing mask saved to {args.save}")


if __name__ == "__main__":
    main()
