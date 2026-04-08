import math

import numpy as np


def freq_to_midi(freq):
    """Convert frequency in Hz to closest MIDI note number (0-127)."""
    if freq <= 0:
        return 0
    return int(round(69 + 12 * math.log2(freq / 440.0)))

def get_midi_pitch(freq_list):
    """Get the average MIDI pitch from a list of frequencies."""
    if isinstance(freq_list, (list, tuple)) and len(freq_list) > 0:
        avg_freq = sum(freq_list) / len(freq_list)
        return freq_to_midi(avg_freq)
    else:
        return 60 # Default middle C if parsing fails


def recompute_note_pitches(notes, f0, sr=24000, hop=480):
    """Recompute note_pitch for each note from target F0 within its time window.

    For each note, extracts the F0 slice covering [start_s, start_s + note_dur),
    computes the mean over voiced frames (F0 > 0), and converts to MIDI.
    Also updates mean_f0_hz and voiced_ratio metadata.

    Falls back to existing note_pitch if no voiced frames are found.
    Mutates notes in-place.

    Args:
        notes: List of note dicts with 'start_s' and 'note_dur' keys.
        f0: 1D array of frame-level F0 in Hz (0 = unvoiced).
        sr: Sample rate (default 24000).
        hop: Hop size in samples (default 480).
    """
    hop_s = hop / sr
    for note in notes:
        start_frame = int(note["start_s"] / hop_s)
        end_frame = int((note["start_s"] + note["note_dur"]) / hop_s)

        if len(f0) > 0 and start_frame < len(f0):
            f0_slice = f0[start_frame:min(end_frame, len(f0))]
            voiced = f0_slice[f0_slice > 0]
            if len(voiced) > 0:
                avg_hz = float(np.mean(voiced))
                note["note_pitch"] = freq_to_midi(avg_hz)
                note["mean_f0_hz"] = round(avg_hz, 2)
                note["voiced_ratio"] = round(len(voiced) / max(len(f0_slice), 1), 4)
                continue

        # Fallback: keep existing note_pitch, ensure metadata fields exist
        note.setdefault("mean_f0_hz", 0.0)
        note.setdefault("voiced_ratio", 0.0)
