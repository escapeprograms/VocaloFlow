"""
Generate a frame-level phoneme identity mask (mel2note) for training.

Replicates the tokenisation and frame-assignment logic from
SoulX-Singer's DataProcessor.preprocess(), but uses:
  - adjusted_notes.json  (DTW-corrected per-note durations/timestamps)
  - music.json           (phoneme strings from g2p, matched by note index)

Produces two files per chunk:
  phoneme_ids.npy   – int32, shape (P,): the expanded phoneme-token ID sequence
                      (with <PAD>, <BOW>, <EOW> markers, matching preprocess())
  phoneme_mask.npy  – int32, shape (T,): per-mel-frame index into phoneme_ids

Usage (standalone, for reprocessing existing data):
    python utils/phoneme_mask.py --chunk_dir ../Data/<dali_id>/<chunk>
"""

import argparse
import json
import os
import sys

import numpy as np

# Default phone_set.json location (relative to this file → SoulX-Singer)
_DEFAULT_PHONESET = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "SoulX-Singer",
    "soulxsinger", "utils", "phoneme", "phone_set.json",
))

SR = 24000
HOP = 480


def _load_phone2idx(phoneset_path: str) -> dict:
    with open(phoneset_path, "r", encoding="utf-8") as f:
        phoneset = json.load(f)
    return {ph: idx for idx, ph in enumerate(phoneset)}


def _extract_phonemes_from_music_json(music_json_path: str):
    """Extract flat per-note phoneme strings from music.json.

    Iterates segments in the same order as
    note_extraction_batch.build_notes_from_music_json so that indices
    align 1:1 with adjusted_notes.
    """
    with open(music_json_path, "r", encoding="utf-8") as f:
        meta_list = json.load(f)

    phonemes = []
    for segment in meta_list:
        phs = segment["phoneme"].split()
        types = [int(t) for t in segment["note_type"].split()]
        n = min(len(phs), len(types))
        for i in range(n):
            phonemes.append(phs[i])
    return phonemes


def _build_mel2note(
    note_durations: list[float],
    phonemes: list[str],
    phone2idx: dict,
    sr: int = SR,
    hop: int = HOP,
):
    """Replicate DataProcessor.preprocess() mel2note construction.

    Returns:
        phoneme_ids: 1-D int32 array of expanded phoneme token IDs (length P).
        mel2note:    1-D int32 array of per-frame indices into phoneme_ids (length T).
    """
    total_frames = int(np.round(sum(note_durations) * sr / hop))
    if total_frames <= 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)

    mel2note = np.zeros(total_frames, dtype=np.int32)

    # --- Phase 1: build expanded phoneme list + per-note frame locations ---
    ph_locations = []  # (start_frame, n_inner_tokens) per note
    new_phonemes = []
    dur_sum = 0.0

    for note_idx in range(len(phonemes)):
        frame_start = int(np.round(dur_sum * sr / hop))
        frame_start = min(frame_start, total_frames - 1)

        new_phonemes.append("<BOW>")

        ph = phonemes[note_idx]
        if ph.startswith("en_"):
            # English: split on '-' → individual phonemes + <SEP>
            en_phs = ["en_" + x for x in ph[3:].split("-")] + ["<SEP>"]
            ph_locations.append((frame_start, max(1, len(en_phs))))
            new_phonemes.extend(en_phs)
        else:
            ph_locations.append((frame_start, 1))
            new_phonemes.append(ph)

        new_phonemes.append("<EOW>")
        dur_sum += note_durations[note_idx]

    # --- Phase 2: fill mel2note (indices into new_phonemes + 1 for <PAD>) ---
    ph_idx = 1  # 1-based because new_phonemes is prepended with <PAD>
    for loc_idx, (i, j) in enumerate(ph_locations):
        next_start = (
            ph_locations[loc_idx + 1][0]
            if loc_idx < len(ph_locations) - 1
            else total_frames
        )
        if i >= total_frames or i + j > total_frames:
            break
        # Handle overlap with previous note
        while i < total_frames and mel2note[i] > 0:
            i += 1
        if i >= total_frames:
            break
        mel2note[i] = ph_idx
        k = i + 1
        while k + j < next_start:
            mel2note[k: k + j] = np.arange(ph_idx, ph_idx + j) + 1
            k += j
        if next_start - 1 >= 0 and next_start - 1 < total_frames:
            mel2note[next_start - 1] = ph_idx + j + 1
        ph_idx += j + 2  # <BOW> + j inner tokens + <EOW>

    # Prepend <PAD> to match preprocess() convention
    new_phonemes = ["<PAD>"] + new_phonemes

    # Map to integer IDs (unknown tokens → <UNK>)
    unk_idx = phone2idx.get("<UNK>", 3)
    phoneme_ids = np.array(
        [phone2idx.get(ph, unk_idx) for ph in new_phonemes], dtype=np.int32
    )

    return phoneme_ids, mel2note


def generate_phoneme_mask(
    chunk_dir: str,
    phoneset_path: str = _DEFAULT_PHONESET,
    sr: int = SR,
    hop: int = HOP,
) -> bool:
    """Generate and save phoneme_ids.npy + phoneme_mask.npy for a chunk.

    Reads adjusted_notes.json (DTW-corrected durations) and music.json
    (phoneme strings).  Falls back to extracted_notes.json if adjusted
    notes don't exist yet.

    Returns True on success.
    """
    # Load notes (prefer adjusted, fall back to extracted)
    adjusted_path = os.path.join(chunk_dir, "adjusted_notes.json")
    extracted_path = os.path.join(chunk_dir, "extracted_notes.json")
    notes_path = adjusted_path if os.path.exists(adjusted_path) else extracted_path
    if not os.path.exists(notes_path):
        print(f"  [phoneme_mask] No notes file in {chunk_dir}, skipping.")
        return False

    music_json_path = os.path.join(chunk_dir, "music.json")
    if not os.path.exists(music_json_path):
        print(f"  [phoneme_mask] No music.json in {chunk_dir}, skipping.")
        return False

    with open(notes_path, "r", encoding="utf-8") as f:
        notes_data = json.load(f)
    notes = notes_data.get("notes", [])
    if not notes:
        return False

    # Extract per-note phoneme strings from music.json (same order as notes)
    phoneme_strings = _extract_phonemes_from_music_json(music_json_path)

    # Lengths should match; truncate to shorter if they don't
    n = min(len(notes), len(phoneme_strings))
    if len(notes) != len(phoneme_strings):
        print(f"  [phoneme_mask] WARNING: note count mismatch in {os.path.basename(chunk_dir)}: "
              f"{len(notes)} notes vs {len(phoneme_strings)} phonemes in music.json. "
              f"Truncating to {n}.")
    note_durations = [note["note_dur"] for note in notes[:n]]
    phonemes = phoneme_strings[:n]

    phone2idx = _load_phone2idx(phoneset_path)
    phoneme_ids, mel2note = _build_mel2note(note_durations, phonemes, phone2idx, sr, hop)

    if len(mel2note) == 0:
        print(f"  [phoneme_mask] Zero-length mel2note for {chunk_dir}, skipping.")
        return False

    np.save(os.path.join(chunk_dir, "phoneme_ids.npy"), phoneme_ids)
    np.save(os.path.join(chunk_dir, "phoneme_mask.npy"), mel2note)
    print(f"  [phoneme_mask] Saved phoneme_ids ({phoneme_ids.shape}) "
          f"+ phoneme_mask ({mel2note.shape}) in {os.path.basename(chunk_dir)}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate phoneme identity mask for a single chunk."
    )
    parser.add_argument("--chunk_dir", required=True)
    parser.add_argument("--phoneset_path", default=_DEFAULT_PHONESET)
    args = parser.parse_args()
    generate_phoneme_mask(args.chunk_dir, args.phoneset_path)
