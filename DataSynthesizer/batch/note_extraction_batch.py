"""
Batch note extraction using music.json + RMVPE F0.

Derives extracted_notes.json from:
  - music.json: note count, text, type, and initial timing (from DALI)
  - target_f0.npy: actual sung pitch from RMVPE (frame-level F0)

ROSVOT is no longer used — music.json provides reliable note structure
(it's what SoulX-Singer was given) and iterative_align.py corrects any
timing deviations afterward.

Usage:
    python note_extraction_batch.py \
        --tasks_json /path/to/tasks.json \
        --rmvpe_model_path .../rmvpe/rmvpe.pt \
        [--device cuda]

tasks.json is a list of dicts, each with keys:
    chunk_dir   – output directory for this chunk (must contain music.json)
    audio_path  – path to target.wav
    item_name   – chunk identifier (e.g. "line_0")
"""

import argparse
import json
import os
import sys
import traceback

import numpy as np

SOULX_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "SoulX-Singer"))
DATASYNTHESIZER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SOULX_DIR)
sys.path.insert(0, DATASYNTHESIZER_DIR)

from preprocess.tools import F0Extractor
from utils.grab_midi import recompute_note_pitches
from utils.voiced_unvoiced import get_voiced_mask, save_voicing


def build_notes_from_music_json(music_json_path):
    """Build extracted_notes from music.json structure.

    Parses music.json's space-separated fields to get per-note duration, text,
    type, and fallback pitch.  Pitch is set to the fallback value from
    music.json; call recompute_note_pitches() afterward to refine from F0.

    Args:
        music_json_path: Path to the SoulX-Singer music.json metadata.

    Returns:
        List of note dicts in extracted_notes format.
    """
    with open(music_json_path, "r", encoding="utf-8") as f:
        meta_list = json.load(f)

    if not meta_list:
        return []

    notes = []
    for segment in meta_list:
        durations = [float(d) for d in segment["duration"].split()]
        texts = segment["text"].split()
        types = [int(t) for t in segment["note_type"].split()]
        fallback_pitches = [int(p) for p in segment["note_pitch"].split()]
        seg_start_s = segment["time"][0] / 1000.0  # ms → s

        n = min(len(durations), len(texts), len(types))

        t = 0.0  # accumulated time within segment
        for i in range(n):
            abs_start = seg_start_s + t
            dur = durations[i]
            note_text = texts[i]
            note_type = types[i]
            fallback_pitch = fallback_pitches[i] if i < len(fallback_pitches) else 60

            # For continuation notes, use "-" (matching synthesizePrior convention)
            if note_type == 3:
                note_text = "-"

            notes.append({
                "start_s": round(abs_start, 6),
                "note_dur": round(dur, 6),
                "note_text": note_text,
                "note_pitch": fallback_pitch,
                "note_type": note_type,
            })

            t += dur

    return notes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks_json", required=True, help="JSON file listing extraction tasks")
    parser.add_argument("--rmvpe_model_path", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(args.tasks_json, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    if not tasks:
        print("No tasks provided.")
        return

    print(f"Loading F0 model (once) for {len(tasks)} chunk(s)...")
    f0_extractor = F0Extractor(
        model_path=args.rmvpe_model_path,
        device=args.device,
    )

    for i, task in enumerate(tasks):
        chunk_dir = task["chunk_dir"]
        audio_path = task["audio_path"]
        item_name = task["item_name"]

        # Resumability: skip chunks that already completed
        notes_path = os.path.join(chunk_dir, "extracted_notes.json")
        f0_path = os.path.join(chunk_dir, "target_f0.npy")
        if os.path.exists(notes_path) and os.path.exists(f0_path):
            print(f"\n[{i+1}/{len(tasks)}] Skipping {item_name} (already done).")
            continue

        if not os.path.exists(audio_path):
            print(f"\n[{i+1}/{len(tasks)}] Skipping {item_name} (no audio at {audio_path}).")
            continue

        music_json_path = os.path.join(chunk_dir, "music.json")
        if not os.path.exists(music_json_path):
            print(f"\n[{i+1}/{len(tasks)}] Skipping {item_name} (no music.json).")
            continue

        print(f"\n[{i+1}/{len(tasks)}] Extracting notes + F0 for {item_name}...")

        try:
            # Extract F0 first (needed by build_notes_from_music_json)
            f0_extractor.process(audio_path, f0_path=f0_path, verbose=False)
            print(f"  F0 saved to {f0_path}")

            # Save voiced/unvoiced mask alongside F0
            f0_data = np.load(f0_path)
            voicing_path = os.path.join(chunk_dir, "target_voicing.npy")
            save_voicing(voicing_path, get_voiced_mask(f0_data))
            print(f"  Voicing mask saved to {voicing_path}")
        except Exception as e:
            print(f"  F0 extraction failed for {item_name}: {e}")
            traceback.print_exc()
            continue

        try:
            # Build notes from music.json structure, then refine pitch from F0
            extracted_notes = build_notes_from_music_json(music_json_path)
            f0_data = np.load(f0_path)
            recompute_note_pitches(extracted_notes, f0_data)

            with open(notes_path, "w", encoding="utf-8") as f:
                json.dump({"notes": extracted_notes, "source": "music_json+f0"}, f, indent=2)

            print(f"  Notes: {len(extracted_notes)} extracted.")
        except Exception as e:
            print(f"  Note extraction failed for {item_name}: {e}")
            traceback.print_exc()

    print(f"\nBatch extraction complete ({len(tasks)} chunk(s)).")


if __name__ == "__main__":
    main()
