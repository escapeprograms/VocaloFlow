# Batch Memory Palace

This directory contains **subprocess entry points** — scripts that are launched as separate processes (typically in the `soulxsinger` conda environment) to run GPU-intensive model inference. They are never imported directly; instead they are invoked via `subprocess.run()` from pipeline or stage scripts.

## File Details

### `soulxsinger_batch_infer.py`
Subprocess entry point that loads the SoulX-Singer model once and runs inference for a list of chunk tasks.
- Reads a `--tasks_json` file (list of dicts with `target_metadata_path`, `save_dir`, `control`, `save_mel`).
- **Resumable**: skips any chunk whose expected output (`generated_mel.npy` or `generated.wav`) already exists on disk.
- Called by both `stages/synthesizePost.py` (per-song) and `pipelines/synthesize_dataset.py` (dataset batches).
- Must be run under the `soulxsinger` conda environment.

### `note_extraction_batch.py` (v2 only)
Subprocess entry point that loads RMVPE once and extracts note timings/pitches + F0 from target audio.
- Reads a `--tasks_json` file (list of dicts with `chunk_dir`, `audio_path`, `item_name`).
- `build_notes_from_music_json(music_json_path)`: Parses music.json's space-separated fields to build note dicts with timing, text, type, and fallback pitch. Does NOT compute pitch from F0 — that is delegated to `recompute_note_pitches()` from `utils/grab_midi.py`, called afterward.
- After building notes, calls `recompute_note_pitches(notes, f0)` to refine `note_pitch`, `mean_f0_hz`, and `voiced_ratio` from the target F0 curve.
- **Resumable**: skips chunks where both `extracted_notes.json` and `target_f0.npy` exist.
- Must be run under the `soulxsinger` conda environment.