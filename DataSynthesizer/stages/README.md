# Stages Memory Palace

This directory contains the **core pipeline stages** — the major processing steps called by the pipeline entry points in `pipelines/`.

## Overview

Each file represents one major step in the synthesis pipeline:
1. **Target synthesis** (`synthesizeTarget.py`): Generate vocals via SoulX-Singer.
2. **Prior synthesis** (`synthesizePrior.py`): Generate vocals via OpenUtau API.

## File Details

### `synthesizeTarget.py`
Generates vocal synthesis using the SoulX-Singer model.
- `get_soulx_inference_config(provider=None)`: Module-level function returning the dict of model/config/prompt paths for the SoulX-Singer subprocess. Looks up voice prompt paths from `voice_providers.VOICE_PROVIDERS` by provider name (defaults to `DEFAULT_PROVIDER`). Single source of truth used by both `process_dali_to_target` and `pipelines/synthesize_dataset_v2.py`.
- `process_dali_to_target(...)`: Parses DALI annotations into `music.json` per chunk, optionally injects F0 curves.
  - `defer_inference=False` (default): immediately launches `batch/soulxsinger_batch_infer.py` subprocess (per-song use).
  - `defer_inference=True`: skips the subprocess and **returns** the list of inference task dicts for the caller to batch across songs.
- Supports `save_mel=True` to also export `target_mel.npy` alongside `target.wav`.
- Imports: `utils.grab_midi`, `utils.determine_chunks`, `utils.prompt_selection`, `voice_providers`.

### `synthesizePrior.py`
Generates vocal synthesis using the OpenUtau API.
- `generate_prior_from_notes(chunk_dir, extracted_notes_path, player, use_phonemes)`: Reads `extracted_notes.json` (produced by `batch/note_extraction_batch.py` from music.json + RMVPE F0), groups notes by word (note_type 2 = word start, 3 = continuation), applies g2p phoneme distribution (`_distribute_phonemes`), converts to OpenUtau ticks, exports `prior.wav` + `prior.ustx` via `player.export()` (single validate call for both outputs).
- `rerender_prior_with_adjusted_durations(chunk_dir, notes, player)`: Fast re-render path for iterative alignment iterations 2+. Uses `player.clearNotes()` to preserve the part object (and phonemizer cache), re-adds notes with updated durations, then renders via `player.exportFast()` which uses lightweight validation (skips manual phonemizer setup). Produces both `prior.wav` and `prior.ustx`.
- Uses `pythonnet` (`clr`) to interface with `UtauGenerate.dll` (a C# wrapper for OpenUtau).
- **Optimized**: Reuses a single `Player` instance. Sleep 0.2s as race-condition buffer for OpenUtau C# thread.
- G2P helpers: `_normalize_arpabet`, `_distribute_phonemes` (coda-dominant syllabification), `_build_text_fallback`.
- Note: `utils.grab_midi.get_midi_pitch` is imported but currently unused (dead import).
