# Alignment Memory Palace

This directory contains **alignment-related code** — specifically the iterative alignment approach used to reconcile timing differences between prior (OpenUtau) and target (SoulX-Singer) audio.

## Overview

Alignment reconciles timing differences between prior and target audio. Rather than warping audio after the fact, the approach modifies *source note durations* so OpenUtau naturally renders audio whose timing matches SoulX-Singer's output.

| File | Approach | Description |
|------|----------|-------------|
| `iterative_align.py` | Iterative note adjustment | Adjusts source note durations instead of warping audio |

## File Details

### `iterative_align.py`
Iterative alignment: repeatedly adjusts UTAU note durations until prior timing converges to match target timing. Used by both single-song (`synthesize_v2.py`) and dataset (`synthesize_dataset_v2.py`) pipelines.
- Uses DTW (not MFA) to measure timing discrepancies — directly compares mel-spectrograms frame-by-frame.
- Algorithm: generate prior → compute DTW warp path → measure per-note time scale ratios → adjust durations → repeat until convergence.
- **Speed optimizations**:
  - Target mel is pre-computed once before the iteration loop and cached across iterations via `cached_target_mel` parameter.
  - Iteration 1 uses the full `generate_prior_from_notes()` path (`resetParts` + `player.export()` with full validation).
  - Iterations 2+ use `rerender_prior_with_adjusted_durations()` which calls `clearNotes()` + `player.exportFast()` (lightweight validation, skips manual phonemizer setup). Both wav and ustx are produced on every iteration.
- **Best-notes tracking**: Across all iterations, tracks the note set with the lowest `max_deviation`. If max iterations are reached without convergence, regenerates the prior from best notes using the full `generate_prior_from_notes()` path (not the fast path).
- `_save_prior_mel` is called on both convergence and non-convergence exits, ensuring `prior_mel.npy` is always produced.
- Temp file `iter_notes.json` is written during alignment and cleaned up on both exit paths.
- `iterative_align(chunk_dir, notes, target_audio_path, lyrics_text, player, ...)`: Main entry point. Returns `(adjusted_notes, metrics_dict)`.
  - `lyrics_text` is unused by DTW, kept for API compatibility.
  - `adjusted_notes` is a list of note dicts (same schema as `extracted_notes.json` notes) with `start_s` and `note_dur` refined through iterative DTW. Each iteration: render prior via OpenUtau → DTW against target mel → compute per-note time-scale ratios → apply damped correction (clamped [0.3, 3.0]). Converges when all ratios within `duration_threshold` or `max_iterations` hit.
  - `metrics_dict` contains final alignment quality info (iteration count, max deviation, convergence flag).
  - The caller (pipeline) writes the output as `adjusted_notes.json` with structure `{"notes": [...], "source": "iterative"}`.
- `_compute_dtw_warp_path(prior_audio_path, target_audio_path, sr, cached_target_mel=None)`: Computes DTW warp path between prior and target mel-spectrograms (cosine metric). Returns `(prior_mel, warp_path, dtw_cost, hop_length)`. Note: `prior_mel` is returned but currently unused by the caller. Accepts optional pre-computed target mel to avoid redundant reloading.
- `_compute_per_note_ratios(notes, warp_path, hop_length, sr)`: Maps DTW warp-path entries to per-note frame ranges. For each note, finds warp-path entries whose prior index falls in that note's frame span, then computes ratio = target_span / prior_span. Returns list of dicts with `note_index`, `ratio`, `prior_frames`, `target_frames`.
- `_adjust_note_durations(notes, per_note_ratios, damping)`: Applies damped duration scaling per note — `ratio = 1 + damping * (raw_ratio - 1)`, clamped to [0.3, 3.0]. Recomputes `start_s` by accumulating durations while preserving original inter-note gaps.
- **Pitch recalculation**: After each duration adjustment, calls `recompute_note_pitches(current_notes, target_f0)` from `utils.grab_midi` to update `note_pitch` based on the target F0 within the new time windows. This ensures the prior sings the correct pitch for each note's adjusted position. `target_f0.npy` is loaded once before the iteration loop (like `cached_target_mel`). Gracefully skipped if F0 file is absent.
- `_save_prior_mel(prior_path, chunk_dir, sr)`: Extracts mel from final prior.wav and saves as `prior_mel.npy` in **(T, 128)** shape (time-first, matching target_mel convention).
- Imports: `utils.grab_midi` (`recompute_note_pitches`), `utils.vocoders` (`mel_to_soulx_mel`, `SOULX_MEL_CONFIG`), `stages.synthesizePrior` (lazy — both `generate_prior_from_notes` and `rerender_prior_with_adjusted_durations`).
