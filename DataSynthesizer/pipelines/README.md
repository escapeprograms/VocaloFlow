# Pipelines Memory Palace

This directory contains the **entry-point scripts** for the DataSynthesizer module. These are the files you actually run to kick off synthesis.

## Overview

| Script | Scale | Description |
|--------|-------|-------------|
| `synthesize_v2.py` | Single song | Target-first: generates target first, extracts notes via ROSVOT, iteratively aligns prior |
| `synthesize_dataset_v2.py` | Dataset | Five-phase pipeline: target-first with iterative alignment + manifest generation |

## File Details

### `synthesize_v2.py`
Single-song driver — runs the target-first pipeline for one DALI entry.
- **Four-phase pipeline** (all in one script, subprocesses for GPU phases):
  - **Phase 1**: Generate `music.json` + `chunk_words.json` from DALI annotations.
  - **Phase 2**: SoulX-Singer inference (subprocess in `soulxsinger` env, launched via `SOULX_PYTHON`).
  - **Phase 3**: `music.json` note extraction + F0 (subprocess via `batch/note_extraction_batch.py`).
  - **Phase 4**: Prior generation from extracted notes via OpenUtau + iterative alignment until DTW convergence.
- Helper functions: `extract_chunk_words`, `save_chunk_words` (also used by `synthesize_dataset_v2.py`).
- CLI args: `--dali_id`, `--mode`, `--use_phonemes`, `--use_continuations`, `--use_f0`.

### `synthesize_dataset_v2.py`
Dataset-scale driver for the target-first pipeline.
- **Batch-oriented architecture**: Songs are split into batches of `songs_per_batch` (default 100). Each batch completes Phases 1–4 fully before the next batch starts, so you get complete entries early without waiting for the entire dataset to finish inference.
- **Five-phase pipeline** amortising all expensive model loads:
  - **Phase 1 — Target Metadata**: Generate `music.json` + `chunk_words.json` from DALI. Collect inference tasks. Merges into cumulative `pending_inference_tasks.json` cache.
  - **Phase 2 — Inference**: SoulX-Singer inference for this batch via `_launch_inference_subprocess`. If Phase 1 was not run in this invocation, loads and filters the cumulative cache.
  - **Phase 3 — Extraction**: `music.json`+ F0 extraction via `batch/note_extraction_batch.py` subprocess for this batch.
  - **Phase 4 — Iterative Alignment**: Generate prior from `extracted_notes.json` via OpenUtau, iteratively adjusting note durations until DTW convergence. OpenUtau Player is initialised once before the batch loop and reused. Accepts optional `player` parameter.
  - **Phase 5 — Manifest**: Generate `manifest.csv` for ML training. Runs once after all batches complete.
- `--phases` accepts `'12345'`, any subset. Each selected phase runs per-batch (except Phase 5). All phases independently resumable via sentinel files.
- CLI args include `--max_iterations` (default 3) and `--duration_threshold` (default 0.15) for alignment tuning.
- `get_english_dali_ids(dali_annot_dir)`: scans DALI annotations and returns IDs where `metadata.language == 'english'`.

## Running Scripts

```bash
# Single song
conda run -n vocaloflow-datasynthesizer python pipelines/synthesize_v2.py --dali_id <id> --mode line

# Full English dataset
conda run -n vocaloflow-datasynthesizer python pipelines/synthesize_dataset_v2.py --phases 12345 --songs_per_batch 100

# Resume a specific phase
conda run -n vocaloflow-datasynthesizer python pipelines/synthesize_dataset_v2.py --phases 34 --songs_per_batch 50
```
