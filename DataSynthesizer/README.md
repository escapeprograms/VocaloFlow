# DataSynthesizer Memory Palace

This document serves as the "memory palace" for the `DataSynthesizer` module. It describes the current up-to-date system details, architecture, and directory structure.

## Overview

The `DataSynthesizer` module is responsible for taking DALI dataset annotations (which include notes, words, lines, and paragraphs) and synthesizing audio using two core stages:
1. **Target Synthesis** (`stages/synthesizeTarget.py`): Generates vocals via the SoulX-Singer machine learning model (usually outputting `target.wav`).
2. **Prior Synthesis** (`stages/synthesizePrior.py`): Generates vocals via the OpenUtau API (usually outputting `prior.wav`).

The pipeline (`pipelines/synthesize_v2.py` / `pipelines/synthesize_dataset_v2.py`) is a **target-first pipeline**: generates target first, extracts note structure from `music.json` with F0-derived pitches from RMVPE, generates prior from those notes, then iteratively adjusts note durations via DTW until the prior timing converges to match the target. No audio warping — the prior is regenerated until it naturally aligns.

## Directory Structure

```
DataSynthesizer/
├── voice_providers.py  # Provider config: maps provider name -> prompt paths + prompt_name
│
├── pipelines/          # Entry points — what you run
│   ├── synthesize_v2.py        # Single-song driver (target-first)
│   └── synthesize_dataset_v2.py# Dataset-scale driver
│
├── stages/             # Core pipeline steps (called by pipelines)
│   ├── synthesizeTarget.py     # SoulX-Singer vocal generation
│   └── synthesizePrior.py      # OpenUtau vocal generation
│
├── alignment/          # All alignment-related code
│   └── iterative_align.py      # Iterative note duration adjustment
│
├── utils/              # Standalone helpers (no local deps)
│   ├── vocoders.py             # Mel-to-audio inversion (Griffin-Lim, HiFiGAN, Vocos)
│   ├── phoneme_mask.py         # Frame-level phoneme identity mask (mel2note) generation
│   ├── voiced_unvoiced.py      # Voiced/unvoiced mask utilities
│   ├── generate_manifest.py    # Training manifest CSV generation
│   ├── prompt_selection.py     # Probabilistic voice prompt selection
│   ├── grab_midi.py            # Frequency-to-MIDI conversion
│   └── determine_chunks.py     # Song segmentation by line/paragraph/n-line
│
├── batch/              # Subprocess entry points (run in `soulxsinger` env)
│   ├── soulxsinger_batch_infer.py  # Batch SoulX-Singer inference
│   └── note_extraction_batch.py    # Batch note extraction (music.json + RMVPE F0)
│
├── scripts/            # One-off utilities
│   └── migrate_to_providers.py # Migration: flat Data/ -> Data/{provider}/ layout
│
└── scratch/            # Testing, debugging, exploration
    ├── VisualizeSegment.py     # Pitch curve visualization
    ├── test_hifigan.py         # Vocoder roundtrip testing
    ├── inspect_player.py       # OpenUtau Player introspection
    └── dali_test.ipynb         # DALI dataset exploration notebook
```

Each subdirectory has its own `README.md` memory palace with detailed per-file documentation.

## Configuration & Usage

### Conda Environments

The pipeline uses **two** conda environments:

- **`vocaloflow-datasynthesizer`** — parent orchestration env. Runs all `pipelines/`, `stages/`, `alignment/`, and `utils/` code. Lightweight — no torch, no SoulX-Singer package.
- **`soulxsinger`** — dedicated subprocess env for the GPU-heavy SoulX-Singer inference and RMVPE note extraction. Contains torch (cu118) + the SoulX-Singer package. Built from `SoulX-Singer/requirements.txt`, with two pinned overrides against upstream drift: `torch==2.2.0+cu118` (from PyTorch cu118 index) and `transformers==4.42.4` + `tokenizers==0.19.1` (upstream `>=4.53.0` breaks SoulX-Singer's `LlamaAttention` call site).

The parent env launches the subprocess via `config.SOULX_PYTHON`, which points at the `soulxsinger` env's `python.exe`. You never activate `soulxsinger` directly — run everything via `conda run -n vocaloflow-datasynthesizer ...` and the subprocess dispatch happens automatically.

### Dependencies
Ensure prerequisites like Python, Conda, `librosa`, `mido`, `soundfile`, and `pyworld` python packages are available.

### Running Scripts

All scripts should be run from the `DataSynthesizer/` directory:

**Single song** (development / testing):
```bash
conda run -n vocaloflow-datasynthesizer python pipelines/synthesize_v2.py --dali_id <id> --mode line --provider WillStetson
```

**Full English dataset** (production):
```bash
conda run -n vocaloflow-datasynthesizer python pipelines/synthesize_dataset_v2.py --phases 12345 --songs_per_batch 100 --provider Rachie
```

**Resume a specific phase** (e.g. extraction + alignment only, after a crash):
```bash
conda run -n vocaloflow-datasynthesizer python pipelines/synthesize_dataset_v2.py --phases 34 --songs_per_batch 50 --provider WillStetson
```

Key Arguments:
- `--mode`: `[line, n-line, paragraph, test]` — chunk boundary granularity.
- `--use_continuations`: Extend note durations to fill intra-word gaps (does not affect phoneme hints).
- `--use_phonemes`: Inject ARPAbet phoneticHints via g2p_en to fix cross-note mispronunciation.
- `--use_f0`: Use DALI F0 curves for pitch prompting instead of flat MIDI.
- `--songs_per_batch` *(dataset only)*: Songs per SoulX-Singer subprocess (default 100 ≈ 5200 chunks). Controls model-load amortisation vs. crash recovery window.

### Output Per Chunk

```
Data/<provider>/<dali_id>/<chunk_name>/
  music.json          # SoulX metadata from DALI
  chunk_words.json     # DALI word texts for lyric mapping
  prompt_info.json     # Voice prompt selection: {"provider", "prompt_name"}
  target.wav           # Target from SoulX-Singer
  target_mel.npy       # Target mel-spectrogram (pre-vocoder)
  extracted_notes.json # Notes from music.json structure + F0-derived pitches
  target_f0.npy        # Frame-level F0 from target audio (RMVPE, Hz, 0=unvoiced)
  target_voicing.npy   # Boolean voiced/unvoiced mask (f0 > 0)
  prior.wav            # Prior from OpenUtau (iteratively aligned to target)
  prior.ustx           # OpenUtau project file
  prior_mel.npy        # Mel-spectrogram of the final prior, shape (T, 128)
  adjusted_notes.json  # Final note durations after iterative alignment (see below)
  alignment.json       # Iterative alignment convergence metrics (includes provider/prompt_name)
  phoneme_ids.npy      # Expanded phoneme token ID sequence, shape (P,), int32
  phoneme_mask.npy     # Frame-level phoneme identity mask (mel2note), shape (T,), int32
```

#### `adjusted_notes.json` structure

Produced by `iterative_align()` in Phase 4. Contains the converged note timing that makes the prior naturally match the target's rhythm — no audio warping needed.

```json
{
  "notes": [
    {
      "note_text": "C4",        // pitch as note name
      "start_s": 0.0,           // note onset in seconds (6 decimal places)
      "note_dur": 0.52,         // adjusted duration in seconds (6 decimal places)
      "lyric": "the",           // syllable text
      ...                       // other fields inherited from extracted_notes.json
    }
  ],
  "source": "iterative"         // always "iterative" — marks origin as iterative alignment
}
```

The `notes` array mirrors `extracted_notes.json` but with `start_s` and `note_dur` values refined through iterative DTW convergence. Each iteration generates a prior via OpenUtau, computes DTW against the target mel-spectrogram, measures per-note time-scale ratios, and applies damped corrections (clamped to [0.3, 3.0]). Iteration stops when all per-note ratios fall within `duration_threshold` (default 15%) or `max_iterations` is reached.

**Downstream consumers:**
- `utils/phoneme_mask.py`: reads `note_dur` values to build frame-level phoneme masks (`phoneme_ids.npy`, `phoneme_mask.npy`). Falls back to `extracted_notes.json` if adjusted version is absent.
- `utils/generate_manifest.py`: records the path in the `adjusted_notes_path` column of `manifest.csv`. Chunks missing this file are excluded from the manifest.

All mel-spectrograms use SoulX-Singer settings: 24kHz, 128 mels, hop=480, n_fft=1920, log+z-score normalized (mean=-4.92, var=8.14). Shape convention is **(T, 128)** (time-first). F0 and voicing are on the same 50fps frame grid as the mel.

**Dataset-level output (multi-provider layout):**
```
Data/
├── WillStetson/
│   ├── manifest.csv              # Training manifest (provider + prompt_name columns)
│   ├── pending_inference_tasks.json
│   └── <dali_id>/<chunk_name>/   # Per-chunk artifacts as above
└── Rachie/
    ├── manifest.csv
    ├── pending_inference_tasks.json
    └── <dali_id>/<chunk_name>/
```

Each provider directory is self-contained. The `alignment.json` per chunk records `"provider"` and `"prompt_name"` fields for traceability. The manifest mirrors these as columns.

**voice_providers.py** (config-only) maps provider names to prompt lists:
- `WillStetson` -> single prompt from `SoulX-Singer/example/transcriptions/WillStetsonSample/`
- `Rachie` -> 3 prompts with `midi_range` tuples: `rachie_low` (55-60), `rachie_mid` (62-67), `rachie_high` (67-74)

**utils/prompt_selection.py** selects which prompt to use per chunk via softmax over the distance from the chunk's median MIDI pitch to each register's range. Single-prompt providers skip the selection. The choice is persisted in `prompt_info.json` per chunk.
