# VocaloFlow Memory Palace

Conditional flow matching model that maps low-quality Vocaloid mel-spectrograms to high-quality singing voice mel-spectrograms using Optimal Transport CFM (OT-CFM).

## Overview

VocaloFlow takes a paired dataset of Vocaloid (prior) and high-quality (target) mel-spectrograms, along with phoneme, F0, and voicing conditioning, and learns a velocity field that transforms the prior into the target via an ODE.

**Architecture**: 2-layer Diffusion Transformer (DiT) with AdaLN-Zero conditioning, ~35-50M parameters.
**Training objective**: OT-CFM with linear interpolation path x_t = (1-(1-σ)t)x_0 + tx_1.
**Inference**: ODE integration (Euler or midpoint) from t=0 to t=1.

## Directory Structure

```
VocaloFlow/
├── README.md                # This file (memory palace root)
├── __init__.py
├── environment.yml          # Conda env: vocaloflow (Python 3.11, PyTorch 2.2+)
│
├── configs/
│   ├── __init__.py          # Re-exports VocaloFlowConfig
│   └── default.py           # VocaloFlowConfig dataclass (all hyperparameters)
│
├── utils/                   # Data extraction and loading
│   ├── __init__.py          # Re-exports load_manifest, filter_manifest, split_by_song, resample_*, etc.
│   ├── README.md            # Memory palace (utils)
│   ├── data_helpers.py      # Manifest I/O, DTW filtering, song-level splitting
│   ├── dataset.py           # VocaloFlowDataset (PyTorch Dataset)
│   ├── resample.py          # resample_1d(), resample_2d(), resolve_phoneme_indirection()
│   └── collate.py           # vocaloflow_collate_fn for DataLoader
│
├── model/                   # Neural network architecture
│   ├── __init__.py          # Re-exports VocaloFlow
│   ├── README.md            # Memory palace (model)
│   ├── vocaloflow.py        # Top-level VocaloFlow nn.Module
│   ├── dit_block.py         # DiTBlock + AdaLNZero
│   ├── embeddings.py        # TimestepMLP + PhonemeEmbedding
│   └── rope.py              # Rotary Position Embeddings
│
├── training/                # Training loop and loss
│   ├── __init__.py
│   ├── README.md            # Memory palace (training)
│   ├── train.py             # Main entry point (argparse CLI)
│   ├── flow_matching.py     # FlowMatchingLoss (OT-CFM)
│   └── lr_schedule.py       # get_lr() with warmup + cosine decay
│
└── inference/               # Sampling and evaluation
    ├── __init__.py
    ├── inference.py          # sample_ode() with Euler/midpoint integration
    ├── pipeline.py           # End-to-end inference: USTX → enhanced WAV
    └── evaluate.py           # mel_mse(), mel_mae() metrics
```

## Data Format

All data comes from `../../Data/` with `manifest.csv` as the index. Per chunk:
- `prior_mel.npy`: (T, 128) float32 — Vocaloid mel (sr=24000, hop=480)
- `target_mel.npy`: (T, 128) float32 — High-quality target mel
- `target_f0.npy`: (T,) float32 — F0 in Hz (0 = unvoiced)
- `target_voicing.npy`: (T,) float32 — Binary V/UV mask
- `phoneme_ids.npy`: (P,) int32 — Expanded phoneme token sequence
- `phoneme_mask.npy`: (T,) int32 — Per-frame INDEX into phoneme_ids (not a direct token ID)

**Critical**: phoneme_mask is an indirection — resolve via `token = phoneme_ids[phoneme_mask[t]]`.
**Critical**: T differs across prior_mel, target_mel, and phoneme_mask — resample to target T.

## Conda Environment

Name: `vocaloflow`. Activate with `conda activate vocaloflow`.
Key deps: PyTorch 2.2+, CUDA 12.1, numpy, pandas, einops, tensorboard.

## How to Train

From the `VocaloFlow/` directory:

```bash
conda activate vocaloflow
python -m training.train
```

CLI overrides (all optional):
- `--data-dir PATH` — Root data directory (default `../../Data`)
- `--manifest PATH` — Path to manifest.csv (default `../../Data/manifest.csv`)
- `--batch-size N` — Batch size (default 32)
- `--lr FLOAT` — Peak learning rate (default 1e-4)
- `--total-steps N` — Total training steps (default 200,000)
- `--max-dtw-cost FLOAT` — DTW quality filter threshold (default 200.0)

## How to Run Inference

From the `VocaloFlow/` directory:

```bash
conda activate vocaloflow
python -m inference.pipeline \
  --ustx path/to/prior.ustx \
  --prior-wav path/to/prior.wav \
  --checkpoint checkpoints/checkpoint_200000.pt \
  --output output.wav
```

Key flags:
- `--ustx` (required) — OpenUTAU .ustx project file
- `--checkpoint` (required) — VocaloFlow .pt checkpoint
- `--prior-wav` — Pre-rendered WAV of the USTX (recommended; if omitted, attempts C# Player API render)
- `--f0` — Pre-extracted F0 .npy (if omitted, uses RMVPE or MIDI pitch fallback)
- `--device auto|cuda|cpu` — Compute device (default auto)
- `--chunk-size 256` — Frames per inference chunk (256 = 5.12s)
- `--overlap 16` — Crossfade overlap frames between chunks

Outputs:

## utils/resample.py

Shared resampling and phoneme resolution utilities used by both `dataset.py` and `inference/pipeline.py`.

### Functions
- `resample_1d(arr, target_len, mode)` — Resample 1D signal (F0, voicing, phoneme IDs) via `F.interpolate`. Accepts tensor or numpy. Mode: `"linear"` for continuous, `"nearest"` for discrete.
- `resample_2d(arr, target_len, mode)` — Resample 2D `(T, C)` signal (mel-spectrograms) along time axis.
- `resolve_phoneme_indirection(phoneme_ids, phoneme_mask)` — Maps frame-level mask indices to actual phoneme token IDs. Clips out-of-bounds indices.

## inference/pipeline.py

End-to-end inference pipeline: USTX -> enhanced WAV.

### Key Functions
- `parse_ustx(ustx_path)` — Loads USTX YAML, extracts notes (position, duration, tone, lyric), BPM, resolution
- `render_ustx_to_wav(ustx_path, notes_data, output_wav)` — Optional: renders USTX via C# Player API (pythonnet)
- `extract_prior_mel(wav_path)` — Extracts (T, 128) normalized log-mel via `mel_to_soulx_mel()` from DataSynthesizer
- `extract_or_load_f0(...)` — 3-tier fallback: provided .npy -> RMVPE extraction -> MIDI pitch synthesis
- `build_phoneme_ids(notes, ms_per_tick, total_frames, phoneset_path)` — USTX lyrics -> `g2p_transform()` from SoulX-Singer -> `_build_mel2note()` -> `resolve_phoneme_indirection()` -> frame-level IDs
- `load_model(checkpoint_path, device)` — Loads VocaloFlow from checkpoint, prefers EMA weights
- `infer_chunked(model, prior_mel, f0, voicing, phoneme_ids, ...)` — Overlap-add ODE inference with linear crossfade
- `mel_to_wav(mel)` — Vocodes output mel via SoulX-Singer Vocos

### External Dependencies (imported via importlib to avoid utils namespace collision)
- `DataSynthesizer/utils/vocoders.py` — `mel_to_soulx_mel()`, `invert_mel_to_audio_soulxsinger()`
- `DataSynthesizer/utils/phoneme_mask.py` — `_build_mel2note()`, `_load_phone2idx()`
- `DataSynthesizer/utils/voiced_unvoiced.py` — `get_voiced_mask()`
- `SoulX-Singer/preprocess/tools/g2p.py` — `g2p_transform()` for lyric -> phoneme string conversion
- `SoulX-Singer/preprocess/tools/f0_extraction.py` — `F0Extractor` (RMVPE)
- `SoulX-Singer/soulxsinger/utils/phoneme/phone_set.json` — 2820-token phoneme vocabulary

Outputs:
- Checkpoints: `./checkpoints/checkpoint_{step}.pt`
- TensorBoard logs: `./logs/` (view with `tensorboard --logdir logs`)

## configs/default.py

### VocaloFlowConfig
Single dataclass holding all hyperparameters. Key values:
- `data_dir="../../Data"`, `manifest_path="../../Data/manifest.csv"`
- `max_dtw_cost=200.0` — Quality filter threshold
- `phoneme_vocab_size=2820`, `phoneme_embed_dim=256`
- `hidden_dim=1024`, `num_heads=16`, `ffn_dim=4096`, `num_dit_blocks=2`
- `input_channels=514` (128+128+1+1+256)
- `max_seq_len=256` — Covers p95 of sequence lengths
- `batch_size=32`, `learning_rate=1e-4`, `total_steps=200_000`
