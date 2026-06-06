# PL-BERT

Precomputes frozen 768-dim contextual phoneme embeddings from a pretrained ALBERT model. Used when VocaloFlow is configured with `use_plbert=True`.

## Environment

Uses the `vocaloflow` conda environment. If you haven't created it yet, see [VocaloFlow/HUMAN_README.md](../VocaloFlow/HUMAN_README.md#environment).

## Prerequisites

- Training data with `phoneme_ids.npy` in each chunk directory (produced by DataSynthesizer)
- The pretrained PL-BERT checkpoint at `PL-BERT/checkpoints/step_1000000.t7` (must be obtained separately)

## Usage

Run from the **repository root**:

```bash
conda activate vocaloflow
python PL-BERT/precompute_features.py --data-dir Data/Rachie --manifest Data/Rachie/manifest.csv
```

Use `--overwrite` to recompute existing features. Use `--device cpu` to force CPU inference.

## Outputs

Creates `plbert_features.npy` (shape: P x 768, float16) in each chunk directory. Chunks that already have this file are skipped unless `--overwrite` is passed.
