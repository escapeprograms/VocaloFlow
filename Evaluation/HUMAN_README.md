# Evaluation

Evaluation pipeline computing thesis metrics (F0 RMSE/Pearson, FFE, MCD, WER, MOS, Onset MAE) across VocaloFlow, RVC baseline, and SoulX-Singer.

## Environment

This pipeline spans **three environments** across its steps. Create the eval-specific one with:

```bash
conda env create -f Evaluation/environment.yml
```

Verify:
```bash
conda activate vocaloflow-eval
python -c "import whisper, jiwer; print('eval env ok')"
```

## Prerequisites

- A trained VocaloFlow checkpoint
- Training data in `Data/` with `manifest.csv`
- (Optional) Trained RVC model for baseline comparison — see [RVCBaseline/HUMAN_README.md](../RVCBaseline/HUMAN_README.md)

## Usage

The full pipeline has three steps, each in a different environment.

### Step 1: Pre-generate RVC baseline audio (one-time, optional)

See [RVCBaseline/HUMAN_README.md](../RVCBaseline/HUMAN_README.md) for training and inference. Produces WAV files in `RVCBaseline/output/` and `RVCBaseline/output_dali/`.

### Step 2: Pre-generate DALI USTX priors (one-time, for controllability eval)

Run from `Evaluation/` in the **datasynthesizer** env:

```bash
conda activate vocaloflow-datasynthesizer
python generate_dali_ustx.py --output-dir dali_ustx --max-chunks 50
```

### Step 3: Run evaluation

Run from `Evaluation/` in the **eval** env:

```bash
conda activate vocaloflow-eval
```

**Full evaluation** (both quality and controllability contexts):
```bash
python run_eval.py \
    --checkpoint ../VocaloFlow/checkpoints/<run_name>/checkpoint_*.pt \
    --rvc-audio-dir ../RVCBaseline/output \
    --rvc-dali-audio-dir ../RVCBaseline/output_dali \
    --eval-context both \
    --max-eval-chunks 50
```

**Quick smoke test** (skip heavy Whisper and MOS models):
```bash
python run_eval.py \
    --checkpoint ../VocaloFlow/checkpoints/<run_name>/checkpoint_*.pt \
    --eval-context quality \
    --max-eval-chunks 5 \
    --disable-wer --disable-mos
```

**Quality only** (no controllability):
```bash
python run_eval.py \
    --checkpoint ../VocaloFlow/checkpoints/<run_name>/checkpoint_*.pt \
    --eval-context quality \
    --max-eval-chunks 50
```

## Outputs

Results are written to `eval_output/`:
- `per_sample_*.csv` — per-chunk metric values
- `summary_*.csv` — aggregated statistics
- `eval_config.yaml` — config snapshot for reproducibility
