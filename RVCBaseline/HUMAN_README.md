# RVCBaseline

RVC (Retrieval-based Voice Conversion) baseline for thesis evaluation. Fine-tunes an RVC model on singing data and runs inference for comparison against VocaloFlow.

## Environment

```bash
conda env create -f RVCBaseline/environment.yml
conda activate rvc
```

## Prerequisites

1. **Clone Applio** (RVC training framework):
   ```bash
   git clone https://github.com/IAHispano/Applio.git ../Applio
   ```

2. **Install Applio dependencies** — from the Applio directory, run:
   - Windows: `run-install.bat`
   - Linux/Mac: `run-install.sh`

3. Training data in `Data/` with `manifest.csv` (produced by DataSynthesizer)

## Usage

All commands run from `RVCBaseline/`:

```bash
conda activate rvc
```

### Step 1: Prepare training data

```bash
python prepare_training_data.py --config configs/default.yaml
```

Copies `target.wav` files from the training split into `training_data/` for RVC consumption.

### Step 2: Train RVC model

```bash
python run_training.py --config configs/default.yaml
```

Drives Applio's 3-step pipeline (preprocess, extract features, train). Use `--dry-run` to preview commands without executing. Output model lands in `../Applio/logs/<model_name>/`.

### Step 3: Run inference

**Quality context** (convert Teto priors):
```bash
python run_inference.py \
    --rvc-model-path "../Applio/logs/Rachie-RVC/Rachie-RVC_170e_47260s_best_epoch.pth" \
    --max-eval-chunks 50
```

**Controllability context** (convert DALI priors):
```bash
python run_inference.py \
    --rvc-model-path "../Applio/logs/Rachie-RVC/Rachie-RVC_170e_47260s_best_epoch.pth" \
    --mode controllability \
    --max-eval-chunks 50
```

## Outputs

- `output/` — RVC WAVs for quality evaluation (named `{dali_id}_{chunk_name}.wav`)
- `output_dali/` — RVC WAVs for controllability evaluation

These directories are consumed by the [Evaluation pipeline](../Evaluation/HUMAN_README.md) via `--rvc-audio-dir` and `--rvc-dali-audio-dir`.
