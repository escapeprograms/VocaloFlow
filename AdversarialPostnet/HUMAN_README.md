# AdversarialPostnet

Experiment 1: a lightweight post-network trained with adversarial + reconstruction losses to sharpen VocaloFlow's mel-spectrogram output.

## Environment

Uses the `vocaloflow` conda environment. If you haven't created it yet, see [VocaloFlow/HUMAN_README.md](../VocaloFlow/HUMAN_README.md#environment).

## Prerequisites

- A pretrained VocaloFlow checkpoint
- Training data in `Data/` with `manifest.csv`

## Usage

All commands run from `AdversarialPostnet/`:

```bash
conda activate vocaloflow
```

### Step 1: Generate predicted mels from frozen VocaloFlow

```bash
python -m data.generate_predictions \
    --checkpoint ../VocaloFlow/checkpoints/<run_name>/checkpoint_*.pt \
    --data-dir ../Data/Rachie \
    --manifest ../Data/Rachie/manifest.csv
```

This creates `predicted_mel.npy` in each chunk directory and writes `predicted_mel_manifest.csv`.

### Step 2: Train the postnet

```bash
python -m training.train_postnet --name my-postnet
```

Optional flags: `--config <yaml>`, `--predicted-mel-manifest <path>`, `--batch-size`, `--lr`, `--total-steps`, `--resume <name>`.

### Step 3: Evaluate

```bash
python evaluate_postnet.py \
    --input-mel path/to/predicted_mel.npy \
    --checkpoint checkpoints/<run_name>/checkpoint_*.pt \
    --output-dir eval_output
```

## Outputs

- Predicted mels: `predicted_mel.npy` in each chunk directory (Step 1)
- Checkpoints: `checkpoints/<run_name>/checkpoint_{step}.pt` (Step 2)
