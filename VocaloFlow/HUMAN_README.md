# VocaloFlow

Core conditional flow matching model. Transforms Vocaloid mel-spectrograms into high-quality singing voice mel-spectrograms.

## Environment

The `vocaloflow` environment is shared by this module, AdversarialFinetune, AdversarialPostnet, PL-BERT, SpeakerEmbedding, and DataAnalysis.

```bash
conda env create -f VocaloFlow/environment.yml
conda activate vocaloflow
```

Verify:
```bash
python -c "import torch; print(torch.__version__, 'cuda:', torch.cuda.is_available())"
```

## Prerequisites

- Training data in `Data/` with a `manifest.csv` (produced by DataSynthesizer)
- Optional: precomputed PL-BERT features and/or speaker embeddings in each chunk directory

## Usage

All commands run from `VocaloFlow/`:

```bash
conda activate vocaloflow
```

**Train** (fresh run):
```bash
python -m training.train --name my-experiment
```

**Train** with config overrides:
```bash
python -m training.train --name my-experiment --config configs/cur_config.yaml
```

**Resume** training:
```bash
python -m training.train --resume my-experiment
```

**Inference** (single song):
```bash
python -m inference.pipeline \
    --ustx path/to/song.ustx \
    --prior-wav path/to/prior.wav \
    --checkpoint checkpoints/<run_name>/checkpoint_200000.pt \
    --output output.wav
```

Key inference flags: `--num-ode-steps` (default 16), `--save-mels`, `--speaker-embedding <path>`, `--chunk-size` (default 256), `--overlap` (default 16).

**TensorBoard**:
```bash
tensorboard --logdir logs
```

## Outputs

- Checkpoints: `checkpoints/<run_name>/checkpoint_{step}.pt`
- Config snapshots: `configs/<run_name>/config.yaml`
- TensorBoard logs: `logs/<run_name>/`
