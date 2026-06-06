# AdversarialFinetune

Experiment 2: adversarial flow-matching fine-tuning of a pretrained VocaloFlow model. Backpropagates adversarial loss through a differentiable 4-step ODE unroll into the velocity field.

## Environment

Uses the `vocaloflow` conda environment. If you haven't created it yet, see [VocaloFlow/HUMAN_README.md](../VocaloFlow/HUMAN_README.md#environment).

## Prerequisites

- A pretrained VocaloFlow checkpoint (e.g. `VocaloFlow/checkpoints/<run_name>/checkpoint_*.pt`)
- Training data in `Data/` with `manifest.csv`

## Usage

All commands run from `AdversarialFinetune/`:

```bash
conda activate vocaloflow
```

**Smoke test** (verify ODE gradient flow):
```bash
python ode_unroll.py
```

**Train** (full run):
```bash
python train_finetune.py --name exp2-v1 --config configs/exp2_v1.yaml
```

To specify the base checkpoint explicitly:
```bash
python train_finetune.py --name my-run --config configs/exp2_v1.yaml --pretrained-run 4-16-wavenet
```

**Resume**:
```bash
python train_finetune.py --resume my-run
```

Available configs in `configs/`: `exp2_v1.yaml` (full 30k steps), `smoke_cfm_only.yaml` (200 steps), `smoke_full.yaml` (500 steps), `exp4_afm.yaml` (AFM recipe).

## Inference

Use VocaloFlow's inference pipeline with the fine-tuned checkpoint:

```bash
cd ../VocaloFlow
python -m inference.pipeline \
    --ustx path/to/song.ustx \
    --prior-wav path/to/prior.wav \
    --checkpoint ../AdversarialFinetune/checkpoints/<run_name>/checkpoint_*.pt \
    --output output.wav \
    --num-ode-steps 4
```

## Outputs

- Checkpoints: `checkpoints/<run_name>/checkpoint_{step}.pt`
- TensorBoard logs: `logs/<run_name>/`
