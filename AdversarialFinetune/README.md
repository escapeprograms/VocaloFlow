# AdversarialFinetune — VocaloFlow Experiment 2

Adversarial flow-matching fine-tuning of a pretrained VocaloFlow model.
Follows the PeriodWave-Turbo (arXiv:2408.08019) recipe adapted for
mel-domain singing voice synthesis.

Starts from a VocaloFlow checkpoint (default: `4-16-wavenet/checkpoint_55000.pt`)
and fine-tunes the velocity field with four loss components, each independently
toggleable:

1. **L_cfm** — the original OT-CFM velocity loss (anchor, prevents velocity
   field collapse).
2. **L_rec** — masked L1 between the 4-step Euler ODE output and the target
   mel (supervises the generated output directly).
3. **L_adv** — hinge-GAN loss from a `PatchDiscriminator` on random mel crops
   at three scales (32×64, 64×128, 128×128).
4. **L_fm** — L1 distance between discriminator intermediate features for real
   vs fake crops.

The adversarial loss backpropagates through a *differentiable* 4-step Euler
ODE unroll into the model's velocity field — this is the whole point of the
experiment, and the key difference from Experiment 1 (`AdversarialPostnet/`).

## Quick start

```bash
cd AdversarialFinetune/

# Full experiment (30k steps)
python train_finetune.py --name exp2-v1 --config configs/exp2_v1.yaml

python train_finetune.py --name 4-17-adv
    --config configs/exp2_v1.yaml 
    --pretrained-run 4-16-wavenet


# Smoke tests (run before committing to the full run)
python ode_unroll.py                                    # correctness + grad flow
python train_finetune.py --name smoke_cfm_only \
    --config configs/smoke_cfm_only.yaml                # CFM only, 200 steps
python train_finetune.py --name smoke_full \
    --config configs/smoke_full.yaml                    # full pipeline, 500 steps

# Resume
python train_finetune.py --resume 4-17-adv
```

## Directory layout

Flat module layout (no `configs/`, `training/`, `evaluation/` sub-packages)
— intentional, see "Namespace note" below.  The one subpackage here,
`ft_utils/`, has a unique name that doesn't collide with anything in
VocaloFlow or AdversarialPostnet.

```
AdversarialFinetune/
├── README.md                     # this file (memory palace)
├── finetune_config.py            # FinetuneConfig dataclass + YAML round-trip
├── ft_checkpoint.py              # save/load (G + D + 2 optimizers + EMA + step)
├── ode_unroll.py                 # differentiable ODE unroll + smoke tests
├── cfm_loss.py                   # wrapper around VocaloFlow's FlowMatchingLoss
├── evaluate_ft.py                # periodic 32-step vs 4-step inference eval
├── train_finetune.py             # main training entry point
├── dit_discriminator.py          # DiT transformer-based discriminator
├── hf_discriminator.py           # REMOVED — HF disc now uses DiTDiscriminator directly
├── disc_augmentation.py          # discriminator input augmentation (Exp 4)
├── ft_utils/                     # reusable helpers (unique name — no collision)
│   ├── imports.py                # import_from_path + sys.path bootstrap for VocaloFlow
│   ├── batch.py                  # BatchTensors, unpack_batch, timestamp()
│   └── paths.py                  # derive_run_paths(config)
├── configs/                      # YAML configs only (not a Python package)
│   ├── exp2_v1.yaml              # full 30k-step run
│   ├── smoke_cfm_only.yaml       # CFM only, 200 steps
│   ├── smoke_cfm_rec.yaml        # CFM + L1, 500 steps
│   ├── smoke_full.yaml           # all losses, 500 steps (no adv phase reached)
│   └── exp4_afm.yaml             # Exp 4: AFM recipe (grad norm + decay + augmentation)
├── checkpoints/                  # created at run time: checkpoints/<run_name>/
└── logs/                         # created at run time: logs/<run_name>/
```

## Namespace note

Python caches each package name globally.  Both VocaloFlow and AdversarialPostnet
use regular packages called `configs/` and `training/`, which would collide if
we added a sibling `configs/` or `training/` Python package here.  To sidestep
the collision:

* Our own top-level modules live flat at the package root with unique names
  (`finetune_config`, `ft_checkpoint`, `ode_unroll`, `cfm_loss`, `evaluate_ft`).
  Shared helpers live under `ft_utils/`, whose name also doesn't clash.
* VocaloFlow modules are imported normally — `VocaloFlow/` is put on
  `sys.path` via `ft_utils.imports.setup_vocaloflow_sys_path()` so
  `configs.default`, `model.vocaloflow`, `utils.dataset`,
  `utils.config_utils`, etc. resolve cleanly.
* AdversarialPostnet modules (`PatchDiscriminator`, loss functions,
  `extract_random_crops`) are loaded via
  `ft_utils.imports.import_from_path` with unique module names
  (`ap_discriminator`, `ap_losses`, `ap_random_crop`) — because their
  `training/` package would clash with VocaloFlow's.
* Sub-directory `configs/` here holds only YAML files; no `__init__.py`, so
  Python's package machinery walks past it to `VocaloFlow/configs/`.

## Per-file notes

### `finetune_config.py`

`FinetuneConfig` dataclass.  Field groups:

* **Pretrained source** — `pretrained_run`, `pretrained_step` (0 = latest),
  `pretrained_vocaloflow_dir` (defaults to `../VocaloFlow`).
* **Data** — mirrors VocaloFlow's defaults (`data_dir`, `manifest_path`,
  `max_seq_len=256`, `max_dtw_cost=100`, `val_fraction=0.05`,
  `split_mode="song"`).
* **ODE unroll** — `ode_num_steps=4`, `ode_method="euler"`,
  `ode_grad_checkpoint` toggle for activation memory.
* **Loss toggles** — `enable_cfm`, `enable_reconstruction`,
  `enable_adversarial`, `enable_feature_matching`.  Each independently
  turns off its component.
* **Loss weights** — `lambda_cfm=1.0`, `lambda_rec=1.0`, `lambda_adv=0.1`,
  `lambda_fm=2.0`.
* **Warmup** — `disc_warmup_steps=3000` (lambda_adv/fm ramp starts here),
  `adv_ramp_steps=2000` (linear ramp width).
* **Optimizers** — generator LR constant at `2e-5` (no scheduler),
  discriminator LR `2e-4` with linear warmup + cosine decay.
  `reset_gen_optimizer=True` discards pretrained Adam moments (see
  "Hyperparameter rationale" below).
* **EMA** — `ema_decay=0.9999`, `preserve_global_step=True` (continues the
  step counter from the pretrained checkpoint so the EMA warmup formula
  stays saturated — see "EMA warmup caveat").

`to_yaml(path)` / `from_yaml(path)` for YAML round-trip.

### `ft_checkpoint.py`

Named `ft_checkpoint` to avoid any chance of collision with
`VocaloFlow/training/checkpoint.py`.

* `extract_checkpoint_step(path)` — parses the step integer out of
  `checkpoint_<n>.pt`; returns `-1` on no match.  Also reused by
  `train_finetune._find_pretrained_checkpoint` to pick the latest
  pretrained VocaloFlow checkpoint.
* `find_latest_checkpoint(dir)` — highest-step `checkpoint_*.pt` or `None`.
* `save_checkpoint(model, ema_model, discriminator, opt_g, opt_d, step,
  config, vf_config, wandb_run_id)` — writes `checkpoint_<step>.pt` with keys:
  `step, model_state_dict, ema_model_state_dict, discriminator_state_dict,
  opt_g_state_dict, opt_d_state_dict, config` (holds the `vf_config` so
  VocaloFlow's inference pipeline can load this file directly),
  `finetune_config` (holds our own config for resume), `wandb_run_id`.
* `load_checkpoint(path, device)` — thin `torch.load` wrapper.

### `ode_unroll.py`

`unroll_ode(model, x_0, f0, voicing, phoneme_ids, padding_mask,
num_steps, method, grad_checkpoint, plbert_features=None,
speaker_embedding=None)`:

* Mirrors `VocaloFlow/inference/inference.py::sample_ode` but without
  `@torch.no_grad`, without CFG, without diagnostic prints.
* Supports `method="euler"` (one model call per step) or
  `method="midpoint"` (two).
* `grad_checkpoint=True` wraps each model call in
  `torch.utils.checkpoint.checkpoint` to cut activation memory roughly in
  half at the cost of an extra forward pass during backward.
* The returned tensor's gradient flows back through every step into the
  model's velocity-field parameters — this is the core mechanism of the
  experiment.
* `__main__` block runs three smoke tests: (1) numerical equivalence with
  `sample_ode`, (2) gradient reaches >50% of trainable parameters (the
  remainder are AdaLN-Zero-gated attention/FFN weights, which receive
  gradient once pretrained weights are loaded), (3) gradient checkpointing
  preserves numerics.  Run from `AdversarialFinetune/` with
  `python ode_unroll.py`.

### `cfm_loss.py`

`build_cfm_loss(cfg_dropout_prob, sigma_min)` returns a
`FlowMatchingLoss` (imported from `VocaloFlow/training/flow_matching.py`
via `ft_utils.import_from_path`) with the STFT auxiliary loss disabled.

CFG dropout fires internally when `criterion.training`; the validation
helper calls `criterion.eval()` to disable it, following VocaloFlow's
existing pattern.

### `ft_utils/`

Small subpackage of reusable helpers imported by the rest of the module.
Unique name, so it doesn't collide with `VocaloFlow/utils/` or any
AdversarialPostnet package.

* `ft_utils/imports.py`
  * `REPO_ROOT`, `VOCALOFLOW_DIR`, `ADV_POSTNET_DIR` — absolute paths
    resolved from the file's own location at import time.
  * `import_from_path(module_name, file_path)` — loads a file by path and
    registers it in `sys.modules` under a unique name.  Registers
    **before** `exec_module` so decorators that resolve types via
    `cls.__module__` (`@dataclass`) don't see `None`.  Used for every
    AdversarialPostnet import and for `sample_ode` / `FlowMatchingLoss`
    from VocaloFlow.
  * `setup_vocaloflow_sys_path()` — prepends `VocaloFlow/` to `sys.path`
    (idempotent).  Must be called before the first
    `from configs.default import VocaloFlowConfig`.
* `ft_utils/batch.py`
  * `BatchTensors(NamedTuple)` — fields `x_0, x_1, f0, voicing,
    phoneme_ids, padding_mask`.  Supports both attribute access and tuple
    unpacking.
  * `unpack_batch(batch, device)` — returns `BatchTensors`; moves the six
    standard VocaloFlow keys onto `device`.  Used by every training and
    validation loop (replaces four copies of the same `.to(device)` block).
  * `unpack_optional_features(batch, device)` — returns a dict with
    `plbert_features` and/or `speaker_embedding` if present in the batch.
    Keeps `BatchTensors` unchanged (no breaking tuple unpacking).
  * `timestamp()` — `"HH:MM:SS"` string for log-line prefixes.
* `ft_utils/paths.py`
  * `derive_run_paths(config)` — sets `config.checkpoint_dir` and
    `config.log_dir` from `config.run_name`.  No-op when `run_name == ""`.

See also `VocaloFlow/utils/config_utils.py::rebuild_dataclass_tolerant`
(imported into `train_finetune.py` for VocaloFlowConfig / FinetuneConfig
schema-tolerant reconstruction on resume / pretrained-checkpoint load).

### `evaluate_ft.py`

Two entry points:

* `evaluate_inference(ema_model, val_loader, device, num_samples,
  save_plots_dir, log_plots_to_wandb, wandb_step, num_plots)` — heavy eval.
  Runs `sample_ode` on the EMA model at both the inference-time setting
  (32 steps, midpoint) and the training-time setting (4 steps, Euler) on a
  val subset.  Returns:

  * `eval/l1_32step_midpoint` — primary mel L1 quality signal at inference.
  * `eval/l1_4step_euler` — training-time setting L1 (measures how well the
    adversarial sharpening transfers to few-step generation).
  * `eval/l1_ratio` — 4-step / 32-step ratio.

  When `save_plots_dir` is set, also writes 4-panel mel PNGs
  (`prior | target | pred_32step | pred_4step`) to disk, one per sample.
  When `log_plots_to_wandb=True`, uploads the first `num_plots` as
  `eval/mel_comparison_<i>` so they appear inline in the wandb run.

* `validate_recon_4step(ema_model, val_loader, device)` — cheap companion
  to `validate_cfm`.  Runs 4-step Euler on the **full** val set and returns
  `val/l1_reconstruction_4step`.  Complements `val/velocity_mse`: catches
  cases where single-`t` velocity MSE looks fine but the integrated ODE
  output degrades.

Mel-only by design.  Vocoding, F0 RMSE, and Whisper-WER live in a separate
post-run script (not yet written) to keep training fast.

### `dit_discriminator.py`

Transformer-based discriminator for full-sequence mel classification.
`DiTDiscriminator(mel_dim=128, hidden_dim=512, num_blocks=4, num_heads=8,
ffn_dim=2048, max_len=512, feature_block_indices=[1,3])` — ~12.75M params.

Architecture: input projection → prepend learnable [CLS] token →
N `TransformerBlock` layers (pre-norm, RoPE, GELU FFN) → LayerNorm on
[CLS] position → Linear(hidden_dim, 1) scalar logit.

`forward(mel, padding_mask, speaker_embedding=None, plbert_features=None)`
returns `(logits, features)` where `logits` is (B, 1) and `features` is a
list of intermediate block outputs at `feature_block_indices` for feature
matching.  When `use_plbert_input=True`, PL-BERT features are projected
(768→128, zero-init) and concatenated with mel.  When
`use_speaker_input=True`, speaker embedding is projected (192→64,
zero-init), broadcast across time, and concatenated.  `input_proj`
dimension adjusts accordingly.  `TransformerBlock` layers are unchanged.

### HF Discriminator (Exp 7)

**`hf_discriminator.py` has been removed.** The high-frequency auxiliary discriminator now uses `DiTDiscriminator` directly (same class as the full-spectrum discriminator) with `mel_dim=64`, `num_blocks=2`. Mel slicing (`mel[:, :, hf_start:]`) is done externally in `_hf_discriminator_step` and `_hf_generator_adv_loss` before passing to the DiT. No augmentation is applied to HF disc input.

Config fields: `use_hf_discriminator` (master toggle), `hf_disc_hf_start`, `hf_disc_num_blocks`, `hf_disc_hidden_dim`, `hf_disc_num_heads`, `hf_disc_ffn_dim`, `hf_disc_feature_blocks`, `lambda_hf_adv`, `hf_disc_learning_rate`, `hf_disc_lr_warmup_steps`.

Training integration: separate optimizer (`opt_d_hf`), separate step function (`_hf_discriminator_step`), separate generator loss (`_hf_generator_adv_loss`). In gradient normalization mode, HF adversarial gradients are independently L2-normalized before adding to generator `p.grad` (Phase 3, after DiT adv Phase 2). Checkpoint saves HF state conditionally; old checkpoints without HF keys load cleanly (HF disc starts from random init). Architecture-mismatched checkpoints (e.g. old PatchGAN HF state) are caught by try/except and fall back to random init.

### `disc_augmentation.py`

Discriminator input augmentation module (Exp 4).  All ops are
differentiable so gradients flow through during the G update.

* `temporal_shift(mel, shift)` — circular `torch.roll` along time dim.
* `temporal_cutout(mel, start, width)` — zero a contiguous frame block.
* `frequency_cutout(mel, start, width)` — zero a contiguous freq-bin block.
* `augment_mel(mel, config, rng)` — entry point.  Each augmentation fires
  independently with probability `config.disc_aug_prob`.  Uses a
  `torch.Generator` seeded by the caller so real and fake mels receive
  identical augmentation parameters.

Includes `__main__` smoke test: determinism, shape, cutout verification,
gradient flow.

### `train_finetune.py`

Main entry point.  `train()` is a thin control loop that delegates to
small named helpers — none of the loss / step / logging logic lives
inline any more.  High-level flow:

1. Parse CLI (`--name`, `--resume`, `--config`, `--pretrained-run`,
   `--pretrained-step`, `--batch-size`, `--total-finetune-steps`).
2. **Setup**: `derive_run_paths(config)` → `_detect_resume(...)` (looks in
   `./checkpoints/<run_name>/`) → `_init_wandb(...)` (autogenerates run
   name when empty and propagates it back into config) →
   `_save_config_snapshot(...)`.
3. **Pretrained load**: `_find_pretrained_checkpoint(config)` → `torch.load`
   → `rebuild_dataclass_tolerant(saved, VocaloFlowConfig)` to recover a
   schema-drift-tolerant `VocaloFlowConfig` matching the pretrained weights.
4. **Models / data / opts**: instantiate `VocaloFlow` + EMA copy,
   `_build_dataloaders(config)`, instantiate `PatchDiscriminator` at random
   init (plan §4.3), `_build_optimizers(config, model, discriminator)`.
5. **State restore**: `_restore_state(...)` branches on resume — on resume
   everything comes from our own checkpoint including `global_step` (and
   `start_global_step` recovered via `_infer_elapsed_ft_steps` from the
   sidecar `_ft_start_step.txt`); on fresh runs, weights come from the
   pretrained checkpoint, discriminator stays random, and `global_step`
   either preserves the pretrained step (`preserve_global_step=True`,
   default — keeps the EMA decay saturated) or restarts at 0.
6. **Per-step loop** (`while global_step < end_global_step: for batch in …`):
   * `_training_step(...)` — one full forward + D update + G update step.
     Internally: `unpack_batch`, LR update, CFM phase, ODE unroll,
     discriminator step (with optional augmentation via `augment_mel`),
     then generator losses.  Two G-update paths: (a) **gradient
     normalization** (`enable_grad_norm=True`): separate backward for
     CFM/rec, `torch.autograd.grad` for adversarial, L2-normalize
     adversarial grads to unit norm, add to CFM grads, clip, step;
     (b) **single backward** (default): weighted sum of all losses,
     `backward()`, clip, step.  Lambda_cfm is computed via
     `get_lambda_cfm()` which supports linear decay.  Returns a dict
     of scalars including `adv_grad_norm` and `cfm_grad_norm`.
   * `_ema_update(ema_model, model, global_step, config)` — warmup-safe
     `min(ema_decay, (step+1)/(step+10))` formula.
   * `_log_train_metrics(...)` every `log_every` steps (TB + wandb + console).
   * `_run_validation(...)` every `val_every` fine-tune steps (CFM MSE
     on full val set + 4-step Euler L1).
   * `_run_heavy_evaluation(...)` every `eval_every` fine-tune steps
     (32-step vs 4-step inference on val subset + optional mel plots).
   * `save_checkpoint(...)` every `save_every` fine-tune steps.
7. Final `save_checkpoint` + `writer.close()` + `wandb.finish()`.

The Module also defines the top-level `validate_cfm`, `get_effective_lambda`,
and `_load_config_from_checkpoint` helpers used on the resume path.  YAML
override merging lives in `_apply_yaml_overrides`.

## Configuration reference

Every field on `FinetuneConfig` ([finetune_config.py](finetune_config.py)).
All are settable via YAML overrides; a handful also have CLI flags (noted in
the rightmost column).  CLI precedence: dataclass defaults < `--config YAML`
< individual CLI flags < (on resume) saved checkpoint config.

### Pretrained source

| Field | Default | CLI | What it does |
|---|---|---|---|
| `pretrained_run` | `"4-16-wavenet"` | `--pretrained-run` | Subdirectory under `<pretrained_vocaloflow_dir>/checkpoints/` containing the VocaloFlow checkpoint to fine-tune from. |
| `pretrained_step` | `0` | `--pretrained-step` | Exact step to load (`checkpoint_<step>.pt`).  `0` means "latest in the directory". |
| `pretrained_vocaloflow_dir` | `"../VocaloFlow"` | — | Path to the VocaloFlow project root, relative to `AdversarialFinetune/`. |

Resolves to `<pretrained_vocaloflow_dir>/checkpoints/<pretrained_run>/checkpoint_<step>.pt`.

### Data

| Field | Default | CLI | What it does |
|---|---|---|---|
| `data_dir` | `"../Data/Rachie"` | — | Root directory containing per-chunk `.npy` files (mirrors VocaloFlow's default). |
| `manifest_path` | `"../Data/Rachie/manifest.csv"` | — | Manifest CSV listing all chunks with DTW costs and paths. |
| `max_seq_len` | `256` | — | Fixed sequence length per batch item (frames).  Shorter chunks pad; longer crop. |
| `max_dtw_cost` | `100.0` | — | Drop chunks whose DTW alignment cost exceeds this (quality filter). |
| `val_fraction` | `0.05` | — | Fraction of songs (or chunks, in random split) held out for validation. |
| `split_mode` | `"song"` | — | `"song"` splits by `dali_id` (no leakage); `"random"` splits per-chunk. |
| `seed` | `42` | — | Seed for the train/val split.  Match VocaloFlow's pretraining seed to keep val songs consistent. |

### ODE unrolling (training-time)

| Field | Default | CLI | What it does |
|---|---|---|---|
| `ode_num_steps` | `4` | — | Number of ODE integration steps used during training.  PeriodWave-Turbo default. |
| `ode_method` | `"euler"` | — | `"euler"` (1 model call per step) or `"midpoint"` (2). |
| `ode_grad_checkpoint` | `False` | — | Wrap each step's model call in `torch.utils.checkpoint` to roughly halve activation memory (costs one extra forward pass during backward).  Enable if OOM at `batch_size=8`. |

### Loss toggles

Each toggle turns its component off entirely — useful for smoke-testing
individual losses in isolation.

| Field | Default | What it does |
|---|---|---|
| `enable_cfm` | `True` | OT-CFM velocity-matching loss.  Anchors the velocity field; turn off only for ablation. |
| `enable_reconstruction` | `True` | L1 between the 4-step ODE output and the target mel. |
| `enable_adversarial` | `True` | Hinge-GAN loss on discriminator.  When `False`, D is not instantiated in the update loop. |
| `enable_feature_matching` | `True` | L1 between D's intermediate features for real vs fake crops.  Only active if `enable_adversarial=True` as well. |

### Loss weights

| Field | Default | What it does |
|---|---|---|
| `lambda_cfm` | `1.0` | Multiplier on the CFM velocity loss in the generator total. |
| `lambda_rec` | `1.0` | Multiplier on L1 reconstruction. |
| `lambda_adv` | `0.1` | Target multiplier on adversarial loss (ramps from 0 after warmup). |
| `lambda_fm` | `2.0` | Target multiplier on feature matching (ramps on the same schedule as `lambda_adv`). |
| `lambda_cfm_final` | `1.0` | End-of-training value for `lambda_cfm`.  When equal to `lambda_cfm` (default), no decay.  Set to e.g. `0.2` for linear decay over the full run. |

### Gradient normalization (Exp 4)

| Field | Default | What it does |
|---|---|---|
| `enable_grad_norm` | `False` | When `True`, compute adversarial gradients separately via `torch.autograd.grad`, normalize to unit L2 norm, then add to CFM gradients.  Prevents the adversarial signal from overwhelming the CFM anchor regardless of discriminator confidence. |

### Discriminator augmentation (Exp 4)

| Field | Default | What it does |
|---|---|---|
| `enable_disc_augmentation` | `False` | Apply augmentations to mels before feeding to discriminator. |
| `disc_aug_prob` | `0.5` | Per-augmentation fire probability. |
| `disc_aug_max_shift` | `16` | Max circular temporal shift (frames). |
| `disc_aug_cutout_min` | `8` | Min temporal cutout width (frames). |
| `disc_aug_cutout_max` | `32` | Max temporal cutout width (frames). |
| `enable_freq_cutout` | `False` | Enable frequency-band cutout augmentation. |
| `disc_aug_freq_cutout_min` | `8` | Min frequency cutout width (bins). |
| `disc_aug_freq_cutout_max` | `32` | Max frequency cutout width (bins). |

### Speaker embedding + conditioned discriminator (Exp 8)

| Field | Default | What it does |
|---|---|---|
| `use_speaker_embedding` | `False` | Load per-chunk speaker embeddings and enable generator `speaker_proj` conditioning. Requires `VocaloFlowConfig.use_speaker_embedding=True` on the pretrained model. |
| `disc_use_plbert_input` | `False` | Concatenate PL-BERT features (projected 768→`disc_plbert_proj_dim`) to discriminator mel input. |
| `disc_plbert_proj_dim` | `128` | Discriminator PL-BERT projection output dim. Zero-init. |
| `disc_use_speaker_input` | `False` | Concatenate speaker embedding (projected 192→`disc_speaker_proj_dim`) to discriminator mel input. |
| `disc_speaker_proj_dim` | `64` | Discriminator speaker embedding projection output dim. Zero-init. |

Generator: `speaker_proj = Linear(192, 512)` (zero-init) adds to timestep conditioning `c`.
Discriminator: projections broadcast across time and concatenate with mel before `input_proj`.
PL-BERT features receive the same `augment_mel` augmentations as mel (same RNG seed).
Checkpoint loading uses `strict=False` for new generator keys (`speaker_proj.*`).
Gradient norms for all projection layers logged as `train/{gen,disc}_{speaker,plbert}_proj_grad_norm`.

### CFG (applies only to the CFM phase)

The ODE unroll phase always uses full conditioning; these knobs only affect
the single-timestep velocity prediction done inside `FlowMatchingLoss`.

| Field | Default | What it does |
|---|---|---|
| `cfg_dropout_prob` | `0.2` | Probability of zeroing (f0, voicing, phoneme_ids) for a given sample during the CFM phase.  Matches pretraining. |
| `sigma_min` | `1.0e-4` | Small noise floor for OT-CFM numerical stability. |

### Warmup schedule (fine-tune-local steps)

Measured from the fine-tune start, not from the absolute `global_step`.

| Field | Default | What it does |
|---|---|---|
| `disc_warmup_steps` | `3000` | Steps with `lambda_adv = lambda_fm = 0`.  Lets the model adapt to fixed-step ODE and gives D a chance to learn on warm-up mels before providing signal. |
| `adv_ramp_steps` | `2000` | After warmup, linearly ramp `lambda_adv` and `lambda_fm` from 0 to their target values across this many steps. |

### Optimizers

| Field | Default | CLI | What it does |
|---|---|---|---|
| `gen_learning_rate` | `2.0e-5` | — | Generator (VocaloFlow) LR.  **Constant** — no schedule.  5× lower than pretraining (1e-4). |
| `disc_learning_rate` | `2.0e-4` | — | Discriminator peak LR.  Linear warmup then cosine decay. |
| `disc_lr_warmup_steps` | `1000` | — | Linear warmup width for D's LR. |
| `adam_beta1` | `0.8` | — | β₁ for both AdamWs.  HiFi-GAN convention, lower than the default 0.9 for faster adaptation. |
| `adam_beta2` | `0.99` | — | β₂ for both AdamWs. |
| `weight_decay` | `0.01` | — | AdamW weight decay (both). |
| `grad_clip` | `1.0` | — | `clip_grad_norm_` max norm (applied separately to G and D). |
| `reset_gen_optimizer` | `True` | — | If `True`, discard the pretrained checkpoint's Adam moments (recommended — see "Hyperparameter rationale" below).  If `False`, try to load them. |

### EMA

| Field | Default | What it does |
|---|---|---|
| `ema_decay` | `0.9999` | Target EMA decay.  Effective decay is `min(ema_decay, (step+1)/(step+10))` to avoid clobbering the EMA at step 0 of a fresh run. |
| `preserve_global_step` | `True` | Continue the pretrained checkpoint's step counter (so the EMA warmup formula saturates immediately).  See "EMA warmup caveat". |

### Training loop

| Field | Default | CLI | What it does |
|---|---|---|---|
| `batch_size` | `8` | `--batch-size` | Micro-batch size.  Reduced from pretraining's 32 because the 4-step ODE unroll holds ~4× activation memory. |
| `total_finetune_steps` | `30000` | `--total-finetune-steps` | Number of fine-tune steps (added on top of the pretrained step count when computing `end_global_step`). |
| `log_every` | `50` | — | Steps between TensorBoard/W&B scalar logs + console print. |
| `val_every` | `1000` | — | Fine-tune steps between cheap CFM-loss validation on the full val set. |
| `eval_every` | `3000` | — | Fine-tune steps between heavy evaluation (true ODE inference at both 32 and 4 steps on a val subset). |
| `save_every` | `5000` | — | Fine-tune steps between checkpoint writes. |
| `eval_num_samples` | `8` | — | Number of val items used for heavy evaluation. |
| `save_mel_plots` | `True` | — | When `True`, every `eval_every` steps save 4-panel mel PNGs (`prior \| target \| pred_32step \| pred_4step`) to `logs/<run_name>/mels/step_<global_step>/sample_<i>.png` AND upload the first `num_mel_plots` figures to wandb as `eval/mel_comparison_<i>`. |
| `num_mel_plots` | `4` | — | Cap on the number of per-sample mel PNGs saved / uploaded per eval invocation. |

### Discriminator

| Field | Default | What it does |
|---|---|---|
| `disc_channels` | `[32, 64, 128, 256]` | Per-layer output channel counts for `PatchDiscriminator`'s 4 strided convs.  ~1.2M params total. |
| `crop_specs` | `[[32,64], [64,128], [128,128]]` | List of `[time_frames, mel_bins]` scales.  One random crop per scale, per batch item. |

### Paths

| Field | Default | CLI | What it does |
|---|---|---|---|
| `run_name` | `""` | `--name`, `--resume` | Subdirectory name under `checkpoints/`, `logs/`, and `configs/` for this run.  Also used as W&B run name.  Empty → W&B auto-generates. |
| `checkpoint_dir` | `"./checkpoints"` | — | Parent directory for checkpoints.  Actual path: `<checkpoint_dir>/<run_name>/`. |
| `log_dir` | `"./logs"` | — | Parent directory for TensorBoard + W&B logs. |

---

## Hyperparameter rationale

| Knob | Value | Why |
|---|---|---|
| gen_learning_rate | 2e-5 | PeriodWave-Turbo setting; 5× lower than VocaloFlow pretraining (1e-4).  Fine-tuning regime — preserves velocity field. |
| disc_learning_rate | 2e-4 | HiFi-GAN convention; 10× higher than G. |
| betas | (0.8, 0.99) | HiFi-GAN convention.  Lower β₁ for faster adaptation. |
| batch_size | 8 | Reduced from pretraining's 32.  4-step ODE unroll holds 4 forward-pass graphs simultaneously (~4× memory). |
| ode_num_steps / method | 4 / Euler | PeriodWave-Turbo default.  Fixed-step specialisation improves few-step inference. |
| disc_warmup_steps | 3000 | Let the model adapt to fixed-step ODE before the discriminator starts providing signal. |
| adv_ramp_steps | 2000 | Linear ramp of lambda_adv and lambda_fm. |
| total_finetune_steps | 30000 | PeriodWave-Turbo achieves SOTA in 1k–100k; our dataset is small (~200 songs), 30k is a reasonable start. |
| lambda_adv / lambda_fm | 0.1 / 2.0 | Conservative start; tune up if sharpening insufficient, down if artifacts appear. |
| reset_gen_optimizer | True | Pretrained Adam moments were built at 1e-4 LR; at the new 2e-5 LR they push 5× too hard.  Fresh optimizer is safer.  (Deviates from spec; see `plans/today-we-are-running-wobbly-moon.md` §3.) |
| preserve_global_step | True | See EMA warmup caveat below. |

## EMA warmup caveat

The EMA update formula is `effective_decay = min(ema_decay, (step+1)/(step+10))`.
At `step=0` it yields 0.1, which would **wipe the pretrained EMA in a single
update**.  Two solutions:

1. Continue the global step counter from the pretrained run (default,
   `preserve_global_step=True`).  If pretrained step is 55000, the formula
   immediately returns `0.9999`, matching our target decay.
2. Override the formula — not implemented here.

On resume, the step counter is loaded from our own checkpoint (which already
reflects the continued counter), so the invariant holds.

## Reused components

No copies — everything imports live:

| Component | Source |
|---|---|
| `VocaloFlow` model class | `VocaloFlow/model/vocaloflow.py` |
| `FlowMatchingLoss` (CFM anchor) | `VocaloFlow/training/flow_matching.py` (via importlib in `cfm_loss.py`) |
| `sample_ode` (for eval) | `VocaloFlow/inference/inference.py` (via importlib in `evaluate_ft.py`) |
| Dataset / collate / splits | `VocaloFlow/utils/` |
| `get_lr` (for disc LR schedule) | `VocaloFlow/training/lr_schedule.py` |
| `PatchDiscriminator` | `AdversarialPostnet/model/discriminator.py` (via importlib) |
| `hinge_d_loss`, `hinge_g_loss`, `feature_matching_loss`, `masked_l1` | `AdversarialPostnet/training/losses.py` (via importlib) |
| `CropSpec`, `extract_random_crops` | `AdversarialPostnet/training/random_crop.py` (via importlib) |

## Smoke tests and verification

Run in order:

1. `python ode_unroll.py` — verifies ODE correctness and gradient flow.
   Takes ~30s on CPU, seconds on GPU.  All three tests should print
   `max|diff|=0.000e+00` or similar and "All ODE unroll smoke tests passed."

2. `python train_finetune.py --name smoke_cfm_only --config configs/smoke_cfm_only.yaml`
   — runs 200 steps with CFM loss only.  Should behave like resumed
   VocaloFlow training.  Expected velocity loss: same order as the
   pretraining's validation loss at step 55000.

3. `python train_finetune.py --name smoke_cfm_rec --config configs/smoke_cfm_rec.yaml`
   — 500 steps, CFM + reconstruction.  Should show decreasing `loss_rec`
   and stable `loss_cfm`.

4. `python train_finetune.py --name smoke_full --config configs/smoke_full.yaml`
   — 500 steps, full pipeline with shortened warmup (100+100) so the
   adversarial phase actually activates.  Should show non-zero `loss_g_adv`
   and `loss_fm` after step 100, `loss_d` in [0.5, 1.5].

5. `python train_finetune.py --name exp2-v1 --config configs/exp2_v1.yaml` —
   the real experiment (30k steps).

## Inference with a fine-tuned checkpoint

Our checkpoint schema is **cross-compatible with VocaloFlow's inference
pipeline**: we save both a `VocaloFlowConfig` (under the `config` key, so
`VocaloFlow/inference/pipeline.py::load_model` can instantiate the model)
and our `FinetuneConfig` (under `finetune_config`, for our own resume
logic).  The checkpoint also includes `ema_model_state_dict` — VocaloFlow's
loader prefers EMA weights when present, which is what you want for
inference.

Run the normal VocaloFlow pipeline with our checkpoint path:

```bash
cd VocaloFlow/
python -m inference.pipeline \
    --ustx path/to/song.ustx \
    --prior-wav path/to/song_prior.wav \
    --checkpoint ../AdversarialFinetune/checkpoints/exp2-v1/checkpoint_85000.pt \
    --output path/to/song_out.wav
```

Other VocaloFlow CLI flags (`--num-ode-steps`, `--ode-method`, `--cfg-scale`,
`--chunk-size`, `--overlap`) work unchanged.  Defaults — 32 midpoint steps,
cfg_scale=2.0 — match pretraining.  Try `--num-ode-steps 4 --ode-method euler
--cfg-scale 1.0` to hear the training-time setting that the adversarial loss
specifically optimised; comparing the two is part of the distillation-transfer
story in the thesis write-up.

## Success criteria (for thesis)

* WER (via Whisper Large v3 on vocoded output) decreases ≥10% relative to
  the pretrained baseline on the same val songs.
* F0 RMSE (pyworld DIO) increases by <10% — pitch accuracy must not degrade.
* Mel spectrograms show visibly sharper harmonic structure.
* Bonus: `eval/l1_4step_euler` approaches `eval/l1_32step_midpoint` (implicit
  distillation for few-step inference).

## Monitoring warnings

Kill a run and reduce `lambda_adv` if:

* `val/velocity_mse` climbs >20% above its step-0 baseline — velocity field
  destabilising.
* `train/d_real_mean - train/d_fake_mean > 4.0` sustained — discriminator
  overpowering; lower `disc_learning_rate`.
* `train/loss_g_adv` diverges or collapses to 0.

## Not included in this experiment

* No changes to the VocaloFlow architecture.
* No new conditioning signals (e.g., HuBERT).
* No data-pipeline changes.
* No multi-step discriminator (only the final 4-step output is discriminated).
* No vocoder-domain losses (everything in mel space).
