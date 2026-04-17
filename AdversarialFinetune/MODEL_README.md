# Adversarial Flow Matching Fine-Tuning: Model Specification

This document describes the adversarial fine-tuning methodology applied to a pretrained VocaloFlow OT-CFM singing voice synthesis model. It is written so that an outsider can understand every design decision that affects model behavior and output quality without reading source code.

---

## 1. Problem Statement

VocaloFlow is a ~51M-parameter conditional flow matching model (WaveNet backbone with DiT blocks using AdaLN-Zero) that transforms Vocaloid-synthesized mel-spectrograms into human-quality mel-spectrograms. After pretraining with velocity MSE, the model produces over-smoothed output: blurred formants, mumbled consonants, and poor intelligibility. A prior experiment (AdversarialPostnet) showed that post-processing cannot recover this detail because the missing phonetic information never enters the generated mel in the first place. This experiment pushes adversarial feedback into the generative model itself during flow matching.

## 2. Base Model: VocaloFlow OT-CFM

### 2.1 Flow Matching Formulation

VocaloFlow uses Optimal Transport Conditional Flow Matching (OT-CFM). The interpolation path is:

```
x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1
```

where `x_0` is the Vocaloid prior mel, `x_1` is the human target mel, `t ~ U[0,1]`, and `sigma_min = 1e-4`.

The velocity target (constant along the OT path) is:

```
v_target = x_1 - (1 - sigma_min) * x_0
```

The model `v_theta(x_t, t, x_0, f0, voicing, phoneme_ids)` is trained to predict this velocity field. At inference, an ODE integrator evolves `x_0` from `t=0` to `t=1` by querying the learned velocity field at each step.

### 2.2 Conditioning

The model receives:
- `x_0`: (B, T, 128) Vocaloid prior mel-spectrogram (also the ODE initial state)
- `f0`: (B, T) fundamental frequency in Hz
- `voicing`: (B, T) binary voicing flag (0 or 1, stored as float)
- `phoneme_ids`: (B, T) per-frame phoneme token IDs (resolved from an indirection table; see Section 7.2)
- `padding_mask`: (B, T) boolean, True = valid frame

### 2.3 Classifier-Free Guidance (CFG)

During training, CFG dropout randomly zeros out `f0`, `voicing`, and `phoneme_ids` for 20% of samples in each batch (per-sample, not per-frame). At inference, the velocity is blended:

```
v = v_uncond + cfg_scale * (v_cond - v_uncond)
```

Default inference `cfg_scale = 2.0`.

### 2.4 Architecture Summary

- WaveNet blocks (dilated causal convolutions) + DiT blocks with AdaLN-Zero conditioning
- AdaLN-Zero gates attention and FFN residuals via a modulation vector initialized to zero. In a fresh (random) model, this means QKV/FFN weights receive zero gradient; in the pretrained model, the modulation is trained away from zero and all parameters receive gradient normally.

## 3. Fine-Tuning Methodology

### 3.1 Overview

The fine-tuning combines four loss signals:

1. **CFM velocity loss** (anchor): same MSE objective as pretraining, preserving the learned velocity field
2. **Reconstruction loss**: L1 between the 4-step ODE output and the target mel
3. **Adversarial loss**: hinge GAN loss from a patch discriminator on random mel crops
4. **Feature matching loss**: L1 on discriminator intermediate activations

This follows the PeriodWave-Turbo recipe (arXiv:2408.08019) adapted for mel-domain SVS.

### 3.2 Training Step Ordering

Each training step proceeds as follows:

1. **CFM phase**: Sample random `t ~ U[0,1]`, interpolate `x_t`, predict velocity, compute MSE against `v_target`. CFG dropout (20%) fires internally during this phase. This is the *only* phase where CFG dropout is active.

2. **ODE unroll phase**: Integrate the velocity field from `t=0` to `t=1` in 4 Euler steps (dt=0.25) to produce `x_1_hat`. This is differentiable (no `@torch.no_grad`), so gradients from downstream losses flow back through all 4 forward passes into the model weights. Full conditioning is always used (no CFG dropout) because CFG dropout during ODE integration would produce incoherent intermediate states.

3. **Random cropping**: Extract multi-scale 2D crops from both target and predicted mels at 3 scales: 32x64, 64x128, 128x128 (time_frames x mel_bins). Same random coordinates used for real and fake. Crops are extracted per batch item; if the mel is shorter than the crop in either dimension, it is zero-padded. This is the WeSinger 2 random area cropping approach.

4. **Discriminator update**: For each crop scale, compute hinge discriminator loss on real vs detached fake crops. One optimizer step on the discriminator.

5. **Generator update**: Sum all generator losses with their effective weights:
   ```
   L_total = lambda_cfm * L_cfm
            + lambda_rec * L_rec
            + eff_lambda_adv * L_adv
            + eff_lambda_fm * L_fm
   ```
   One optimizer step on the generator (the VocaloFlow model).

6. **EMA update**: Update exponential moving average of model weights.

### 3.3 Differentiable ODE Unrolling

The ODE unroll replicates the inference-time integration but with gradients enabled:

```python
dt = 1.0 / num_steps  # 0.25 for 4 steps
x_t = x_0
for i in range(num_steps):
    t = i * dt
    v = model(x_t, t, x_0, f0, voicing, phoneme_ids, padding_mask)
    x_t = x_t + dt * v
return x_t  # this is x_1_hat
```

Key differences from inference:
- **No `@torch.no_grad`**: the full computation graph across all N steps is retained
- **No CFG**: always uses full conditioning (cfg_scale=1.0 equivalent)
- **No diagnostics**: no velocity norm tracking
- **Optional gradient checkpointing**: `torch.utils.checkpoint` can wrap each model call to halve peak activation memory at ~33% forward recompute cost

The output `x_1_hat` feeds the reconstruction, adversarial, and feature matching losses. Backpropagation through `x_1_hat` propagates gradients through every model forward pass in the integration chain.

### 3.4 Loss Functions

**CFM Velocity Loss** (`loss_cfm`):
Masked MSE between predicted and target velocity, identical to pretraining. Masking uses the padding mask to exclude padded frames.

**Reconstruction Loss** (`loss_rec`):
Masked L1 between `x_1_hat` (4-step Euler output) and `x_1` (target mel). Masking excludes padded frames.

**Hinge Adversarial Loss** (`loss_g_adv`):
```
D loss = mean(relu(1 - D(real))) + mean(relu(1 + D(fake_detached)))
G loss = -mean(D(fake))  # fake NOT detached for generator update
```
Computed on each crop scale and averaged.

**Feature Matching Loss** (`loss_fm`):
L1 between discriminator intermediate activations (post-LeakyReLU) at each layer, averaged across layers and crop scales. Real activations are detached (no gradient flows through the discriminator from this loss).

### 3.5 Warmup Schedule

The adversarial and feature matching losses are gated by a two-phase warmup schedule based on `ft_step` (steps elapsed since fine-tuning began, not global step):

- **Phase 1** (`ft_step < disc_warmup_steps = 3000`): `eff_lambda_adv = 0`, `eff_lambda_fm = 0`. Only CFM and reconstruction losses are active. The discriminator still trains (sees real/fake crops) but its signal does not flow into the generator.
- **Phase 2** (`ft_step in [3000, 5000)`): Linear ramp from 0 to `lambda_adv` / `lambda_fm` over `adv_ramp_steps = 2000` steps.
- **Phase 3** (`ft_step >= 5000`): Full `lambda_adv` and `lambda_fm`.

Formula:
```
if ft_step < disc_warmup_steps:
    eff_lambda = 0
else:
    progress = (ft_step - disc_warmup_steps) / adv_ramp_steps
    eff_lambda = base_lambda * min(1.0, progress)
```

### 3.6 EMA (Exponential Moving Average)

```
effective_decay = min(ema_decay, (global_step + 1) / (global_step + 10))
p_ema = effective_decay * p_ema + (1 - effective_decay) * p_model
```

**Critical**: At `global_step=0`, this formula gives `decay = 0.1`, which would **destroy the pretrained EMA in one update**. The system avoids this by continuing the global step counter from the pretrained checkpoint (e.g., 55000), where `decay = min(0.9999, 55001/55010) = 0.9999`. The `preserve_global_step=True` default enforces this.

### 3.7 Discriminator Architecture

Reuses `PatchDiscriminator` from AdversarialPostnet (fresh random weights, not pretrained):
- 4 downsampling layers: Conv2d(kernel=4, stride=2) with weight normalization and LeakyReLU(0.2)
- 1 output layer: Conv2d(kernel=3, stride=1) producing raw logits
- Channel progression: `[32, 64, 128, 256]` (configurable via `disc_channels`)
- Returns both final logits and list of intermediate activations for feature matching
- Input: (B, 1, T_crop, F_crop) mel crop (unsqueezed channel dim)

## 4. Optimizer Configuration

### 4.1 Generator Optimizer

- **AdamW** with lr=2e-5 (constant, no scheduler), betas=(0.8, 0.99), weight_decay=0.01
- **Fresh optimizer state** (pretrained Adam moments discarded). Rationale: the pretrained optimizer was trained at lr=1e-4; its accumulated momentum would push 5x harder than appropriate at the new lr=2e-5. `reset_gen_optimizer=True` is the default.
- Gradient clipping: max_norm=1.0

### 4.2 Discriminator Optimizer

- **AdamW** with lr=2e-4, betas=(0.8, 0.99), weight_decay=0.01
- LR schedule: linear warmup over `disc_lr_warmup_steps=1000` steps, then cosine decay to 0 over `total_finetune_steps`
- Gradient clipping: max_norm=1.0

## 5. Hyperparameter Table

| Parameter | Value | Rationale |
|---|---|---|
| `lambda_cfm` | 1.0 | Velocity anchor prevents catastrophic forgetting of the flow field |
| `lambda_rec` | 1.0 | L1 on ODE output; drives mel-level fidelity |
| `lambda_adv` | 0.1 | Low relative to rec/cfm; adversarial signal is a sharpening nudge, not the primary driver |
| `lambda_fm` | 2.0 | Feature matching stabilizes GAN training; higher weight than adversarial per PeriodWave-Turbo |
| `gen_learning_rate` | 2e-5 | 5x lower than pretraining lr to limit drift from pretrained weights |
| `disc_learning_rate` | 2e-4 | 10x higher than generator; discriminator must learn faster to provide useful signal |
| `batch_size` | 8 | Reduced from pretraining (32) due to 4x memory from ODE unroll holding 4 forward-pass graphs |
| `ode_num_steps` | 4 | Matches the intended few-step inference setting; more steps = more memory + compute |
| `ode_method` | euler | Simpler, cheaper; midpoint available but doubles model calls per step |
| `disc_warmup_steps` | 3000 | Let CFM+rec stabilize before introducing adversarial signal |
| `adv_ramp_steps` | 2000 | Gradual ramp avoids shocking the generator with sudden adversarial gradient |
| `ema_decay` | 0.9999 | Standard for fine-tuning; EMA model is used for all evaluation and inference |
| `max_seq_len` | 256 | ~6.5 seconds at 256 mel frames (hop_size=512, sr=22050) |
| `sigma_min` | 1e-4 | OT-CFM noise floor; identical to pretraining |
| `cfg_dropout_prob` | 0.2 | 20% unconditional samples for CFG; identical to pretraining |
| `grad_clip` | 1.0 | Both generator and discriminator |
| `total_finetune_steps` | 30000 | Added on top of the pretrained global step |

## 6. Training-Time vs Inference-Time ODE

| Setting | Steps | Method | CFG | Used for |
|---|---|---|---|---|
| Training ODE unroll | 4 | Euler | None (full conditioning) | Producing x_1_hat for rec/adv/fm losses |
| Inference (default) | 32 | Midpoint | cfg_scale=2.0 | Final output generation |
| Eval (training-time check) | 4 | Euler | cfg_scale=1.0 | Monitoring distillation transfer |
| Eval (inference-time check) | 32 | Midpoint | cfg_scale=1.0 | Primary quality metric |

A key risk is that adversarial signal shaped by the 4-step Euler output may not transfer to the 32-step midpoint inference output. The evaluation tracks both settings to detect this gap. If `eval/l1_ratio` (4-step / 32-step) grows significantly above 1.0, the adversarial signal is not transferring.

## 7. Data Pipeline

### 7.1 Dataset

Uses `VocaloFlowDataset` from VocaloFlow, identical to pretraining. Each sample contains:
- `target_mel`: (256, 128) human mel-spectrogram (the learning target)
- `prior_mel`: (256, 128) Vocaloid-synthesized mel-spectrogram (the conditioning input / ODE initial state)
- `f0`: (256,) fundamental frequency in Hz
- `voicing`: (256,) binary voicing flag
- `phoneme_ids`: (256,) per-frame resolved phoneme token IDs
- `padding_mask`: (256,) boolean mask
- `length`: actual valid length before padding

### 7.2 Sequence Length Handling (affects results)

All sequences are fixed to `max_seq_len=256` frames:
- **Longer sequences**: randomly cropped to 256 contiguous frames during training; cropped from the start during validation. This means the model never sees full songs, only ~6.5s windows.
- **Shorter sequences**: zero-padded to 256 frames. Padded frames are excluded from all losses via `padding_mask`.

This is **padding, not resampling** — temporal resolution is preserved. A frame always represents the same physical duration (hop_size/sr seconds).

### 7.3 T-Mismatch Handling (affects results)

Prior mel, target mel, f0, voicing, and phoneme mask may have slightly different time dimensions after DTW alignment (off by a few frames). All signals are padded or truncated to match `target_mel`'s length *before* the max_seq_len crop/pad. This means:
- If prior_mel is shorter than target_mel, trailing frames are zero-padded
- If prior_mel is longer, trailing frames are truncated

### 7.4 Phoneme Indirection

Raw data stores `phoneme_mask[t]` as an index into a separate `phoneme_ids` array (not a direct token ID). The dataset resolves this indirection at load time via `resolve_phoneme_indirection`, producing per-frame token IDs.

### 7.5 Data Filtering

Manifest entries are filtered by `max_dtw_cost=100.0` — pairs with poor DTW alignment (high warping cost) are excluded. Train/val split is by song (`split_mode="song"`, `val_fraction=0.05`) to prevent data leakage between song segments.

## 8. Checkpoint Schema

Each checkpoint stores:

| Key | Type | Purpose |
|---|---|---|
| `step` | int | Global step counter (continues from pretrained checkpoint) |
| `model_state_dict` | dict | Live model weights |
| `ema_model_state_dict` | dict | EMA model weights (used for inference) |
| `discriminator_state_dict` | dict | Discriminator weights |
| `opt_g_state_dict` | dict | Generator optimizer state |
| `opt_d_state_dict` | dict | Discriminator optimizer state |
| `config` | VocaloFlowConfig **instance** | Architecture config as a pickled object |
| `finetune_config` | dict | FinetuneConfig as a plain dict via `dataclasses.asdict()` |
| `wandb_run_id` | str | For resuming wandb logging |

### 8.1 Dual-Config Asymmetry (critical for portability)

The `config` key stores a `VocaloFlowConfig` as a **pickled class instance** (not a dict). VocaloFlow's inference pipeline reads `.num_wavenet_blocks`, `.hidden_dim`, etc. as attributes directly. Converting to dict would silently break attribute access.

The `finetune_config` key stores `FinetuneConfig` as a **plain dict** via `dataclasses.asdict()`. This is necessary because `torch.load` unpickles the entire checkpoint atomically — if `FinetuneConfig` were stored as a class instance, `torch.load` would try to resolve the `finetune_config` module on the reader's sys.path. VocaloFlow's inference pipeline does not have `AdversarialFinetune/` on its sys.path, so this would crash with `ModuleNotFoundError`.

### 8.2 Inference Compatibility

Fine-tuned checkpoints are directly compatible with VocaloFlow's inference pipeline:
```
cd VocaloFlow/
python -m inference.pipeline --checkpoint ../AdversarialFinetune/checkpoints/<run>/checkpoint_<step>.pt ...
```
The pipeline loads the `config` key (VocaloFlowConfig instance) to build the model architecture, then loads `ema_model_state_dict` for weights. The `finetune_config` key (plain dict) is ignored but does not cause errors.

## 9. Evaluation Metrics

### 9.1 Per val_every (1000 steps) — Cheap

- `val/velocity_mse`: CFM velocity MSE on the full val set (EMA model). Should remain stable; >20% increase signals velocity field destabilization.
- `val/l1_reconstruction_4step`: L1 on 4-step Euler output vs target mel on full val set.

### 9.2 Per eval_every (3000 steps) — Expensive

- `eval/l1_32step_midpoint`: L1 at inference setting (32-step midpoint, cfg_scale=1.0)
- `eval/l1_4step_euler`: L1 at training setting (4-step Euler, cfg_scale=1.0)
- `eval/l1_ratio`: 4-step / 32-step. Expected >= 1.0. Growing ratio = poor transfer.
- 4-panel mel comparison plots: prior | target | 32-step pred | 4-step pred (saved to disk and wandb)

### 9.3 Post-Run (manual, final checkpoint)

- Vocode with Vocos, compute Whisper WER against lyrics
- pyworld DIO F0 extraction, F0 RMSE against conditioning F0
- A/B listening comparison: pretrained vs fine-tuned

## 10. Expected Loss Dynamics

- **loss_cfm** (~0.001 scale): noisy at batch_size=8, should stay roughly stable. Any sustained upward trend signals velocity field damage.
- **loss_rec** (~0.025 scale): noisy, should not increase. Roughly 25x larger than loss_cfm because it measures L1 mel error vs MSE velocity error on different scales.
- **loss_d**: should settle to [0.5, 1.5] range once the discriminator is trained. Values near 0 = discriminator overpowering.
- **loss_g_adv**: emerges at ft_step=3000, ramps over 2000 steps.
- **loss_fm**: follows the same ramp schedule as loss_g_adv.
- **D(real), D(fake)**: gap of 1-3 is healthy; gap > 4 sustained = discriminator too strong.

## 11. Implementation Details That Affect Results

1. **Mel representation**: 128-bin log-mel spectrograms, sr=22050, hop_size=512, computed upstream (not by this system).

2. **Random crop zero-padding**: If a mel crop region extends beyond valid frames (due to padding_mask or sequence boundary), it is zero-padded. The discriminator sees these zero regions. This could bias the discriminator at sequence boundaries.

3. **CFG dropout only in CFM phase**: The ODE unroll always uses full conditioning. If CFG dropout were applied during ODE integration, intermediate states would be incoherent (half-conditioned, half-unconditioned velocity steps). This means the adversarial/reconstruction losses only see fully-conditioned outputs, while the CFM loss sees both conditioned and unconditioned.

4. **Discriminator trains during warmup**: Even during Phase 1 (ft_step < 3000), the discriminator receives optimizer steps on real/fake crops. It just doesn't influence the generator. This means the discriminator is partially trained by the time its signal starts flowing to the generator.

5. **Same crop coordinates for D and G**: The `extract_random_crops` call happens once per step; both the discriminator update (on detached fakes) and the generator adversarial loss (on non-detached fakes) use the same crop coordinates. The discriminator does re-forward the real crops during the generator step (for feature matching), but with the same coordinates.

6. **Global step for x-axis**: wandb and TensorBoard use `global_step` (continuing from pretrained, e.g., 55000+), not `ft_step`. This makes the fine-tune x-axis continuous with the pretraining run.

7. **Train loader `drop_last=True`**: The last incomplete batch of each epoch is dropped to avoid variable batch sizes affecting loss scaling.

8. **Validation uses start-crop, not random-crop**: Val samples take frames `[0:256]`, not a random window. This is deterministic across evaluations.

9. **All evaluation uses the EMA model**: Both cheap validation and expensive eval run on `ema_model`, never the live `model`.

10. **Feature matching detaches real**: `feature_matching_loss(real_feats, fake_feats)` detaches the real activations internally. Gradient only flows through the fake path into the generator, not through real into the discriminator.
