# VocaloFlow Training Memory Palace

Training loop, loss computation, and learning rate scheduling for VocaloFlow.

## flow_matching.py

### `FlowMatchingLoss(sigma_min=1e-4, cfg_dropout_prob=0.0, stft_loss=None, stft_lambda=0.0)`
Optimal Transport Conditional Flow Matching loss with classifier-free guidance (CFG) dropout and optional multi-resolution STFT auxiliary loss.

**Formulation**:
- Interpolation: `x_t = (1 - (1-σ)*t) * x_0 + t * x_1` where σ = sigma_min
- Target velocity: `v = x_1 - (1-σ) * x_0` (constant along the OT path)
- Velocity loss: `MSE(v_pred, v_target)` masked by padding_mask
- Optional STFT aux: `stft_loss(x_1_hat, x_1, padding_mask)` where `x_1_hat = x_t.detach() + (1-t)*v_pred` is the one-step output estimate. The `x_t.detach()` is intentional — prevents redundant grad paths through the interpolation.
- Total: `velocity + stft_lambda * stft`

**CFG dropout**: During training, with probability `cfg_dropout_prob`, conditioning signals (f0, voicing, phoneme_ids, and plbert_features if present) are replaced with zeros for randomly selected batch elements. This trains the unconditional path needed for classifier-free guidance at inference. Prior mel (x_0) and x_t are never dropped.

**Important — eval mode**: `FlowMatchingLoss` is an `nn.Module`, so its `.training` flag is independent of the model. Validation must call `criterion.eval()` (not just `model.eval()`) before running the val loop, otherwise CFG dropout will fire during validation and inflate val loss. `validate()` in `train.py` handles this with a `try/finally`.

**Forward**: `forward(model, x_0, x_1, f0, voicing, phoneme_ids, padding_mask, plbert_features=None) -> dict[str, Tensor]`
- Samples `t ~ Uniform(0,1)` per batch element
- Applies CFG dropout (if training and cfg_dropout_prob > 0)
- Constructs x_t from the interpolation formula
- Calls `model(x_t, t, x_0, ...)` to get v_pred
- Computes masked velocity MSE (normalized by valid frame count * mel channels)
- If `stft_loss is not None`, computes STFT loss on `x_1_hat`
- Returns `{"total", "velocity", "stft"}` — callers use `losses["total"]` for backward and log all three components

## stft_loss.py

### `MultiResolutionSTFTLoss(resolutions=((16,4,16),(32,8,32),(64,16,64)), eps=1e-7)`
Auxiliary spectral loss on the mel **time axis**. Treats each of the 128 mel channels as an independent 1-D signal, runs `torch.stft` along time at each `(n_fft, hop, win)` resolution, and averages spectral-convergence + log-magnitude L1 losses. Penalizes mismatches in short-time temporal modulation — targets perceived spectral blur in predicted mels.

- Hann windows are pre-registered as non-persistent buffers (one per resolution) so device/dtype handling is automatic.
- If `padding_mask` is provided, padded frames are zeroed in both tensors before STFT. Errors at the pad boundary cancel because both inputs have the same zeroed regions.
- Skips resolutions where `T < win_size` (cheap size check, no `.item()` sync). Returns a zero tensor if all resolutions are skipped.

## lr_schedule.py

### `get_lr(step, warmup_steps, total_steps, base_lr) -> float`
Linear warmup for `warmup_steps`, then cosine decay to 0 over the remaining steps.

## train.py

### `train(config: VocaloFlowConfig)`
Main training loop. Steps:
1. Derive paths from `config.run_name` — sets `checkpoint_dir` and `log_dir` to `./checkpoints/<run_name>/` and `./logs/<run_name>/`
2. Resume detection — if `run_name` is set and checkpoints exist, loads the latest via `find_latest_checkpoint()` and extracts `start_step` and `wandb_run_id`
3. Initialize wandb (with `name=run_name` if set; with `id`/`resume="must"` if resuming). If no name was given, adopts wandb's auto-generated name
4. Save config snapshot to `configs/<run_name>/config.yaml` (once, skipped on resume)
5. Load and filter manifest (DTW cost ≤ config.max_dtw_cost)
6. Train/val split via `split_by_song` (default) or `split_random` based on `config.split_mode`
7. Create DataLoaders (4 workers, pin_memory, drop_last for train)
8. Initialize VocaloFlow + EMA copy + AdamW optimizer + FlowMatchingLoss
9. Restore model/EMA/optimizer state dicts if resuming, then free checkpoint from memory
10. Initialize TensorBoard SummaryWriter
11. Loop from `start_step`: forward → loss → backward → clip grads (1.0) → step → EMA update (with warmup)
12. Log to TensorBoard + wandb every `log_every` steps, validate every `val_every`, checkpoint every `save_every`

**EMA warmup**: The EMA update uses `effective_decay = min(config.ema_decay, (step+1)/(step+10))` so that early in training the EMA tracks the live model tightly (decay ≈ 0.1 at step 0, ≈ 0.918 at step 100, ≈ 0.991 at step 1000) and only converges to the configured `ema_decay=0.9999` after ~100k steps. **Without this warmup, val/loss would reflect the random init for thousands of steps** because `0.9999^N` decays very slowly — a previously-observed bug where val_loss was stuck near the random-init baseline (~2.0) while train loss dropped to ~0.4.

### `validate(model, val_loader, criterion, device) -> dict[str, float]`
Runs the given model on the val set under `torch.no_grad()` and returns the average per-batch losses as `{"total", "velocity", "stft"}`. Sets `model.eval()` and `criterion.eval()` for the duration of the call (the criterion toggle is critical — see `FlowMatchingLoss` note above), and restores `criterion.train()` in a `try/finally` block. Typically called with the EMA model.

### `main()`
CLI entry point with argparse overrides for name, resume, config, data_dir, manifest, batch_size, lr, total_steps, max_dtw_cost, cfg_dropout, split_mode. `--resume NAME` and `--name NAME` both set `config.run_name`; resume is auto-detected by the presence of existing checkpoints.

**Precedence**: dataclass defaults < `--config` YAML overrides < individual CLI flags.

**Usage**: `python -m training.train --name my-run --config configs/experiments/deep-convnext.yaml`

### `_apply_yaml_overrides(config, path)`
Loads a YAML file of partial overrides and merges them onto an existing config via `setattr`. Validates each key against `dataclasses.fields(config)` and raises `ValueError` on unknown keys (catches typos like `learing_rate`) and `TypeError` on wrong value types (e.g., a string where a float is expected). Any subset of config fields may be specified; unspecified fields keep their default values. Saved `configs/<run_name>/config.yaml` snapshots from past runs are valid override files and can be copied as a starting point for new experiments.

### `_load_config_from_checkpoint(run_name)`
Finds the latest checkpoint under `checkpoints/<run_name>/` and reconstructs a `VocaloFlowConfig` from the `config` field embedded in the checkpoint. Merges saved values into a fresh `VocaloFlowConfig()` so that fields added to the dataclass after the checkpoint was saved pick up their defaults. Raises `FileNotFoundError` if no checkpoint exists and `ValueError` if the checkpoint has no embedded config. Called from `main()` whenever `--resume` is passed — this guarantees the resumed architecture matches what's on disk regardless of current `default.py` values.

## checkpoint.py

Checkpoint save/load/discovery utilities, extracted from train.py for modularity.

### `find_latest_checkpoint(checkpoint_dir) -> Optional[str]`
Globs for `checkpoint_*.pt` in the directory, parses step numbers from filenames, returns the path with the highest step. Returns None if the directory is empty or doesn't exist.

### `save_checkpoint(model, ema_model, optimizer, step, config, wandb_run_id=None) -> str`
Saves `checkpoint_{step}.pt` to `config.checkpoint_dir` with model, EMA, optimizer state dicts, config, and `wandb_run_id`. Returns the written path.

### `load_checkpoint(path, device) -> dict`
Thin wrapper around `torch.load(..., weights_only=False)`. Returns the raw checkpoint dict.

## Logging

Training logs to both TensorBoard (`logs/<run_name>/`) and Weights & Biases (`entity="archimedesli"`, `project="VocaloFlow"`). View TensorBoard with `tensorboard --logdir logs`. Wandb uses the API key from `wandb login`.

Logged scalars:
- `train/loss` (= total), `train/loss_velocity`, `train/loss_stft`, `train/lr` — every `log_every` steps from the live model
- `val/loss` (= total), `val/loss_velocity`, `val/loss_stft` — every `val_every` steps from the EMA model

When `config.stft_loss_enabled=False`, `loss_stft` is logged as 0 and `loss` == `loss_velocity` (zero-overhead disabled path — the STFT module is never instantiated).
