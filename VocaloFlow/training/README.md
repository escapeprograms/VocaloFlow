# VocaloFlow Training Memory Palace

Training loop, loss computation, and learning rate scheduling for VocaloFlow.

## flow_matching.py

### `FlowMatchingLoss(sigma_min=1e-4, cfg_dropout_prob=0.0)`
Optimal Transport Conditional Flow Matching loss with classifier-free guidance (CFG) dropout.

**Formulation**:
- Interpolation: `x_t = (1 - (1-σ)*t) * x_0 + t * x_1` where σ = sigma_min
- Target velocity: `v = x_1 - (1-σ) * x_0` (constant along the OT path)
- Loss: `MSE(v_pred, v_target)` masked by padding_mask

**CFG dropout**: During training, with probability `cfg_dropout_prob`, conditioning signals (f0, voicing, phoneme_ids) are replaced with zeros for randomly selected batch elements. This trains the unconditional path needed for classifier-free guidance at inference. Prior mel (x_0) and x_t are never dropped.

**Important — eval mode**: `FlowMatchingLoss` is an `nn.Module`, so its `.training` flag is independent of the model. Validation must call `criterion.eval()` (not just `model.eval()`) before running the val loop, otherwise CFG dropout will fire during validation and inflate val loss. `validate()` in `train.py` handles this with a `try/finally`.

**Forward**: `forward(model, x_0, x_1, f0, voicing, phoneme_ids, padding_mask) -> loss`
- Samples `t ~ Uniform(0,1)` per batch element
- Applies CFG dropout (if training and cfg_dropout_prob > 0)
- Constructs x_t from the interpolation formula
- Calls `model(x_t, t, x_0, ...)` to get v_pred
- Returns masked MSE loss (normalized by valid frame count * mel channels)

## lr_schedule.py

### `get_lr(step, warmup_steps, total_steps, base_lr) -> float`
Linear warmup for `warmup_steps`, then cosine decay to 0 over the remaining steps.

## train.py

### `train(config: VocaloFlowConfig)`
Main training loop. Steps:
1. Load and filter manifest (DTW cost ≤ config.max_dtw_cost)
2. Train/val split via `split_by_song` (default) or `split_random` based on `config.split_mode`
3. Create DataLoaders (4 workers, pin_memory, drop_last for train)
4. Initialize VocaloFlow + EMA copy + AdamW optimizer
5. Initialize FlowMatchingLoss with cfg_dropout_prob from config
6. Initialize wandb (entity="archimedesli", project="VocaloFlow") + TensorBoard SummaryWriter
7. Loop: forward → loss → backward → clip grads (1.0) → step → EMA update (with warmup)
8. Log to TensorBoard + wandb every `log_every` steps, validate every `val_every`, checkpoint every `save_every`

**EMA warmup**: The EMA update uses `effective_decay = min(config.ema_decay, (step+1)/(step+10))` so that early in training the EMA tracks the live model tightly (decay ≈ 0.1 at step 0, ≈ 0.918 at step 100, ≈ 0.991 at step 1000) and only converges to the configured `ema_decay=0.9999` after ~100k steps. **Without this warmup, val/loss would reflect the random init for thousands of steps** because `0.9999^N` decays very slowly — a previously-observed bug where val_loss was stuck near the random-init baseline (~2.0) while train loss dropped to ~0.4.

### `validate(model, val_loader, criterion, device) -> float`
Runs the given model on the val set under `torch.no_grad()` and returns the average per-batch loss. Sets `model.eval()` and `criterion.eval()` for the duration of the call (the criterion toggle is critical — see `FlowMatchingLoss` note above), and restores `criterion.train()` in a `try/finally` block. Typically called with the EMA model.

### `save_checkpoint(model, ema_model, optimizer, step, config)`
Saves `checkpoint_{step}.pt` with model, EMA, optimizer state dicts and config.

### `main()`
CLI entry point with argparse overrides for data_dir, manifest, batch_size, lr, total_steps, max_dtw_cost, cfg_dropout, split_mode.

**Usage**: `python -m training.train --batch-size 16 --lr 5e-5 --cfg-dropout 0.2 --split-mode song`

## Logging

Training logs to both TensorBoard (`config.log_dir`, default `./logs`) and Weights & Biases (`entity="archimedesli"`, `project="VocaloFlow"`). View TensorBoard with `tensorboard --logdir logs`. Wandb uses the API key from `wandb login`.

Logged scalars:
- `train/loss`, `train/lr` — every `log_every` steps from the live model
- `val/loss` — every `val_every` steps from the EMA model
