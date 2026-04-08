# VocaloFlow Training Memory Palace

Training loop, loss computation, and learning rate scheduling for VocaloFlow.

## flow_matching.py

### `FlowMatchingLoss(sigma_min=1e-4)`
Optimal Transport Conditional Flow Matching loss.

**Formulation**:
- Interpolation: `x_t = (1 - (1-σ)*t) * x_0 + t * x_1` where σ = sigma_min
- Target velocity: `v = x_1 - (1-σ) * x_0` (constant along the OT path)
- Loss: `MSE(v_pred, v_target)` masked by padding_mask

**Forward**: `forward(model, x_0, x_1, f0, voicing, phoneme_ids, padding_mask) -> loss`
- Samples `t ~ Uniform(0,1)` per batch element
- Constructs x_t from the interpolation formula
- Calls `model(x_t, t, x_0, ...)` to get v_pred
- Returns masked MSE loss (normalized by valid frame count * mel channels)

## lr_schedule.py

### `get_lr(step, warmup_steps, total_steps, base_lr) -> float`
Linear warmup for `warmup_steps`, then cosine decay to 0 over the remaining steps.

## train.py

### `train(config: VocaloFlowConfig)`
Main training loop. Steps:
1. Load and filter manifest (DTW cost ≤ 200)
2. Song-level train/val split (by dali_id, 5% val)
3. Create DataLoaders (4 workers, pin_memory, drop_last for train)
4. Initialize VocaloFlow + EMA copy + AdamW optimizer
5. Loop: forward → loss → backward → clip grads (1.0) → step → EMA update
6. Log to TensorBoard every 50 steps, validate every 2000, checkpoint every 5000

### `validate(model, val_loader, criterion, device) -> float`
Runs EMA model on val set, returns average loss.

### `save_checkpoint(model, ema_model, optimizer, step, config)`
Saves `checkpoint_{step}.pt` with model, EMA, optimizer state dicts and config.

### `main()`
CLI entry point with argparse overrides for data_dir, manifest, batch_size, lr, total_steps, max_dtw_cost.

**Usage**: `python -m training.train --batch-size 16 --lr 5e-5`
