"""Main training entry point for VocaloFlow."""

import argparse
import copy
import dataclasses
import os
import sys
from datetime import datetime

import torch
import wandb
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.default import VocaloFlowConfig
from model.vocaloflow import VocaloFlow
from training.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from training.flow_matching import FlowMatchingLoss
from training.stft_loss import MultiResolutionSTFTLoss
from training.lr_schedule import get_lr
from utils.data_helpers import load_manifest, filter_manifest, split_by_song, split_random
from utils.dataset import VocaloFlowDataset
from utils.collate import vocaloflow_collate_fn


def train(config: VocaloFlowConfig) -> None:
    """Run VocaloFlow training loop.

    Args:
        config: Training configuration.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Device: {device}")

    # ── Derive paths from run_name (before any I/O) ──────────────────────
    if config.run_name:
        config.checkpoint_dir = os.path.join("./checkpoints", config.run_name)
        config.log_dir = os.path.join("./logs", config.run_name)

    # ── Resume detection ─────────────────────────────────────────────────
    start_step = 0
    wandb_run_id = None
    resume_ckpt = None

    if config.run_name:
        latest = find_latest_checkpoint(config.checkpoint_dir)
        if latest:
            resume_ckpt = load_checkpoint(latest, device)
            start_step = resume_ckpt["step"]
            wandb_run_id = resume_ckpt.get("wandb_run_id")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Resuming from step {start_step}")

    # ── Logging (wandb first so we can adopt its name) ───────────────────
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    wandb_kwargs = dict(
        entity="archimedesli",
        project="VocaloFlow",
        config=vars(config),
        dir=config.log_dir,
    )
    if config.run_name:
        wandb_kwargs["name"] = config.run_name
    if wandb_run_id:
        wandb_kwargs["id"] = wandb_run_id
        wandb_kwargs["resume"] = "must"
    wandb.init(**wandb_kwargs)

    # If no name was given, adopt wandb's auto-generated name
    if not config.run_name:
        config.run_name = wandb.run.name
        config.checkpoint_dir = os.path.join("./checkpoints", config.run_name)
        config.log_dir = os.path.join("./logs", config.run_name)
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Save config snapshot for this run (once, at first launch)
    config_dir = os.path.join("./configs", config.run_name)
    config_path = os.path.join(config_dir, "config.yaml")
    if not os.path.exists(config_path):
        os.makedirs(config_dir, exist_ok=True)
        config.to_yaml(config_path)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved config: {config_path}")

    # ── Data ──────────────────────────────────────────────────────────────
    manifest = load_manifest(config.manifest_path, config.data_dir)
    manifest = filter_manifest(manifest, max_dtw_cost=config.max_dtw_cost)
    if config.split_mode == "random":
        train_df, val_df = split_random(manifest, config.val_fraction, config.seed)
    elif config.split_mode == "song":
        train_df, val_df = split_by_song(manifest, config.val_fraction, config.seed)
    else:
        raise ValueError(f"Unknown split_mode: {config.split_mode!r} (expected 'song' or 'random')")

    train_ds = VocaloFlowDataset(train_df, config.data_dir, config.max_seq_len, training=True)
    val_ds = VocaloFlowDataset(val_df, config.data_dir, config.max_seq_len, training=False)

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=vocaloflow_collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=vocaloflow_collate_fn,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = VocaloFlow(config).to(device)
    ema_model = copy.deepcopy(model)
    ema_model.eval()

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Model parameters: {param_count:,}")

    # ── Optimizer & Loss ──────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    stft_module = None
    if config.stft_loss_enabled:
        stft_module = MultiResolutionSTFTLoss(
            resolutions=tuple(tuple(r) for r in config.stft_resolutions),
        ).to(device)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"STFT aux loss enabled (lambda={config.stft_loss_lambda}, "
              f"resolutions={config.stft_resolutions})")

    criterion = FlowMatchingLoss(
        sigma_min=config.sigma_min,
        cfg_dropout_prob=config.cfg_dropout_prob,
        stft_loss=stft_module,
        stft_lambda=config.stft_loss_lambda if config.stft_loss_enabled else 0.0,
    )

    # ── Restore state if resuming ─────────────────────────────────────────
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state_dict"])
        ema_model.load_state_dict(resume_ckpt["ema_model_state_dict"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        del resume_ckpt  # free memory

    writer = SummaryWriter(log_dir=config.log_dir)
    wandb.watch(model, log="gradients", log_freq=config.log_every * 10)

    # ── Training Loop ─────────────────────────────────────────────────────
    global_step = start_step
    model.train()

    while global_step < config.total_steps:
        for batch in train_loader:
            if global_step >= config.total_steps:
                break

            # Move to device
            x_0 = batch["prior_mel"].to(device)
            x_1 = batch["target_mel"].to(device)
            f0 = batch["f0"].to(device)
            voicing = batch["voicing"].to(device)
            phoneme_ids = batch["phoneme_ids"].to(device)
            padding_mask = batch["padding_mask"].to(device)

            # Update learning rate
            lr = get_lr(global_step, config.warmup_steps, config.total_steps, config.learning_rate)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Forward + backward
            losses = criterion(model, x_0, x_1, f0, voicing, phoneme_ids, padding_mask)
            loss = losses["total"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            # EMA update with warmup so the EMA tracks the live model tightly
            # at the start of training (otherwise val/loss reflects the random
            # init for ~10k steps with decay=0.9999).
            effective_decay = min(
                config.ema_decay,
                (global_step + 1) / (global_step + 10),
            )
            with torch.no_grad():
                for p_ema, p_model in zip(ema_model.parameters(), model.parameters()):
                    p_ema.lerp_(p_model, 1.0 - effective_decay)

            # Logging
            if global_step % config.log_every == 0:
                loss_val = loss.item()
                velocity_val = losses["velocity"].item()
                stft_val = losses["stft"].item()
                writer.add_scalar("train/loss", loss_val, global_step)
                writer.add_scalar("train/loss_velocity", velocity_val, global_step)
                writer.add_scalar("train/loss_stft", stft_val, global_step)
                writer.add_scalar("train/lr", lr, global_step)
                wandb.log(
                    {
                        "train/loss": loss_val,
                        "train/loss_velocity": velocity_val,
                        "train/loss_stft": stft_val,
                        "train/lr": lr,
                    },
                    step=global_step,
                )
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"step={global_step}  loss={loss_val:.4f}  "
                      f"vel={velocity_val:.4f}  stft={stft_val:.4f}  lr={lr:.2e}")

            # Validation
            if global_step > 0 and global_step % config.val_every == 0:
                val_losses = validate(ema_model, val_loader, criterion, device)
                writer.add_scalar("val/loss", val_losses["total"], global_step)
                writer.add_scalar("val/loss_velocity", val_losses["velocity"], global_step)
                writer.add_scalar("val/loss_stft", val_losses["stft"], global_step)
                wandb.log(
                    {
                        "val/loss": val_losses["total"],
                        "val/loss_velocity": val_losses["velocity"],
                        "val/loss_stft": val_losses["stft"],
                    },
                    step=global_step,
                )
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"step={global_step}  val_loss={val_losses['total']:.4f}  "
                      f"val_vel={val_losses['velocity']:.4f}  "
                      f"val_stft={val_losses['stft']:.4f}")
                model.train()

            # Checkpoint
            if global_step > 0 and global_step % config.save_every == 0:
                save_checkpoint(model, ema_model, optimizer, global_step, config, wandb.run.id)

            global_step += 1

    # Final save
    save_checkpoint(model, ema_model, optimizer, global_step, config, wandb.run.id)
    writer.close()
    wandb.finish()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Training complete.")


@torch.no_grad()
def validate(
    model: VocaloFlow,
    val_loader: DataLoader,
    criterion: FlowMatchingLoss,
    device: torch.device,
) -> dict[str, float]:
    """Run validation and return average loss components {total, velocity, stft}."""
    model.eval()
    # Disable CFG dropout during validation — criterion is an nn.Module
    # whose .training flag is independent of model.training.
    criterion.eval()
    sums = {"total": 0.0, "velocity": 0.0, "stft": 0.0}
    n_batches = 0

    try:
        for batch in val_loader:
            x_0 = batch["prior_mel"].to(device)
            x_1 = batch["target_mel"].to(device)
            f0 = batch["f0"].to(device)
            voicing = batch["voicing"].to(device)
            phoneme_ids = batch["phoneme_ids"].to(device)
            padding_mask = batch["padding_mask"].to(device)

            losses = criterion(model, x_0, x_1, f0, voicing, phoneme_ids, padding_mask)
            for k in sums:
                sums[k] += losses[k].item()
            n_batches += 1
    finally:
        criterion.train()

    n = max(1, n_batches)
    return {k: v / n for k, v in sums.items()}


def _load_config_from_checkpoint(run_name: str) -> VocaloFlowConfig:
    """Reconstruct a VocaloFlowConfig from the latest checkpoint of *run_name*.

    Uses the config embedded in the checkpoint as ground truth for architecture
    and training hyperparameters, so resumed runs always match the model shape
    on disk. Fields that exist on the current dataclass but not in the saved
    config (schema additions) keep their defaults.
    """
    checkpoint_dir = os.path.join("./checkpoints", run_name)
    latest = find_latest_checkpoint(checkpoint_dir)
    if latest is None:
        raise FileNotFoundError(
            f"No checkpoint found in {checkpoint_dir} — cannot resume {run_name!r}"
        )
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading saved config from {latest}")
    ckpt = torch.load(latest, map_location="cpu", weights_only=False)
    saved = ckpt.get("config")
    if saved is None:
        raise ValueError(
            f"Checkpoint {latest} has no embedded config — cannot safely resume. "
            f"Pass --config explicitly if you know the original hyperparameters."
        )

    # Merge saved values into a fresh instance so any newly-added dataclass
    # fields pick up their defaults.
    fresh = VocaloFlowConfig()
    valid = {f.name for f in dataclasses.fields(fresh)}
    saved_dict = dataclasses.asdict(saved) if dataclasses.is_dataclass(saved) else vars(saved)
    for k, v in saved_dict.items():
        if k in valid:
            setattr(fresh, k, v)
    return fresh


def _apply_yaml_overrides(config: VocaloFlowConfig, path: str) -> None:
    """Merge a YAML file of partial overrides onto an existing config.

    Validates that every key is a real field of the config and that every
    value matches the field's declared type (with int -> float coercion).
    """
    import yaml

    with open(path) as f:
        overrides = yaml.safe_load(f) or {}

    fields_by_name = {f.name: f for f in dataclasses.fields(config)}
    for key, value in overrides.items():
        if key not in fields_by_name:
            raise ValueError(
                f"Unknown config field in {path}: {key!r}. "
                f"Valid fields: {sorted(fields_by_name)}"
            )
        expected_type = fields_by_name[key].type
        if expected_type is float and isinstance(value, int):
            value = float(value)
        elif not isinstance(value, expected_type):
            raise TypeError(
                f"Config field {key!r} in {path} must be {expected_type.__name__}, "
                f"got {type(value).__name__}: {value!r}"
            )
        setattr(config, key, value)


def main():
    parser = argparse.ArgumentParser(description="Train VocaloFlow")
    parser.add_argument("--name", type=str, default=None,
                        help="Run name (used for checkpoint/log/config subdirs and wandb name)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training for the given run name")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML file of config overrides (applied on top of defaults)")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--max-dtw-cost", type=float, default=None)
    parser.add_argument("--cfg-dropout", type=float, default=None,
                        help="CFG conditioning dropout probability (default from config)")
    parser.add_argument("--split-mode", type=str, default=None, choices=["song", "random"],
                        help="Train/val split strategy: 'song' (default) or 'random' (per-chunk)")
    args = parser.parse_args()

    # Precedence (fresh run):  dataclass defaults < --config YAML < CLI flags
    # Precedence (resume):     saved checkpoint config < --config YAML < CLI flags
    if args.resume:
        config = _load_config_from_checkpoint(args.resume)
        config.run_name = args.resume
    else:
        config = VocaloFlowConfig()
        if args.name:
            config.run_name = args.name

    if args.config:
        _apply_yaml_overrides(config, args.config)
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.manifest:
        config.manifest_path = args.manifest
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.total_steps:
        config.total_steps = args.total_steps
    if args.max_dtw_cost:
        config.max_dtw_cost = args.max_dtw_cost
    if args.cfg_dropout is not None:
        config.cfg_dropout_prob = args.cfg_dropout
    if args.split_mode:
        config.split_mode = args.split_mode

    train(config)


if __name__ == "__main__":
    main()
