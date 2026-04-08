"""Main training entry point for VocaloFlow."""

import argparse
import copy
import os
import sys
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.default import VocaloFlowConfig
from model.vocaloflow import VocaloFlow
from training.flow_matching import FlowMatchingLoss
from training.lr_schedule import get_lr
from utils.data_helpers import load_manifest, filter_manifest, split_by_song
from utils.dataset import VocaloFlowDataset
from utils.collate import vocaloflow_collate_fn


def train(config: VocaloFlowConfig) -> None:
    """Run VocaloFlow training loop.

    Args:
        config: Training configuration.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    manifest = load_manifest(config.manifest_path, config.data_dir)
    manifest = filter_manifest(manifest, max_dtw_cost=config.max_dtw_cost)
    train_df, val_df = split_by_song(manifest, config.val_fraction, config.seed)

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
    criterion = FlowMatchingLoss(sigma_min=config.sigma_min)

    # ── Logging ───────────────────────────────────────────────────────────
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=config.log_dir)

    # ── Training Loop ─────────────────────────────────────────────────────
    global_step = 0
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
            loss = criterion(model, x_0, x_1, f0, voicing, phoneme_ids, padding_mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            # EMA update
            with torch.no_grad():
                for p_ema, p_model in zip(ema_model.parameters(), model.parameters()):
                    p_ema.lerp_(p_model, 1.0 - config.ema_decay)

            # Logging
            if global_step % config.log_every == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", lr, global_step)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"step={global_step}  loss={loss.item():.4f}  lr={lr:.2e}")

            # Validation
            if global_step > 0 and global_step % config.val_every == 0:
                val_loss = validate(ema_model, val_loader, criterion, device)
                writer.add_scalar("val/loss", val_loss, global_step)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"step={global_step}  val_loss={val_loss:.4f}")
                model.train()

            # Checkpoint
            if global_step > 0 and global_step % config.save_every == 0:
                save_checkpoint(model, ema_model, optimizer, global_step, config)

            global_step += 1

    # Final save
    save_checkpoint(model, ema_model, optimizer, global_step, config)
    writer.close()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Training complete.")


@torch.no_grad()
def validate(
    model: VocaloFlow,
    val_loader: DataLoader,
    criterion: FlowMatchingLoss,
    device: torch.device,
) -> float:
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in val_loader:
        x_0 = batch["prior_mel"].to(device)
        x_1 = batch["target_mel"].to(device)
        f0 = batch["f0"].to(device)
        voicing = batch["voicing"].to(device)
        phoneme_ids = batch["phoneme_ids"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        loss = criterion(model, x_0, x_1, f0, voicing, phoneme_ids, padding_mask)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


def save_checkpoint(
    model: VocaloFlow,
    ema_model: VocaloFlow,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: VocaloFlowConfig,
) -> None:
    """Save training checkpoint."""
    path = os.path.join(config.checkpoint_dir, f"checkpoint_{step}.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "ema_model_state_dict": ema_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }, path)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved checkpoint: {path}")


def main():
    parser = argparse.ArgumentParser(description="Train VocaloFlow")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--max-dtw-cost", type=float, default=None)
    args = parser.parse_args()

    config = VocaloFlowConfig()
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

    train(config)


if __name__ == "__main__":
    main()
