"""Main training entry point for the adversarial post-network.

Usage (from AdversarialPostnet/):
    python -m training.train_postnet \
        --name my-postnet-run \
        --predicted-mel-manifest predicted_mel_manifest.csv

    # Resume:
    python -m training.train_postnet --resume my-postnet-run
"""

import argparse
import dataclasses
import os
import sys
from datetime import datetime

import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------------------------
# Path setup — import from VocaloFlow via importlib (avoids namespace
# collision with AdversarialPostnet's own training/ and utils/ packages)
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ADVERSARIAL_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_ADVERSARIAL_DIR, ".."))
_VOCALOFLOW_DIR = os.path.join(_REPO_ROOT, "VocaloFlow")

import importlib.util as _ilu


def _import_from_path(module_name: str, file_path: str):
    spec = _ilu.spec_from_file_location(module_name, file_path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_lr_mod = _import_from_path(
    "vf_lr_schedule",
    os.path.join(_VOCALOFLOW_DIR, "training", "lr_schedule.py"),
)
get_lr = _lr_mod.get_lr

_dh_mod = _import_from_path(
    "vf_data_helpers",
    os.path.join(_VOCALOFLOW_DIR, "utils", "data_helpers.py"),
)
split_by_song = _dh_mod.split_by_song
split_random = _dh_mod.split_random

# AdversarialPostnet imports
from configs.postnet_config import PostnetConfig
from data.collate import postnet_collate_fn
from data.postnet_dataset import PostnetDataset
from model.discriminator import PatchDiscriminator
from model.postnet import PostNet
from training.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from training.losses import (
    feature_matching_loss,
    hinge_d_loss,
    hinge_g_loss,
    masked_l1,
)
from training.random_crop import CropSpec, extract_random_crops


# ═══════════════════════════════════════════════════════════════════════════
# Lambda warmup
# ═══════════════════════════════════════════════════════════════════════════

def get_effective_lambda_adv(step: int, config: PostnetConfig) -> float:
    """Compute effective adversarial loss weight with warmup + ramp.

    Returns 0 during discriminator warmup, then linearly ramps to
    ``config.lambda_adv`` over ``config.adv_ramp_steps``.
    """
    if step < config.disc_warmup_steps:
        return 0.0
    ramp_progress = (step - config.disc_warmup_steps) / max(1, config.adv_ramp_steps)
    return config.lambda_adv * min(1.0, ramp_progress)


# ═══════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(
    postnet: PostNet,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Run validation and return average metrics."""
    postnet.eval()
    rec_sum = 0.0
    n_batches = 0

    for batch in val_loader:
        predicted_mel = batch["predicted_mel"].to(device)
        target_mel = batch["target_mel"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        sharpened_mel = postnet(predicted_mel)
        l_rec = masked_l1(sharpened_mel, target_mel, padding_mask)
        rec_sum += l_rec.item()
        n_batches += 1

    n = max(1, n_batches)
    return {"loss_rec": rec_sum / n}


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

def train(config: PostnetConfig) -> None:
    """Run adversarial postnet training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Device: {device}")

    # -- Derive paths from run_name ----------------------------------------
    if config.run_name:
        config.checkpoint_dir = os.path.join("./checkpoints", config.run_name)
        config.log_dir = os.path.join("./logs", config.run_name)

    # -- Resume detection --------------------------------------------------
    start_step = 0
    wandb_run_id = None
    resume_ckpt = None

    if config.run_name:
        latest = find_latest_checkpoint(config.checkpoint_dir)
        if latest:
            resume_ckpt = load_checkpoint(latest, device)
            start_step = resume_ckpt["step"]
            wandb_run_id = resume_ckpt.get("wandb_run_id")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Resuming from step {start_step}")

    # -- Logging -----------------------------------------------------------
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    wandb_kwargs = dict(
        entity="archimedesli",
        project="AdversarialPostnet",
        config=vars(config),
        dir=config.log_dir,
    )
    if config.run_name:
        wandb_kwargs["name"] = config.run_name
    if wandb_run_id:
        wandb_kwargs["id"] = wandb_run_id
        wandb_kwargs["resume"] = "must"
    wandb.init(**wandb_kwargs)

    if not config.run_name:
        config.run_name = wandb.run.name
        config.checkpoint_dir = os.path.join("./checkpoints", config.run_name)
        config.log_dir = os.path.join("./logs", config.run_name)
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Save config snapshot
    config_dir = os.path.join("./configs", config.run_name)
    config_path = os.path.join(config_dir, "config.yaml")
    if not os.path.exists(config_path):
        os.makedirs(config_dir, exist_ok=True)
        config.to_yaml(config_path)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved config: {config_path}")

    # -- Data --------------------------------------------------------------
    manifest = pd.read_csv(config.predicted_mel_manifest)

    # The output manifest from generate_predictions already has fully resolved
    # paths for most columns (via load_manifest).  Only predicted_mel_path is
    # stored relative to data_dir — resolve it here.
    if "predicted_mel_path" in manifest.columns:
        manifest["predicted_mel_path"] = manifest["predicted_mel_path"].apply(
            lambda p: os.path.join(config.data_dir, p) if pd.notna(p) else p
        )

    if config.split_mode == "song":
        train_df, val_df = split_by_song(manifest, config.val_fraction, config.seed)
    elif config.split_mode == "random":
        train_df, val_df = split_random(manifest, config.val_fraction, config.seed)
    else:
        raise ValueError(f"Unknown split_mode: {config.split_mode!r}")

    train_ds = PostnetDataset(train_df, config.max_seq_len, training=True)
    val_ds = PostnetDataset(val_df, config.max_seq_len, training=False)

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=postnet_collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=postnet_collate_fn,
    )

    # -- Models ------------------------------------------------------------
    postnet = PostNet(
        mel_channels=128,
        num_blocks=config.postnet_num_blocks,
        kernel_size=config.postnet_kernel_size,
    ).to(device)

    discriminator = PatchDiscriminator(
        channels=config.disc_channels,
    ).to(device)

    pn_params = sum(p.numel() for p in postnet.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
          f"PostNet: {pn_params:,} params, Discriminator: {disc_params:,} params")

    # -- Optimizers --------------------------------------------------------
    opt_g = torch.optim.AdamW(
        postnet.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.weight_decay,
    )
    opt_d = torch.optim.AdamW(
        discriminator.parameters(),
        lr=config.disc_learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.weight_decay,
    )

    # -- Restore state if resuming -----------------------------------------
    if resume_ckpt is not None:
        postnet.load_state_dict(resume_ckpt["postnet_state_dict"])
        discriminator.load_state_dict(resume_ckpt["discriminator_state_dict"])
        opt_g.load_state_dict(resume_ckpt["opt_g_state_dict"])
        opt_d.load_state_dict(resume_ckpt["opt_d_state_dict"])
        del resume_ckpt

    writer = SummaryWriter(log_dir=config.log_dir)
    crop_specs = [CropSpec(t, f) for t, f in config.crop_specs]

    # -- Training loop -----------------------------------------------------
    global_step = start_step
    postnet.train()
    discriminator.train()

    while global_step < config.total_steps:
        for batch in train_loader:
            if global_step >= config.total_steps:
                break

            predicted_mel = batch["predicted_mel"].to(device)
            target_mel = batch["target_mel"].to(device)
            padding_mask = batch["padding_mask"].to(device)

            # LR schedule
            lr_g = get_lr(global_step, config.warmup_steps,
                          config.total_steps, config.learning_rate)
            lr_d = get_lr(global_step, config.warmup_steps,
                          config.total_steps, config.disc_learning_rate)
            for pg in opt_g.param_groups:
                pg["lr"] = lr_g
            for pg in opt_d.param_groups:
                pg["lr"] = lr_d

            # --- PostNet forward ---
            sharpened_mel = postnet(predicted_mel)

            # --- Random crops ---
            crop_pairs = extract_random_crops(
                target_mel, sharpened_mel, crop_specs, padding_mask,
            )

            # --- Discriminator update ---
            d_loss = torch.zeros((), device=device)
            d_real_mean = 0.0
            d_fake_mean = 0.0

            for real_crop, fake_crop in crop_pairs:
                real_scores, _ = discriminator(real_crop)
                fake_scores, _ = discriminator(fake_crop.detach())
                d_loss = d_loss + hinge_d_loss(real_scores, fake_scores)
                d_real_mean += real_scores.mean().item()
                d_fake_mean += fake_scores.mean().item()
            d_loss = d_loss / len(crop_pairs)
            d_real_mean /= len(crop_pairs)
            d_fake_mean /= len(crop_pairs)

            opt_d.zero_grad()
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                discriminator.parameters(), config.grad_clip,
            )
            opt_d.step()

            # --- Generator (PostNet) update ---
            l_rec = masked_l1(sharpened_mel, target_mel, padding_mask)

            eff_lambda = get_effective_lambda_adv(global_step, config)
            g_adv = torch.zeros((), device=device)
            l_fm = torch.zeros((), device=device)

            if eff_lambda > 0:
                for real_crop, fake_crop in crop_pairs:
                    real_scores, real_feats = discriminator(real_crop)
                    fake_scores, fake_feats = discriminator(fake_crop)
                    g_adv = g_adv + hinge_g_loss(fake_scores)
                    l_fm = l_fm + feature_matching_loss(real_feats, fake_feats)
                g_adv = g_adv / len(crop_pairs)
                l_fm = l_fm / len(crop_pairs)

            g_loss = l_rec + eff_lambda * g_adv + config.lambda_fm * l_fm

            opt_g.zero_grad()
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                postnet.parameters(), config.grad_clip,
            )
            opt_g.step()

            # --- Logging ---
            if global_step % config.log_every == 0:
                metrics = {
                    "train/loss_total": g_loss.item(),
                    "train/loss_rec": l_rec.item(),
                    "train/loss_g_adv": g_adv.item(),
                    "train/loss_fm": l_fm.item(),
                    "train/loss_d": d_loss.item(),
                    "train/lambda_adv": eff_lambda,
                    "train/lr": lr_g,
                    "train/d_real_mean": d_real_mean,
                    "train/d_fake_mean": d_fake_mean,
                }
                for k, v in metrics.items():
                    writer.add_scalar(k, v, global_step)
                wandb.log(metrics, step=global_step)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"step={global_step}  rec={l_rec.item():.4f}  "
                      f"g_adv={g_adv.item():.4f}  fm={l_fm.item():.4f}  "
                      f"d={d_loss.item():.4f}  "
                      f"D(r)={d_real_mean:.3f}  D(f)={d_fake_mean:.3f}  "
                      f"λ={eff_lambda:.4f}")

            # --- Validation ---
            if global_step > 0 and global_step % config.val_every == 0:
                val_metrics = validate(postnet, val_loader, device)
                for k, v in val_metrics.items():
                    writer.add_scalar(f"val/{k}", v, global_step)
                wandb.log(
                    {f"val/{k}": v for k, v in val_metrics.items()},
                    step=global_step,
                )
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"step={global_step}  val_rec={val_metrics['loss_rec']:.4f}")
                postnet.train()
                discriminator.train()

            # --- Checkpoint ---
            if global_step > 0 and global_step % config.save_every == 0:
                save_checkpoint(
                    postnet, discriminator, opt_g, opt_d,
                    global_step, config, wandb.run.id,
                )

            global_step += 1

    # Final save
    save_checkpoint(
        postnet, discriminator, opt_g, opt_d,
        global_step, config, wandb.run.id,
    )
    writer.close()
    wandb.finish()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Training complete.")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _load_config_from_checkpoint(run_name: str) -> PostnetConfig:
    """Reconstruct config from the latest checkpoint of *run_name*."""
    checkpoint_dir = os.path.join("./checkpoints", run_name)
    latest = find_latest_checkpoint(checkpoint_dir)
    if latest is None:
        raise FileNotFoundError(
            f"No checkpoint found in {checkpoint_dir} — cannot resume {run_name!r}"
        )
    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
          f"Loading saved config from {latest}")
    ckpt = torch.load(latest, map_location="cpu", weights_only=False)
    saved = ckpt.get("config")
    if saved is None:
        raise ValueError(f"Checkpoint {latest} has no embedded config.")

    fresh = PostnetConfig()
    valid = {f.name for f in dataclasses.fields(fresh)}
    saved_dict = (dataclasses.asdict(saved) if dataclasses.is_dataclass(saved)
                  else vars(saved))
    for k, v in saved_dict.items():
        if k in valid:
            setattr(fresh, k, v)
    return fresh


def _apply_yaml_overrides(config: PostnetConfig, path: str) -> None:
    """Merge a YAML file of partial overrides onto a config."""
    import yaml

    with open(path) as f:
        overrides = yaml.safe_load(f) or {}

    fields_by_name = {f.name: f for f in dataclasses.fields(config)}
    for key, value in overrides.items():
        if key not in fields_by_name:
            raise ValueError(f"Unknown config field in {path}: {key!r}")
        expected_type = fields_by_name[key].type
        if expected_type is float and isinstance(value, int):
            value = float(value)
        setattr(config, key, value)


def main():
    parser = argparse.ArgumentParser(
        description="Train adversarial post-network",
    )
    parser.add_argument("--name", type=str, default=None,
                        help="Run name")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from run name")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config overrides")
    parser.add_argument("--predicted-mel-manifest", type=str, default=None,
                        help="Path to predicted_mel_manifest.csv")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    args = parser.parse_args()

    if args.resume:
        config = _load_config_from_checkpoint(args.resume)
        config.run_name = args.resume
    else:
        config = PostnetConfig()
        if args.name:
            config.run_name = args.name

    if args.config:
        _apply_yaml_overrides(config, args.config)
    if args.predicted_mel_manifest:
        config.predicted_mel_manifest = args.predicted_mel_manifest
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.total_steps:
        config.total_steps = args.total_steps

    if not config.predicted_mel_manifest:
        parser.error("--predicted-mel-manifest is required "
                     "(or set predicted_mel_manifest in config)")

    train(config)


if __name__ == "__main__":
    main()
