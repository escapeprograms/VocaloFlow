"""Checkpoint save / load / discovery for adversarial postnet training."""

import glob
import os
import re
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn

from configs.postnet_config import PostnetConfig


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Return the path to the highest-step checkpoint, or None."""
    if not os.path.isdir(checkpoint_dir):
        return None

    pattern = os.path.join(checkpoint_dir, "checkpoint_*.pt")
    paths = glob.glob(pattern)
    if not paths:
        return None

    def _step(p: str) -> int:
        m = re.search(r"checkpoint_(\d+)\.pt$", p)
        return int(m.group(1)) if m else -1

    return max(paths, key=_step)


def save_checkpoint(
    postnet: nn.Module,
    discriminator: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    step: int,
    config: PostnetConfig,
    wandb_run_id: Optional[str] = None,
) -> str:
    """Save a training checkpoint and return the written path."""
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    path = os.path.join(config.checkpoint_dir, f"checkpoint_{step}.pt")
    torch.save(
        {
            "step": step,
            "postnet_state_dict": postnet.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "opt_g_state_dict": opt_g.state_dict(),
            "opt_d_state_dict": opt_d.state_dict(),
            "config": config,
            "wandb_run_id": wandb_run_id,
        },
        path,
    )
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved checkpoint: {path}")
    return path


def load_checkpoint(path: str, device: torch.device) -> dict:
    """Load a checkpoint dict from disk."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading checkpoint: {path}")
    return torch.load(path, map_location=device, weights_only=False)
