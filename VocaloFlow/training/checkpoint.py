"""Checkpoint save / load / discovery utilities for VocaloFlow."""

import glob
import os
import re
from datetime import datetime
from typing import Optional

import torch

from configs.default import VocaloFlowConfig
from model.vocaloflow import VocaloFlow


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Return the path to the highest-step checkpoint in *checkpoint_dir*, or None."""
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
    model: VocaloFlow,
    ema_model: VocaloFlow,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: VocaloFlowConfig,
    wandb_run_id: Optional[str] = None,
) -> str:
    """Save a training checkpoint and return the written path."""
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    path = os.path.join(config.checkpoint_dir, f"checkpoint_{step}.pt")
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "ema_model_state_dict": ema_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
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
