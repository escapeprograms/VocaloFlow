"""Checkpoint save / load / discovery for adversarial fine-tuning.

Named ``ft_checkpoint`` (not ``checkpoint``) to avoid any chance of collision
with ``VocaloFlow/training/checkpoint.py`` if this module and VocaloFlow's
``training`` package ever end up on the same ``sys.path``.
"""

import dataclasses
import glob
import os
import re
from typing import Optional

import torch
import torch.nn as nn

from finetune_config import FinetuneConfig
from ft_utils import timestamp


def extract_checkpoint_step(path: str) -> int:
    """Return the integer step encoded in ``checkpoint_<step>.pt``, or -1."""
    m = re.search(r"checkpoint_(\d+)\.pt$", path)
    return int(m.group(1)) if m else -1


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Return the path to the highest-step checkpoint, or None."""
    if not os.path.isdir(checkpoint_dir):
        return None

    pattern = os.path.join(checkpoint_dir, "checkpoint_*.pt")
    paths = glob.glob(pattern)
    if not paths:
        return None

    return max(paths, key=extract_checkpoint_step)


def save_checkpoint(
    model: nn.Module,
    ema_model: nn.Module,
    discriminator: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    step: int,
    config: FinetuneConfig,
    vf_config,
    wandb_run_id: Optional[str] = None,
) -> str:
    """Save an adversarial-fine-tune checkpoint and return the written path.

    The checkpoint is cross-compatible with VocaloFlow's inference pipeline.
    Two config keys are stored, on purpose, in different forms:

    ``config`` — ``VocaloFlowConfig`` **instance** (not dict).
        ``VocaloFlow/inference/pipeline.py::load_model`` reads this key as an
        object, accessing ``.num_wavenet_blocks`` etc. directly to build the
        model.  A dict would silently break that access.  Unpickling on
        VocaloFlow's sys.path works because ``VocaloFlowConfig`` lives at
        ``configs.default.VocaloFlowConfig``, which is importable from any
        process that has ``VocaloFlow/`` on its path — the only expected
        consumer of this key.  **Do not change to asdict().**

    ``finetune_config`` — plain dict (via ``dataclasses.asdict``).
        ``torch.load`` unpickles the whole checkpoint atomically, so every
        stored object has to be reconstructible in the reader's process.
        Storing as a dict removes the ``finetune_config`` module reference
        that a pickled ``FinetuneConfig`` instance would carry — VocaloFlow's
        inference pipeline doesn't have our module on its sys.path and would
        otherwise crash.  Our own resume logic rebuilds the dataclass with
        ``rebuild_dataclass_tolerant``.

    The ``ema_model_state_dict`` and ``model_state_dict`` keys match
    VocaloFlow's convention; its loader prefers EMA when present.
    """
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    path = os.path.join(config.checkpoint_dir, f"checkpoint_{step}.pt")
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "ema_model_state_dict": ema_model.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "opt_g_state_dict": opt_g.state_dict(),
            "opt_d_state_dict": opt_d.state_dict(),
            # INSTANCE on purpose — VocaloFlow's load_model reads .attributes.  See docstring.
            "config": vf_config,
            # DICT on purpose — portable across sys.path contexts.  See docstring.
            "finetune_config": dataclasses.asdict(config),
            "wandb_run_id": wandb_run_id,
        },
        path,
    )
    print(f"[{timestamp()}] Saved checkpoint: {path}")
    return path


def load_checkpoint(path: str, device: torch.device) -> dict:
    """Load a checkpoint dict from disk."""
    print(f"[{timestamp()}] Loading checkpoint: {path}")
    return torch.load(path, map_location=device, weights_only=False)
