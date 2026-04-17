"""Thin wrapper around VocaloFlow's FlowMatchingLoss for the fine-tune loop.

Keeps the CFM anchor loss logic in one place and guarantees we never enable the
STFT auxiliary loss here — the adversarial fine-tune replaces STFT auxiliary
signal with the discriminator.
"""

from __future__ import annotations

import os

import torch.nn as nn

from ft_utils import VOCALOFLOW_DIR, import_from_path


# Resolve VocaloFlow/training/flow_matching.py via file path to avoid any
# coupling to sys.path state.
_fm_mod = import_from_path(
    "vf_flow_matching",
    os.path.join(VOCALOFLOW_DIR, "training", "flow_matching.py"),
)
FlowMatchingLoss = _fm_mod.FlowMatchingLoss


def build_cfm_loss(cfg_dropout_prob: float, sigma_min: float = 1.0e-4) -> nn.Module:
    """Return a FlowMatchingLoss with STFT disabled.

    Callers use this as ``criterion(model, x_0, x_1, f0, voicing, phoneme_ids,
    padding_mask)``; it returns ``{"total", "velocity", "stft"}`` with ``stft``
    always zero.  ``total`` equals ``velocity`` for our setup.

    CFG dropout fires internally during ``criterion.training``; call
    ``criterion.eval()`` for validation.
    """
    return FlowMatchingLoss(
        sigma_min=sigma_min,
        cfg_dropout_prob=cfg_dropout_prob,
        stft_loss=None,
        stft_lambda=0.0,
    )
