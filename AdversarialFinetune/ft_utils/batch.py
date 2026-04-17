"""Batch unpacking + a few small logging helpers shared across the module."""

from __future__ import annotations

from datetime import datetime
from typing import NamedTuple

import torch
from torch import Tensor


class BatchTensors(NamedTuple):
    """The six tensors every AdversarialFinetune training / eval step needs.

    Field order matches tuple unpacking throughout the module:
        x_0, x_1, f0, voicing, phoneme_ids, padding_mask = unpack_batch(...)
    """

    x_0: Tensor
    x_1: Tensor
    f0: Tensor
    voicing: Tensor
    phoneme_ids: Tensor
    padding_mask: Tensor


def unpack_batch(batch: dict, device: torch.device) -> BatchTensors:
    """Move the six standard keys of a VocaloFlow batch to ``device``."""
    return BatchTensors(
        x_0=batch["prior_mel"].to(device),
        x_1=batch["target_mel"].to(device),
        f0=batch["f0"].to(device),
        voicing=batch["voicing"].to(device),
        phoneme_ids=batch["phoneme_ids"].to(device),
        padding_mask=batch["padding_mask"].to(device),
    )


def timestamp() -> str:
    """Return ``HH:MM:SS`` for use as a log-line prefix."""
    return datetime.now().strftime("%H:%M:%S")
