"""Evaluation utilities for VocaloFlow inference quality."""

import torch
from torch import Tensor


def mel_mse(pred: Tensor, target: Tensor, padding_mask: Tensor | None = None) -> float:
    """Compute MSE between predicted and target mel-spectrograms.

    Args:
        pred: (B, T, 128) predicted mel.
        target: (B, T, 128) ground-truth mel.
        padding_mask: (B, T) bool, True = valid frame.

    Returns:
        Scalar MSE value.
    """
    diff = (pred - target) ** 2
    if padding_mask is not None:
        mask = padding_mask.unsqueeze(-1).float()
        return (diff * mask).sum().item() / (mask.sum().item() * diff.shape[-1])
    return diff.mean().item()


def mel_mae(pred: Tensor, target: Tensor, padding_mask: Tensor | None = None) -> float:
    """Compute MAE between predicted and target mel-spectrograms.

    Args:
        pred: (B, T, 128) predicted mel.
        target: (B, T, 128) ground-truth mel.
        padding_mask: (B, T) bool, True = valid frame.

    Returns:
        Scalar MAE value.
    """
    diff = (pred - target).abs()
    if padding_mask is not None:
        mask = padding_mask.unsqueeze(-1).float()
        return (diff * mask).sum().item() / (mask.sum().item() * diff.shape[-1])
    return diff.mean().item()
