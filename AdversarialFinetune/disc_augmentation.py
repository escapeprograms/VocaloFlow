"""Discriminator input augmentation for adversarial flow-matching fine-tuning.

Implements temporal shift, temporal cutout, and optional frequency cutout.
All ops are differentiable so gradients flow through during the G update.
"""

from __future__ import annotations

import torch
from torch import Tensor


def temporal_shift(mel: Tensor, shift: int) -> Tensor:
    """Circular shift along the time axis."""
    return torch.roll(mel, shifts=shift, dims=1)


def temporal_cutout(mel: Tensor, start: int, width: int) -> Tensor:
    """Zero out a contiguous block of frames."""
    mel = mel.clone()
    mel[:, start : start + width, :] = 0.0
    return mel


def frequency_cutout(mel: Tensor, start: int, width: int) -> Tensor:
    """Zero out a contiguous block of frequency bins."""
    mel = mel.clone()
    mel[:, :, start : start + width] = 0.0
    return mel


def augment_mel(mel: Tensor, config, rng: torch.Generator) -> Tensor:
    """Apply discriminator augmentations using a seeded RNG.

    Caller must seed ``rng`` identically for real and fake mels to ensure
    the same augmentation parameters are applied to both.

    Args:
        mel: (B, T, C) mel spectrogram.
        config: FinetuneConfig with disc_aug_* fields.
        rng: torch.Generator seeded by the caller.
    """
    B, T, C = mel.shape
    p = config.disc_aug_prob

    # Temporal shift
    if torch.rand(1, generator=rng).item() < p:
        shift = torch.randint(
            -config.disc_aug_max_shift, config.disc_aug_max_shift + 1,
            (1,), generator=rng,
        ).item()
        mel = temporal_shift(mel, shift)

    # Temporal cutout
    if torch.rand(1, generator=rng).item() < p:
        width = torch.randint(
            config.disc_aug_cutout_min, config.disc_aug_cutout_max + 1,
            (1,), generator=rng,
        ).item()
        width = min(width, T)
        start = torch.randint(0, max(1, T - width + 1), (1,), generator=rng).item()
        mel = temporal_cutout(mel, start, width)

    # Frequency cutout (optional)
    if config.enable_freq_cutout and torch.rand(1, generator=rng).item() < p:
        fw = torch.randint(
            config.disc_aug_freq_cutout_min, config.disc_aug_freq_cutout_max + 1,
            (1,), generator=rng,
        ).item()
        fw = min(fw, C)
        fs = torch.randint(0, max(1, C - fw + 1), (1,), generator=rng).item()
        mel = frequency_cutout(mel, fs, fw)

    return mel


if __name__ == "__main__":
    print("=== disc_augmentation smoke test ===")

    from dataclasses import dataclass

    @dataclass
    class _FakeConfig:
        disc_aug_prob: float = 1.0
        disc_aug_max_shift: int = 16
        disc_aug_cutout_min: int = 8
        disc_aug_cutout_max: int = 32
        enable_freq_cutout: bool = True
        disc_aug_freq_cutout_min: int = 8
        disc_aug_freq_cutout_max: int = 24

    cfg = _FakeConfig()
    mel = torch.randn(2, 64, 128)

    rng1 = torch.Generator()
    rng1.manual_seed(42)
    out1 = augment_mel(mel, cfg, rng1)

    rng2 = torch.Generator()
    rng2.manual_seed(42)
    out2 = augment_mel(mel, cfg, rng2)

    assert out1.shape == mel.shape, f"Shape mismatch: {out1.shape} vs {mel.shape}"
    assert torch.equal(out1, out2), "Determinism check failed"
    assert not torch.equal(out1, mel), "Augmentation had no effect"
    assert (out1 == 0).any(), "No cutout regions found"

    mel_grad = mel.clone().requires_grad_(True)
    rng3 = torch.Generator()
    rng3.manual_seed(99)
    out3 = augment_mel(mel_grad, cfg, rng3)
    out3.sum().backward()
    assert mel_grad.grad is not None, "Gradient flow broken"

    print("All checks passed.")
