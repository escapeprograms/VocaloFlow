"""Random area cropping for the mel-spectrogram patch discriminator.

Extracts random rectangular 2D patches from mel spectrograms at multiple
scales (following WeSinger 2).  The same crop coordinates are applied to
both real and fake mels so the discriminator compares matching regions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class CropSpec:
    """Specification for a single crop scale.

    Attributes:
        time_frames: Crop height in time frames.
        mel_bins: Crop width in mel frequency bins.
    """
    time_frames: int
    mel_bins: int


def extract_random_crops(
    real_mel: Tensor,
    fake_mel: Tensor,
    crop_specs: list[CropSpec],
    padding_mask: Tensor | None = None,
) -> list[tuple[Tensor, Tensor]]:
    """Extract multi-scale random crops from real and fake mels.

    For each crop spec, samples random (time_start, freq_start) coordinates
    per batch item and extracts the same region from both tensors.  If the
    mel is shorter than the crop size in either dimension, it is zero-padded.

    Args:
        real_mel:  (B, T, 128) target mel-spectrogram.
        fake_mel:  (B, T, 128) post-network output mel (same shape).
        crop_specs: List of CropSpec defining each scale.
        padding_mask: (B, T) bool, True = valid frame.  Used only to
            determine valid time extent for randomizing crop start.

    Returns:
        List of (real_crop, fake_crop) tuples, one per crop spec.
        Each crop tensor has shape (B, 1, time_frames, mel_bins).
    """
    B, T, n_mels = real_mel.shape
    device = real_mel.device
    results: list[tuple[Tensor, Tensor]] = []

    # Determine per-sample valid time lengths for crop start randomization
    if padding_mask is not None:
        # lengths[i] = number of True entries in padding_mask[i]
        lengths = padding_mask.sum(dim=1)  # (B,)
    else:
        lengths = torch.full((B,), T, device=device)

    for spec in crop_specs:
        ct, cf = spec.time_frames, spec.mel_bins

        # --- Pad mel if shorter than crop size ---
        pad_t = max(0, ct - T)
        pad_f = max(0, cf - n_mels)
        if pad_t > 0 or pad_f > 0:
            # F.pad expects (left, right, top, bottom) for last two dims
            real_padded = F.pad(real_mel, (0, pad_f, 0, pad_t))
            fake_padded = F.pad(fake_mel, (0, pad_f, 0, pad_t))
            T_eff = T + pad_t
            F_eff = n_mels + pad_f
        else:
            real_padded = real_mel
            fake_padded = fake_mel
            T_eff = T
            F_eff = n_mels

        # --- Sample per-item crop coordinates ---
        # Time start: randomize within valid region when possible
        max_t_start = (lengths - ct).clamp(min=0)  # (B,)
        t_starts = (torch.rand(B, device=device) * (max_t_start.float() + 1)).long()
        t_starts = t_starts.clamp(max=T_eff - ct)

        # Frequency start
        max_f_start = F_eff - cf
        if max_f_start > 0:
            f_starts = torch.randint(0, max_f_start + 1, (B,), device=device)
        else:
            f_starts = torch.zeros(B, dtype=torch.long, device=device)

        # --- Extract crops ---
        real_crops = []
        fake_crops = []
        for i in range(B):
            ts = t_starts[i].item()
            fs = f_starts[i].item()
            real_crops.append(real_padded[i, ts:ts + ct, fs:fs + cf])
            fake_crops.append(fake_padded[i, ts:ts + ct, fs:fs + cf])

        # Stack and add channel dim for Conv2d: (B, ct, cf) -> (B, 1, ct, cf)
        real_crop = torch.stack(real_crops).unsqueeze(1)  # (B, 1, ct, cf)
        fake_crop = torch.stack(fake_crops).unsqueeze(1)  # (B, 1, ct, cf)

        results.append((real_crop, fake_crop))

    return results
