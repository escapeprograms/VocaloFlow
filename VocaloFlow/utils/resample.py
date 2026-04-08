"""Resampling and phoneme resolution utilities shared by dataset and inference."""

import numpy as np
import torch
import torch.nn.functional as F


def resample_1d(arr: torch.Tensor | np.ndarray, target_len: int, mode: str = "linear") -> torch.Tensor:
    """Resample a 1-D signal to *target_len* frames.

    Args:
        arr: 1-D tensor or numpy array.
        target_len: Desired output length.
        mode: ``"linear"`` for continuous signals (F0, voicing),
              ``"nearest"`` for discrete signals (phoneme IDs).

    Returns:
        1-D tensor of length *target_len*, same dtype convention as input
        (float for linear, long/int for nearest when input is integer).
    """
    if isinstance(arr, np.ndarray):
        arr = torch.from_numpy(arr)
    if arr.shape[0] == target_len:
        return arr
    was_int = arr.dtype in (torch.int32, torch.int64, torch.long)
    x = arr.float().unsqueeze(0).unsqueeze(0)  # (1, 1, T)
    kwargs = {"align_corners": False} if mode == "linear" else {}
    x = F.interpolate(x, size=target_len, mode=mode, **kwargs)
    out = x.squeeze()  # (target_len,)
    if was_int or mode == "nearest":
        out = out.long()
    return out


def resample_2d(arr: torch.Tensor | np.ndarray, target_len: int, mode: str = "linear") -> torch.Tensor:
    """Resample a 2-D ``(T, C)`` signal along the time axis.

    Useful for mel-spectrograms of shape ``(T, 128)``.

    Args:
        arr: 2-D tensor or numpy array of shape ``(T, C)``.
        target_len: Desired output time length.
        mode: Interpolation mode (default ``"linear"``).

    Returns:
        Tensor of shape ``(target_len, C)``.
    """
    if isinstance(arr, np.ndarray):
        arr = torch.from_numpy(arr)
    if arr.shape[0] == target_len:
        return arr
    # (T, C) -> (1, C, T) -> interpolate -> (1, C, target_len) -> (target_len, C)
    x = arr.float().T.unsqueeze(0)  # (1, C, T)
    kwargs = {"align_corners": False} if mode == "linear" else {}
    x = F.interpolate(x, size=target_len, mode=mode, **kwargs)
    return x.squeeze(0).T  # (target_len, C)


def resolve_phoneme_indirection(phoneme_ids: np.ndarray, phoneme_mask: np.ndarray) -> np.ndarray:
    """Map frame-level mask indices to actual phoneme token IDs.

    ``phoneme_mask[t]`` is an index into ``phoneme_ids``, not a direct
    token ID.  This function resolves that indirection.

    Args:
        phoneme_ids: 1-D int array of expanded phoneme tokens (length P).
        phoneme_mask: 1-D int array of per-frame indices into *phoneme_ids* (length T).

    Returns:
        1-D int array of resolved per-frame phoneme IDs (length T).
    """
    mask_clipped = np.clip(phoneme_mask, 0, len(phoneme_ids) - 1)
    return phoneme_ids[mask_clipped]
