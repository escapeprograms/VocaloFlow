"""Rotary Position Embeddings (RoPE) for transformer self-attention."""

import torch
from torch import Tensor


def precompute_freqs_cis(
    dim: int, max_len: int, theta: float = 10000.0
) -> Tensor:
    """Precompute cosine and sine tables for RoPE.

    Args:
        dim: Per-head dimension (must be even).
        max_len: Maximum sequence length.
        theta: Base frequency.

    Returns:
        (max_len, dim) tensor with interleaved [cos, sin] pairs.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)  # (max_len, dim//2)
    return torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)  # (max_len, dim//2, 2)


def apply_rotary_emb(
    q: Tensor, k: Tensor, freqs_cis: Tensor
) -> tuple:
    """Apply rotary position embeddings to query and key tensors.

    Args:
        q: (B, H, T, D) query tensor.
        k: (B, H, T, D) key tensor.
        freqs_cis: (max_len, D//2, 2) precomputed cos/sin.

    Returns:
        Tuple of (q_rotated, k_rotated) with same shapes.
    """
    T = q.shape[2]
    freqs = freqs_cis[:T].to(q.device, dtype=q.dtype)  # (T, D//2, 2)
    cos = freqs[:, :, 0]  # (T, D//2)
    sin = freqs[:, :, 1]  # (T, D//2)

    def rotate(x: Tensor) -> Tensor:
        # x: (B, H, T, D) -> split into pairs
        x1 = x[..., 0::2]  # (B, H, T, D//2)
        x2 = x[..., 1::2]  # (B, H, T, D//2)
        # Apply rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
        r1 = x1 * cos + x2 * (-sin)
        r2 = x1 * sin + x2 * cos
        # Interleave back
        out = torch.stack([r1, r2], dim=-1)  # (B, H, T, D//2, 2)
        return out.flatten(-2)  # (B, H, T, D)

    return rotate(q), rotate(k)
