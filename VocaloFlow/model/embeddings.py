"""Timestep and phoneme embedding modules for VocaloFlow."""

import math

import torch
import torch.nn as nn
from torch import Tensor


def sinusoidal_timestep_embedding(t: Tensor, dim: int, max_period: float = 10000.0) -> Tensor:
    """Sinusoidal positional embedding for continuous timestep t in [0, 1].

    Args:
        t: (B,) tensor of timestep values.
        dim: Embedding dimension (must be even).
        max_period: Controls the frequency range.

    Returns:
        (B, dim) embedding tensor.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimestepMLP(nn.Module):
    """Sinusoidal embedding -> 2-layer MLP with SiLU -> conditioning vector.

    Args:
        hidden_dim: Output dimension (default 1024).
        sinusoidal_dim: Dimension of the sinusoidal embedding (default 256).
    """

    def __init__(self, hidden_dim: int = 1024, sinusoidal_dim: int = 256) -> None:
        super().__init__()
        self.sinusoidal_dim = sinusoidal_dim
        self.mlp = nn.Sequential(
            nn.Linear(sinusoidal_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        """Map timestep t to conditioning vector.

        Args:
            t: (B,) timestep in [0, 1].

        Returns:
            (B, hidden_dim) conditioning vector.
        """
        emb = sinusoidal_timestep_embedding(t, self.sinusoidal_dim)
        return self.mlp(emb)


class F0Embedding(nn.Module):
    """Learned embedding for continuous F0 values.

    Small MLP that projects scalar F0 (in Hz) to a dense representation,
    giving the model a richer pitch signal than a single raw channel.

    Args:
        embed_dim: Output embedding dimension (default 64).
    """

    def __init__(self, embed_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, f0: Tensor) -> Tensor:
        """Embed continuous F0 values.

        Args:
            f0: (B, T) F0 contour in Hz.

        Returns:
            (B, T, embed_dim) dense F0 embedding.
        """
        return self.mlp(f0.unsqueeze(-1))


class PhonemeEmbedding(nn.Module):
    """Learned embedding table for phoneme token IDs.

    Args:
        vocab_size: Number of phoneme tokens (default 2820).
        embed_dim: Output embedding dimension (default 64).
    """

    def __init__(self, vocab_size: int = 2820, embed_dim: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

    def forward(self, ids: Tensor) -> Tensor:
        """Look up phoneme embeddings.

        Args:
            ids: (B, T) int64 phoneme token IDs.

        Returns:
            (B, T, embed_dim) dense embeddings.
        """
        return self.embedding(ids)
