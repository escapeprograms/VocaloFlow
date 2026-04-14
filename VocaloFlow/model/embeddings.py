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
    """Learned embedding table for phoneme token IDs (hard lookup).

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


class BlurredPhonemeEmbedding(PhonemeEmbedding):
    """Phoneme embedding with duration-proportional boundary blending.

    Near phoneme boundaries, produces a weighted average of adjacent phoneme
    embeddings instead of a hard lookup. The blend region scales with the
    shorter adjacent phoneme's duration, so short consonants get small blend
    windows and long vowels get larger ones.

    Args:
        vocab_size: Number of phoneme tokens (default 2820).
        embed_dim: Output embedding dimension (default 64).
        blend_fraction: Fraction of shorter adjacent phoneme's duration used
            as blend radius on each side of the boundary (default 0.3).
    """

    def __init__(
        self,
        vocab_size: int = 2820,
        embed_dim: int = 64,
        blend_fraction: float = 0.3,
    ) -> None:
        super().__init__(vocab_size, embed_dim)
        self.blend_fraction = blend_fraction

    def forward(self, ids: Tensor) -> Tensor:
        """Look up phoneme embeddings with boundary blurring.

        Args:
            ids: (B, T) int64 phoneme token IDs.

        Returns:
            (B, T, embed_dim) blurred embeddings.
        """
        # Base embeddings from parent's table
        emb = self.embedding(ids)  # (B, T, D)

        if self.blend_fraction <= 0.0:
            return emb

        B, T, D = emb.shape
        if T < 2:
            return emb

        # Detect boundaries: positions where the next frame has a different phoneme
        # boundary_mask[:, t] = True means ids[:, t] != ids[:, t+1]
        boundary_mask = ids[:, :-1] != ids[:, 1:]  # (B, T-1)

        # Build blend weights for each sample in the batch
        blend_weights = torch.zeros(B, T, device=ids.device, dtype=emb.dtype)
        # neighbor_ids: for each frame, the phoneme to blend toward
        # (defaults to same as current = no blend)
        neighbor_ids = ids.clone()

        for b in range(B):
            boundary_positions = boundary_mask[b].nonzero(as_tuple=True)[0]  # 1D
            if boundary_positions.numel() == 0:
                continue

            # Compute segment durations from boundary positions
            # Segments are: [0, bp0], [bp0+1, bp1], ..., [bpN+1, T-1]
            bp = boundary_positions
            seg_starts = torch.cat([
                bp.new_zeros(1),
                bp + 1,
            ])
            seg_ends = torch.cat([
                bp + 1,
                bp.new_full((1,), T),
            ])
            seg_durations = seg_ends - seg_starts  # (num_segments,)

            # For each boundary, compute blend radius from adjacent segment durations
            # boundary i is between segment i and segment i+1
            for i, bp_pos in enumerate(boundary_positions):
                left_dur = seg_durations[i].item()
                right_dur = seg_durations[i + 1].item()
                radius = self.blend_fraction * min(left_dur, right_dur)

                if radius < 0.5:
                    continue  # Too small to affect any frame

                radius_int = max(1, int(round(radius)))

                # Left side of boundary: frames in phoneme A blending toward B
                left_start = max(0, bp_pos.item() + 1 - radius_int)
                left_end = bp_pos.item() + 1  # exclusive; bp_pos is last frame of phoneme A
                if left_end > left_start:
                    positions = torch.arange(left_start, left_end, device=ids.device)
                    # Linear ramp: 0 at left_start -> 0.5 at boundary
                    weights = 0.5 * (positions - left_start + 1).float() / radius_int
                    # Clamp to max 0.5
                    weights = weights.clamp(max=0.5)
                    # Only apply if stronger than existing weight at that position
                    existing = blend_weights[b, left_start:left_end]
                    mask = weights > existing
                    blend_weights[b, left_start:left_end] = torch.where(
                        mask, weights, existing
                    )
                    # Neighbor for left-side frames is the right phoneme
                    right_phoneme_id = ids[b, bp_pos + 1]
                    neighbor_ids[b, left_start:left_end] = torch.where(
                        mask, right_phoneme_id, neighbor_ids[b, left_start:left_end]
                    )

                # Right side of boundary: frames in phoneme B blending toward A
                right_start = bp_pos.item() + 1
                right_end = min(T, right_start + radius_int)
                if right_end > right_start:
                    positions = torch.arange(right_start, right_end, device=ids.device)
                    # Linear ramp: 0.5 at boundary -> 0 at right_end
                    weights = 0.5 * (right_end - positions).float() / radius_int
                    weights = weights.clamp(max=0.5)
                    existing = blend_weights[b, right_start:right_end]
                    mask = weights > existing
                    blend_weights[b, right_start:right_end] = torch.where(
                        mask, weights, existing
                    )
                    # Neighbor for right-side frames is the left phoneme
                    left_phoneme_id = ids[b, bp_pos]
                    neighbor_ids[b, right_start:right_end] = torch.where(
                        mask, left_phoneme_id, neighbor_ids[b, right_start:right_end]
                    )

        # Look up neighbor embeddings and blend
        neighbor_emb = self.embedding(neighbor_ids)  # (B, T, D)
        w = blend_weights.unsqueeze(-1)  # (B, T, 1)
        return (1.0 - w) * emb + w * neighbor_emb
