"""Loss functions for adversarial post-network training.

Plain functions (not nn.Module) since they have no learnable parameters.
All operate on tensors and return scalar loss values.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def masked_l1(
    pred: Tensor,
    target: Tensor,
    padding_mask: Tensor | None = None,
) -> Tensor:
    """L1 reconstruction loss, optionally masked for padding.

    Args:
        pred: (B, T, 128) predicted mel.
        target: (B, T, 128) target mel.
        padding_mask: (B, T) bool, True = valid frame.  If None, all valid.

    Returns:
        Scalar L1 loss averaged over valid frames and mel bins.
    """
    diff = (pred - target).abs()  # (B, T, 128)

    if padding_mask is not None:
        mask = padding_mask.unsqueeze(-1).float()  # (B, T, 1)
        return (diff * mask).sum() / (mask.sum() * diff.shape[-1])

    return diff.mean()


def hinge_d_loss(real_scores: Tensor, fake_scores: Tensor) -> Tensor:
    """Hinge loss for the discriminator.

    Args:
        real_scores: D(real) logits, any shape.
        fake_scores: D(fake) logits, any shape.

    Returns:
        Scalar discriminator loss.
    """
    return (F.relu(1.0 - real_scores).mean()
            + F.relu(1.0 + fake_scores).mean())


def hinge_g_loss(fake_scores: Tensor) -> Tensor:
    """Hinge loss for the generator (post-network).

    Args:
        fake_scores: D(fake) logits, any shape.

    Returns:
        Scalar generator adversarial loss.
    """
    return -fake_scores.mean()


def feature_matching_loss(
    real_features: list[Tensor],
    fake_features: list[Tensor],
) -> Tensor:
    """Feature matching loss between discriminator intermediate activations.

    For each layer, computes the L1 distance between real (detached) and
    fake feature maps.  Summed across layers, divided by layer count.

    Args:
        real_features: List of D intermediate activations for real input.
        fake_features: List of D intermediate activations for fake input.

    Returns:
        Scalar feature matching loss.
    """
    loss = torch.zeros((), device=fake_features[0].device)
    for real_feat, fake_feat in zip(real_features, fake_features):
        loss = loss + F.l1_loss(fake_feat, real_feat.detach())
    return loss / len(real_features)
