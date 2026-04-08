"""Learning rate schedule with linear warmup and cosine decay."""

import math


def get_lr(step: int, warmup_steps: int, total_steps: int, base_lr: float) -> float:
    """Compute learning rate with linear warmup and cosine decay.

    Args:
        step: Current training step.
        warmup_steps: Number of warmup steps.
        total_steps: Total number of training steps.
        base_lr: Peak learning rate after warmup.

    Returns:
        Learning rate for the given step.
    """
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
