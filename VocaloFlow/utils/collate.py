"""Custom collate function for VocaloFlow DataLoader."""

from typing import Dict, List

import torch


def vocaloflow_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack a list of dataset items into batched tensors.

    Args:
        batch: List of dicts from VocaloFlowDataset.__getitem__.

    Returns:
        Dict with batched tensors:
            target_mel:   (B, T, 128)
            prior_mel:    (B, T, 128)
            f0:           (B, T)
            voicing:      (B, T)
            phoneme_ids:  (B, T)
            length:       (B,)
            padding_mask: (B, T)
    """
    return {
        "target_mel": torch.stack([item["target_mel"] for item in batch]),
        "prior_mel": torch.stack([item["prior_mel"] for item in batch]),
        "f0": torch.stack([item["f0"] for item in batch]),
        "voicing": torch.stack([item["voicing"] for item in batch]),
        "phoneme_ids": torch.stack([item["phoneme_ids"] for item in batch]),
        "length": torch.tensor([item["length"] for item in batch], dtype=torch.long),
        "padding_mask": torch.stack([item["padding_mask"] for item in batch]),
    }
