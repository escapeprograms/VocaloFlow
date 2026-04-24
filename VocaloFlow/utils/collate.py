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
    result = {
        "target_mel": torch.stack([item["target_mel"] for item in batch]),
        "prior_mel": torch.stack([item["prior_mel"] for item in batch]),
        "f0": torch.stack([item["f0"] for item in batch]),
        "voicing": torch.stack([item["voicing"] for item in batch]),
        "phoneme_ids": torch.stack([item["phoneme_ids"] for item in batch]),
        "length": torch.tensor([item["length"] for item in batch], dtype=torch.long),
        "padding_mask": torch.stack([item["padding_mask"] for item in batch]),
    }
    if "plbert_features" in batch[0]:
        result["plbert_features"] = torch.stack(
            [item["plbert_features"] for item in batch]
        )
    if "speaker_embedding" in batch[0]:
        result["speaker_embedding"] = torch.stack(
            [item["speaker_embedding"] for item in batch]
        )
    return result


def validate_batch_signals(
    batch: Dict[str, torch.Tensor],
    *,
    expect_plbert: bool = False,
    expect_speaker_embedding: bool = False,
) -> None:
    """Raise loudly if config-expected optional signals are missing from a batch."""
    if expect_plbert and "plbert_features" not in batch:
        raise RuntimeError(
            "use_plbert=True but batch has no 'plbert_features'. "
            "Check that plbert_features.npy exists in each chunk directory."
        )
    if expect_speaker_embedding and "speaker_embedding" not in batch:
        raise RuntimeError(
            "use_speaker_embedding=True but batch has no 'speaker_embedding'. "
            "Check that global_speaker_embedding_path points to a valid .pt file."
        )
