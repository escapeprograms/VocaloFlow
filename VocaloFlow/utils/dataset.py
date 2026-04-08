"""PyTorch Dataset for VocaloFlow training data."""

import os
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils.resample import resample_1d, resample_2d, resolve_phoneme_indirection


class VocaloFlowDataset(Dataset):
    """Loads paired singing voice data for conditional flow matching.

    Handles three key data challenges:
      1. T-mismatch: prior_mel, target_mel, and phoneme_mask may have
         different time dimensions. All are resampled to target_mel's T.
      2. phoneme_mask indirection: phoneme_mask[t] is an index into
         phoneme_ids, not a direct token ID. Resolved during loading.
      3. Variable lengths: Sequences are randomly cropped or zero-padded
         to max_seq_len.

    Args:
        manifest_df: Filtered manifest DataFrame with absolute paths.
        data_dir: Root data directory (used to locate phoneme_ids.npy).
        max_seq_len: Fixed output sequence length.
        training: If True, use random crops; if False, take the start.
    """

    def __init__(
        self,
        manifest_df,
        data_dir: str,
        max_seq_len: int = 256,
        training: bool = True,
    ) -> None:
        self.records = manifest_df.to_dict("records")
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.training = training

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.records[idx]

        # ── Load raw arrays ───────────────────────────────────────────────
        target_mel = np.load(row["target_mel_path"]).astype(np.float32)   # (T_target, 128)
        prior_mel = np.load(row["prior_mel_path"]).astype(np.float32)     # (T_prior, 128)
        f0 = np.load(row["f0_path"]).astype(np.float32)                   # (T_target,)
        voicing = np.load(row["voicing_path"]).astype(np.float32)         # (T_target,)

        phoneme_mask = np.load(row["phoneme_mask_path"]).astype(np.int64) # (T_mask,)

        # Load phoneme_ids from the same chunk directory
        chunk_dir = os.path.dirname(row["phoneme_mask_path"])
        phoneme_ids_path = os.path.join(chunk_dir, "phoneme_ids.npy")
        phoneme_ids = np.load(phoneme_ids_path).astype(np.int64)          # (P,)

        # ── Resolve phoneme indirection ───────────────────────────────────
        # phoneme_mask[t] is an index into phoneme_ids
        resolved_phonemes = resolve_phoneme_indirection(phoneme_ids, phoneme_mask)

        # ── Convert to tensors ────────────────────────────────────────────
        target_mel = torch.from_numpy(target_mel)       # (T_target, 128)
        prior_mel = torch.from_numpy(prior_mel)         # (T_prior, 128)
        f0 = torch.from_numpy(f0)                       # (T_target,)
        voicing = torch.from_numpy(voicing)              # (T_target,)
        resolved_phonemes = torch.from_numpy(resolved_phonemes)  # (T_mask,)

        T_target = target_mel.shape[0]

        # ── Resample all signals to T_target ─────────────────────────────
        prior_mel = resample_2d(prior_mel, T_target, mode="linear")
        resolved_phonemes = resample_1d(resolved_phonemes, T_target, mode="nearest")
        f0 = resample_1d(f0, T_target, mode="linear")
        voicing = resample_1d(voicing, T_target, mode="nearest")

        # ── Crop or pad to max_seq_len ────────────────────────────────────
        length = min(T_target, self.max_seq_len)

        if T_target > self.max_seq_len:
            if self.training:
                start = torch.randint(0, T_target - self.max_seq_len + 1, (1,)).item()
            else:
                start = 0
            target_mel = target_mel[start:start + self.max_seq_len]
            prior_mel = prior_mel[start:start + self.max_seq_len]
            f0 = f0[start:start + self.max_seq_len]
            voicing = voicing[start:start + self.max_seq_len]
            resolved_phonemes = resolved_phonemes[start:start + self.max_seq_len]
        elif T_target < self.max_seq_len:
            pad_len = self.max_seq_len - T_target
            target_mel = F.pad(target_mel, (0, 0, 0, pad_len))
            prior_mel = F.pad(prior_mel, (0, 0, 0, pad_len))
            f0 = F.pad(f0, (0, pad_len))
            voicing = F.pad(voicing, (0, pad_len))
            resolved_phonemes = F.pad(resolved_phonemes, (0, pad_len))

        # ── Padding mask ──────────────────────────────────────────────────
        padding_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        padding_mask[:length] = True

        return {
            "target_mel": target_mel,           # (max_seq_len, 128)
            "prior_mel": prior_mel,             # (max_seq_len, 128)
            "f0": f0,                           # (max_seq_len,)
            "voicing": voicing,                 # (max_seq_len,)
            "phoneme_ids": resolved_phonemes,   # (max_seq_len,)
            "length": length,                   # int
            "padding_mask": padding_mask,        # (max_seq_len,)
        }
