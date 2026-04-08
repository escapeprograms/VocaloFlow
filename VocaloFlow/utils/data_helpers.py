"""Manifest loading, quality filtering, and train/val splitting utilities."""

import os
from typing import Tuple

import numpy as np
import pandas as pd


def load_manifest(manifest_path: str, data_dir: str) -> pd.DataFrame:
    """Load manifest.csv and resolve relative paths to absolute.

    Args:
        manifest_path: Path to manifest.csv.
        data_dir: Root directory containing the per-chunk subdirectories.

    Returns:
        DataFrame with absolute paths for all *_path columns.
    """
    df = pd.read_csv(manifest_path)

    path_cols = [c for c in df.columns if c.endswith("_path")]
    for col in path_cols:
        df[col] = df[col].apply(lambda p: os.path.join(data_dir, p) if pd.notna(p) else p)

    return df


def filter_manifest(df: pd.DataFrame, max_dtw_cost: float = 200.0) -> pd.DataFrame:
    """Filter chunks by DTW alignment quality.

    Args:
        df: Manifest DataFrame.
        max_dtw_cost: Maximum DTW cost to include (default 200).

    Returns:
        Filtered DataFrame.
    """
    mask = df["dtw_cost"].notna() & (df["dtw_cost"] <= max_dtw_cost)
    filtered = df[mask].reset_index(drop=True)
    print(f"[data_helpers] Filtered {len(df)} -> {len(filtered)} chunks "
          f"(max_dtw_cost={max_dtw_cost})")
    return filtered


def split_by_song(
    df: pd.DataFrame,
    val_fraction: float = 0.05,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split manifest into train/val by dali_id to prevent data leakage.

    All chunks from a given song go entirely into train or val.

    Args:
        df: Manifest DataFrame.
        val_fraction: Fraction of songs to reserve for validation.
        seed: Random seed for reproducibility.

    Returns:
        (train_df, val_df) tuple.
    """
    rng = np.random.RandomState(seed)
    song_ids = np.array(df["dali_id"].unique())
    rng.shuffle(song_ids)

    n_val = max(1, int(len(song_ids) * val_fraction))
    val_ids = set(song_ids[:n_val])

    val_mask = df["dali_id"].isin(val_ids)
    train_df = df[~val_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)

    print(f"[data_helpers] Split: {len(train_df)} train chunks "
          f"({len(song_ids) - n_val} songs), "
          f"{len(val_df)} val chunks ({n_val} songs)")
    return train_df, val_df


def check_chunk_complete(row: pd.Series) -> bool:
    """Check if all required .npy files exist for a chunk.

    Args:
        row: A row from the manifest DataFrame.

    Returns:
        True if all required files exist.
    """
    required = ["prior_mel_path", "target_mel_path", "f0_path",
                 "voicing_path", "phoneme_mask_path"]
    return all(
        pd.notna(row.get(col)) and os.path.isfile(row[col])
        for col in required
    )
