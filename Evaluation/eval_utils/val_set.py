"""Validation set loading and curation utilities."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from eval_utils.imports import (
    VOCALOFLOW_DIR,
    import_from_path,
    timestamp,
)

_data_helpers = import_from_path(
    "vf_data_helpers",
    os.path.join(VOCALOFLOW_DIR, "utils", "data_helpers.py"),
)
load_manifest = _data_helpers.load_manifest
filter_manifest = _data_helpers.filter_manifest
split_by_song = _data_helpers.split_by_song


def get_val_manifest(
    data_dir: str,
    manifest_path: str,
    max_dtw_cost: float = 100.0,
    val_fraction: float = 0.05,
    seed: int = 42,
    max_eval_chunks: int = 0,
) -> pd.DataFrame:
    """Load manifest, filter, split by song, return val portion.

    Args:
        max_eval_chunks: Cap on number of val chunks (0 = all).

    Returns:
        Validation DataFrame with absolute paths.
    """
    df = load_manifest(manifest_path, data_dir)
    df = filter_manifest(df, max_dtw_cost)
    _, val_df = split_by_song(df, val_fraction, seed)

    if max_eval_chunks > 0 and len(val_df) > max_eval_chunks:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(val_df), max_eval_chunks, replace=False)
        val_df = val_df.iloc[sorted(idx)].reset_index(drop=True)
        print(f"[{timestamp()}] Capped eval set to {max_eval_chunks} chunks")

    print(f"[{timestamp()}] Eval set: {len(val_df)} chunks")
    return val_df


def get_chunk_dir(data_dir: str, dali_id: str, chunk_name: str) -> str:
    """Absolute path to a chunk's data directory."""
    return os.path.join(data_dir, dali_id, chunk_name)
