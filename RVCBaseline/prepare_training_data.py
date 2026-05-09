"""Aggregate SoulX-Singer target.wav files from the training split into a flat
directory that RVC/Applio can consume for fine-tuning.

Usage (from RVCBaseline/):
    python prepare_training_data.py
    python prepare_training_data.py --config configs/default.yaml
    python prepare_training_data.py --max-training-chunks 500
"""

from __future__ import annotations

import argparse
import os
import shutil

import numpy as np

from rvc_config import RVCBaselineConfig
from rvc_utils import (
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


def _parse_args() -> RVCBaselineConfig:
    parser = argparse.ArgumentParser(
        description="Prepare RVC training data from SoulX-Singer target.wav files"
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--max-training-chunks", type=int, default=None)
    parser.add_argument("--training-data-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    config = RVCBaselineConfig()
    if args.config:
        config = RVCBaselineConfig.from_yaml(args.config)

    if args.data_dir:
        config.data_dir = args.data_dir
    if args.manifest:
        config.manifest_path = args.manifest
    if args.max_training_chunks is not None:
        config.max_training_chunks = args.max_training_chunks
    if args.training_data_dir:
        config.training_data_dir = args.training_data_dir
    if args.seed is not None:
        config.seed = args.seed

    return config


def main() -> None:
    config = _parse_args()

    print(f"[{timestamp()}] Loading manifest from {config.manifest_path}")
    df = load_manifest(config.manifest_path, config.data_dir)
    df = filter_manifest(df, config.max_dtw_cost)

    train_df, val_df = split_by_song(df, config.val_fraction, config.seed)
    print(f"[{timestamp()}] Train: {len(train_df)} chunks, Val: {len(val_df)} chunks")

    if config.max_training_chunks > 0 and len(train_df) > config.max_training_chunks:
        rng = np.random.RandomState(config.seed)
        idx = rng.choice(len(train_df), config.max_training_chunks, replace=False)
        train_df = train_df.iloc[sorted(idx)].reset_index(drop=True)
        print(f"[{timestamp()}] Subsampled to {config.max_training_chunks} training chunks")

    os.makedirs(config.training_data_dir, exist_ok=True)

    copied = 0
    skipped = 0
    for i, (_, row) in enumerate(train_df.iterrows()):
        dali_id = row["dali_id"]
        chunk_name = row["chunk_name"]
        src = os.path.join(config.data_dir, dali_id, chunk_name, "target.wav")

        if not os.path.exists(src):
            skipped += 1
            continue

        dst = os.path.join(config.training_data_dir, f"{i:05d}.wav")
        shutil.copy2(src, dst)
        copied += 1

        if (i + 1) % 200 == 0:
            print(f"[{timestamp()}] Copied {copied}/{i + 1} files...")

    print(f"\n[{timestamp()}] Done. Copied: {copied}, Skipped: {skipped}")
    print(f"[{timestamp()}] Training data directory: {os.path.abspath(config.training_data_dir)}")


if __name__ == "__main__":
    main()
