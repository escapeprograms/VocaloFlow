"""
Generate a training manifest (manifest.csv) from iterative alignment results.

Walks Data/<dali_id>/<chunk_name>/ directories, reads each alignment.json
(v2 iterative format), and writes a single manifest.csv for fast ML training
data loading.

Usage (standalone):
    python utils/generate_manifest.py --data_dir ../Data
"""

import argparse
import csv
import json
import os
from typing import Optional


MANIFEST_COLUMNS = [
    "provider",
    "prompt_name",
    "dali_id",
    "chunk_name",
    "prior_mel_path",
    "target_mel_path",
    "f0_path",
    "voicing_path",
    "phoneme_mask_path",
    "adjusted_notes_path",
    "converged",
    "iterations",
    "dtw_cost",
    "mean_deviation",
    "max_deviation",
]


def generate_manifest(data_dir: str, manifest_path: Optional[str] = None) -> str:
    """Walk data_dir and produce manifest.csv from iterative alignment results.

    Only processes chunks whose alignment.json contains the ``dtw_cost`` key
    (v2 iterative format).  Chunks with the v1 segmented DTW format
    (``mean_dtw_cost``) are silently skipped.

    Args:
        data_dir:      Root data directory (contains <dali_id>/ subdirs).
        manifest_path: Output CSV path.  Defaults to ``data_dir/manifest.csv``.

    Returns:
        The absolute path to the written manifest file.
    """
    if manifest_path is None:
        manifest_path = os.path.join(data_dir, "manifest.csv")

    rows = []

    for dali_id in sorted(os.listdir(data_dir)):
        dali_dir = os.path.join(data_dir, dali_id)
        if not os.path.isdir(dali_dir):
            continue

        for chunk_name in sorted(os.listdir(dali_dir)):
            chunk_dir = os.path.join(dali_dir, chunk_name)
            if not os.path.isdir(chunk_dir):
                continue

            alignment_path = os.path.join(chunk_dir, "alignment.json")
            prior_mel = os.path.join(chunk_dir, "prior_mel.npy")
            target_mel = os.path.join(chunk_dir, "target_mel.npy")
            f0_file = os.path.join(chunk_dir, "target_f0.npy")
            voicing_file = os.path.join(chunk_dir, "target_voicing.npy")
            phoneme_mask_file = os.path.join(chunk_dir, "phoneme_mask.npy")
            adjusted_notes_file = os.path.join(chunk_dir, "adjusted_notes.json")

            # Skip incomplete chunks — require all training artifacts
            required = [alignment_path, prior_mel, target_mel,
                        f0_file, voicing_file, phoneme_mask_file, adjusted_notes_file]
            if not all(os.path.exists(p) for p in required):
                continue

            with open(alignment_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            # Only process v2 iterative format (has "dtw_cost", not "mean_dtw_cost")
            if "dtw_cost" not in meta:
                continue

            converged = meta.get("converged", False)

            # Use forward slashes for cross-platform path portability in CSV
            rel = f"{dali_id}/{chunk_name}"
            row = {
                "provider": meta.get("provider", "WillStetson"),
                "prompt_name": meta.get("prompt_name", "WillStetson"),
                "dali_id": dali_id,
                "chunk_name": chunk_name,
                "prior_mel_path": f"{rel}/prior_mel.npy",
                "target_mel_path": f"{rel}/target_mel.npy",
                "f0_path": f"{rel}/target_f0.npy",
                "voicing_path": f"{rel}/target_voicing.npy",
                "phoneme_mask_path": f"{rel}/phoneme_mask.npy",
                "adjusted_notes_path": f"{rel}/adjusted_notes.json",
                "converged": converged,
                "iterations": meta.get("iterations", ""),
                "dtw_cost": meta.get("dtw_cost", ""),
                "mean_deviation": meta.get("mean_deviation", ""),
                "max_deviation": meta.get("max_deviation", "") if converged else meta.get("best_max_deviation", ""),
            }
            rows.append(row)

    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return os.path.abspath(manifest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training manifest from iterative alignment results.")
    parser.add_argument("--data_dir", required=True, help="Root data directory containing <dali_id>/ subdirs.")
    parser.add_argument("--manifest_path", default=None, help="Output CSV path (default: <data_dir>/manifest.csv).")
    args = parser.parse_args()

    path = generate_manifest(args.data_dir, args.manifest_path)
    print(f"Manifest written to {path}")
