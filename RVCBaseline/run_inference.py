"""Run RVC inference on validation-set prior.wav files, producing output
audio with the exact naming convention the eval pipeline expects.

Output files: {output_dir}/{dali_id}_{chunk_name}.wav

This uses the same val-set selection logic (get_val_manifest) as
Evaluation/run_eval.py, so the two scripts operate on identical chunks.

Usage (from RVCBaseline/):
    python run_inference.py --rvc-model-path path/to/model.pth
    python run_inference.py --config configs/default.yaml --max-eval-chunks 50
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

from rvc_config import RVCBaselineConfig
from rvc_utils import (
    APPLIO_DIR,
    VOCALOFLOW_DIR,
    import_from_path,
    timestamp,
)

# Import VocaloFlow data helpers directly (same functions val_set.py wraps).
import numpy as np
import pandas as pd

_data_helpers = import_from_path(
    "vf_data_helpers",
    os.path.join(VOCALOFLOW_DIR, "utils", "data_helpers.py"),
)


def get_val_manifest(
    data_dir: str,
    manifest_path: str,
    max_dtw_cost: float = 100.0,
    val_fraction: float = 0.05,
    seed: int = 42,
    max_eval_chunks: int = 0,
) -> pd.DataFrame:
    df = _data_helpers.load_manifest(manifest_path, data_dir)
    df = _data_helpers.filter_manifest(df, max_dtw_cost)
    _, val_df = _data_helpers.split_by_song(df, val_fraction, seed)

    if max_eval_chunks > 0 and len(val_df) > max_eval_chunks:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(val_df), max_eval_chunks, replace=False)
        val_df = val_df.iloc[sorted(idx)].reset_index(drop=True)

    print(f"[{timestamp()}] Eval set: {len(val_df)} chunks")
    return val_df


def _find_index_file(model_name: str) -> str:
    """Locate the .index file Applio generated during training."""
    pattern = os.path.join(APPLIO_DIR, "logs", model_name, "*.index")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    return ""


def _run_single_infer(
    input_path: str,
    output_path: str,
    config: RVCBaselineConfig,
) -> None:
    """Run Applio CLI infer on a single file."""
    cmd = [
        sys.executable,
        os.path.join(APPLIO_DIR, "core.py"),
        "infer",
        "--input_path", os.path.abspath(input_path),
        "--output_path", os.path.abspath(output_path),
        "--pth_path", os.path.abspath(config.rvc_model_path),
        "--index_path", os.path.abspath(config.rvc_index_path) if config.rvc_index_path else "",
        "--pitch", str(config.f0_shift),
        "--index_rate", str(config.index_ratio),
        "--f0_method", config.f0_method,
        "--export_format", "WAV",
        "--clean_audio", "False",
        "--f0_autotune", "False",
        "--embedder_model", "contentvec",
        "--split_audio", "False",
        "--volume_envelope", "1.0",
        "--protect", "0.33",
    ]
    result = subprocess.run(cmd, cwd=APPLIO_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Applio infer failed (exit {result.returncode}):\n"
            f"  stderr: {result.stderr[-500:] if result.stderr else '(none)'}"
        )


def _parse_args() -> RVCBaselineConfig:
    parser = argparse.ArgumentParser(
        description="Run RVC inference on eval-set prior audio"
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--rvc-model-path", type=str, default=None)
    parser.add_argument("--rvc-index-path", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--max-eval-chunks", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--f0-shift", type=int, default=None)
    parser.add_argument("--index-ratio", type=float, default=None)

    args = parser.parse_args()

    config = RVCBaselineConfig()
    if args.config:
        config = RVCBaselineConfig.from_yaml(args.config)

    if args.rvc_model_path:
        config.rvc_model_path = args.rvc_model_path
    if args.rvc_index_path:
        config.rvc_index_path = args.rvc_index_path
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.manifest:
        config.manifest_path = args.manifest
    if args.max_eval_chunks is not None:
        config.max_eval_chunks = args.max_eval_chunks
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.f0_shift is not None:
        config.f0_shift = args.f0_shift
    if args.index_ratio is not None:
        config.index_ratio = args.index_ratio

    return config


def main() -> None:
    config = _parse_args()

    if not config.rvc_model_path or not os.path.isfile(config.rvc_model_path):
        print(f"ERROR: RVC model not found at '{config.rvc_model_path}'")
        print("Provide --rvc-model-path or set rvc_model_path in config YAML.")
        sys.exit(1)

    if not config.rvc_index_path:
        config.rvc_index_path = _find_index_file(config.rvc_model_name)
        if config.rvc_index_path:
            print(f"[{timestamp()}] Auto-detected index: {config.rvc_index_path}")
        else:
            print(f"[{timestamp()}] No index file found (index_ratio={config.index_ratio})")

    if not os.path.isdir(APPLIO_DIR):
        print(f"ERROR: Applio not found at {APPLIO_DIR}")
        sys.exit(1)

    print(f"[{timestamp()}] Loading validation manifest...")
    val_df = get_val_manifest(
        config.data_dir,
        config.manifest_path,
        config.max_dtw_cost,
        config.val_fraction,
        config.seed,
        config.max_eval_chunks,
    )

    os.makedirs(config.output_dir, exist_ok=True)

    total = len(val_df)
    succeeded = 0
    failed = 0
    skipped = 0

    print(f"[{timestamp()}] Processing {total} chunks")
    print(f"[{timestamp()}] Model: {config.rvc_model_path}")
    print(f"[{timestamp()}] Output: {os.path.abspath(config.output_dir)}")
    print(f"[{timestamp()}] F0 shift: {config.f0_shift}, Index ratio: {config.index_ratio}")
    print()

    for i, (_, row) in enumerate(val_df.iterrows()):
        dali_id = row["dali_id"]
        chunk_name = row["chunk_name"]

        input_path = os.path.join(config.data_dir, dali_id, chunk_name, "prior.wav")
        output_path = os.path.join(config.output_dir, f"{dali_id}_{chunk_name}.wav")

        if not os.path.exists(input_path):
            print(f"  [{i+1}/{total}] SKIP (no prior.wav): {dali_id}/{chunk_name}")
            skipped += 1
            continue

        if os.path.exists(output_path):
            print(f"  [{i+1}/{total}] SKIP (already exists): {dali_id}/{chunk_name}")
            skipped += 1
            continue

        try:
            _run_single_infer(input_path, output_path, config)
            succeeded += 1
            print(f"  [{i+1}/{total}] OK: {dali_id}/{chunk_name}")
        except Exception as e:
            failed += 1
            print(f"  [{i+1}/{total}] FAILED: {dali_id}/{chunk_name} — {e}")

    print(f"\n[{timestamp()}] Done. Succeeded: {succeeded}, Failed: {failed}, Skipped: {skipped}")
    print(f"[{timestamp()}] Output directory: {os.path.abspath(config.output_dir)}")


if __name__ == "__main__":
    main()
