"""Drive Applio's 3-step RVC training pipeline: preprocess -> extract -> train.

Applio must be cloned to ../Applio/ and its dependencies installed.
Training data must be prepared first via prepare_training_data.py.

Usage (from RVCBaseline/):
    python run_training.py
    python run_training.py --config configs/default.yaml
    python run_training.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from rvc_config import RVCBaselineConfig
from rvc_utils import APPLIO_DIR, timestamp


def _run_applio_command(args: list[str], dry_run: bool = False) -> None:
    cmd = [sys.executable, os.path.join(APPLIO_DIR, "core.py")] + args
    print(f"[{timestamp()}] CMD: {' '.join(cmd)}")
    if dry_run:
        print(f"[{timestamp()}] (dry run — skipping)")
        return
    result = subprocess.run(cmd, cwd=APPLIO_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"Applio command failed (exit {result.returncode}): {' '.join(args[:2])}")


def run_preprocess(config: RVCBaselineConfig, cpu_cores: int, dry_run: bool) -> None:
    training_data_abs = os.path.abspath(config.training_data_dir)
    _run_applio_command([
        "preprocess",
        "--model_name", config.rvc_model_name,
        "--dataset_path", training_data_abs,
        "--sample_rate", str(config.rvc_sample_rate),
        "--cpu_cores", str(cpu_cores),
        "--cut_preprocess", "Automatic",
        "--process_effects", "False",
        "--noise_reduction", "False",
    ], dry_run)


def run_extract(config: RVCBaselineConfig, cpu_cores: int, gpu: str, dry_run: bool) -> None:
    _run_applio_command([
        "extract",
        "--model_name", config.rvc_model_name,
        "--f0_method", config.f0_method,
        "--cpu_cores", str(cpu_cores),
        "--gpu", gpu,
        "--sample_rate", str(config.rvc_sample_rate),
        "--embedder_model", "contentvec",
        "--include_mutes", "2",
    ], dry_run)


def run_train(config: RVCBaselineConfig, gpu: str, dry_run: bool) -> None:
    _run_applio_command([
        "train",
        "--model_name", config.rvc_model_name,
        "--save_every_epoch", "50",
        "--save_only_latest", "False",
        "--save_every_weights", "True",
        "--total_epoch", str(config.rvc_epochs),
        "--sample_rate", str(config.rvc_sample_rate),
        "--batch_size", "4",
        "--gpu", gpu,
        "--pretrained", "True",
        "--overtraining_detector", "True",
        "--overtraining_threshold", "50",
        "--cleanup", "False",
    ], dry_run)


def _parse_args() -> tuple[RVCBaselineConfig, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="Run Applio RVC training pipeline")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--gpu", type=str, default="0", help="GPU index (default: 0)")
    parser.add_argument("--cpu-cores", type=int, default=4)
    parser.add_argument("--rvc-model-name", type=str, default=None)
    parser.add_argument("--rvc-epochs", type=int, default=None)
    parser.add_argument("--step", type=str, default="all",
                        choices=["all", "preprocess", "extract", "train"],
                        help="Run a single step or all steps")

    args = parser.parse_args()

    config = RVCBaselineConfig()
    if args.config:
        config = RVCBaselineConfig.from_yaml(args.config)

    if args.rvc_model_name:
        config.rvc_model_name = args.rvc_model_name
    if args.rvc_epochs is not None:
        config.rvc_epochs = args.rvc_epochs

    return config, args


def main() -> None:
    config, args = _parse_args()

    if not os.path.isdir(APPLIO_DIR):
        print(f"ERROR: Applio not found at {APPLIO_DIR}")
        print("Clone it first: git clone https://github.com/IAHispano/Applio.git ../Applio")
        sys.exit(1)

    training_data_abs = os.path.abspath(config.training_data_dir)
    if not os.path.isdir(training_data_abs):
        print(f"ERROR: Training data not found at {training_data_abs}")
        print("Run prepare_training_data.py first.")
        sys.exit(1)

    n_files = len([f for f in os.listdir(training_data_abs) if f.endswith(".wav")])
    print(f"[{timestamp()}] Model: {config.rvc_model_name}")
    print(f"[{timestamp()}] Training data: {n_files} WAV files in {training_data_abs}")
    print(f"[{timestamp()}] Epochs: {config.rvc_epochs}, Sample rate: {config.rvc_sample_rate}")
    print(f"[{timestamp()}] F0 method: {config.f0_method}")
    print()

    step = args.step

    if step in ("all", "preprocess"):
        print(f"[{timestamp()}] === Step 1/3: Preprocess ===")
        run_preprocess(config, args.cpu_cores, args.dry_run)

    if step in ("all", "extract"):
        print(f"[{timestamp()}] === Step 2/3: Extract features ===")
        run_extract(config, args.cpu_cores, args.gpu, args.dry_run)

    if step in ("all", "train"):
        print(f"[{timestamp()}] === Step 3/3: Train ===")
        run_train(config, args.gpu, args.dry_run)

    model_dir = os.path.join(APPLIO_DIR, "logs", config.rvc_model_name)
    print(f"\n[{timestamp()}] Done. Check model outputs in: {model_dir}")


if __name__ == "__main__":
    main()
