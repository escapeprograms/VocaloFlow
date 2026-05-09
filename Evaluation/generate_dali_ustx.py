"""Pre-generate USTX files from DALI annotations for fair controllability evaluation.

Creates a ``dali_prior.ustx`` and ``dali_prior.wav`` for each validation chunk,
stored under ``{output_dir}/{dali_id}/{chunk_name}/``.  These are later consumed
by ``run_eval.py --eval-context fair``.

Usage (from repo root):
    python -m Evaluation.generate_dali_ustx \\
        --output-dir Evaluation/dali_ustx

    # Limit to N chunks for a quick test
    python -m Evaluation.generate_dali_ustx \\
        --output-dir Evaluation/dali_ustx --max-chunks 5
"""

from __future__ import annotations

import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from eval_config import EvalConfig
from eval_utils.val_set import get_val_manifest
from eval_utils.dali_ustx import generate_all_dali_ustx


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate DALI-path USTX priors for fair controllability eval"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./dali_ustx",
        help="Directory for generated USTX/WAV artifacts (default: ./dali_ustx)",
    )
    parser.add_argument("--config", type=str, default=None, help="YAML eval config")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--dali-annot-dir", type=str, default=None)
    parser.add_argument("--max-chunks", type=int, default=0, help="0 = all")
    parser.add_argument("--max-dtw-cost", type=float, default=None)
    parser.add_argument("--no-phonemes", action="store_true")
    args = parser.parse_args()

    config = EvalConfig()
    if args.config:
        config = EvalConfig.from_yaml(args.config)
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.manifest:
        config.manifest_path = args.manifest
    if args.dali_annot_dir:
        config.dali_annot_dir = args.dali_annot_dir
    if args.max_dtw_cost is not None:
        config.max_dtw_cost = args.max_dtw_cost

    print("Loading validation manifest...")
    val_df = get_val_manifest(
        config.data_dir,
        config.manifest_path,
        config.max_dtw_cost,
        config.val_fraction,
        config.seed,
        args.max_chunks,
    )
    print(f"  {len(val_df)} chunks to process")

    output_dir = args.output_dir
    print(f"Output directory: {output_dir}")
    print(f"DALI annotations: {config.dali_annot_dir}")
    print(f"Phoneme hints: {'off' if args.no_phonemes else 'on'}")
    print()

    generate_all_dali_ustx(
        val_df,
        config.dali_annot_dir,
        output_dir,
        use_phonemes=not args.no_phonemes,
    )


if __name__ == "__main__":
    main()
