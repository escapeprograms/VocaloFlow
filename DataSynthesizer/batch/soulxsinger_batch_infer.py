"""
Batch inference runner for SoulX-Singer.

Loads the model once and runs inference for all provided chunks, avoiding the
expensive checkpoint reload that occurs when invoking cli.inference per-segment.

Usage:
    python soulxsinger_batch_infer.py \
        --tasks_json /path/to/tasks.json \
        --model_path .../model.pt \
        --config .../soulxsinger.yaml \
        --prompt_wav_path .../vocal.wav \
        --prompt_metadata_path .../metadata.json \
        --phoneset_path .../phone_set.json \
        [--device cuda] [--auto_shift] [--pitch_shift 0]

tasks.json is a list of dicts, each with keys:
    target_metadata_path, save_dir, control, save_mel
"""

import argparse
import json
import os
import sys
import types

SOULX_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "SoulX-Singer"))
sys.path.insert(0, SOULX_DIR)

from cli.inference import build_model, process as soulx_process
from soulxsinger.utils.file_utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks_json", required=True, help="JSON file listing inference tasks")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt_wav_path", required=True)
    parser.add_argument("--prompt_metadata_path", required=True)
    parser.add_argument("--phoneset_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--auto_shift", action="store_true")
    parser.add_argument("--pitch_shift", type=int, default=0)
    args = parser.parse_args()

    with open(args.tasks_json, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    if not tasks:
        print("No tasks provided.")
        return

    print(f"Loading SoulX-Singer model (once) for {len(tasks)} chunk(s)...")
    config = load_config(args.config)
    model = build_model(model_path=args.model_path, config=config, device=args.device)

    for i, task in enumerate(tasks):
        target_metadata_path = task["target_metadata_path"]
        save_dir = task["save_dir"]
        control = task["control"]
        save_mel = task.get("save_mel", False)

        chunk_name = os.path.basename(save_dir)

        # Resumability: skip chunks that already completed successfully
        expected_output = os.path.join(save_dir, "target_mel.npy" if save_mel else "target.wav")
        if os.path.exists(expected_output):
            print(f"\n[{i+1}/{len(tasks)}] Skipping {chunk_name} (already done).")
            continue

        print(f"\n[{i+1}/{len(tasks)}] Running inference for {chunk_name} (control={control})...")

        chunk_args = types.SimpleNamespace(
            device=args.device,
            model_path=args.model_path,
            config=args.config,
            prompt_wav_path=task.get("prompt_wav_path", args.prompt_wav_path),
            prompt_metadata_path=task.get("prompt_metadata_path", args.prompt_metadata_path),
            target_metadata_path=target_metadata_path,
            phoneset_path=args.phoneset_path,
            save_dir=save_dir,
            auto_shift=args.auto_shift,
            pitch_shift=args.pitch_shift,
            save_mel=save_mel,
            control=control,
        )

        try:
            soulx_process(chunk_args, config, model)
            print(f"Successfully generated audio for {chunk_name} in {save_dir}")
        except Exception as e:
            print(f"Inference failed for {chunk_name}: {e}")

    print(f"\nBatch inference complete ({len(tasks)} chunk(s)).")


if __name__ == "__main__":
    main()
