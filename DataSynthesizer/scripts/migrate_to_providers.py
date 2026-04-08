"""One-time migration: move existing Data/ contents into Data/WillStetson/.

Restructures the flat Data/{dali_id}/{chunk_name}/ layout into the new
Data/{provider}/{dali_id}/{chunk_name}/ layout by moving everything under
a WillStetson/ subdirectory.

Dry-run by default — pass --execute to actually move files.

Usage:
    python scripts/migrate_to_providers.py                # dry-run
    python scripts/migrate_to_providers.py --execute      # real migration
"""

import argparse
import json
import os
import re
import shutil
import sys

UUID_RE = re.compile(r"^[0-9a-f]{32}$")

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Data"))
PROVIDER = "WillStetson"


def main():
    parser = argparse.ArgumentParser(description="Migrate Data/ to Data/{provider}/ layout.")
    parser.add_argument("--execute", action="store_true",
                        help="Actually move files (default is dry-run).")
    parser.add_argument("--data_dir", default=DATA_DIR,
                        help="Root data directory (default: ../Data relative to DataSynthesizer).")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    provider_dir = os.path.join(data_dir, PROVIDER)
    dry_run = not args.execute

    if not os.path.isdir(data_dir):
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    if os.path.isdir(provider_dir):
        # Check if already migrated
        existing = [e for e in os.listdir(provider_dir) if UUID_RE.match(e)]
        if existing:
            print(f"ERROR: {provider_dir} already exists with {len(existing)} UUID dirs.")
            print("Migration may have already been run.")
            sys.exit(1)

    # Collect items to move
    entries = os.listdir(data_dir)
    uuid_dirs = sorted([e for e in entries if UUID_RE.match(e) and os.path.isdir(os.path.join(data_dir, e))])
    meta_files = []
    for name in ["manifest.csv", "pending_inference_tasks.json"]:
        if name in entries:
            meta_files.append(name)

    print(f"Data directory: {data_dir}")
    print(f"Target:         {provider_dir}")
    print(f"UUID dirs:      {len(uuid_dirs)}")
    print(f"Meta files:     {meta_files}")
    print(f"Mode:           {'DRY RUN' if dry_run else 'EXECUTE'}")
    print()

    if not uuid_dirs and not meta_files:
        print("Nothing to migrate.")
        return

    if dry_run:
        print("--- DRY RUN (pass --execute to apply) ---\n")
        print(f"Would create: {provider_dir}")
        for name in meta_files:
            print(f"  MOVE  {name}  ->  {PROVIDER}/{name}")
        print(f"  MOVE  {len(uuid_dirs)} UUID directories  ->  {PROVIDER}/")
        if uuid_dirs:
            print(f"    e.g. {uuid_dirs[0]}  ->  {PROVIDER}/{uuid_dirs[0]}")
        if "pending_inference_tasks.json" in meta_files:
            print(f"\n  Would patch absolute paths in pending_inference_tasks.json")
        print(f"\nTotal items to move: {len(uuid_dirs) + len(meta_files)}")
        return

    # --- Execute ---
    print("Creating provider directory...")
    os.makedirs(provider_dir, exist_ok=True)

    # Move meta files first
    for name in meta_files:
        src = os.path.join(data_dir, name)
        dst = os.path.join(provider_dir, name)
        print(f"  Moving {name}...")
        shutil.move(src, dst)

    # Move UUID directories (same-filesystem rename = instant)
    print(f"Moving {len(uuid_dirs)} UUID directories...")
    for i, uuid_dir in enumerate(uuid_dirs):
        src = os.path.join(data_dir, uuid_dir)
        dst = os.path.join(provider_dir, uuid_dir)
        shutil.move(src, dst)
        if (i + 1) % 500 == 0 or i == len(uuid_dirs) - 1:
            print(f"  {i + 1}/{len(uuid_dirs)} moved")

    # Patch pending_inference_tasks.json paths
    tasks_path = os.path.join(provider_dir, "pending_inference_tasks.json")
    if os.path.exists(tasks_path):
        print("Patching absolute paths in pending_inference_tasks.json...")
        with open(tasks_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)

        # Normalize Data dir for matching (handle both / and \ on Windows)
        data_norm = data_dir.replace("\\", "/")
        patched = 0
        for task in tasks:
            for key in ["save_dir", "target_metadata_path"]:
                if key in task and isinstance(task[key], str):
                    val = task[key].replace("\\", "/")
                    if data_norm in val and f"/{PROVIDER}/" not in val:
                        task[key] = val.replace(data_norm, f"{data_norm}/{PROVIDER}")
                        patched += 1

        with open(tasks_path, "w", encoding="utf-8") as f:
            json.dump(tasks, f)
        print(f"  Patched {patched} path entries across {len(tasks)} tasks.")

    print(f"\nMigration complete. All data now lives under {provider_dir}")


if __name__ == "__main__":
    main()
