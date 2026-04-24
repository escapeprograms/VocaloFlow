"""Precompute per-chunk speaker embeddings for VocaloFlow training data.

For each chunk directory, loads target.wav, extracts an ECAPA-TDNN
embedding (192,), and saves it as speaker_embedding.npy.

Unlike PL-BERT features (which vary per phoneme), the speaker embedding
is a single global vector per utterance.  It is identical across all
chunks from the same speaker, but is stored per-chunk for consistency
with the data loading pattern.

Usage:
    cd "Honors Thesis"
    python SpeakerEmbedding/precompute_embeddings.py \
        --data-dir Data/Rachie --manifest Data/Rachie/manifest.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from extract_embedding import extract_single, load_ecapa_model


def main():
    parser = argparse.ArgumentParser(
        description="Precompute ECAPA-TDNN speaker embeddings for training chunks"
    )
    parser.add_argument("--data-dir", required=True, help="Root data directory (e.g. Data/Rachie)")
    parser.add_argument("--manifest", required=True, help="Path to manifest.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if speaker_embedding.npy exists")
    args = parser.parse_args()

    import pandas as pd

    print(f"Loading ECAPA-TDNN on {args.device}...")
    model = load_ecapa_model(args.device)

    manifest = pd.read_csv(args.manifest)
    print(f"Manifest: {len(manifest)} chunks")

    done = 0
    skipped = 0
    failed = 0

    for idx, row in manifest.iterrows():
        chunk_dir = os.path.join(args.data_dir, os.path.dirname(row["phoneme_mask_path"]))
        output_path = os.path.join(chunk_dir, "speaker_embedding.npy")

        if os.path.exists(output_path) and not args.overwrite:
            skipped += 1
            continue

        target_wav = os.path.join(chunk_dir, "target.wav")
        if not os.path.exists(target_wav):
            failed += 1
            continue

        try:
            emb = extract_single(model, target_wav)
            np.save(output_path, emb.cpu().numpy().astype(np.float32))
            done += 1
        except Exception as e:
            print(f"  FAILED {chunk_dir}: {e}")
            failed += 1

        total = done + skipped + failed
        if total % 500 == 0:
            print(f"  Progress: {total}/{len(manifest)} "
                  f"(done={done}, skipped={skipped}, failed={failed})")

    print(f"\nFinished: done={done}, skipped={skipped}, failed={failed}")

    # Spot-check
    sample_dir = os.path.join(args.data_dir, os.path.dirname(manifest.iloc[0]["phoneme_mask_path"]))
    sample_path = os.path.join(sample_dir, "speaker_embedding.npy")
    if os.path.exists(sample_path):
        emb = np.load(sample_path)
        print(f"\nSpot-check: {sample_path}")
        print(f"  Shape: {emb.shape}, dtype: {emb.dtype}")
        print(f"  L2 norm: {np.linalg.norm(emb):.6f}")
        print(f"  Mean: {emb.mean():.6f}, Std: {emb.std():.6f}")


if __name__ == "__main__":
    main()
