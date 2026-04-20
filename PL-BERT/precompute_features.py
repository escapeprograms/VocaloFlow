"""Precompute PL-BERT features for all training chunks.

For each chunk, reads phoneme_ids.npy, converts the phoneme sequence to IPA,
runs PL-BERT to get contextual embeddings, and saves plbert_features.npy
with shape (P, 768) where P = len(phoneme_ids).

Structural tokens (<PAD>, <BOW>, <EOW>, <SEP>, <SP>, etc.) and non-English
phonemes get zero vectors. English phonemes get their PL-BERT contextual
embedding (averaged across multi-char IPA representations).

Usage:
    cd "Honors Thesis"
    python PL-BERT/precompute_features.py --data-dir Data/Rachie --manifest Data/Rachie/manifest.csv
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_DIR = Path(__file__).parent


def load_plbert_modules():
    """Import PL-BERT modules by file path (directory name has a hyphen)."""
    import importlib.util

    spec1 = importlib.util.spec_from_file_location("plbert", str(_DIR / "plbert.py"))
    plbert_mod = importlib.util.module_from_spec(spec1)
    spec1.loader.exec_module(plbert_mod)

    spec2 = importlib.util.spec_from_file_location("text_utils", str(_DIR / "text_utils.py"))
    text_utils = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(text_utils)

    spec3 = importlib.util.spec_from_file_location("arpabet_ipa", str(_DIR / "arpabet_ipa.py"))
    arpabet_ipa = importlib.util.module_from_spec(spec3)
    spec3.loader.exec_module(arpabet_ipa)

    return plbert_mod, text_utils, arpabet_ipa


def precompute_chunk(
    chunk_dir: str,
    phone_set: list[str],
    extractor,
    text_utils,
    arpabet_ipa,
    hidden_dim: int = 768,
) -> bool:
    """Precompute PL-BERT features for one chunk.

    Reads:  phoneme_ids.npy (P,)
    Writes: plbert_features.npy (P, 768) float16

    Returns True if successful, False if skipped/failed.
    """
    phoneme_ids_path = os.path.join(chunk_dir, "phoneme_ids.npy")
    output_path = os.path.join(chunk_dir, "plbert_features.npy")

    if not os.path.exists(phoneme_ids_path):
        return False

    phoneme_ids = np.load(phoneme_ids_path).astype(np.int64)
    P = len(phoneme_ids)
    features = np.zeros((P, hidden_dim), dtype=np.float32)

    # Identify English phoneme positions and build the IPA sequence
    en_positions = []  # (phoneme_ids index, arpabet symbol)
    for i, pid in enumerate(phoneme_ids):
        entry = phone_set[pid]
        arpa = arpabet_ipa.phone_set_entry_to_arpabet(entry)
        if arpa is not None:
            en_positions.append((i, arpa))

    if not en_positions:
        np.save(output_path, features.astype(np.float16))
        return True

    # Build the full IPA sequence from English phonemes, tracking
    # which IPA chars belong to which ARPAbet phoneme.
    ipa_chars = []
    char_to_phoneme_idx = []  # maps each IPA char position -> phoneme_ids index
    for pid_idx, arpa in en_positions:
        ipa = arpabet_ipa.arpabet_to_ipa(arpa)
        if ipa is None:
            continue
        for c in ipa:
            ipa_chars.append(c)
            char_to_phoneme_idx.append(pid_idx)

    if not ipa_chars:
        np.save(output_path, features.astype(np.float16))
        return True

    # Tokenize and run PL-BERT
    ipa_string = "".join(ipa_chars)
    token_ids = text_utils.tokenize_ipa(ipa_string)
    plbert_output = extractor.extract(token_ids)  # (num_ipa_chars, 768)
    plbert_np = plbert_output.cpu().numpy()

    # Aggregate: average-pool IPA char embeddings back to ARPAbet phonemes
    for ipa_pos, pid_idx in enumerate(char_to_phoneme_idx):
        features[pid_idx] += plbert_np[ipa_pos]

    # Normalize by count (for multi-char IPA like aɪ → 2 chars)
    from collections import Counter
    counts = Counter(char_to_phoneme_idx)
    for pid_idx, count in counts.items():
        features[pid_idx] /= count

    np.save(output_path, features.astype(np.float16))
    return True


def main():
    parser = argparse.ArgumentParser(description="Precompute PL-BERT features for VocaloFlow training data")
    parser.add_argument("--data-dir", required=True, help="Root data directory (e.g. Data/Rachie)")
    parser.add_argument("--manifest", required=True, help="Path to manifest.csv")
    parser.add_argument("--phoneset", default=str(_DIR.parent / "SoulX-Singer" / "soulxsinger" / "utils" / "phoneme" / "phone_set.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if plbert_features.npy exists")
    args = parser.parse_args()

    import pandas as pd

    plbert_mod, text_utils, arpabet_ipa = load_plbert_modules()

    with open(args.phoneset) as f:
        phone_set = json.load(f)
    print(f"Loaded phone_set: {len(phone_set)} entries")

    extractor = plbert_mod.PLBertFeatureExtractor(device=args.device)
    print(f"PL-BERT loaded on {args.device}")

    manifest = pd.read_csv(args.manifest)
    print(f"Manifest: {len(manifest)} chunks")

    done = 0
    skipped = 0
    failed = 0
    for idx, row in manifest.iterrows():
        chunk_dir = os.path.join(args.data_dir, os.path.dirname(row["phoneme_mask_path"]))
        output_path = os.path.join(chunk_dir, "plbert_features.npy")

        if os.path.exists(output_path) and not args.overwrite:
            skipped += 1
            continue

        ok = precompute_chunk(chunk_dir, phone_set, extractor, text_utils, arpabet_ipa)
        if ok:
            done += 1
        else:
            failed += 1

        if (done + skipped + failed) % 500 == 0:
            print(f"  Progress: {done + skipped + failed}/{len(manifest)} "
                  f"(done={done}, skipped={skipped}, failed={failed})")

    print(f"\nFinished: done={done}, skipped={skipped}, failed={failed}")

    # Spot-check
    sample_dir = os.path.join(args.data_dir, os.path.dirname(manifest.iloc[0]["phoneme_mask_path"]))
    sample_path = os.path.join(sample_dir, "plbert_features.npy")
    if os.path.exists(sample_path):
        feats = np.load(sample_path)
        pids = np.load(os.path.join(sample_dir, "phoneme_ids.npy"))
        print(f"\nSpot-check: {sample_path}")
        print(f"  phoneme_ids shape: {pids.shape}")
        print(f"  plbert_features shape: {feats.shape}")
        print(f"  dtype: {feats.dtype}")
        non_zero = np.any(feats != 0, axis=1)
        print(f"  Non-zero rows: {non_zero.sum()}/{len(feats)}")
        if non_zero.any():
            print(f"  Feature stats (non-zero): mean={feats[non_zero].mean():.4f}, "
                  f"std={feats[non_zero].std():.4f}")


if __name__ == "__main__":
    main()
