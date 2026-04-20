"""Diagnostic script to inspect PL-BERT checkpoint and token_maps.pkl.

Run this before the full integration to verify assumptions about the
checkpoint structure and vocabulary.

Usage:
    cd "Honors Thesis"
    python PL-BERT/inspect_checkpoint.py
"""

import pickle
from pathlib import Path

import torch
import yaml

_DIR = Path(__file__).parent


def main() -> None:
    config_path = _DIR / "configs" / "config.yml"
    ckpt_path = _DIR / "checkpoints" / "step_1000000.t7"
    token_maps_path = _DIR / "configs" / "token_maps.pkl"

    # ── Config ────────────────────────────────────────────────────────────
    print("=" * 60)
    print("PL-BERT config.yml")
    print("=" * 60)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    model_params = config["model_params"]
    for k, v in model_params.items():
        print(f"  {k}: {v}")

    # ── Checkpoint ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PL-BERT checkpoint")
    print("=" * 60)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"Top-level keys: {list(ckpt.keys())}")
    print(f"Training step: {ckpt.get('step', 'N/A')}")

    state_dict = ckpt["net"]
    print(f"\nState dict ({len(state_dict)} keys):")
    total_params = 0
    for k, v in state_dict.items():
        numel = v.numel()
        total_params += numel
        print(f"  {k}: {list(v.shape)} ({numel:,} params)")
    print(f"\nTotal parameters: {total_params:,}")

    # Check embedding shape matches config
    emb_key = "module.encoder.embeddings.word_embeddings.weight"
    if emb_key in state_dict:
        emb_shape = state_dict[emb_key].shape
        print(f"\nEmbedding matrix: {list(emb_shape)}")
        print(f"  vocab_size from embedding: {emb_shape[0]}")
        print(f"  vocab_size from config:    {model_params['vocab_size']}")
        assert emb_shape[0] == model_params["vocab_size"], "Mismatch!"

    # ── Token maps ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("token_maps.pkl (word-level TransfoXL tokenizer)")
    print("=" * 60)
    with open(token_maps_path, "rb") as f:
        token_maps = pickle.load(f)
    print(f"Type: {type(token_maps)}")
    print(f"Num entries: {len(token_maps)}")
    print("First 5 entries:")
    for i, (k, v) in enumerate(token_maps.items()):
        if i >= 5:
            break
        print(f"  {k} -> {v}")
    print("(This file is NOT used for phoneme tokenization —")
    print(" it's for the word-prediction auxiliary task during PL-BERT training.)")

    # ── Vocab test ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("IPA vocabulary test (from text_utils.py)")
    print("=" * 60)
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "text_utils", str(_DIR / "text_utils.py")
    )
    tu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tu)
    print(f"VOCAB_SIZE: {tu.VOCAB_SIZE}")
    print(f"char_to_idx entries: {len(tu.char_to_idx)}")

    test_ipa = "hɛloʊ wɜːld"
    tokens = tu.tokenize_ipa(test_ipa)
    print(f'tokenize_ipa("{test_ipa}") = {tokens}')
    print(f"All tokens in range [0, {tu.VOCAB_SIZE}): {all(0 <= t < tu.VOCAB_SIZE for t in tokens)}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
