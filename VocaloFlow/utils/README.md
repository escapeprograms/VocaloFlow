# VocaloFlow Utils Memory Palace

Data extraction and loading utilities for the VocaloFlow training pipeline.

## data_helpers.py

Manifest loading, quality filtering, and train/val splitting.

### `load_manifest(manifest_path, data_dir) -> pd.DataFrame`
Loads `manifest.csv` and resolves all `*_path` columns from relative to absolute paths using `data_dir` as the root.

### `filter_manifest(df, max_dtw_cost=200.0) -> pd.DataFrame`
Filters chunks by DTW alignment quality. Removes rows where `dtw_cost` is NaN or exceeds the threshold. Prints before/after counts.

### `split_by_song(df, val_fraction=0.05, seed=42) -> (train_df, val_df)`
Song-level split by `dali_id` to prevent data leakage — all chunks from a song go into the same split. Shuffles song IDs with a fixed seed, takes the first `val_fraction` as validation.

### `split_random(df, val_fraction=0.05, seed=42) -> (train_df, val_df)`
Random per-chunk (line) split that does NOT respect song boundaries — chunks from the same `dali_id` may appear in both train and val. Provided as a leakage-tolerant baseline for comparison against `split_by_song`. Selected at training time via `config.split_mode="random"` or the `--split-mode random` CLI flag in `training/train.py`. Logs how many songs end up overlapping the two splits.

### `check_chunk_complete(row) -> bool`
Verifies all 5 required `.npy` files exist on disk for a manifest row.

## dataset.py

### `VocaloFlowDataset(manifest_df, data_dir, max_seq_len=256, training=True, use_plbert=False, use_speaker_embedding=False, global_speaker_embedding_path="")`
PyTorch Dataset that handles three data challenges:

1. **Phoneme indirection**: `phoneme_mask.npy` contains indices into `phoneme_ids.npy`. Resolved at load time: `resolved = phoneme_ids[clip(phoneme_mask, 0, len-1)]`.

2. **T-mismatch**: prior_mel, target_mel, f0, voicing, and phoneme_mask can have slightly different time dimensions (off by a few frames after DTW alignment). All are padded or truncated to `target_mel`'s T — no interpolation, since DTW already aligned them temporally.

3. **Variable lengths**: Random crop (training) or start crop (eval) to `max_seq_len`. Shorter sequences are zero-padded. A `padding_mask` (bool, True=valid) is returned.

**Returns dict** with keys: `target_mel` (T,128), `prior_mel` (T,128), `f0` (T,), `voicing` (T,), `phoneme_ids` (T,), `length` (int), `padding_mask` (T,). When `use_plbert=True`, also returns `plbert_features` (T,768) — loaded from `plbert_features.npy` (P,768) and expanded to frame-level via `phoneme_mask` indirection (same mechanism as phoneme_ids resolution), then cropped/padded to `max_seq_len`. When `use_speaker_embedding=True`, also returns `speaker_embedding` (D_spk,) — either loaded once from `global_speaker_embedding_path` (a `.pt` file, shared across all items) or per-chunk from `speaker_embedding.npy`.

## collate.py

### `vocaloflow_collate_fn(batch) -> dict`
Stacks a list of dataset items into batched tensors. All items are already the same length (max_seq_len) so this is a straightforward `torch.stack`. Conditionally includes `plbert_features` and `speaker_embedding` if present in batch items.

### `validate_batch_signals(batch, *, expect_plbert=False, expect_speaker_embedding=False) -> None`
Raises `RuntimeError` if config-expected optional signals are missing from a batch. Called once at training startup (after dataloaders are built) in both VocaloFlow `train.py` and AdversarialFinetune `train_finetune.py` to catch config/data mismatches early.
