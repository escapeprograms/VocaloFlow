# RVCBaseline — Memory Palace

RVC (Retrieval-based Voice Conversion) baseline for VocaloFlow evaluation. Uses Applio (RVC fork) to fine-tune on Rachie vocal data and run inference on prior audio. Supports two modes: **quality** (Context 1, converts Teto `prior.wav`) and **controllability** (Context 2, converts DALI `dali_prior.wav`). The thesis argument: ContentVec cannot extract meaningful content from synthetic Vocaloid singing, so RVC fails where VocaloFlow succeeds.

## Quick Start

All commands run from `RVCBaseline/`.

```bash
# 1. Prepare training data (copies target.wav from train split)
python prepare_training_data.py --config configs/default.yaml

# 2. Train (3-step pipeline: preprocess -> extract -> train)
python run_training.py --config configs/default.yaml

# 3a. Run inference on eval set (Context 1: quality)
python run_inference.py --rvc-model-path <path/to/model.pth> --max-eval-chunks 50

#use this one
python run_inference.py --rvc-model-path "../Applio/logs/Rachie-RVC/Rachie-RVC_170e_47260s_best_epoch.pth" --max-eval-chunks 50

# 3b. Run inference on DALI priors (Context 2: controllability)
python run_inference.py --rvc-model-path "../Applio/logs/Rachie-RVC/Rachie-RVC_170e_47260s_best_epoch.pth" \
    --mode controllability --max-eval-chunks 50

# 4. Evaluate (from Evaluation/)
cd ../Evaluation
python run_eval.py --checkpoint <vf_checkpoint> --rvc-audio-dir ../RVCBaseline/output \
    --rvc-dali-audio-dir ../RVCBaseline/output_dali --eval-context both
```

## Directory Layout

```
RVCBaseline/
├── README.md                        # This file (memory palace)
├── rvc_config.py                    # RVCBaselineConfig dataclass
├── rvc_utils/
│   ├── __init__.py                  # Re-exports
│   └── imports.py                   # Path constants, import_from_path, timestamp
├── prepare_training_data.py         # Step 1: aggregate target.wav for RVC
├── run_training.py                  # Step 2: drive Applio fine-tuning
├── run_inference.py                 # Step 3: batch inference on val set
├── configs/
│   └── default.yaml                 # Default config values
├── training_data/                   # (gitignored) flat WAV dir for RVC training
└── output/                          # (gitignored) RVC inference outputs
```

External dependency: `../Applio/` (cloned repo, gitignored).

## File Descriptions

### rvc_config.py

`RVCBaselineConfig` dataclass with YAML serialization. Data-selection fields (`data_dir`, `manifest_path`, `max_dtw_cost`, `val_fraction`, `seed`) must match `EvalConfig` defaults to ensure the same validation set is used by both `run_inference.py` and `run_eval.py`.

- `to_yaml(path)` — serialize config to YAML
- `from_yaml(path)` — deserialize config from YAML

### rvc_utils/imports.py

Path constants and cross-module import helpers. Same pattern as `Evaluation/eval_utils/imports.py`.

- `RVC_BASELINE_DIR`, `REPO_ROOT`, `VOCALOFLOW_DIR`, `EVAL_DIR`, `APPLIO_DIR`, `DATA_DIR` — absolute paths
- `import_from_path(module_name, file_path)` — load a Python file as a module without polluting sys.path
- `timestamp()` — formatted current time string

### prepare_training_data.py

Aggregates SoulX-Singer `target.wav` files from the training split into a flat directory for RVC consumption.

- Imports `load_manifest`, `filter_manifest`, `split_by_song` from VocaloFlow via `import_from_path`
- Takes the TRAIN split (complement of val)
- Subsamples to `max_training_chunks` (default 1000, ~45 min audio)
- Copies each `target.wav` to `training_data/{index:05d}.wav`

CLI flags: `--config`, `--data-dir`, `--manifest`, `--max-training-chunks`, `--training-data-dir`, `--seed`

### run_training.py

Wraps Applio's 3-step training pipeline via subprocess calls to `core.py`:

1. `preprocess` — slice audio, resample to 40kHz
2. `extract` — extract ContentVec embeddings + RMVPE F0
3. `train` — fine-tune RVC model with pretrained initialization

- `run_preprocess(config, cpu_cores, dry_run)` — calls `core.py preprocess`
- `run_extract(config, cpu_cores, gpu, dry_run)` — calls `core.py extract`
- `run_train(config, gpu, dry_run)` — calls `core.py train`

CLI flags: `--config`, `--dry-run`, `--gpu`, `--cpu-cores`, `--rvc-model-name`, `--rvc-epochs`, `--step`

Output model lands in `../Applio/logs/{model_name}/`.

### run_inference.py

Runs RVC voice conversion on val-set audio files. Produces output files with the exact naming convention the eval pipeline expects: `{dali_id}_{chunk_name}.wav`.

- Supports two modes via `--mode`:
  - `quality` (default): Converts `data_dir/{dali_id}/{chunk_name}/prior.wav` → `output/{dali_id}_{chunk_name}.wav`
  - `controllability`: Converts `dali_ustx_dir/{dali_id}/{chunk_name}/dali_prior.wav` → `output_dali/{dali_id}_{chunk_name}.wav`
- Imports `get_val_manifest` from `Evaluation/eval_utils/val_set.py` to select the same chunks as eval
- Calls Applio CLI `infer` per file via subprocess
- Skips chunks where output already exists (resumable)
- Logs per-chunk success/failure, continues on errors

CLI flags: `--config`, `--rvc-model-path`, `--rvc-index-path`, `--data-dir`, `--manifest`, `--max-eval-chunks`, `--output-dir`, `--f0-shift`, `--index-ratio`, `--mode`, `--dali-ustx-dir`, `--dali-output-dir`

## Eval Pipeline Integration

The eval pipeline (`Evaluation/run_eval.py`) already supports RVC via:

- `EvalConfig.rvc_audio_dir` — directory containing RVC output WAVs (Context 1)
- `EvalConfig.rvc_dali_audio_dir` — directory containing RVC output WAVs from DALI priors (Context 2)
- CLI flags: `--rvc-audio-dir`, `--rvc-dali-audio-dir`
- File naming: `{dali_id}_{chunk_name}.wav`
- Context 1 metrics: `rvc_f0_rmse`, `rvc_f0_pearson`, `rvc_gpe/vde/ffe`, `rvc_mcd`, `rvc_wer`, `rvc_mos`
- Context 2 metrics: `rvc_f0_rmse`, `rvc_f0_pearson`, `rvc_gpe/vde/ffe`, `rvc_onset_mae` (all vs DALI F0)
- Audio auto-resampled to 24kHz via `librosa.load(sr=24000)`

## Configuration Reference

| Field | Default | Description |
|-------|---------|-------------|
| `data_dir` | `../Data/Rachie` | Root data directory |
| `manifest_path` | `../Data/Rachie/manifest.csv` | Manifest CSV |
| `max_dtw_cost` | `100.0` | DTW quality filter |
| `val_fraction` | `0.05` | Validation split fraction |
| `seed` | `42` | Random seed |
| `max_training_chunks` | `1000` | Training data cap |
| `training_data_dir` | `./training_data` | Flat WAV output dir |
| `rvc_model_name` | `Rachie-RVC` | Applio model name |
| `rvc_epochs` | `300` | Training epochs |
| `rvc_sample_rate` | `40000` | RVC internal sample rate |
| `f0_method` | `rmvpe` | Pitch extraction method |
| `max_eval_chunks` | `0` | Eval chunks (0 = all val) |
| `output_dir` | `./output` | Inference output dir |
| `index_ratio` | `0.0` | Retrieval index influence |
| `f0_shift` | `0` | Pitch shift in semitones |
| `mode` | `quality` | `quality` or `controllability` |
| `dali_ustx_dir` | `../Evaluation/dali_ustx` | DALI USTX prior directory |
| `dali_output_dir` | `./output_dali` | Output dir for controllability mode |
| `rvc_model_path` | `""` | Trained model .pth path |
| `rvc_index_path` | `""` | FAISS index path |
