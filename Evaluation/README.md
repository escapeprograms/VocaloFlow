# Evaluation — VocaloFlow Thesis Evaluation Pipeline

Modular, toggleable evaluation pipeline computing thesis metrics across VocaloFlow, RVC (SVC baseline), and SoulX-Singer (SVS baseline).

## Full Pipeline: Start to Finish

### Step 1: Pre-generate RVC baseline audio (one-time)

Requires Applio + a trained RVC model. Run from `RVCBaseline/`:

```bash
cd RVCBaseline

# 1a. Prepare training data (copies target.wav files from training split)
python prepare_training_data.py --config configs/default.yaml

# 1b. Train RVC model (preprocess → extract features → train, ~300 epochs)
python run_training.py --config configs/default.yaml

# 1c. Run RVC inference on validation set priors → output/{dali_id}_{chunk_name}.wav
python run_inference.py \
    --rvc-model-path "../Applio/logs/Rachie-RVC/Rachie-RVC_170e_47260s_best_epoch.pth" \
    --max-eval-chunks 50
```

### Step 2: Pre-generate DALI USTX priors (one-time, needs pythonnet)

Creates USTX + prior WAV from DALI MIDI annotations for controllability eval. Run from `Evaluation/` in the datasynthesizer env:

```bash
conda activate vocaloflow-datasynthesizer
cd Evaluation
python generate_dali_ustx.py --output-dir dali_ustx --max-chunks 50
```

### Step 3: Run evaluation

Run from `Evaluation/` in the eval env:

```bash
conda activate vocaloflow-eval
cd Evaluation

# Both contexts (quality + controllability) with RVC baseline
python run_eval.py \
    --checkpoint ../VocaloFlow/checkpoints/4-27-wn/checkpoint_150000.pt \
    --rvc-audio-dir ../RVCBaseline/output \
    --eval-context both \
    --max-eval-chunks 50

# Quick smoke test (skip heavy Whisper/MOS models)
python run_eval.py \
    --checkpoint ../VocaloFlow/checkpoints/4-27-wn/checkpoint_150000.pt \
    --eval-context quality \
    --max-eval-chunks 5 \
    --disable-wer --disable-mos

# Controllability only
python run_eval.py \
    --checkpoint ../VocaloFlow/checkpoints/4-27-wn/checkpoint_150000.pt \
    --eval-context controllability \
    --max-eval-chunks 50
```

## Directory Layout

```
Evaluation/
├── __init__.py
├── README.md                       # This file (memory palace)
├── run_eval.py                     # Main orchestrator
├── eval_config.py                  # EvalConfig dataclass + YAML
├── metrics/
│   ├── __init__.py
│   ├── f0_metrics.py               # F0 RMSE, Pearson, GPE/VDE/FFE
│   ├── mcd.py                      # Mel Cepstral Distortion
│   ├── wer.py                      # Whisper WER
│   ├── mos.py                      # UTMOS MOS prediction
│   ├── duration.py                 # Onset MAE (F0 onset detection + matching)
│   └── contentvec.py               # ContentVec embedding analysis
├── generate_dali_ustx.py               # CLI: pre-generate DALI USTX priors
└── eval_utils/
    ├── __init__.py
    ├── imports.py                  # Cross-module import helpers
    ├── audio.py                    # Vocoding + F0 extraction wrappers
    ├── dali_ref.py                 # DALI ground-truth extraction
    ├── dali_ustx.py                # DALI → USTX conversion for fair controllability
    └── val_set.py                  # Validation set loading
```

## eval_config.py

`EvalConfig` dataclass with YAML serialization. Key fields:

- `checkpoint_path`: VocaloFlow checkpoint .pt file
- `eval_context`: `"quality"`, `"controllability"`, or `"both"`
- `rvc_audio_dir`: Directory of pre-generated RVC audio (CLI: `--rvc-audio-dir`). Files named `{dali_id}_{chunk_name}.wav`.
- `dali_ustx_dir`: Directory with pre-generated DALI USTX priors (CLI: `--dali-ustx-dir`). Default `./dali_ustx`.
- `enable_*` toggles: Each metric independently activatable
- `max_eval_chunks`: Cap on validation set size (0 = all)
- `needs_audio()`: Returns True if any enabled metric needs vocoded audio
- `needs_f0_extraction()`: Returns True if any metric needs RMVPE F0

## eval_utils/imports.py

Path constants and cross-module import helpers. Follows AdversarialFinetune pattern.

- `REPO_ROOT`, `VOCALOFLOW_DIR`, `DATASYNTHESIZER_DIR`, `SOULX_DIR`: Absolute paths
- `import_from_path(name, path)`: Load a Python file as a named module (avoids utils collision)
- `setup_vocaloflow_sys_path()`: Adds VocaloFlow/ and SoulX-Singer/ to sys.path
- `timestamp()`: HH:MM:SS string for log lines

## eval_utils/audio.py

Wraps vocoding and F0 extraction from existing codebase.

- `vocode_mel(mel)`: (T, 128) normalized log-mel → 1D audio at 24 kHz via SoulX Vocos
- `extract_f0(audio_or_path, rmvpe_path, device)`: RMVPE F0 extraction, accepts array or file path, caches the extractor singleton
- `denormalize_mel(mel)`: Undo SoulX z-score normalization (mean=-4.92, var=8.14)
- `audio_to_mel(audio, sr)`: 1D audio → (T, 128) normalized log-mel via `mel_to_soulx_mel()` from DataSynthesizer vocoders. Used for RVC MCD computation.

## eval_utils/dali_ref.py

DALI ground-truth extraction for Context 2 (controllability). DALI stores frequencies directly in Hz as `(start_Hz, end_Hz)` tuples.

- `load_dali_entry(dir, id)`: Load+cache DALI annotation from .gz file
- `get_chunk_time_range(entry, chunk_name)`: Map `line_N` to absolute seconds
- `extract_dali_f0_reference(entry, start, end, T)`: Frame-level step-function F0 from DALI notes
- `extract_dali_lyrics(entry, start, end)`: Word-level lyrics for WER reference
- `extract_dali_note_timings(entry, start, end)`: Note onset/offset for onset MAE

## eval_utils/val_set.py

Validation set loading using VocaloFlow's existing manifest infrastructure.

- `get_val_manifest(...)`: Load manifest → filter DTW cost → split by song → return val portion
- `get_chunk_dir(data_dir, dali_id, chunk_name)`: Absolute path to chunk directory

## metrics/f0_metrics.py

F0 evaluation metrics operating on (T,) Hz arrays.

- `f0_rmse(pred, ref)`: RMSE in Hz over mutually voiced frames
- `f0_pearson(pred, ref)`: Pearson correlation over mutually voiced frames
- `compute_gpe_vde_ffe(pred, ref)`: Standard definitions — GPE (20% threshold on voiced), VDE (voicing disagreement on all), FFE (union of GPE and VDE on all frames). Returns percentages [0-100].

## metrics/mcd.py

Mel Cepstral Distortion computed directly from log-mel spectrograms (no audio round-trip).

- `compute_mcd(pred_mel, target_mel, n_mfcc=13)`: Denormalize → DCT → MFCC distance → scale to dB. Uses coefficients 1..n_mfcc (skip energy). Scale factor: 10*sqrt(2)/ln(10).

## metrics/wer.py

Word Error Rate via Whisper transcription. Both VocaloFlow and SoulX-Singer are evaluated against DALI ground-truth lyrics (not against each other).

- `transcribe(audio, sr, model_name, device)`: Whisper transcription → normalized text
- `compute_wer(hypothesis, reference)`: WER via jiwer. Normalizes: lowercase, strip punctuation.
- Output columns: `wer_vf` (VocaloFlow vs DALI), `wer_sx` (SoulX vs DALI), `wer_teto` (Teto prior vs DALI), `rvc_wer` (RVC vs DALI)

Whisper model is cached as singleton.

## metrics/mos.py

Mean Opinion Score prediction via UTMOS (SingMOS-Pro fallback).

- `compute_mos(audio, sr, device)`: Resamples to 16 kHz, runs UTMOS, returns [1, 5] score.

UTMOS model loaded from `tarepan/SpeechMOS:v1.2.0` via torch.hub.

## metrics/duration.py

Onset MAE via F0 onset detection. Detects note onsets from F0 contours and compares against DALI reference onsets.

- `detect_note_onsets(f0, sr, hop, semitone_threshold)`: Find onsets via voicing transitions (unvoiced→voiced) and pitch jumps (>1 semitone). Returns onset times in seconds.
- `compute_onset_mae(pred_onsets, ref_onsets, max_dist_s)`: Greedy nearest-neighbor matching of onset times with 200ms max threshold. Returns MAE in milliseconds.
- `extract_dali_onset_times(dali_notes)`: Extract `start_s` values from DALI note timing dicts as a sorted array.

## metrics/contentvec.py

ContentVec embedding analysis for SVC baseline motivation.

- `extract_features(audio, sr, device)`: HuBERT features at 16 kHz → (T, 768)
- `compute_cosine_similarity(feats_a, feats_b)`: Per-frame cosine similarity stats

## eval_utils/dali_ustx.py

DALI annotation → OpenUTAU USTX conversion for fair controllability evaluation. Creates USTX + prior WAV files from DALI MIDI annotations, bypassing SoulX-Singer so VocaloFlow gets independent DALI-derived conditioning. No pythonnet at module level — only loaded inside generation functions.

- `extract_dali_notes_with_words(entry, start, end)`: Like `extract_dali_note_timings` but preserves note→word mapping via `note['index']` for proper word-start/continuation assignment. Extends durations to fill inter-note gaps.
- `format_notes_for_player(notes, use_phonemes)`: Convert notes to `(position_ticks, length_ticks, midi_pitch, text)` tuples with g2p_en phoneme hints (ARPAbet). Copies `_normalize_arpabet()` and `_distribute_phonemes()` from synthesizePrior.py locally.
- `generate_dali_ustx_for_chunk(entry, chunk_name, output_dir, player)`: End-to-end: extract notes → format → Player.addNote() → Player.export() → `dali_prior.ustx` + `dali_prior.wav`.
- `generate_all_dali_ustx(manifest_df, dali_annot_dir, output_dir)`: Initializes OpenUTAU Player, iterates validation manifest, calls per-chunk generation. Returns `{(dali_id, chunk_name): (ustx_path, wav_path)}`.

## generate_dali_ustx.py

CLI script for pre-generating DALI USTX priors. Usage:
```bash
python -m Evaluation.generate_dali_ustx --output-dir Evaluation/dali_ustx [--max-chunks 5]
```
Loads validation manifest, initializes OpenUTAU Player (requires pythonnet), generates `dali_prior.ustx` and `dali_prior.wav` per chunk.

## run_eval.py

Main orchestrator. Processing flow per chunk:

1. Load pre-computed arrays (prior_mel, target_mel, F0, voicing, phonemes, PL-BERT)
2. Resolve phoneme indirection
3. VocaloFlow inference via `infer_chunked()` → pred_mel
4. Vocode pred_mel → pred_audio (if audio metrics enabled)
5. Load target.wav directly (avoids double-vocoding)
6. Extract pred_f0 via RMVPE
7. Compute all enabled metrics (incl. `wer_teto` for Teto prior intelligibility floor)
8. RVC baseline metrics (if `--rvc-audio-dir` provided): load RVC audio, extract F0, compute `rvc_f0_rmse`, `rvc_f0_pearson`, `rvc_gpe/vde/ffe`, `rvc_mcd` (via `audio_to_mel()`), `rvc_wer`, `rvc_mos`
9. Context 2 (controllability): Load pre-generated DALI USTX + prior WAV → parse USTX → extract conditioning (F0 from note pitches, not SoulX) → VF inference → compare output F0 vs DALI reference. Uses `_eval_controllability()` which lazy-loads `pipeline.py` functions. Also includes SX metrics (`sx_*`) for side-by-side comparison. Requires pre-generated USTX via `generate_dali_ustx.py`.

Outputs per-sample CSV, summary CSV, and config YAML to `output_dir/`.

## Audio Parameters

- SR = 24000, HOP = 480, n_mels = 128
- Frame rate: 50 fps (20 ms/frame)
- Mel normalization: log-compressed + z-score (mean=-4.92, var=8.14)
