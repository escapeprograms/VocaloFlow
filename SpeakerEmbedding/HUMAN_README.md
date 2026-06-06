# SpeakerEmbedding

Extracts 192-dim ECAPA-TDNN speaker identity vectors for VocaloFlow conditioning. Used when VocaloFlow is configured with `use_speaker_embedding=True`.

## Environment

Uses the `vocaloflow` conda environment. If you haven't created it yet, see [VocaloFlow/HUMAN_README.md](../VocaloFlow/HUMAN_README.md#environment).

## Prerequisites

- The SpeechBrain ECAPA-TDNN model downloads automatically on first run (cached to `checkpoints/ecapa-tdnn/`)

## Usage

Run from the **repository root**:

```bash
conda activate vocaloflow
```

**Extract a global speaker embedding** from reference audio clips:
```bash
python SpeakerEmbedding/extract_embedding.py --audio-files clip1.wav clip2.wav --name Rachie
```

This creates `SpeakerEmbedding/embeddings/Rachie/speaker_embedding.pt`. Point VocaloFlow's `global_speaker_embedding_path` config to this file.

**Precompute per-chunk embeddings** (multi-speaker mode):
```bash
python SpeakerEmbedding/precompute_embeddings.py --data-dir Data/Rachie --manifest Data/Rachie/manifest.csv
```

Creates `speaker_embedding.npy` (shape: 192) in each chunk directory.
