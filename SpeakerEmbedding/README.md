# SpeakerEmbedding Memory Palace

ECAPA-TDNN speaker embedding extraction for VocaloFlow adversarial fine-tuning. Provides a fixed 192-dim speaker identity vector injected into the generator's AdaLN timestep conditioning and concatenated to the discriminator's mel input.

## Overview

Uses SpeechBrain's `spkrec-ecapa-voxceleb` pretrained model to extract 192-dim speaker embeddings. Each embedding is L2-normalized. The embedding is a global per-speaker vector (not per-frame), representing the target speaker's voice identity.

**Two loading modes** (controlled by `VocaloFlowConfig`):
1. **Global embedding** (recommended for single-speaker): Set `global_speaker_embedding_path` to a `.pt` file. Loaded once in `VocaloFlowDataset.__init__`, returned for every item. No per-chunk files needed.
2. **Per-chunk embedding** (multi-speaker fallback): Leave `global_speaker_embedding_path` empty. Each chunk dir must contain `speaker_embedding.npy`. Use `precompute_embeddings.py` to create these.

**Generator integration**: `speaker_proj = Linear(192, 512)` (zero-init) adds to timestep conditioning `c` in `VocaloFlow/model/vocaloflow.py`. Controlled by `VocaloFlowConfig.use_speaker_embedding`.

**Discriminator integration**: `speaker_proj = Linear(192, 64)` (zero-init) broadcasts across time and concatenates with mel input in `AdversarialFinetune/dit_discriminator.py`. Controlled by `FinetuneConfig.disc_use_speaker_input`.

**CFG**: Speaker embedding is NOT dropped during classifier-free guidance dropout. It's a fixed property of the target voice.

## Directory Structure

```
SpeakerEmbedding/
├── __init__.py
├── README.md                    # This file
├── extract_embedding.py         # Extract averaged embedding from audio clips
├── precompute_embeddings.py     # Batch per-chunk precomputation (multi-speaker)
├── checkpoints/                 # ECAPA-TDNN model cache (auto-downloaded)
│   └── ecapa-tdnn/
└── embeddings/                  # Output directory for extracted embeddings
    └── Rachie/
        └── speaker_embedding.pt # (192,) float32, L2-normalized
```

## extract_embedding.py

### `load_ecapa_model(device) -> EncoderClassifier`
Loads SpeechBrain ECAPA-TDNN from hub. Caches model weights to `checkpoints/ecapa-tdnn/`.

### `extract_single(model, audio_path, target_sr=16000) -> Tensor`
Loads audio, resamples to 16kHz mono, returns L2-normalized (192,) embedding.

### `extract_averaged(model, audio_paths, target_sr=16000) -> Tensor`
Extracts embeddings from multiple clips, prints pairwise cosine similarities, returns L2-normalized average.

### CLI
```bash
# With --name (recommended): creates embeddings/<name>/speaker_embedding.pt
python SpeakerEmbedding/extract_embedding.py \
    --audio-files clip1.wav clip2.wav --name Rachie

# With explicit --output
python SpeakerEmbedding/extract_embedding.py \
    --audio-files clip1.wav clip2.wav --output path/to/output.pt
```

## precompute_embeddings.py

Batch per-chunk precomputation for multi-speaker datasets. Not needed when using global embedding mode.

### CLI
```bash
python SpeakerEmbedding/precompute_embeddings.py \
    --data-dir Data/Rachie --manifest Data/Rachie/manifest.csv
```

## Data Format

- **Global mode**: `.pt` file containing a single `(192,)` float32 tensor, L2-normalized. Pointed to by `VocaloFlowConfig.global_speaker_embedding_path`.
- **Per-chunk mode**: `speaker_embedding.npy` in each chunk directory, shape `(192,)`, dtype float32, L2-normalized.

Both modes produce `(B, 192)` batch tensors after collation.
