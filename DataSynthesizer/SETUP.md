# DataSynthesizer Setup Guide

This guide walks through setting up the DataSynthesizer pipeline on a fresh machine. The pipeline uses two conda environments — a lightweight parent env for orchestration and a GPU-heavy subprocess env for SoulX-Singer inference.

## Prerequisites

- **OS**: Windows 10/11 (the pipeline uses .NET interop via pythonnet for OpenUtau)
- **GPU**: NVIDIA GPU with CUDA 11.8+ support
- **Conda**: Miniconda or Anaconda installed ([install guide](https://docs.conda.io/en/latest/miniconda.html))
- **.NET 9 SDK**: Required to build UtauGenerate.dll ([download](https://dotnet.microsoft.com/download/dotnet/9.0))
- **Git**: For cloning the repository

## 1. Clone the Repository

```bash
git clone <repo-url>
cd "Honors Thesis"
```

The expected top-level layout is:

```
Honors Thesis/
├── DataSynthesizer/    # This pipeline
├── SoulX-Singer/       # Singing voice synthesis model (vendored)
├── VocaloFlow/         # Flow matching model (separate)
├── API/
│   ├── OpenUtau/       # OpenUtau source (for reference)
│   └── UtauGenerate/   # C# wrapper DLL for OpenUtau
├── DALI/               # DALI dataset annotations
├── VocalPrompts/       # Voice prompt audio (Rachie multi-register)
└── Data/               # Pipeline output (created automatically)
```

## 2. Download External Data & Models

### DALI Dataset

Download the DALI v2.0 annotations and extract them:

```
Honors Thesis/DALI/
├── DALI_v2.0/
│   └── annot_tismir/   # .gz annotation files (one per song)
└── README.md
```

The annotations are gzipped pickle files. The pipeline reads them via the `DALI-dataset` pip package.

### SoulX-Singer Pretrained Models

```bash
pip install -U huggingface_hub

# SVS model
huggingface-cli download Soul-AILab/SoulX-Singer --local-dir SoulX-Singer/pretrained_models/SoulX-Singer

# Preprocessing models (RMVPE for F0 extraction)
huggingface-cli download Soul-AILab/SoulX-Singer-Preprocess --local-dir SoulX-Singer/pretrained_models/SoulX-Singer-Preprocess
```

Verify:
```
SoulX-Singer/pretrained_models/
├── SoulX-Singer/
│   └── model.pt
└── SoulX-Singer-Preprocess/
    └── rmvpe/
        └── rmvpe.pt
```

### Voice Prompts

The pipeline uses voice prompts for SoulX-Singer's zero-shot synthesis. WillStetson's prompt ships with SoulX-Singer:

```
SoulX-Singer/example/transcriptions/WillStetsonSample/
├── vocal.wav
└── metadata.json
```

For Rachie (multi-register), place prompts at:

```
VocalPrompts/Rachie/
├── rachie_low/
│   ├── vocal.wav
│   └── metadata.json
├── rachie_mid/
│   ├── vocal.wav
│   └── metadata.json
└── rachie_high/
    ├── vocal.wav
    └── metadata.json
```

See `DataSynthesizer/voice_providers.py` for prompt path definitions and how to add new providers.

### UTAU Singer Voicebank

The pipeline generates prior audio via OpenUtau using the Kasane Teto voicebank. Place it at:

```
API/UtauGenerate/bin/Release/net9.0/Singers/重音テト音声ライブラリー/
```

## 3. Build UtauGenerate.dll

The pipeline calls OpenUtau through a C# wrapper. Build it with .NET 9:

```bash
cd API/UtauGenerate
dotnet build -c Release
```

This produces `API/UtauGenerate/bin/Release/net9.0/UtauGenerate.dll`. The pipeline finds it automatically via the default path in `config.py` (repo root).

## 4. Create Conda Environments

The pipeline uses two environments:

### 4a. Parent Environment (`vocaloflow-datasynthesizer`)

This is the orchestration env. It is CPU-only (no CUDA), but it does include a CPU
build of torch — `DataSynthesizer/utils/vocoders.py` uses it for mel<->audio.

```bash
conda env create -f DataSynthesizer/environment.yml
```

This creates `vocaloflow-datasynthesizer` (~75 exactly-pinned pip packages: librosa, numpy, pythonnet, g2p, DALI, CPU torch, etc.).

### 4b. Subprocess Environment (`soulxsinger`)

This is the GPU-heavy env for SoulX-Singer inference and RMVPE note extraction.
Create it from the environment.yml, which reproduces the exact working env
(CUDA 11.8 torch build + the transformers/tokenizers pins below) in one command:

```bash
conda env create -f SoulX-Singer/environment.yml
```

> **Why a dedicated `SoulX-Singer/environment.yml`?** It is our authoritative spec.
> `SoulX-Singer/requirements.txt` is the **vendored upstream** list and is left untouched;
> installing from it directly gives CPU-only torch + `transformers>=4.53.0` on Windows,
> which breaks SoulX-Singer. The two load-bearing overrides baked into `environment.yml` are:
> - **torch 2.2.0 + CUDA 11.8** (cu118 wheels) instead of upstream's CPU-only torch.
> - **transformers==4.42.4 / tokenizers==0.19.1** — SoulX-Singer's model calls
>   `LlamaAttention.forward()` without the `position_embeddings` arg added in transformers 4.53.

### Verify Both Environments

```bash
# Parent env: imports work; torch is present but CPU-only (no CUDA)
conda run -n vocaloflow-datasynthesizer python -c "import librosa, numpy, pythonnet; print('parent ok')"
conda run -n vocaloflow-datasynthesizer python -c "import torch; print('cpu torch', torch.__version__)"  # 2.10.0+cpu

# Subprocess env: should have torch + CUDA
conda run -n soulxsinger python -c "import torch; print(torch.__version__, 'cuda:', torch.cuda.is_available())"
# Expected: 2.2.0+cu118 cuda: True
```

## 5. Configure Paths (Optional)

The pipeline resolves paths via `config.py` (repo root), which reads from environment variables (or a `.env` file) with sensible defaults:

| Variable | Default | Purpose |
|----------|---------|---------|
| `SOULX_PYTHON` | `<conda>/envs/soulxsinger/python.exe` | Python interpreter for subprocess env |
| `UTAU_GENERATE_DLL` | `../API/UtauGenerate/bin/Release/net9.0/UtauGenerate.dll` | OpenUtau C# bridge |

**If your conda is at a non-standard location**, create a `.env` file in `DataSynthesizer/`:

```bash
cp DataSynthesizer/.env.example DataSynthesizer/.env
# Edit .env with your actual paths
```

Otherwise, the defaults will work if your repo layout matches section 1.

## 6. Run the Pipeline

All commands are run from the repository root.

**Single song** (good for testing setup):
```bash
conda run -n vocaloflow-datasynthesizer python DataSynthesizer/pipelines/synthesize_v2.py \
    --dali_id 00070c7c333849e4a3725b906c339042 --mode line --provider WillStetson
```

**Full English dataset**:
```bash
conda run -n vocaloflow-datasynthesizer python DataSynthesizer/pipelines/synthesize_dataset_v2.py \
    --phases 12345 --songs_per_batch 100 --provider Rachie
```

**Resume after crash** (e.g. re-run extraction + alignment only):
```bash
conda run -n vocaloflow-datasynthesizer python DataSynthesizer/pipelines/synthesize_dataset_v2.py \
    --phases 34 --songs_per_batch 50 --provider Rachie
```

The pipeline is resumable per-chunk — each phase checks for sentinel files before re-processing.

## Troubleshooting

### `AssertionError: Torch not compiled with CUDA enabled`
The `soulxsinger` env has CPU-only torch. Recreate it from `SoulX-Singer/environment.yml`, or force-reinstall the CUDA build:
```bash
pip install torch==2.2.0+cu118 torchaudio==2.2.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118 --force-reinstall --no-deps
```

### `LlamaAttention.forward() missing 1 required positional argument: 'position_embeddings'`
The `transformers` version is too new. Pin it: `pip install transformers==4.42.4 tokenizers==0.19.1`

### `System.IO.IOException: ... prefs.json`
Intermittent OpenUtau file lock race condition. Non-fatal — the pipeline recovers and continues. If a specific chunk's `prior.wav` ends up empty (46 bytes), delete that chunk's directory and re-run phase 4.

### `ModuleNotFoundError: No module named 'DALI'`
The `DALI-dataset` pip package is missing from the parent env. Run: `conda run -n vocaloflow-datasynthesizer pip install DALI-dataset==1.1`

### OpenUtau produces silence / empty priors
Ensure the Teto voicebank is at `API/UtauGenerate/bin/Release/net9.0/Singers/` and that `.NET 9` runtime is installed. The Arpasing+ phonemizer must be discoverable at `API/UtauGenerate/bin/Release/net9.0/OpenUtau.Plugin.Builtin.dll`.
