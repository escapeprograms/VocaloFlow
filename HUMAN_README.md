# VocaloFlow Thesis Project

Conditional flow matching model that transforms low-quality Vocaloid mel-spectrograms into high-quality singing voice audio, conditioned on F0 and phoneme information.

## System Requirements

- Windows 10/11
- NVIDIA GPU with CUDA support
- Conda (Miniconda or Anaconda)
- .NET 9 SDK (for OpenUtau prior generation)
- Git

## Environments

This project uses five conda environments. Create only the ones you need.

| Environment | Create with | Used by |
|---|---|---|
| `vocaloflow` | `conda env create -f VocaloFlow/environment.yml` | VocaloFlow, AdversarialFinetune, AdversarialPostnet, PL-BERT, SpeakerEmbedding, DataAnalysis |
| `vocaloflow-datasynthesizer` | `conda env create -f DataSynthesizer/environment.yml` | DataSynthesizer, Evaluation Step 2 |
| `soulxsinger` | `conda env create -f SoulX-Singer/environment.yml` | SoulX-Singer (subprocess only, never activate directly) |
| `vocaloflow-eval` | `conda env create -f Evaluation/environment.yml` | Evaluation Step 3 |
| `rvc` | `conda env create -f RVCBaseline/environment.yml` | RVCBaseline |

## Modules

| Module | Description | Docs |
|---|---|---|
| [DataSynthesizer](DataSynthesizer/HUMAN_README.md) | Generate paired training data from DALI annotations | Also see [SETUP.md](DataSynthesizer/SETUP.md) |
| [VocaloFlow](VocaloFlow/HUMAN_README.md) | Train and run inference with the core flow matching model | |
| [AdversarialFinetune](AdversarialFinetune/HUMAN_README.md) | Experiment 2: adversarial flow-matching fine-tuning | |
| [AdversarialPostnet](AdversarialPostnet/HUMAN_README.md) | Experiment 1: post-network mel sharpening | |
| [PL-BERT](PL-BERT/HUMAN_README.md) | Precompute frozen phoneme embeddings for training | |
| [SpeakerEmbedding](SpeakerEmbedding/HUMAN_README.md) | Extract ECAPA-TDNN speaker identity vectors | |
| [Evaluation](Evaluation/HUMAN_README.md) | Run thesis evaluation metrics (F0, WER, MOS, MCD, etc.) | |
| [RVCBaseline](RVCBaseline/HUMAN_README.md) | RVC voice conversion baseline for comparison | |
| [DataAnalysis](DataAnalysis/HUMAN_README.md) | Jupyter notebooks for EDA and diagnostics | |

## Typical Workflow

1. **Synthesize training data** — Set up DataSynthesizer and SoulX-Singer, generate paired (prior, target) audio from DALI
2. **Precompute features** (optional) — Run PL-BERT and/or SpeakerEmbedding precomputation
3. **Train VocaloFlow** — Train the base model on synthesized data
4. **Fine-tune** (optional) — Run AdversarialFinetune or AdversarialPostnet experiments
5. **Train RVC baseline** (optional) — Train the RVC comparison model
6. **Evaluate** — Run the evaluation pipeline to compute metrics across all models
