# DataSynthesizer

Generates paired (prior, target) training data from DALI dataset annotations using SoulX-Singer and OpenUtau.

## Environment

This module uses **two** conda environments working together:

- `vocaloflow-datasynthesizer` — orchestration (you activate this one)
- `soulxsinger` — GPU inference subprocess (launched automatically, never activate directly)

For full setup instructions including external data downloads, .NET build, and voicebank installation, see **[SETUP.md](SETUP.md)**.

## Quick Reference

All commands run from `DataSynthesizer/`:

```bash
conda activate vocaloflow-datasynthesizer
```

**Single song** (good for testing your setup):
```bash
python pipelines/synthesize_v2.py --dali_id 00070c7c333849e4a3725b906c339042 --mode line --provider WillStetson
```

**Full dataset**:
```bash
python pipelines/synthesize_dataset_v2.py --phases 12345 --songs_per_batch 100 --provider Rachie
```

**Resume after crash** (e.g. re-run extraction + alignment only):
```bash
python pipelines/synthesize_dataset_v2.py --phases 34 --songs_per_batch 50 --provider Rachie
```

## Outputs

Per-chunk artifacts land in `Data/<provider>/<dali_id>/<chunk_name>/` (target.wav, prior.wav, mels, F0, phoneme masks, etc.). A `manifest.csv` is generated per provider directory.
