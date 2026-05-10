# dataAnalysis — Diagnostic Scripts

Ad-hoc analysis and visualization scripts for debugging VocaloFlow evaluation results.

## F0_comparison.py

Plots F0 curves from three sources on a single chart to diagnose pitch tracking:

| Curve | Source | Color |
|-------|--------|-------|
| DALI GT F0 | Step-function from DALI note annotations (Hz) | Green |
| SoulX target F0 | Pre-extracted `target_f0.npy` from SoulX preprocessing | Blue |
| VocaloFlow F0 | RMVPE extraction from VF inference output | Red |

The script automatically picks the first valid DALI USTX sample, runs VocaloFlow inference using DALI GT notes as F0 conditioning, and saves audio + plot.

### Prerequisites

- DALI USTX priors must be pre-generated (see main Evaluation README, Step 2)
- VocaloFlow checkpoint at `VocaloFlow/checkpoints/4-27-wn/checkpoint_150000.pt`
- RMVPE weights at `SoulX-Singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt`

### Usage

Run from the `Evaluation/` directory:

```bash
conda activate vocaloflow-eval
cd Evaluation
python -m dataAnalysis.F0_comparison
```

### Output

All files are saved to `dataAnalysis/f0_comparison_output/`:

```
f0_comparison_output/
├── {dali_id}_{chunk}_ustx_prior.wav      # DALI USTX prior audio (copy)
├── {dali_id}_{chunk}_soulx_target.wav    # SoulX-Singer target audio (copy)
├── {dali_id}_{chunk}_vocaloflow.wav      # VocaloFlow inference output
└── {dali_id}_{chunk}_f0_comparison.png   # F0 overlay plot
```

### Configuration

Edit constants at the top of the script to change checkpoint, ODE steps, CFG scale, etc.
