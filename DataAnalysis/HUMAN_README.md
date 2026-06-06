# DataAnalysis

Jupyter notebooks for exploratory data analysis and diagnostics.

## Environment

Uses the `vocaloflow` conda environment. If you haven't created it yet, see [VocaloFlow/HUMAN_README.md](../VocaloFlow/HUMAN_README.md#environment).

## Usage

```bash
conda activate vocaloflow
cd DataAnalysis
jupyter notebook
```

## Notebooks

| Notebook | Description |
|---|---|
| `eda.ipynb` | Exploratory data analysis of the training dataset |
| `f0_comparison.ipynb` | F0 pitch analysis and comparison across models |
| `validation_viewer.ipynb` | Interactive viewer for validation set samples |
| `inference_ablation.ipynb` | Ablation study of inference settings (ODE steps, CFG scale, etc.) |
| `mel_analysis.ipynb` | Mel-spectrogram analysis and visualization |
| `alignment_diagnostic.ipynb` | DTW alignment convergence diagnostics |
| `integration_gridsearch.ipynb` | Grid search results for ODE integration parameters |
