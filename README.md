# SymGWR

Gradient-Weighted Regression (SymGWR) applied as a differentiable surrogate for
CALPHAD phase-field Gibbs free energy: composition and temperature in,
Gibbs free energy out, via a symbolic base model corrected with nearest-neighbor
analytical gradient extrapolation.

Code accompanying the SymGWR submission to the IEEE BigData 2026
Undergraduate/REU Consortium.

## Repository layout

- `gwr/` — core SymGWR library (base model fitting, symbolic gradients, neighbor correction)
- `baselines/` — competing methods (XGBoost, MLP, PINN, kNN, LOESS, Gaussian Process, kernel regression, CALPHAD-NN, symbolic regression)
- `experiments/` — reproduction scripts (SymGWR pipeline, baseline comparisons, ablations, timing, CALPHAD reference data)
- `regression/` — legacy exploratory per-system baseline sweeps
- `data/raw/718_all.csv` — primary CALPHAD dataset (composition, temperature, Gibbs energy)
- `results/`, `figures/`, `paper/` — evaluation outputs, plots, and manuscript tables/figures
- `notebooks/` — exploratory analysis and a SymGWR tutorial notebook
- `docs/` — method positioning and reproducibility notes

## Setup

```bash
conda env create -f environment.yml
conda activate gwr
```

See `docs/reproducibility.md` for the experiment commands.

## License

[MIT License](LICENSE).
