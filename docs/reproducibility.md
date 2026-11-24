# Reproducing Experiments

This document is for developers regenerating benchmark outputs after code changes.

## Environment

```bash
conda env create -f environment.yml
conda activate gwr
```

## Main experiment commands

### 1) SymGWR with repeated train/val/test

```bash
python -m experiments.run_gwr \
  --data data/raw/718_all.csv \
  --target Gibbs_energy \
  --base-model polynomial \
  --degrees 2 3 4 5 \
  --neighbors 2 3 5 10 \
  --repeats 5 \
  --output results/gwr_results.csv \
  --raw-output results/gwr_results_raw.csv
```

### 2) Baselines (including LOESS, GPR, Kernel Regression, CALPHAD-NN)

```bash
python -m experiments.run_baselines \
  --data data/raw/718_all.csv \
  --target Gibbs_energy \
  --repeats 5 \
  --output results/baseline_comparison.csv \
  --raw-output results/baseline_comparison_raw.csv
```

### 3) Neighbor ablation

```bash
python -m experiments.ablation_neighbors \
  --data data/raw/718_all.csv \
  --target Gibbs_energy \
  --degree 3 \
  --max-k 20 \
  --repeats 5 \
  --output results/ablation_neighbors.csv
```

### 4) Polynomial degree ablation

```bash
python -m experiments.ablation_polynomial_degree \
  --data data/raw/718_all.csv \
  --target Gibbs_energy \
  --neighbors 5 \
  --max-degree 10 \
  --repeats 5 \
  --output results/ablation_degree.csv
```

### 5) PMLB benchmark

```bash
python -m experiments.pmlb_benchmark \
  --cache data/pmlb \
  --degrees 2 3 4 5 \
  --max-k 5 \
  --repeats 3 \
  --output results/pmlb_benchmark.csv
```

### 6) Derivative evaluation

```bash
python -m experiments.derivative_evaluation \
  --data data/raw/718_all.csv \
  --target Gibbs_energy \
  --degree 3 \
  --neighbors 5 \
  --repeats 5 \
  --output results/derivative_evaluation.csv \
  --summary-output results/derivative_evaluation_summary.csv
```

### 7) Regenerate tables and figures

```bash
python -m experiments.generate_tables --results-dir results
python -m experiments.generate_figures --results-dir results --figures-dir figures
```

### 8) Verify links

```bash
python -m experiments.verify_links --output results/link_check.csv
```

## Output locations

- Raw experiment metrics: `results/*_raw.csv`
- Aggregated summaries: `results/*.csv`
- Generated tables: `results/table_*.csv`
- Generated figures: `figures/*.png`
- Link verification output: `results/link_check.csv`

## Adding a new baseline

1. Add training wrapper in `baselines/`.
2. Register in `experiments/run_baselines.py`:
   - `BASELINES`
   - `PARAM_GRID`
3. Ensure model returns `(model, y_pred, scaler)` or `(model, y_pred)`.
4. Regenerate baseline outputs and derived tables/figures.
