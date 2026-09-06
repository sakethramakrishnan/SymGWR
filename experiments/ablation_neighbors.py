#!/usr/bin/env python
"""
experiments/ablation_neighbors.py
=================================

Ablation study over the number of neighbours *k* in GWR.

Sweeps k = 1 … max_k for both absolute and relative neighbour strategies
and records RMSE / R² / MAPE / SMAPE for each configuration.

Usage
-----
    python -m experiments.ablation_neighbors \
        --data data/raw/718_all.csv \
        --target Gibbs_energy \
        --degree 3 \
        --max-k 20 \
        --output results/ablation_neighbors.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gwr.core import GradientWeightedRegressor
from gwr.models import fit_polynomial_regression, model_to_sympy_equation
from gwr.utils import compute_all_metrics, load_dataset, train_val_test_split_df


def main():
    parser = argparse.ArgumentParser(description="Ablation: number of neighbours")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--target", type=str, default="Gibbs_energy")
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--max-k", type=int, default=20)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/ablation_neighbors.csv")
    parser.add_argument("--no-scale", action="store_true")
    args = parser.parse_args()

    _, _, df = load_dataset(args.data, args.target, scale=not args.no_scale)

    rows = []
    for k in range(1, args.max_k + 1):
        for relative in [False, True]:
            rep_metrics = []
            for rep in range(args.repeats):
                rep_seed = args.seed + rep
                X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split_df(
                    df,
                    target_col=args.target,
                    train_frac=args.train_frac,
                    val_frac=args.val_frac,
                    test_frac=args.test_frac,
                    random_state=rep_seed,
                )
                X_train_full = np.vstack([X_train, X_val])
                y_train_full = np.concatenate([y_train, y_val])
                model, _ = fit_polynomial_regression(X_train_full, y_train_full, degree=args.degree)
                sympy_eq, x_symbols = model_to_sympy_equation(model)
                gwr = GradientWeightedRegressor(
                    sympy_equation=sympy_eq,
                    x_symbols=x_symbols,
                    n_neighbors=k,
                    relative=relative,
                )
                gwr.fit(X_train_full, y_train_full)
                y_pred = gwr.predict(X_test)
                rep_metrics.append(compute_all_metrics(y_test, y_pred))

            metric_keys = rep_metrics[0].keys()
            agg = {
                f"{m}_mean": float(np.mean([r[m] for r in rep_metrics]))
                for m in metric_keys
            }
            agg.update({
                f"{m}_std": float(np.std([r[m] for r in rep_metrics]))
                for m in metric_keys
            })
            agg["n_neighbors"] = k
            agg["relative"] = relative
            agg["degree"] = args.degree
            agg["n_repeats"] = args.repeats
            rows.append(agg)

            label = "relative" if relative else "absolute"
            print(
                f"  k={k:2d} ({label:8s}):  "
                f"MSE={agg['mse_mean']:.6f} MAE={agg['mae_mean_mean']:.6f} "
                f"MAPE={agg['mape_mean_mean']:.6f}"
            )

    df_out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df_out.to_csv(args.output, index=False)
    print(f"\n✓ Ablation results saved to {args.output}")


if __name__ == "__main__":
    main()
