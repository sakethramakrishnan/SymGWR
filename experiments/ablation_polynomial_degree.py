#!/usr/bin/env python
"""
experiments/ablation_polynomial_degree.py
=========================================

Ablation study over polynomial degree of the base model in GWR.

For each degree in [1 … max_degree], fits a polynomial base model,
runs GWR (with a fixed number of neighbours), and reports metrics.

Usage
-----
    python -m experiments.ablation_polynomial_degree \
        --data data/raw/718_all.csv \
        --target Gibbs_energy \
        --neighbors 5 \
        --max-degree 15 \
        --output results/ablation_degree.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gwr.core import GradientWeightedRegressor
from gwr.models import (
    fit_linear_regression,
    fit_polynomial_regression,
    model_to_sympy_equation,
)
from gwr.utils import compute_all_metrics, load_dataset, train_val_test_split_df


def main():
    parser = argparse.ArgumentParser(description="Ablation: polynomial degree")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--target", type=str, default="Gibbs_energy")
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--max-degree", type=int, default=10)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/ablation_degree.csv")
    parser.add_argument("--no-scale", action="store_true")
    args = parser.parse_args()

    _, _, df = load_dataset(args.data, args.target, scale=not args.no_scale)
    rows = []
    for deg in range(1, args.max_degree + 1):
        print(f"\n>>> Degree {deg}")
        base_repeats = []
        gwr_repeats = []
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

            if deg == 1:
                model, y_pred_base = fit_linear_regression(X_train_full, y_train_full, X_test)
            else:
                model, y_pred_base = fit_polynomial_regression(
                    X_train_full, y_train_full, X_test, degree=deg
                )
            base_metrics = compute_all_metrics(y_test, y_pred_base)
            base_repeats.append(base_metrics)

            try:
                sympy_eq, x_symbols = model_to_sympy_equation(model)
                gwr = GradientWeightedRegressor(
                    sympy_equation=sympy_eq,
                    x_symbols=x_symbols,
                    n_neighbors=args.neighbors,
                )
                gwr.fit(X_train_full, y_train_full)
                y_pred_gwr = gwr.predict(X_test)
                gwr_metrics = compute_all_metrics(y_test, y_pred_gwr)
            except Exception as e:
                print(f"    GWR failed for degree {deg}: {e}")
                gwr_metrics = {k: np.nan for k in base_metrics}
            gwr_repeats.append(gwr_metrics)

        row = {"degree": deg, "n_neighbors": args.neighbors, "n_repeats": args.repeats}
        for k in base_repeats[0]:
            row[f"base_{k}_mean"] = float(np.mean([r[k] for r in base_repeats]))
            row[f"base_{k}_std"] = float(np.std([r[k] for r in base_repeats]))
            row[f"gwr_{k}_mean"] = float(np.mean([r[k] for r in gwr_repeats]))
            row[f"gwr_{k}_std"] = float(np.std([r[k] for r in gwr_repeats]))
        rows.append(row)

        print(
            f"    Base MSE={row['base_mse_mean']:.6f}  "
            f"GWR MSE={row['gwr_mse_mean']:.6f}"
        )

    df_out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df_out.to_csv(args.output, index=False)
    print(f"\n✓ Degree ablation saved to {args.output}")


if __name__ == "__main__":
    main()
