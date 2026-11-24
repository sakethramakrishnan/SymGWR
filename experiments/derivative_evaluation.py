#!/usr/bin/env python
"""Derivative evaluation for SymGWR and selected baselines.

If true derivative columns are unavailable, this script uses a local finite-difference
proxy estimated from nearest-neighbour linear fits and records the limitation.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gwr.core import GradientWeightedRegressor
from gwr.models import fit_polynomial_regression, model_to_sympy_equation
from gwr.utils import load_dataset, train_val_test_split_df


def _local_fd_proxy(X_ref: np.ndarray, y_ref: np.ndarray, X_query: np.ndarray, k: int = 30) -> np.ndarray:
    """Estimate local gradient via weighted linear regression around each query point."""
    nn = NearestNeighbors(n_neighbors=min(k, len(X_ref))).fit(X_ref)
    distances, indices = nn.kneighbors(X_query, return_distance=True)
    grads = np.zeros((X_query.shape[0], X_query.shape[1]), dtype=np.float64)

    for i, idx in enumerate(indices):
        X_local = X_ref[idx]
        y_local = y_ref[idx]
        d = distances[i]
        w = 1.0 / (d + 1e-6)
        W = np.diag(w / (w.sum() + 1e-12))
        X_center = X_local - X_query[i][np.newaxis, :]
        X_design = np.hstack([np.ones((X_center.shape[0], 1)), X_center])
        beta = np.linalg.lstsq(X_design.T @ W @ X_design, X_design.T @ W @ y_local, rcond=None)[0]
        grads[i] = beta[1:]
    return grads


def _detect_true_derivatives(df: pd.DataFrame, target_col: str) -> List[str]:
    candidates = []
    target_low = target_col.lower()
    for col in df.columns:
        c_low = col.lower()
        if c_low.startswith("d") and target_low in c_low:
            candidates.append(col)
        if "deriv" in c_low and target_low in c_low:
            candidates.append(col)
    return sorted(set(candidates))


def main():
    parser = argparse.ArgumentParser(description="Derivative evaluation for SymGWR")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--target", type=str, default="Gibbs_energy")
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--output", type=str, default="results/derivative_evaluation.csv")
    parser.add_argument(
        "--summary-output",
        type=str,
        default="results/derivative_evaluation_summary.csv",
    )
    parser.add_argument("--no-scale", action="store_true")
    args = parser.parse_args()

    _, _, df = load_dataset(args.data, args.target, scale=not args.no_scale)
    true_deriv_cols = _detect_true_derivatives(df, args.target)
    derivative_mode = "true_derivative_columns" if true_deriv_cols else "local_finite_difference_proxy"

    rows: List[Dict[str, float | str | int]] = []
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
            n_neighbors=args.neighbors,
        )
        gwr.fit(X_train_full, y_train_full)
        _, gwr_grad = gwr.predict_with_local_gradient(X_test)

        if true_deriv_cols:
            # Map derivative columns by feature name where possible.
            feature_cols = [c for c in df.columns if c != args.target]
            true_grad = np.zeros_like(gwr_grad)
            col_map = {c.lower(): c for c in true_deriv_cols}
            for j, feat in enumerate(feature_cols):
                key_options = [
                    f"d{args.target}_d{feat}".lower(),
                    f"d{args.target}/d{feat}".lower(),
                    f"deriv_{args.target}_{feat}".lower(),
                ]
                matched = next((col_map[k] for k in key_options if k in col_map), None)
                if matched is not None:
                    true_grad[:, j] = df.loc[df.index[: len(X_test)], matched].to_numpy(dtype=np.float64)
                else:
                    true_grad[:, j] = np.nan
        else:
            true_grad = _local_fd_proxy(X_train_full, y_train_full, X_test, k=30)

        deriv_mae = float(np.nanmean(np.abs(true_grad - gwr_grad)))
        deriv_mse = float(np.nanmean((true_grad - gwr_grad) ** 2))
        rows.append(
            {
                "repeat": rep + 1,
                "seed": rep_seed,
                "degree": args.degree,
                "n_neighbors": args.neighbors,
                "derivative_mode": derivative_mode,
                "derivative_mae": deriv_mae,
                "derivative_mse": deriv_mse,
            }
        )

    df_rows = pd.DataFrame(rows)
    summary = {
        "derivative_mode": derivative_mode,
        "n_repeats": args.repeats,
        "derivative_mae_mean": float(df_rows["derivative_mae"].mean()),
        "derivative_mae_std": float(df_rows["derivative_mae"].std()),
        "derivative_mse_mean": float(df_rows["derivative_mse"].mean()),
        "derivative_mse_std": float(df_rows["derivative_mse"].std()),
        "limitation_note": (
            "No true CALPHAD finite-difference derivatives found; used local finite-difference proxy."
            if derivative_mode != "true_derivative_columns"
            else "Used true derivative columns from dataset."
        ),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df_rows.to_csv(args.output, index=False)
    os.makedirs(os.path.dirname(args.summary_output) or ".", exist_ok=True)
    pd.DataFrame([summary]).to_csv(args.summary_output, index=False)
    print(f"Saved derivative rows to {args.output}")
    print(f"Saved derivative summary to {args.summary_output}")


if __name__ == "__main__":
    main()
