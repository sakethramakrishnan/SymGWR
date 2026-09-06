#!/usr/bin/env python
"""Run GWR with repeated train/val/test evaluation."""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

# Ensure the project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gwr.core import GradientWeightedRegressor
from gwr.models import (
    fit_polynomial_regression,
    fit_linear_regression,
    fit_ridge_regression,
    fit_lasso_regression,
    model_to_sympy_equation,
)
from gwr.utils import (
    aggregate_metric_dicts,
    compute_all_metrics,
    load_dataset,
    train_val_test_split_df,
)


def _fit_base_model(base_model: str, X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, degree: int):
    if base_model == "linear":
        return fit_linear_regression(X_train, y_train, X_eval)
    if base_model == "ridge":
        return fit_ridge_regression(X_train, y_train, X_eval)
    if base_model == "lasso":
        return fit_lasso_regression(X_train, y_train, X_eval)
    return fit_polynomial_regression(X_train, y_train, X_eval, degree=degree)


def main():
    parser = argparse.ArgumentParser(description="Run GWR experiment")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--target", type=str, default="Gibbs_energy")
    parser.add_argument("--base-model", type=str, default="polynomial", choices=["polynomial", "linear", "ridge", "lasso"])
    parser.add_argument("--degrees", nargs="+", type=int, default=[2, 3, 4, 5])
    parser.add_argument("--neighbors", nargs="+", type=int, default=[2, 3, 5, 10])
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--relative", action="store_true")
    parser.add_argument("--output", type=str, default="results/gwr_results.csv")
    parser.add_argument("--raw-output", type=str, default="results/gwr_results_raw.csv")
    parser.add_argument("--no-scale", action="store_true")
    args = parser.parse_args()

    _, _, df = load_dataset(args.data, args.target, scale=not args.no_scale)
    per_repeat_rows = []
    base_metrics_repeats = []
    gwr_metrics_repeats = []
    gwr_times = []

    for rep in range(args.repeats):
        rep_seed = args.seed + rep
        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
        ) = train_val_test_split_df(
            df,
            target_col=args.target,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            random_state=rep_seed,
        )

        best_cfg = {"degree": args.degrees[0], "n_neighbors": args.neighbors[0]}
        best_val_mse = float("inf")
        for degree in args.degrees:
            for k in args.neighbors:
                model, _ = _fit_base_model(args.base_model, X_train, y_train, X_val, degree=degree)
                sympy_eq, x_symbols = model_to_sympy_equation(model)
                gwr = GradientWeightedRegressor(
                    sympy_equation=sympy_eq,
                    x_symbols=x_symbols,
                    n_neighbors=k,
                    relative=args.relative,
                )
                gwr.fit(X_train, y_train)
                y_val_pred = gwr.predict(X_val)
                val_mse = float(np.mean((y_val - y_val_pred) ** 2))
                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    best_cfg = {"degree": degree, "n_neighbors": k}

        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.concatenate([y_train, y_val])

        model_full, y_pred_base = _fit_base_model(
            args.base_model,
            X_train_full,
            y_train_full,
            X_test,
            degree=best_cfg["degree"],
        )
        sympy_eq, x_symbols = model_to_sympy_equation(model_full)
        gwr = GradientWeightedRegressor(
            sympy_equation=sympy_eq,
            x_symbols=x_symbols,
            n_neighbors=best_cfg["n_neighbors"],
            relative=args.relative,
        )
        t0 = time.perf_counter()
        gwr.fit(X_train_full, y_train_full)
        y_pred_gwr = gwr.predict(X_test)
        elapsed = time.perf_counter() - t0

        base_metrics = compute_all_metrics(y_test, y_pred_base)
        gwr_metrics = compute_all_metrics(y_test, y_pred_gwr)
        base_metrics_repeats.append(base_metrics)
        gwr_metrics_repeats.append(gwr_metrics)
        gwr_times.append(elapsed)

        per_repeat_rows.append(
            {
                "repeat": rep + 1,
                "seed": rep_seed,
                "base_model": args.base_model,
                "selected_degree": best_cfg["degree"],
                "selected_n_neighbors": best_cfg["n_neighbors"],
                "relative": args.relative,
                "gwr_time_s": elapsed,
                **{f"base_{k}": v for k, v in base_metrics.items()},
                **{f"gwr_{k}": v for k, v in gwr_metrics.items()},
            }
        )
        print(
            f"[{rep + 1}/{args.repeats}] k={best_cfg['n_neighbors']} deg={best_cfg['degree']} "
            f"GWR MSE={gwr_metrics['mse']:.6f} MAPE={gwr_metrics['mape_mean']:.6f}"
        )

    base_agg = aggregate_metric_dicts(base_metrics_repeats)
    gwr_agg = aggregate_metric_dicts(gwr_metrics_repeats)
    summary = {
        "base_model": args.base_model,
        "relative": args.relative,
        "repeats": args.repeats,
        "gwr_time_s_mean": float(np.mean(gwr_times)),
        "gwr_time_s_std": float(np.std(gwr_times)),
        **{f"base_{k}": v for k, v in base_agg.items()},
        **{f"gwr_{k}": v for k, v in gwr_agg.items()},
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    pd.DataFrame([summary]).to_csv(args.output, index=False)
    os.makedirs(os.path.dirname(args.raw_output) or ".", exist_ok=True)
    pd.DataFrame(per_repeat_rows).to_csv(args.raw_output, index=False)
    print(f"\nResults saved to {args.output}")
    print(f"Per-repeat results saved to {args.raw_output}")


if __name__ == "__main__":
    main()
