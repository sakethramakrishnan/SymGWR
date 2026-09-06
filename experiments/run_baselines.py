#!/usr/bin/env python
"""Run baseline models with repeated train/val/test evaluation."""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gwr.utils import (
    aggregate_metric_dicts,
    compute_all_metrics,
    load_dataset,
    train_val_test_split_df,
)
BASELINES = {}


PARAM_GRID = {
    "XGBoost": [
        {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05},
        {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.1},
    ],
    "MLP": [
        {"hidden_layer_sizes": (256, 128, 64), "max_iter": 250},
        {"hidden_layer_sizes": (512, 256, 128), "max_iter": 300},
    ],
    "PINN": [
        {"hidden_sizes": (128, 128, 64), "epochs": 400, "lambda_physics": 0.1},
        {"hidden_sizes": (256, 128, 64), "epochs": 500, "lambda_physics": 0.1},
    ],
    "KNN": [
        {"n_neighbors": 3},
        {"n_neighbors": 5},
        {"n_neighbors": 10},
    ],
    "LOESS": [
        {"frac": 0.2},
        {"frac": 0.3},
        {"frac": 0.4},
    ],
    "GaussianProcess": [
        {"alpha": 1e-6},
        {"alpha": 1e-4},
    ],
    "KernelRegression": [
        {"bandwidth": 0.5},
        {"bandwidth": 1.0},
        {"bandwidth": 2.0},
    ],
    "CALPHAD_NN": [
        {"hidden_sizes": (256, 128, 64), "epochs": 180},
        {"hidden_sizes": (512, 256, 128), "epochs": 220},
    ],
}


def _build_available_baselines():
    baselines = {}
    skipped = []

    try:
        from baselines.xgboost import train_xgboost
        baselines["XGBoost"] = train_xgboost
    except Exception as exc:
        skipped.append(f"XGBoost ({exc})")

    try:
        from baselines.mlp import train_mlp
        baselines["MLP"] = train_mlp
    except Exception as exc:
        skipped.append(f"MLP ({exc})")

    try:
        from baselines.pinn import train_pinn
        baselines["PINN"] = train_pinn
    except Exception as exc:
        skipped.append(f"PINN ({exc})")

    try:
        from baselines.knn import train_knn
        baselines["KNN"] = train_knn
    except Exception as exc:
        skipped.append(f"KNN ({exc})")

    try:
        from baselines.loess import train_loess
        baselines["LOESS"] = train_loess
    except Exception as exc:
        skipped.append(f"LOESS ({exc})")

    try:
        from baselines.gaussian_process import train_gaussian_process
        baselines["GaussianProcess"] = train_gaussian_process
    except Exception as exc:
        skipped.append(f"GaussianProcess ({exc})")

    try:
        from baselines.kernel_regression import train_kernel_regression
        baselines["KernelRegression"] = train_kernel_regression
    except Exception as exc:
        skipped.append(f"KernelRegression ({exc})")

    try:
        from baselines.calphad_nn import train_calphad_nn
        baselines["CALPHAD_NN"] = train_calphad_nn
    except Exception as exc:
        skipped.append(f"CALPHAD_NN ({exc})")

    return baselines, skipped


def _get_predictions(train_fn, X_train, y_train, X_eval, params):
    result = train_fn(X_train, y_train, X_eval, **params)
    return result[1] if isinstance(result, tuple) else result


def _select_hparams(name, train_fn, X_train, y_train, X_val, y_val):
    best_params = PARAM_GRID[name][0]
    best_mse = float("inf")
    for params in PARAM_GRID[name]:
        y_val_pred = _get_predictions(train_fn, X_train, y_train, X_val, params)
        val_mse = float(np.mean((y_val - y_val_pred) ** 2))
        if val_mse < best_mse:
            best_mse = val_mse
            best_params = params
    return best_params


def _subsample_training_data(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_train_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    expensive = {"GaussianProcess", "KernelRegression", "LOESS"}
    if model_name not in expensive or len(X_train) <= max_train_samples:
        return X_train, y_train
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_train), size=max_train_samples, replace=False)
    return X_train[idx], y_train[idx]


def main():
    parser = argparse.ArgumentParser(description="Run baseline evaluations")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--target", type=str, default="Gibbs_energy")
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int, default=2500)
    parser.add_argument("--output", type=str, default="results/baseline_comparison.csv")
    parser.add_argument(
        "--raw-output",
        type=str,
        default="results/baseline_comparison_raw.csv",
        help="Per-repeat raw metrics output path.",
    )
    parser.add_argument("--include-pysr", action="store_true")
    parser.add_argument("--no-scale", action="store_true")
    args = parser.parse_args()

    # Load data
    _, _, df = load_dataset(args.data, args.target, scale=not args.no_scale)

    baselines, skipped = _build_available_baselines()
    if skipped:
        print("Skipping unavailable baselines:")
        for item in skipped:
            print(f"  - {item}")
    if not baselines:
        raise RuntimeError("No available baselines; install required dependencies first.")
    if args.include_pysr:
        try:
            from baselines.symbolic_pysr import train_pysr
            baselines["PySR"] = train_pysr
            PARAM_GRID["PySR"] = [{"niterations": 50, "population_size": 20}]
        except Exception as exc:
            print(f"Skipping PySR baseline ({exc})")

    per_repeat_rows = []
    grouped_metrics = {name: [] for name in baselines}
    grouped_times = {name: [] for name in baselines}

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

        for name, train_fn in baselines.items():
            print(f"\n>>> [{rep + 1}/{args.repeats}] {name}")
            X_train_model, y_train_model = _subsample_training_data(
                name, X_train, y_train, args.max_train_samples, rep_seed
            )
            best_params = _select_hparams(
                name, train_fn, X_train_model, y_train_model, X_val, y_val
            )
            X_train_full = np.vstack([X_train_model, X_val])
            y_train_full = np.concatenate([y_train_model, y_val])

            t0 = time.perf_counter()
            y_pred = _get_predictions(train_fn, X_train_full, y_train_full, X_test, best_params)
            elapsed = time.perf_counter() - t0

            metrics = compute_all_metrics(y_test, y_pred)
            grouped_metrics[name].append(metrics)
            grouped_times[name].append(elapsed)

            row = {
                "repeat": rep + 1,
                "seed": rep_seed,
                "model": name,
                "selected_params": str(best_params),
                "time_s": elapsed,
                **metrics,
            }
            per_repeat_rows.append(row)
            print(
                f"    {name}: MSE={metrics['mse']:.6f} MAE={metrics['mae_mean']:.6f} "
                f"MAPE={metrics['mape_mean']:.6f} ({elapsed:.2f}s)"
            )

    summary_rows = []
    for name in baselines:
        agg = aggregate_metric_dicts(grouped_metrics[name])
        agg["model"] = name
        agg["time_s_mean"] = float(np.mean(grouped_times[name]))
        agg["time_s_std"] = float(np.std(grouped_times[name]))
        summary_rows.append(agg)

    df_out = pd.DataFrame(summary_rows)
    df_raw = pd.DataFrame(per_repeat_rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df_out.to_csv(args.output, index=False)
    os.makedirs(os.path.dirname(args.raw_output) or ".", exist_ok=True)
    df_raw.to_csv(args.raw_output, index=False)
    print(f"\n✓ Results saved to {args.output}")
    print(f"✓ Raw per-repeat results saved to {args.raw_output}")


if __name__ == "__main__":
    main()
