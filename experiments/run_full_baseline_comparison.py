#!/usr/bin/env python
"""Run full per-baseline comparison with fixed Polynomial GWR config.

NOTE: This script is the authoritative baseline table and replaces
`results/baseline_comparison_raw.csv` for manuscript reporting.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from baselines.calphad_nn import train_calphad_nn
from baselines.gaussian_process import train_gaussian_process
from baselines.kernel_regression import train_kernel_regression
from baselines.knn import train_knn
from baselines.loess import train_loess
from baselines.mlp import train_mlp
from gwr.core import GradientWeightedRegressor
from gwr.models import fit_polynomial_regression, model_to_sympy_equation
from gwr.utils import compute_all_metrics, load_dataset, train_val_test_split_df


RANDOM_STATES = [0, 1, 2, 3, 4]
BASELINE_PARAM_GRID = {
    "GaussianProcess": [
        {"alpha": 1e-6},
        {"alpha": 1e-4},
    ],
    "MLP": [
        {"hidden_layer_sizes": (256, 128, 64), "max_iter": 250},
        {"hidden_layer_sizes": (512, 256, 128), "max_iter": 300},
    ],
    "CALPHAD_NN": [
        {"hidden_sizes": (256, 128, 64), "epochs": 180},
        {"hidden_sizes": (512, 256, 128), "epochs": 220},
    ],
    "LOESS": [
        {"frac": 0.2},
        {"frac": 0.3},
        {"frac": 0.4},
    ],
    "KNN": [
        {"n_neighbors": 3},
        {"n_neighbors": 5},
        {"n_neighbors": 10},
    ],
    "KernelRegression": [
        {"bandwidth": 0.5},
        {"bandwidth": 1.0},
        {"bandwidth": 2.0},
    ],
}
BASELINE_FNS = {
    "GaussianProcess": train_gaussian_process,
    "MLP": train_mlp,
    "CALPHAD_NN": train_calphad_nn,
    "LOESS": train_loess,
    "KNN": train_knn,
    "KernelRegression": train_kernel_regression,
}
LOWER_IS_BETTER = {
    "mae",
    "mse",
    "rmse",
    "mape",
    "smape",
    "training_time_s",
    "inference_time_s",
}


def _extract_prediction(result: Any) -> np.ndarray | None:
    if isinstance(result, tuple):
        if len(result) >= 2:
            return result[1]
        return None
    return result


def _select_hparams(
    train_fn: Any,
    param_grid: list[dict[str, Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, Any]:
    best = param_grid[0]
    best_mse = float("inf")
    for params in param_grid:
        y_val_pred = _extract_prediction(train_fn(X_train, y_train, X_val, **params))
        val_mse = float(np.mean((y_val - y_val_pred) ** 2))
        if val_mse < best_mse:
            best_mse = val_mse
            best = params
    return best


def _subsample_training_data(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_train_samples: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    expensive = {"GaussianProcess", "KernelRegression", "LOESS"}
    if model_name not in expensive or len(X_train) <= max_train_samples:
        return X_train, y_train
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X_train), size=max_train_samples, replace=False)
    return X_train[idx], y_train[idx]


def _train_and_predict_with_timing(
    train_fn: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    params: dict[str, Any],
) -> tuple[np.ndarray, float, float]:
    t0 = time.perf_counter()
    result = train_fn(X_train, y_train, None, **params)
    train_time = time.perf_counter() - t0
    if isinstance(result, tuple):
        model = result[0]
        scaler = result[2] if len(result) >= 3 else None
    else:
        model = result
        scaler = None
    X_eval = scaler.transform(X_test) if scaler is not None else X_test

    t1 = time.perf_counter()
    y_pred = model.predict(X_eval)
    infer_time = time.perf_counter() - t1
    return y_pred, float(train_time), float(infer_time)


def _run_polynomial_gwr(
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    X_test: np.ndarray,
    degree: int,
    neighbors: int,
    relative: bool,
) -> tuple[np.ndarray, float, float]:
    t0 = time.perf_counter()
    model, _ = fit_polynomial_regression(
        X_train_full,
        y_train_full,
        X_test=None,
        degree=degree,
    )
    sympy_eq, x_symbols = model_to_sympy_equation(model)
    gwr = GradientWeightedRegressor(
        sympy_equation=sympy_eq,
        x_symbols=x_symbols,
        n_neighbors=neighbors,
        relative=relative,
    )
    gwr.fit(X_train_full, y_train_full)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = gwr.predict(X_test)
    infer_time = time.perf_counter() - t1
    return y_pred, float(train_time), float(infer_time)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full CALPHAD baseline comparison")
    parser.add_argument("--data", type=str, default="data/raw/718_all.csv")
    parser.add_argument("--target", type=str, default="Gibbs_energy")
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--neighbors", type=int, default=2)
    parser.add_argument("--relative", action="store_true")
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--max-train-samples", type=int, default=2500)
    parser.add_argument("--raw-output", type=str, default="results/full_baseline_raw.csv")
    parser.add_argument(
        "--summary-output",
        type=str,
        default="results/full_baseline_summary.csv",
    )
    parser.add_argument(
        "--improvement-output",
        type=str,
        default="results/full_baseline_improvement.csv",
    )
    parser.add_argument("--no-scale", action="store_true")
    args = parser.parse_args()

    _, _, df = load_dataset(args.data, args.target, scale=not args.no_scale)
    raw_rows: list[dict[str, float | int | str]] = []

    for random_state in RANDOM_STATES:
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split_df(
            df,
            target_col=args.target,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            random_state=random_state,
        )

        for model_name, train_fn in BASELINE_FNS.items():
            X_train_model, y_train_model = _subsample_training_data(
                model_name,
                X_train,
                y_train,
                args.max_train_samples,
                random_state,
            )
            best_params = _select_hparams(
                train_fn,
                BASELINE_PARAM_GRID[model_name],
                X_train_model,
                y_train_model,
                X_val,
                y_val,
            )
            X_train_full = np.vstack([X_train_model, X_val])
            y_train_full = np.concatenate([y_train_model, y_val])
            y_pred, train_t, infer_t = _train_and_predict_with_timing(
                train_fn,
                X_train_full,
                y_train_full,
                X_test,
                best_params,
            )
            metrics = compute_all_metrics(y_test, y_pred)
            raw_rows.append(
                {
                    "model": model_name,
                    "random_state": random_state,
                    "mae": float(metrics["mae_mean"]),
                    "mse": float(metrics["mse"]),
                    "rmse": float(metrics["rmse"]),
                    "mape": float(metrics["mape_mean"]),
                    "smape": float(metrics["smape_mean"]),
                    "r2": float(metrics["r2"]),
                    "training_time_s": float(train_t),
                    "inference_time_s": float(infer_t),
                }
            )
            print(f"seed={random_state} model={model_name} done")

        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.concatenate([y_train, y_val])
        y_pred, train_t, infer_t = _run_polynomial_gwr(
            X_train_full,
            y_train_full,
            X_test,
            degree=args.degree,
            neighbors=args.neighbors,
            relative=args.relative,
        )
        metrics = compute_all_metrics(y_test, y_pred)
        raw_rows.append(
            {
                "model": "Polynomial GWR",
                "random_state": random_state,
                "mae": float(metrics["mae_mean"]),
                "mse": float(metrics["mse"]),
                "rmse": float(metrics["rmse"]),
                "mape": float(metrics["mape_mean"]),
                "smape": float(metrics["smape_mean"]),
                "r2": float(metrics["r2"]),
                "training_time_s": float(train_t),
                "inference_time_s": float(infer_t),
            }
        )
        print(f"seed={random_state} model=Polynomial GWR done")

    raw_df = pd.DataFrame(raw_rows)
    summary_rows: list[dict[str, float | str]] = []
    metric_cols = [
        "mae",
        "mse",
        "rmse",
        "mape",
        "smape",
        "r2",
        "training_time_s",
        "inference_time_s",
    ]
    for model_name, group in raw_df.groupby("model"):
        row: dict[str, float | str] = {"model": model_name, "n_repeats": float(len(group))}
        for col in metric_cols:
            vals = group[col].to_numpy(dtype=np.float64)
            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_std"] = float(np.std(vals))
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).sort_values("model").reset_index(drop=True)

    gwr_row = summary_df.loc[summary_df["model"] == "Polynomial GWR"]
    if gwr_row.empty:
        raise RuntimeError("Polynomial GWR row missing from summary output.")
    gwr_row = gwr_row.iloc[0]

    improvement_rows: list[dict[str, Any]] = []
    for row in summary_df.itertuples(index=False):
        model_name = str(row.model)
        if model_name == "Polynomial GWR":
            continue
        for metric in metric_cols:
            baseline_val = float(getattr(row, f"{metric}_mean"))
            gwr_val = float(gwr_row[f"{metric}_mean"])

            if metric == "r2" and np.isclose(gwr_val, 1.0) and np.isclose(baseline_val, 1.0):
                abs_improvement: float | str = "at_ceiling"
                pct_improvement: float | str = "at_ceiling"
            else:
                if metric in LOWER_IS_BETTER:
                    abs_improvement = baseline_val - gwr_val
                else:
                    abs_improvement = gwr_val - baseline_val
                pct_improvement = (
                    np.nan if np.isclose(baseline_val, 0.0) else (abs_improvement / abs(baseline_val)) * 100.0
                )

            improvement_rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "gwr_mean": gwr_val,
                    "baseline_mean": baseline_val,
                    "absolute_improvement": abs_improvement,
                    "percent_improvement": pct_improvement,
                }
            )
    improvement_df = pd.DataFrame(improvement_rows)

    os.makedirs(os.path.dirname(args.raw_output) or ".", exist_ok=True)
    raw_df.to_csv(args.raw_output, index=False)
    os.makedirs(os.path.dirname(args.summary_output) or ".", exist_ok=True)
    summary_df.to_csv(args.summary_output, index=False)
    os.makedirs(os.path.dirname(args.improvement_output) or ".", exist_ok=True)
    improvement_df.to_csv(args.improvement_output, index=False)
    print(f"Saved raw baseline results: {args.raw_output}")
    print(f"Saved summary baseline results: {args.summary_output}")
    print(f"Saved improvements table: {args.improvement_output}")


if __name__ == "__main__":
    main()
