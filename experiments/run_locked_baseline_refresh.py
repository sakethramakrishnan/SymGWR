#!/usr/bin/env python
"""Recompute locked CALPHAD baseline table and RQ4 t-test (n=10 seeds).

Outputs:
  - results/baseline_table_locked.csv
  - results/rq4_ttest_locked.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from baselines.mlp import train_mlp
from baselines.pinn import train_pinn
from baselines.symbolic_pysr import train_pysr
from baselines.xgboost import train_xgboost
from gwr.core import GradientWeightedRegressor
from gwr.models import fit_polynomial_regression, model_to_sympy_equation
from gwr.utils import compute_all_metrics, load_dataset, train_val_test_split_df


RANDOM_STATES = list(range(10))

BASELINES: dict[str, Any] = {
    "XGBoost": train_xgboost,
    "MLP": train_mlp,
    "Decision Tree": None,  # Special-cased below.
    "PySR": train_pysr,
    "PINN": train_pinn,
}

PARAM_GRID: dict[str, list[dict[str, Any]]] = {
    "XGBoost": [
        {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05, "random_state": 42},
        {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.1, "random_state": 42},
    ],
    "MLP": [
        {"hidden_layer_sizes": (256, 128, 64), "max_iter": 250, "random_state": 42},
        {"hidden_layer_sizes": (512, 256, 128), "max_iter": 300, "random_state": 42},
    ],
    "Decision Tree": [
        {"max_depth": None, "min_samples_leaf": 1, "random_state": 42},
        {"max_depth": 20, "min_samples_leaf": 2, "random_state": 42},
        {"max_depth": 40, "min_samples_leaf": 1, "random_state": 42},
    ],
    "PySR": [
        {"niterations": 50, "population_size": 20, "populations": 10, "random_state": 42},
    ],
    "PINN": [
        {"hidden_sizes": (128, 128, 64), "epochs": 400, "lambda_physics": 0.1, "random_state": 42},
        {"hidden_sizes": (256, 128, 64), "epochs": 500, "lambda_physics": 0.1, "random_state": 42},
    ],
}


def _extract_prediction(result: Any) -> np.ndarray:
    if isinstance(result, tuple):
        if len(result) >= 2 and result[1] is not None:
            return np.asarray(result[1], dtype=np.float64)
        raise RuntimeError("Baseline returned tuple without prediction array.")
    return np.asarray(result, dtype=np.float64)


def _train_decision_tree_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    model = DecisionTreeRegressor(**params)
    model.fit(X_train, y_train)
    return model.predict(X_eval)


def _predict_model(
    model_name: str,
    train_fn: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    if model_name == "Decision Tree":
        return _train_decision_tree_predict(X_train, y_train, X_eval, params)
    result = train_fn(X_train, y_train, X_eval, **params)
    return _extract_prediction(result)


def _select_hparams(
    model_name: str,
    train_fn: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, Any]:
    best_params = PARAM_GRID[model_name][0]
    best_mse = float("inf")
    for params in PARAM_GRID[model_name]:
        y_val_pred = _predict_model(model_name, train_fn, X_train, y_train, X_val, params)
        val_mse = float(np.mean((y_val - y_val_pred) ** 2))
        if val_mse < best_mse:
            best_mse = val_mse
            best_params = params
    return best_params


def _run_locked_gwr(
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    X_test: np.ndarray,
    degree: int,
    neighbors: int,
    relative: bool,
) -> np.ndarray:
    model, _ = fit_polynomial_regression(X_train_full, y_train_full, X_test=None, degree=degree)
    sympy_eq, x_symbols = model_to_sympy_equation(model)
    gwr = GradientWeightedRegressor(
        sympy_equation=sympy_eq,
        x_symbols=x_symbols,
        n_neighbors=neighbors,
        relative=relative,
    )
    gwr.fit(X_train_full, y_train_full)
    return np.asarray(gwr.predict(X_test), dtype=np.float64)


def _error_vectors(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    abs_err = np.abs(y_true - y_pred)
    non_zero_mask = y_true != 0
    ape = np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])
    return abs_err.astype(np.float64), ape.astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Locked baseline refresh for CALPHAD")
    parser.add_argument("--data", type=str, default="data/raw/718_all.csv")
    parser.add_argument("--target", type=str, default="Gibbs_energy")
    parser.add_argument("--degree", type=int, default=5)
    parser.add_argument("--neighbors", type=int, default=1)
    parser.add_argument("--relative", action="store_true")
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--table-output", type=str, default="results/baseline_table_locked.csv")
    parser.add_argument("--ttest-output", type=str, default="results/rq4_ttest_locked.csv")
    parser.add_argument("--no-scale", action="store_true")
    args = parser.parse_args()

    _, _, df = load_dataset(args.data, args.target, scale=not args.no_scale)
    records: list[dict[str, Any]] = []
    ape_by_model: dict[str, list[np.ndarray]] = {name: [] for name in [*BASELINES.keys(), "Polynomial GWR"]}

    for seed in RANDOM_STATES:
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split_df(
            df,
            target_col=args.target,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            random_state=seed,
        )

        for model_name, train_fn in BASELINES.items():
            t0 = time.perf_counter()
            best_params = _select_hparams(model_name, train_fn, X_train, y_train, X_val, y_val)
            X_train_full = np.vstack([X_train, X_val])
            y_train_full = np.concatenate([y_train, y_val])
            y_pred = _predict_model(model_name, train_fn, X_train_full, y_train_full, X_test, best_params)
            elapsed_s = time.perf_counter() - t0

            metrics = compute_all_metrics(y_test, y_pred)
            _, ape = _error_vectors(y_test, y_pred)
            ape_by_model[model_name].append(ape)
            records.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "split": "test",
                    "degree": args.degree,
                    "k": args.neighbors,
                    "train_frac": args.train_frac,
                    "val_frac": args.val_frac,
                    "test_frac": args.test_frac,
                    "selected_params": str(best_params),
                    "mae": float(metrics["mae_mean"]),
                    "rmse": float(metrics["rmse"]),
                    "mape": float(metrics["mape_mean"]),
                    "r2": float(metrics["r2"]),
                    "runtime_s": float(elapsed_s),
                    "source_script": "experiments/run_locked_baseline_refresh.py",
                }
            )
            print(f"seed={seed} model={model_name} done")

        t0 = time.perf_counter()
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.concatenate([y_train, y_val])
        y_pred_gwr = _run_locked_gwr(
            X_train_full,
            y_train_full,
            X_test,
            degree=args.degree,
            neighbors=args.neighbors,
            relative=args.relative,
        )
        gwr_elapsed = time.perf_counter() - t0
        gwr_metrics = compute_all_metrics(y_test, y_pred_gwr)
        _, gwr_ape = _error_vectors(y_test, y_pred_gwr)
        ape_by_model["Polynomial GWR"].append(gwr_ape)
        records.append(
            {
                "model": "Polynomial GWR",
                "seed": seed,
                "split": "test",
                "degree": args.degree,
                "k": args.neighbors,
                "train_frac": args.train_frac,
                "val_frac": args.val_frac,
                "test_frac": args.test_frac,
                "selected_params": "fixed_locked_config",
                "mae": float(gwr_metrics["mae_mean"]),
                "rmse": float(gwr_metrics["rmse"]),
                "mape": float(gwr_metrics["mape_mean"]),
                "r2": float(gwr_metrics["r2"]),
                "runtime_s": float(gwr_elapsed),
                "source_script": "experiments/run_locked_baseline_refresh.py",
            }
        )
        print(f"seed={seed} model=Polynomial GWR done")

    raw_df = pd.DataFrame(records)
    summary = (
        raw_df.groupby("model", as_index=False)
        .agg(
            n_repeats=("mape", "count"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mape_mean=("mape", "mean"),
            mape_std=("mape", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            runtime_s_mean=("runtime_s", "mean"),
            runtime_s_std=("runtime_s", "std"),
        )
        .fillna(0.0)
        .sort_values("mape_mean")
        .reset_index(drop=True)
    )
    summary["locked_degree"] = args.degree
    summary["locked_k"] = args.neighbors
    summary["seed_convention"] = "0..9"
    summary["split_protocol"] = f"{args.train_frac:.1f}/{args.val_frac:.1f}/{args.test_frac:.1f}"
    summary["source_raw_table"] = "results/baseline_table_locked.csv (same file, per-model aggregate)"
    summary["source_script"] = "experiments/run_locked_baseline_refresh.py"

    baseline_only = summary[summary["model"] != "Polynomial GWR"].copy()
    best_baseline_model = str(baseline_only.sort_values("mape_mean").iloc[0]["model"])
    gwr_err = np.concatenate(ape_by_model["Polynomial GWR"])
    base_err = np.concatenate(ape_by_model[best_baseline_model])
    t_res = stats.ttest_ind(gwr_err, base_err, equal_var=False)

    ttest_df = pd.DataFrame(
        [
            {
                "comparison": "Polynomial GWR vs best_black_box_baseline",
                "metric": "absolute_percentage_error",
                "best_baseline_model": best_baseline_model,
                "t_statistic": float(t_res.statistic),
                "p_value_two_sided": float(t_res.pvalue),
                "n_gwr": int(gwr_err.size),
                "n_baseline": int(base_err.size),
                "seed_convention": "0..9",
                "split": "test",
                "split_protocol": f"{args.train_frac:.1f}/{args.val_frac:.1f}/{args.test_frac:.1f}",
                "locked_degree": args.degree,
                "locked_k": args.neighbors,
                "source_error_vectors": "in-memory from per-seed test predictions in run_locked_baseline_refresh.py",
                "source_table_for_best_baseline": "results/baseline_table_locked.csv:mape_mean",
                "source_script": "experiments/run_locked_baseline_refresh.py",
            }
        ]
    )

    os.makedirs(os.path.dirname(args.table_output) or ".", exist_ok=True)
    summary.to_csv(args.table_output, index=False)
    os.makedirs(os.path.dirname(args.ttest_output) or ".", exist_ok=True)
    ttest_df.to_csv(args.ttest_output, index=False)
    print(f"Saved locked baseline table: {args.table_output}")
    print(f"Saved locked RQ4 t-test: {args.ttest_output}")


if __name__ == "__main__":
    main()
