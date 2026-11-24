#!/usr/bin/env python
"""Run CALPHAD headline SymGWR config across 10 split seeds.

NOTE: This script is the authoritative source for headline CALPHAD results and
replaces `results/gwr_results_raw.csv` for manuscript reporting. The old file
is kept only for historical comparison.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gwr.core import GradientWeightedRegressor
from gwr.models import fit_polynomial_regression, model_to_sympy_equation
from gwr.utils import compute_all_metrics, load_dataset, train_val_test_split_df


RANDOM_STATES = list(range(10))


def main() -> None:
    parser = argparse.ArgumentParser(description="CALPHAD repeats for fixed SymGWR config")
    parser.add_argument("--data", type=str, default="data/raw/718_all.csv")
    parser.add_argument("--target", type=str, default="Gibbs_energy")
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--neighbors", type=int, default=2)
    parser.add_argument("--relative", action="store_true")
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument(
        "--raw-output",
        type=str,
        default="results/calphad_repeats_raw.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default="results/calphad_repeats_summary.csv",
    )
    parser.add_argument("--no-scale", action="store_true")
    args = parser.parse_args()

    _, _, df = load_dataset(args.data, args.target, scale=not args.no_scale)
    rows: list[dict[str, float | int]] = []

    for random_state in RANDOM_STATES:
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split_df(
            df,
            target_col=args.target,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            random_state=random_state,
        )
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.concatenate([y_train, y_val])

        t_train_0 = time.perf_counter()
        base_model, _ = fit_polynomial_regression(
            X_train_full,
            y_train_full,
            X_test=None,
            degree=args.degree,
        )
        sympy_eq, x_symbols = model_to_sympy_equation(base_model)
        gwr = GradientWeightedRegressor(
            sympy_equation=sympy_eq,
            x_symbols=x_symbols,
            n_neighbors=args.neighbors,
            relative=args.relative,
        )
        gwr.fit(X_train_full, y_train_full)
        training_time_s = time.perf_counter() - t_train_0

        t_infer_0 = time.perf_counter()
        y_pred = gwr.predict(X_test)
        inference_time_s = time.perf_counter() - t_infer_0
        total_time_s = training_time_s + inference_time_s

        metrics = compute_all_metrics(y_test, y_pred)
        row = {
            "random_state": random_state,
            "mae": float(metrics["mae_mean"]),
            "mse": float(metrics["mse"]),
            "rmse": float(metrics["rmse"]),
            "mape": float(metrics["mape_mean"]),
            "smape": float(metrics["smape_mean"]),
            "r2": float(metrics["r2"]),
            "training_time_s": float(training_time_s),
            "inference_time_s": float(inference_time_s),
            "total_time_s": float(total_time_s),
        }
        rows.append(row)
        print(
            f"seed={random_state} "
            f"MAPE={row['mape']:.6g} MAE={row['mae']:.6g} R2={row['r2']:.6g} "
            f"train={training_time_s:.3f}s infer={inference_time_s:.4f}s"
        )

    raw_df = pd.DataFrame(rows)
    summary = {"n_repeats": float(len(rows))}
    for col in [
        "mae",
        "mse",
        "rmse",
        "mape",
        "smape",
        "r2",
        "training_time_s",
        "inference_time_s",
        "total_time_s",
    ]:
        values = raw_df[col].to_numpy(dtype=np.float64)
        summary[f"{col}_mean"] = float(np.mean(values))
        summary[f"{col}_std"] = float(np.std(values))
    summary_df = pd.DataFrame([summary])

    os.makedirs(os.path.dirname(args.raw_output) or ".", exist_ok=True)
    raw_df.to_csv(args.raw_output, index=False)
    os.makedirs(os.path.dirname(args.summary_output) or ".", exist_ok=True)
    summary_df.to_csv(args.summary_output, index=False)
    print(f"Saved raw results: {args.raw_output}")
    print(f"Saved summary: {args.summary_output}")


if __name__ == "__main__":
    main()
