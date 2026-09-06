#!/usr/bin/env python
"""Benchmark CALPHAD GWR inference latency on single-point queries."""

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
from gwr.utils import load_dataset, train_val_test_split_df


CALPHAD_QUERY_TIME_US = None  # TO BE FILLED MANUALLY
# Ask co-author or CALPHAD software vendor for single-point CALPHAD
# evaluation time in microseconds on equivalent hardware. This is the
# critical comparison GWR_timing vs CALPHAD_timing that motivates the
# paper. If unavailable, this gap must appear in the Limitations section.


def _timing_stats_us(samples_us: np.ndarray) -> dict[str, float]:
    return {
        "mean_us": float(np.mean(samples_us)),
        "std_us": float(np.std(samples_us)),
        "min_us": float(np.min(samples_us)),
        "max_us": float(np.max(samples_us)),
        "p95_us": float(np.percentile(samples_us, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CALPHAD GWR timing benchmark")
    parser.add_argument("--data", type=str, default="data/raw/718_all.csv")
    parser.add_argument("--target", type=str, default="Gibbs_energy")
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--neighbors", type=int, default=2)
    parser.add_argument("--relative", action="store_true")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--n-queries", type=int, default=100)
    parser.add_argument("--output", type=str, default="results/gwr_timing_detail.csv")
    parser.add_argument(
        "--placeholder-output",
        type=str,
        default="results/calphad_timing_placeholder.csv",
    )
    parser.add_argument("--no-scale", action="store_true")
    args = parser.parse_args()

    _, _, df = load_dataset(args.data, args.target, scale=not args.no_scale)
    X_train, X_val, X_test, y_train, y_val, _ = train_val_test_split_df(
        df,
        target_col=args.target,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        random_state=args.random_state,
    )
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    base_model, _ = fit_polynomial_regression(
        X_train_full, y_train_full, X_test=None, degree=args.degree
    )
    sympy_eq, x_symbols = model_to_sympy_equation(base_model)
    gwr = GradientWeightedRegressor(
        sympy_equation=sympy_eq,
        x_symbols=x_symbols,
        n_neighbors=args.neighbors,
        relative=args.relative,
    )
    gwr.fit(X_train_full, y_train_full)

    n_queries = min(args.n_queries, X_test.shape[0])
    query_points = X_test[:n_queries]

    single_query_times_us: list[float] = []
    for i in range(n_queries):
        t0 = time.perf_counter_ns()
        _ = gwr.predict(query_points[i : i + 1])
        elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
        single_query_times_us.append(float(elapsed_us))
    single_query_times_us_arr = np.array(single_query_times_us, dtype=np.float64)
    single_stats = _timing_stats_us(single_query_times_us_arr)

    t_batch_0 = time.perf_counter_ns()
    _ = gwr.predict(query_points)
    batch_total_us = (time.perf_counter_ns() - t_batch_0) / 1000.0
    batch_per_query_us = batch_total_us / max(n_queries, 1)
    batch_stats = {
        "mean_us": float(batch_per_query_us),
        "std_us": 0.0,
        "min_us": float(batch_per_query_us),
        "max_us": float(batch_per_query_us),
        "p95_us": float(batch_per_query_us),
    }

    out_df = pd.DataFrame(
        [
            {"mode": "single_query", "n_queries": n_queries, **single_stats},
            {"mode": "batch", "n_queries": n_queries, **batch_stats},
        ]
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_df.to_csv(args.output, index=False)

    placeholder_df = out_df.copy()
    placeholder_df["gwr_mean_us"] = placeholder_df["mean_us"]
    placeholder_df["gwr_std_us"] = placeholder_df["std_us"]
    placeholder_df["gwr_min_us"] = placeholder_df["min_us"]
    placeholder_df["gwr_max_us"] = placeholder_df["max_us"]
    placeholder_df["gwr_p95_us"] = placeholder_df["p95_us"]
    placeholder_df["calphad_query_time_us"] = np.nan
    placeholder_df["calphad_query_time_us_note"] = (
        "manual_input_required_from_vendor_or_coauthor"
        if CALPHAD_QUERY_TIME_US is None
        else str(CALPHAD_QUERY_TIME_US)
    )
    placeholder_df = placeholder_df[
        [
            "mode",
            "n_queries",
            "gwr_mean_us",
            "gwr_std_us",
            "gwr_min_us",
            "gwr_max_us",
            "gwr_p95_us",
            "calphad_query_time_us",
            "calphad_query_time_us_note",
        ]
    ]
    os.makedirs(os.path.dirname(args.placeholder_output) or ".", exist_ok=True)
    placeholder_df.to_csv(args.placeholder_output, index=False)

    print(f"Saved timing detail: {args.output}")
    print(f"Saved timing placeholder: {args.placeholder_output}")


if __name__ == "__main__":
    main()
