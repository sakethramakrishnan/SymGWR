#!/usr/bin/env python
"""Reproducible locked SymGWR single-query timing with independent runs.

This script writes:
  - per-run summary table (independent process invocations)
  - locked canonical table from one single 10k-query warmed run
  - a compact provenance JSON blob
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gwr.core import GradientWeightedRegressor
from gwr.models import fit_polynomial_regression, model_to_sympy_equation
from gwr.utils import load_dataset, train_val_test_split_df


def _build_model(
    data: str,
    target: str,
    degree: int,
    neighbors: int,
    random_state: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    no_scale: bool,
) -> tuple[GradientWeightedRegressor, np.ndarray]:
    _, _, df = load_dataset(data, target, scale=not no_scale)
    X_train, X_val, X_test, y_train, y_val, _ = train_val_test_split_df(
        df,
        target_col=target,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        random_state=random_state,
    )
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    model, _ = fit_polynomial_regression(X_train_full, y_train_full, X_test=None, degree=degree)
    sympy_eq, x_symbols = model_to_sympy_equation(model)
    gwr = GradientWeightedRegressor(
        sympy_equation=sympy_eq,
        x_symbols=x_symbols,
        n_neighbors=neighbors,
        relative=False,
    )
    gwr.fit(X_train_full, y_train_full)
    return gwr, X_test


def _measure_single_run(
    data: str,
    target: str,
    degree: int,
    neighbors: int,
    random_state: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    n_queries: int,
    n_warmup: int,
    query_seed: int,
    no_scale: bool,
) -> np.ndarray:
    gwr, query_pool = _build_model(
        data=data,
        target=target,
        degree=degree,
        neighbors=neighbors,
        random_state=random_state,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        no_scale=no_scale,
    )

    # Warm up model path before timing to reduce one-time artifacts.
    warmup_count = max(1, min(n_warmup, query_pool.shape[0]))
    for i in range(warmup_count):
        _ = gwr.predict(query_pool[i : i + 1])

    rng = np.random.default_rng(query_seed)
    query_idx = rng.choice(query_pool.shape[0], size=n_queries, replace=True)
    samples_us = np.zeros(n_queries, dtype=np.float64)
    for i, idx in enumerate(query_idx):
        t0 = time.perf_counter_ns()
        _ = gwr.predict(query_pool[idx : idx + 1])
        samples_us[i] = (time.perf_counter_ns() - t0) / 1000.0
    return samples_us


def _single_run_mode(args: argparse.Namespace) -> None:
    samples_us = _measure_single_run(
        data=args.data,
        target=args.target,
        degree=args.degree,
        neighbors=args.neighbors,
        random_state=args.random_state,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        n_queries=args.n_queries,
        n_warmup=args.n_warmup,
        query_seed=args.query_seed,
        no_scale=args.no_scale,
    )
    np.save(args.samples_out, samples_us)


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def _aggregate(run_samples: list[np.ndarray]) -> tuple[pd.DataFrame, dict[str, Any]]:
    run_rows: list[dict[str, Any]] = []
    run_means = []
    for i, samples in enumerate(run_samples, start=1):
        mean_us = float(np.mean(samples))
        run_means.append(mean_us)
        run_rows.append(
            {
                "run_id": i,
                "mean_us": mean_us,
                "std_us": float(np.std(samples)),
                "min_us": float(np.min(samples)),
                "p50_us": float(np.percentile(samples, 50)),
                "p95_us": float(np.percentile(samples, 95)),
                "p99_us": float(np.percentile(samples, 99)),
                "max_us": float(np.max(samples)),
            }
        )

    canonical_samples = run_samples[0]
    p50 = float(np.percentile(canonical_samples, 50))
    p95 = float(np.percentile(canonical_samples, 95))
    p99 = float(np.percentile(canonical_samples, 99))
    canonical_row = {
        "mode": "single_query",
        "locked_convention": "single_canonical_run_distribution",
        "canonical_run_id": 1,
        "n_runs_independent_support": int(len(run_samples)),
        "n_queries": int(canonical_samples.shape[0]),
        "mean_us": float(np.mean(canonical_samples)),
        "std_us": float(np.std(canonical_samples)),
        "min_us": float(np.min(canonical_samples)),
        "p50_us": p50,
        "p95_us": p95,
        "p99_us": p99,
        "max_us": float(np.max(canonical_samples)),
        "quantiles_monotonic": bool(p50 <= p95 <= p99),
        "across_run_mean_of_means_us": float(np.mean(run_means)),
        "across_run_std_of_means_us": float(np.std(run_means)),
        "across_run_min_of_means_us": float(np.min(run_means)),
        "across_run_max_of_means_us": float(np.max(run_means)),
        "sampling": "with_replacement_from_test_split_same_run",
        "warmup_note": "quantiles from one warmed canonical run (10,000 queries)",
    }
    return pd.DataFrame(run_rows), canonical_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible locked SymGWR timing")
    parser.add_argument("--data", type=str, default="data/raw/718_all.csv")
    parser.add_argument("--target", type=str, default="Gibbs_energy")
    parser.add_argument("--degree", type=int, default=5)
    parser.add_argument("--neighbors", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--n-queries", type=int, default=10000)
    parser.add_argument("--n-warmup", type=int, default=1)
    parser.add_argument("--query-seed-base", type=int, default=7000)
    parser.add_argument(
        "--run-summary-out",
        type=str,
        default="results/gwr_timing_runs_v2.csv",
    )
    parser.add_argument(
        "--locked-out",
        type=str,
        default="results/gwr_timing_detail_v2_locked.csv",
    )
    parser.add_argument(
        "--provenance-out",
        type=str,
        default="results/gwr_timing_detail_v2_provenance.json",
    )
    parser.add_argument("--no-scale", action="store_true")

    # Internal mode used to force independent Python processes per run.
    parser.add_argument("--single-run", action="store_true")
    parser.add_argument("--query-seed", type=int, default=0)
    parser.add_argument("--samples-out", type=str, default="")

    args = parser.parse_args()

    if args.single_run:
        _single_run_mode(args)
        return

    run_samples: list[np.ndarray] = []
    for run_idx in range(args.n_runs):
        query_seed = args.query_seed_base + run_idx
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            tmp_path = tmp.name
        cmd = [
            sys.executable,
            "-m",
            "experiments.run_gwr_timing_final_v2",
            "--single-run",
            "--data",
            args.data,
            "--target",
            args.target,
            "--degree",
            str(args.degree),
            "--neighbors",
            str(args.neighbors),
            "--random-state",
            str(args.random_state),
            "--train-frac",
            str(args.train_frac),
            "--val-frac",
            str(args.val_frac),
            "--test-frac",
            str(args.test_frac),
            "--n-queries",
            str(args.n_queries),
            "--n-warmup",
            str(args.n_warmup),
            "--query-seed",
            str(query_seed),
            "--samples-out",
            tmp_path,
        ]
        if args.no_scale:
            cmd.append("--no-scale")
        subprocess.run(cmd, check=True)
        samples = np.load(tmp_path)
        os.unlink(tmp_path)
        run_samples.append(samples)

    run_df, canonical_row = _aggregate(run_samples)
    os.makedirs(Path(args.run_summary_out).parent, exist_ok=True)
    run_df.to_csv(args.run_summary_out, index=False)

    locked_df = pd.DataFrame([canonical_row])
    os.makedirs(Path(args.locked_out).parent, exist_ok=True)
    locked_df.to_csv(args.locked_out, index=False)

    provenance = {
        "source_script": "experiments/run_gwr_timing_final_v2.py",
        "python_executable": sys.executable,
        "parameters": {
            "data": args.data,
            "target": args.target,
            "degree": args.degree,
            "neighbors": args.neighbors,
            "random_state": args.random_state,
            "train_frac": args.train_frac,
            "val_frac": args.val_frac,
            "test_frac": args.test_frac,
            "n_runs": args.n_runs,
            "n_queries": args.n_queries,
            "n_warmup": args.n_warmup,
            "query_seed_base": args.query_seed_base,
            "no_scale": bool(args.no_scale),
        },
        "outputs": {
            "run_summary_csv": args.run_summary_out,
            "locked_csv": args.locked_out,
        },
        "hardware_environment_disclosure": {
            "methodology_note": "Absolute latency is hardware and environment dependent; do not treat as universal.",
            "cpu_model": _cpu_model(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
        },
    }
    os.makedirs(Path(args.provenance_out).parent, exist_ok=True)
    with open(args.provenance_out, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, sort_keys=True)

    print(f"Saved run summary: {args.run_summary_out}")
    print(f"Saved locked timing: {args.locked_out}")
    print(f"Saved provenance: {args.provenance_out}")


if __name__ == "__main__":
    main()
