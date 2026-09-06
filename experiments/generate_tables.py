#!/usr/bin/env python
"""Generate reproducible summary tables from experiment CSV outputs."""

from __future__ import annotations

import argparse
import os

import pandas as pd


def _maybe_read(path: str):
    return pd.read_csv(path) if os.path.exists(path) else None


def main():
    parser = argparse.ArgumentParser(description="Generate result tables")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    baseline = _maybe_read(os.path.join(results_dir, "baseline_comparison.csv"))
    gwr = _maybe_read(os.path.join(results_dir, "gwr_results.csv"))
    pmlb = _maybe_read(os.path.join(results_dir, "pmlb_benchmark.csv"))
    deriv = _maybe_read(os.path.join(results_dir, "derivative_evaluation_summary.csv"))

    if baseline is not None:
        cols = [c for c in baseline.columns if c.startswith(("mse_", "mae_", "mape_")) or c == "model"]
        baseline[cols].to_csv(os.path.join(results_dir, "table_baselines_metrics.csv"), index=False)

    if gwr is not None:
        cols = [
            c for c in gwr.columns
            if c.startswith(("base_mse", "base_mae", "base_mape", "gwr_mse", "gwr_mae", "gwr_mape"))
            or c in {"base_model", "repeats"}
        ]
        gwr[cols].to_csv(os.path.join(results_dir, "table_gwr_vs_base.csv"), index=False)

    if pmlb is not None and {"dataset", "model"}.issubset(pmlb.columns):
        agg = (
            pmlb.groupby("model")[["mse", "mae_mean", "mape_mean"]]
            .agg(["mean", "std"])
            .reset_index()
        )
        agg.columns = ["_".join([x for x in col if x]).strip("_") for col in agg.columns.to_flat_index()]
        agg.to_csv(os.path.join(results_dir, "table_pmlb_summary.csv"), index=False)

    if deriv is not None:
        deriv.to_csv(os.path.join(results_dir, "table_derivative_summary.csv"), index=False)

    print("Generated table CSVs in", results_dir)


if __name__ == "__main__":
    main()
