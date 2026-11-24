#!/usr/bin/env python
"""Generate figures from experiment CSV outputs."""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def _maybe_read(path: str):
    return pd.read_csv(path) if os.path.exists(path) else None


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark figures")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--figures-dir", type=str, default="figures")
    args = parser.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)

    baseline = _maybe_read(os.path.join(args.results_dir, "baseline_comparison.csv"))
    if baseline is not None and {"model", "mape_mean_mean", "mape_mean_std"}.issubset(baseline.columns):
        order = baseline.sort_values("mape_mean_mean")
        plt.figure(figsize=(9, 5))
        plt.bar(order["model"], order["mape_mean_mean"], yerr=order["mape_mean_std"], capsize=4)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("MAPE (mean ± std)")
        plt.title("Baseline Comparison")
        plt.tight_layout()
        plt.savefig(os.path.join(args.figures_dir, "baseline_comparison_mape.png"), dpi=200)
        plt.close()

    gwr = _maybe_read(os.path.join(args.results_dir, "gwr_results.csv"))
    if gwr is not None and {"gwr_mape_mean_mean", "base_mape_mean_mean"}.issubset(gwr.columns):
        vals = [float(gwr["base_mape_mean_mean"].iloc[0]), float(gwr["gwr_mape_mean_mean"].iloc[0])]
        plt.figure(figsize=(5, 4))
        plt.bar(["Base", "SymGWR"], vals)
        plt.ylabel("MAPE")
        plt.title("SymGWR vs Base")
        plt.tight_layout()
        plt.savefig(os.path.join(args.figures_dir, "gwr_vs_base_mape.png"), dpi=200)
        plt.close()

    pmlb = _maybe_read(os.path.join(args.results_dir, "pmlb_benchmark.csv"))
    if pmlb is not None and {"model", "mape_mean"}.issubset(pmlb.columns):
        top = (
            pmlb.groupby("model")["mape_mean"]
            .mean()
            .sort_values()
            .head(10)
            .reset_index()
        )
        plt.figure(figsize=(10, 5))
        plt.bar(top["model"], top["mape_mean"])
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("MAPE")
        plt.title("Top-10 models on PMLB")
        plt.tight_layout()
        plt.savefig(os.path.join(args.figures_dir, "pmlb_top10_mape.png"), dpi=200)
        plt.close()

    print("Generated figures in", args.figures_dir)


if __name__ == "__main__":
    main()
