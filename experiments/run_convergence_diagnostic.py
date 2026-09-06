#!/usr/bin/env python
"""Plot gradient-norm convergence diagnostic from precomputed CSV."""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def _plot(trace_df: pd.DataFrame, grad_threshold: float, out_png: str) -> None:
    fig, ax = plt.subplots(figsize=(2.9, 5.2), constrained_layout=True)

    grouped = trace_df.groupby(["seed", "start_id"], sort=True)
    for idx, ((_seed, _start_id), sub) in enumerate(grouped):
        sub = sub.sort_values("iteration")
        ax.plot(
            sub["iteration"],
            sub["grad_norm"],
            alpha=0.75,
            linewidth=1.2,
            label="individual runs" if idx == 0 else None,
        )

    ax.axhline(grad_threshold, color="red", linestyle="--", linewidth=1.2, label=f"grad_tol={grad_threshold:g}")

    ax.set_yscale("log")
    ax.set_title("Gradient norm trajectories\n(alpha=0.01, 10 representative runs)", fontsize=9)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("grad_norm")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9)

    plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot convergence diagnostic figure")
    parser.add_argument("--input-csv", type=str, default="results/convergence_diagnostic.csv")
    parser.add_argument("--output-figure", type=str, default="figures/convergence_diagnostic.png")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    trace_df = df[df["row_type"] == "trace"].copy()
    if trace_df.empty:
        raise ValueError(f"No trace rows found in {args.input_csv}")

    grad_threshold = float(trace_df["grad_threshold"].iloc[0])
    os.makedirs(os.path.dirname(args.output_figure) or ".", exist_ok=True)
    _plot(trace_df, grad_threshold, args.output_figure)
    print(f"Saved: {args.output_figure}")


if __name__ == "__main__":
    main()
