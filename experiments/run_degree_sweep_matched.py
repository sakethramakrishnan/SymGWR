#!/usr/bin/env python
"""Matched train/val MAPE degree sweep via full SymGWR pipeline (k=1 locked).

Same protocol as ``run_corrected_sweeps.run_k_sweep`` / ``joint_grid_n10_full.csv``:
60/20/20 split, fit base polynomial + SymGWR on X_train, evaluate MAPE on train and val.
Writes ``results/degree_sweep_matched_train_val_n10.csv`` only (no locked CSV edits).
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.run_corrected_sweeps import (  # noqa: E402
    DEGREE_VALUES,
    LOCKED_K,
    SEEDS,
    _eval_mape,
    _fit_gwr,
    _iter_seed_splits,
    _parse_degrees,
    _parse_seeds,
)
from gwr.utils import load_dataset  # noqa: E402

OUTPUT_FIELDS = ["degree", "seed", "train_mape", "val_mape"]
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(__file__), "..", "results/degree_sweep_matched_train_val_n10.csv"
)


def _load_done_keys(path: str) -> set[tuple[int, int]]:
    done: set[tuple[int, int]] = set()
    if os.path.exists(path):
        existing = pd.read_csv(path, usecols=OUTPUT_FIELDS)
        for r in existing.itertuples(index=False):
            done.add((int(r.seed), int(r.degree)))
        print(f"Resuming from {len(done)} cached rows in {path}", flush=True)
    return done


def _append_row(path: str, row: dict[str, int | float]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def run_matched_degree_sweep(
    df: pd.DataFrame,
    target_col: str,
    seeds: list[int],
    degrees: list[int],
    output_path: str,
) -> pd.DataFrame:
    done = _load_done_keys(output_path)
    needed = {(seed, degree) for seed in seeds for degree in degrees}
    if needed.issubset(done):
        print("Matched degree sweep already complete; skipping recompute", flush=True)
        return pd.read_csv(output_path)

    for seed in seeds:
        _, split = next(_iter_seed_splits(df, target_col, [seed]))
        for degree in degrees:
            key = (seed, degree)
            if key in done:
                continue
            gwr = _fit_gwr(split.X_train, split.y_train, degree, LOCKED_K)
            if LOCKED_K == 1:
                train_mape = 0.0
            else:
                train_mape = _eval_mape(gwr, split.X_train, split.y_train)
            val_mape = _eval_mape(gwr, split.X_val, split.y_val)
            row = {
                "degree": degree,
                "seed": seed,
                "train_mape": train_mape,
                "val_mape": val_mape,
            }
            _append_row(output_path, row)
            done.add(key)
            print(
                f"matched degree-sweep seed={seed} degree={degree} "
                f"train_mape={train_mape:.6g} val_mape={val_mape:.6g}",
                flush=True,
            )
            del gwr
            gc.collect()
        del split
        gc.collect()

    return pd.read_csv(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched SymGWR train/val degree sweep")
    parser.add_argument("--data", type=str, default="data/raw/718_all.csv")
    parser.add_argument("--target", type=str, default="Gibbs_energy")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-scale", action="store_true")
    parser.add_argument("--degrees", type=str, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    degrees = _parse_degrees(args.degrees)
    t0 = time.perf_counter()

    _, _, df = load_dataset(args.data, args.target, scale=not args.no_scale)
    raw = run_matched_degree_sweep(df, args.target, seeds, degrees, args.output)
    raw.to_csv(args.output, index=False)

    summary = (
        raw.groupby("degree", as_index=False)
        .agg(
            train_mape_mean=("train_mape", "mean"),
            train_mape_std=("train_mape", "std"),
            val_mape_mean=("val_mape", "mean"),
            val_mape_std=("val_mape", "std"),
        )
        .sort_values("degree")
    )
    print("\nPer-degree means (n=10 seeds):", flush=True)
    print(summary.to_string(index=False), flush=True)

    train_means = summary["train_mape_mean"].to_numpy()
    monotonic = all(train_means[i] >= train_means[i + 1] for i in range(len(train_means) - 1))
    print(f"\ntrain_mape monotonically non-increasing over degrees 1-15: {monotonic}", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"Saved: {args.output}", flush=True)
    print(f"Total wall-clock time: {elapsed:.1f}s ({elapsed/60:.2f} min)", flush=True)


if __name__ == "__main__":
    main()
