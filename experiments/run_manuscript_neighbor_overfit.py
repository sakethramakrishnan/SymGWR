#!/usr/bin/env python
"""Neighbor-sweep (multi base regressor) + matched train/val overfitting figures.

Protocol matches ``run_corrected_sweeps`` (60/20/20 split, seeds 0..9, absolute k).
Writes only new deliverable files requested for the manuscript figures.
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
import warnings
from collections import Counter
from typing import Any, Iterator

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.run_corrected_sweeps import (  # noqa: E402
    K_VALUES,
    LOCKED_K,
    SEEDS,
    _eval_mape,
    _iter_seed_splits,
    _parse_degrees,
    _parse_seeds,
)
from gwr.core import GradientWeightedRegressor, PipelineGradientWeightedRegressor
from gwr.models import (
    fit_linear_regression,
    fit_polynomial_regression,
    fit_ridge_regression,
    model_to_sympy_equation,
)
from gwr.utils import compute_all_metrics, load_dataset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RIDGE_ALPHA = 1.0
POLY_DEGREES = list(range(1, 16))
UNSTABLE_MAPE_THRESHOLD = 1.0

NEIGHBOR_RAW_FIELDS = [
    "base_regressor_type",
    "degree",
    "k",
    "seed",
    "val_mape",
    "unstable",
]
DEGREE_RAW_FIELDS = ["degree", "seed", "train_mape", "val_mape", "unstable"]

DEFAULT_NEIGHBOR_RAW = os.path.join(REPO_ROOT, "results/neighbor_sweep_all_regressors_raw_n10.csv")
DEFAULT_NEIGHBOR_OUT = os.path.join(REPO_ROOT, "results/neighbor_sweep_all_regressors_n10.csv")
DEFAULT_NEIGHBOR_FIG = os.path.join(REPO_ROOT, "figures/gwr_neighbor_sweep_plot.png")
DEFAULT_DEGREE_RAW = os.path.join(REPO_ROOT, "results/degree_sweep_matched_train_val_raw_n10.csv")
DEFAULT_DEGREE_OUT = os.path.join(REPO_ROOT, "results/degree_sweep_matched_train_val_n10.csv")
DEFAULT_DEGREE_FIG = os.path.join(REPO_ROOT, "figures/overfitting_comparison_plot.png")


def _is_unstable(mape: float) -> bool:
    if not np.isfinite(mape):
        return True
    return abs(mape) > UNSTABLE_MAPE_THRESHOLD


def _fit_sympy_gwr(
    sympy_eq: Any,
    x_symbols: list,
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
) -> GradientWeightedRegressor:
    gwr = GradientWeightedRegressor(
        sympy_equation=sympy_eq,
        x_symbols=x_symbols,
        n_neighbors=k,
        relative=False,
    )
    gwr.fit(X_train, y_train)
    return gwr


def _fit_poly_gwr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    degree: int,
    k: int,
) -> PipelineGradientWeightedRegressor:
    base_model, _ = fit_polynomial_regression(
        X_train,
        y_train,
        X_test=None,
        degree=degree,
    )
    gwr = PipelineGradientWeightedRegressor(
        pipeline=base_model,
        n_neighbors=k,
        relative=False,
    )
    gwr.fit(X_train, y_train)
    return gwr


def _eval_gwr_at_k(
    gwr: GradientWeightedRegressor | PipelineGradientWeightedRegressor,
    k: int,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> float:
    gwr.n_neighbors = k
    return _eval_mape(gwr, X_eval, y_eval)


def _load_done_keys(path: str, fields: list[str], key_cols: list[str]) -> set[tuple]:
    done: set[tuple] = set()
    if os.path.exists(path):
        existing = pd.read_csv(path, usecols=fields)
        for row in existing.itertuples(index=False):
            done.add(tuple(getattr(row, c) for c in key_cols))
        print(f"Resuming {path}: {len(done)} cached rows", flush=True)
    return done


def _append_row(path: str, fields: list[str], row: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def _regressor_specs(degrees: list[int]) -> list[tuple[str, int | None]]:
    specs: list[tuple[str, int | None]] = [("linear", None), ("ridge", None)]
    for degree in degrees:
        specs.append(("polynomial", degree))
    return specs


def run_neighbor_sweep(
    df: pd.DataFrame,
    target_col: str,
    seeds: list[int],
    degrees: list[int],
    raw_path: str,
) -> pd.DataFrame:
    done = _load_done_keys(raw_path, NEIGHBOR_RAW_FIELDS, ["base_regressor_type", "degree", "k", "seed"])
    specs = _regressor_specs(degrees)

    for seed, split in _iter_seed_splits(df, target_col, seeds):
        for base_type, degree in specs:
            degree_val = -1 if degree is None else int(degree)
            needed_ks = [
                k
                for k in K_VALUES
                if (base_type, degree_val, k, seed) not in done
            ]
            if not needed_ks:
                continue

            unstable_fit = False
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    if base_type == "linear":
                        model, _ = fit_linear_regression(split.X_train, split.y_train)
                        sympy_eq, x_symbols = model_to_sympy_equation(model)
                        gwr: Any = _fit_sympy_gwr(
                            sympy_eq, x_symbols, split.X_train, split.y_train, k=1
                        )
                    elif base_type == "ridge":
                        model, _ = fit_ridge_regression(
                            split.X_train, split.y_train, alpha=RIDGE_ALPHA
                        )
                        sympy_eq, x_symbols = model_to_sympy_equation(model)
                        gwr = _fit_sympy_gwr(
                            sympy_eq, x_symbols, split.X_train, split.y_train, k=1
                        )
                    else:
                        gwr = _fit_poly_gwr(
                            split.X_train, split.y_train, int(degree), k=1
                        )
                    if caught:
                        unstable_fit = True
            except Exception as exc:
                print(
                    f"neighbor-sweep FAILED seed={seed} {base_type} degree={degree}: {exc}",
                    flush=True,
                )
                for k in needed_ks:
                    row = {
                        "base_regressor_type": base_type,
                        "degree": degree_val,
                        "k": k,
                        "seed": seed,
                        "val_mape": float("nan"),
                        "unstable": True,
                    }
                    _append_row(raw_path, NEIGHBOR_RAW_FIELDS, row)
                    done.add((base_type, degree_val, k, seed))
                continue

            for k in needed_ks:
                try:
                    val_mape = _eval_gwr_at_k(gwr, k, split.X_val, split.y_val)
                    unstable = unstable_fit or _is_unstable(val_mape)
                except Exception as exc:
                    print(
                        f"neighbor-sweep eval FAILED seed={seed} {base_type} "
                        f"degree={degree} k={k}: {exc}",
                        flush=True,
                    )
                    val_mape = float("nan")
                    unstable = True

                row = {
                    "base_regressor_type": base_type,
                    "degree": degree_val,
                    "k": k,
                    "seed": seed,
                    "val_mape": val_mape,
                    "unstable": unstable,
                }
                _append_row(raw_path, NEIGHBOR_RAW_FIELDS, row)
                done.add((base_type, degree_val, k, seed))
                print(
                    f"neighbor seed={seed} {base_type} deg={degree} k={k} "
                    f"val_mape={val_mape:.6g} unstable={unstable}",
                    flush=True,
                )

            del gwr
            gc.collect()

        del split
        gc.collect()

    return pd.read_csv(raw_path)


def run_matched_degree_sweep(
    df: pd.DataFrame,
    target_col: str,
    seeds: list[int],
    degrees: list[int],
    raw_path: str,
) -> pd.DataFrame:
    done = _load_done_keys(raw_path, DEGREE_RAW_FIELDS, ["degree", "seed"])

    for seed, split in _iter_seed_splits(df, target_col, seeds):
        for degree in degrees:
            if (degree, seed) in done:
                continue
            unstable = False
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    gwr = _fit_poly_gwr(split.X_train, split.y_train, degree, LOCKED_K)
                    if caught:
                        unstable = True
                train_mape = _eval_gwr_at_k(gwr, LOCKED_K, split.X_train, split.y_train)
                val_mape = _eval_gwr_at_k(gwr, LOCKED_K, split.X_val, split.y_val)
                unstable = unstable or _is_unstable(train_mape) or _is_unstable(val_mape)
            except Exception as exc:
                print(
                    f"matched degree-sweep FAILED seed={seed} degree={degree}: {exc}",
                    flush=True,
                )
                train_mape = float("nan")
                val_mape = float("nan")
                unstable = True

            row = {
                "degree": degree,
                "seed": seed,
                "train_mape": train_mape,
                "val_mape": val_mape,
                "unstable": unstable,
            }
            _append_row(raw_path, DEGREE_RAW_FIELDS, row)
            done.add((degree, seed))
            print(
                f"matched seed={seed} degree={degree} train={train_mape:.6g} "
                f"val={val_mape:.6g} unstable={unstable}",
                flush=True,
            )
            del gwr
            gc.collect()

        del split
        gc.collect()

    return pd.read_csv(raw_path)


def summarize_neighbor_sweep(raw: pd.DataFrame, n_seeds: int) -> pd.DataFrame:
    stable = raw[~raw["unstable"].astype(bool)].copy()
    summary = (
        stable.groupby(["base_regressor_type", "degree", "k"], as_index=False)
        .agg(
            mean_val_mape=("val_mape", "mean"),
            std_val_mape=("val_mape", "std"),
            n_seeds=("seed", "nunique"),
        )
        .fillna(0.0)
    )
    summary["n_seeds"] = summary["n_seeds"].astype(int)
    unstable_counts = (
        raw[raw["unstable"].astype(bool)]
        .groupby(["base_regressor_type", "degree", "k"], as_index=False)
        .size()
        .rename(columns={"size": "n_unstable"})
    )
    if not unstable_counts.empty:
        summary = summary.merge(
            unstable_counts,
            on=["base_regressor_type", "degree", "k"],
            how="left",
        )
        summary["n_unstable"] = summary["n_unstable"].fillna(0).astype(int)
    else:
        summary["n_unstable"] = 0
    return summary.sort_values(["base_regressor_type", "degree", "k"]).reset_index(drop=True)


def summarize_degree_sweep(raw: pd.DataFrame, n_seeds: int) -> pd.DataFrame:
    stable = raw[~raw["unstable"].astype(bool)].copy()
    summary = (
        stable.groupby("degree", as_index=False)
        .agg(
            mean_train_mape=("train_mape", "mean"),
            std_train_mape=("train_mape", "std"),
            mean_val_mape=("val_mape", "mean"),
            std_val_mape=("val_mape", "std"),
            n_seeds=("seed", "nunique"),
        )
        .fillna(0.0)
    )
    summary["n_seeds"] = summary["n_seeds"].astype(int)
    return summary.sort_values("degree").reset_index(drop=True)


def _line_label(base_type: str, degree: int) -> str:
    if base_type == "linear":
        return "Linear / degree=1"
    if base_type == "ridge":
        return f"Ridge (alpha={RIDGE_ALPHA})"
    return f"Polynomial degree {int(degree)}"


def plot_neighbor_sweep(summary: pd.DataFrame, out_png: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig, ax = plt.subplots()
    poly = summary[summary["base_regressor_type"] == "polynomial"].copy()
    # degree=1 polynomial identical to linear — plot once via linear series
    poly_degrees = sorted(d for d in poly["degree"].unique() if d != 1)
    try:
        cmap = matplotlib.colormaps["viridis"].resampled(max(len(poly_degrees), 1))
    except AttributeError:
        cmap = cm.get_cmap("viridis", max(len(poly_degrees), 1))

    for idx, degree in enumerate(poly_degrees):
        sub = poly[poly["degree"] == degree].sort_values("k")
        color = cmap(idx / max(len(poly_degrees) - 1, 1))
        ax.plot(
            sub["k"],
            sub["mean_val_mape"],
            color=color,
            linewidth=1.5,
            label=_line_label("polynomial", int(degree)),
        )

    for base_type, color, linestyle in [
        ("linear", "#1f77b4", "--"),
        ("ridge", "#d62728", ":"),
    ]:
        sub = summary[summary["base_regressor_type"] == base_type].sort_values("k")
        if sub.empty:
            continue
        ax.plot(
            sub["k"],
            sub["mean_val_mape"],
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            marker="o",
            markersize=4,
            label=_line_label(base_type, -1),
        )

    ax.set_yscale("log")
    ax.set_xlabel("Number of absolute neighbors k")
    ax.set_ylabel("Mean validation MAPE (log scale)")
    ax.set_title(
        f"SymGWR validation MAPE vs. k (n={summary['n_seeds'].max()} seeds, 60/20/20 split)"
    )
    ax.set_xticks(K_VALUES)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_overfitting_comparison(summary: pd.DataFrame, out_png: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    degrees = summary["degree"].to_numpy()
    train = summary["mean_train_mape"].to_numpy()
    val = summary["mean_val_mape"].to_numpy()

    fig, ax = plt.subplots()
    ax.plot(degrees, train, "-o", color="#1f77b4", linewidth=1.8, label="Training MAPE")
    ax.plot(degrees, val, "-s", color="#d62728", linewidth=1.8, label="Validation MAPE")
    ax.set_yscale("log")
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("Mean MAPE (log scale)")
    ax.set_title(
        f"Matched GWR overfitting (k={LOCKED_K}, n={summary['n_seeds'].iloc[0]} seeds)"
    )
    ax.set_xticks(list(degrees))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def analyze_neighbor_sweep(summary: pd.DataFrame, raw: pd.DataFrame) -> dict[str, Any]:
    poly5 = summary[
        (summary["base_regressor_type"] == "polynomial") & (summary["degree"] == 5)
    ].copy()
    best_k_poly5 = int(poly5.loc[poly5["mean_val_mape"].idxmin(), "k"]) if not poly5.empty else None

    best_ks: list[int] = []
    for (_, degree), grp in summary.groupby(["base_regressor_type", "degree"]):
        if grp["mean_val_mape"].isna().all():
            continue
        best_ks.append(int(grp.loc[grp["mean_val_mape"].idxmin(), "k"]))
    modal_k = Counter(best_ks).most_common(1)[0][0] if best_ks else None

    unstable = raw[raw["unstable"].astype(bool)]
    unstable_specs = sorted(
        {
            (r.base_regressor_type, int(r.degree))
            for r in unstable.itertuples(index=False)
        }
    )
    return {
        "best_k_poly5": best_k_poly5,
        "modal_k": modal_k,
        "best_ks": best_ks,
        "unstable_specs": unstable_specs,
    }


def analyze_degree_sweep(summary: pd.DataFrame, raw: pd.DataFrame) -> dict[str, Any]:
    degrees = summary["degree"].to_list()
    train = summary["mean_train_mape"].to_list()
    val = summary["mean_val_mape"].to_list()

    train_monotonic = all(train[i] >= train[i + 1] for i in range(len(train) - 1))
    val_increase_start = None
    for i in range(1, len(val)):
        if val[i] > val[i - 1]:
            val_increase_start = degrees[i]
            break

    deg5 = summary[summary["degree"] == 5]
    unstable_degrees = sorted(raw[raw["unstable"].astype(bool)]["degree"].unique().tolist())

    return {
        "train_monotonic": train_monotonic,
        "val_increase_start": val_increase_start,
        "deg5_train": float(deg5["mean_train_mape"].iloc[0]) if not deg5.empty else None,
        "deg5_val": float(deg5["mean_val_mape"].iloc[0]) if not deg5.empty else None,
        "max_degree": int(degrees[-1]),
        "train_at_max": float(train[-1]),
        "val_at_max": float(val[-1]),
        "gap_at_max": float(val[-1] - train[-1]),
        "unstable_degrees": unstable_degrees,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Manuscript neighbor + overfitting sweeps")
    parser.add_argument("--data", type=str, default="data/raw/718_all.csv")
    parser.add_argument("--target", type=str, default="Gibbs_energy")
    parser.add_argument("--no-scale", action="store_true")
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--degrees", type=str, default=None)
    parser.add_argument("--neighbor-only", action="store_true")
    parser.add_argument("--degree-only", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--skip-plot", action="store_true", help="Skip figure generation")
    parser.add_argument("--neighbor-raw", type=str, default=DEFAULT_NEIGHBOR_RAW)
    parser.add_argument("--neighbor-output", type=str, default=DEFAULT_NEIGHBOR_OUT)
    parser.add_argument("--neighbor-figure", type=str, default=DEFAULT_NEIGHBOR_FIG)
    parser.add_argument("--degree-raw", type=str, default=DEFAULT_DEGREE_RAW)
    parser.add_argument("--degree-output", type=str, default=DEFAULT_DEGREE_OUT)
    parser.add_argument("--degree-figure", type=str, default=DEFAULT_DEGREE_FIG)
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    degrees = _parse_degrees(args.degrees)
    t0 = time.perf_counter()

    run_neighbor = not args.degree_only
    run_degree = not args.neighbor_only

    if args.plot_only:
        neighbor_summary = pd.DataFrame()
        degree_summary = pd.DataFrame()
        neighbor_raw = pd.DataFrame()
        degree_raw = pd.DataFrame()
        if run_neighbor and os.path.exists(args.neighbor_raw):
            neighbor_raw = pd.read_csv(args.neighbor_raw)
            neighbor_summary = summarize_neighbor_sweep(neighbor_raw, len(seeds))
            os.makedirs(os.path.dirname(args.neighbor_output) or ".", exist_ok=True)
            neighbor_summary.to_csv(args.neighbor_output, index=False)
            plot_neighbor_sweep(neighbor_summary, args.neighbor_figure)
        if run_degree and os.path.exists(args.degree_raw):
            degree_raw = pd.read_csv(args.degree_raw)
            degree_summary = summarize_degree_sweep(degree_raw, len(seeds))
            os.makedirs(os.path.dirname(args.degree_output) or ".", exist_ok=True)
            degree_summary.to_csv(args.degree_output, index=False)
            plot_overfitting_comparison(degree_summary, args.degree_figure)
        elapsed = time.perf_counter() - t0
        print(f"Plot-only done in {elapsed:.1f}s", flush=True)
        return

    _, _, df = load_dataset(args.data, args.target, scale=not args.no_scale)

    neighbor_raw = pd.DataFrame()
    degree_raw = pd.DataFrame()
    if run_neighbor:
        neighbor_raw = run_neighbor_sweep(df, args.target, seeds, degrees, args.neighbor_raw)
        neighbor_summary = summarize_neighbor_sweep(neighbor_raw, len(seeds))
        if not args.skip_plot:
            neighbor_summary.to_csv(args.neighbor_output, index=False)
            plot_neighbor_sweep(neighbor_summary, args.neighbor_figure)
        stats = analyze_neighbor_sweep(neighbor_summary, neighbor_raw)
        print("\n=== Neighbor sweep analysis ===", flush=True)
        print(f"Ridge alpha used: {RIDGE_ALPHA}", flush=True)
        print(f"Polynomial degree=5 best k: {stats['best_k_poly5']}", flush=True)
        print(f"Modal best k across regressors: {stats['modal_k']}", flush=True)
        if stats["unstable_specs"]:
            print(f"Unstable (flagged) specs: {stats['unstable_specs']}", flush=True)
        print(f"Saved: {args.neighbor_output}", flush=True)
        if not args.skip_plot:
            print(f"Saved: {args.neighbor_figure}", flush=True)

    if run_degree:
        degree_raw = run_matched_degree_sweep(df, args.target, seeds, degrees, args.degree_raw)
        degree_summary = summarize_degree_sweep(degree_raw, len(seeds))
        if not args.skip_plot:
            degree_summary.to_csv(args.degree_output, index=False)
            plot_overfitting_comparison(degree_summary, args.degree_figure)
        stats = analyze_degree_sweep(degree_summary, degree_raw)
        print("\n=== Matched degree sweep analysis ===", flush=True)
        print(f"Train MAPE monotonically non-increasing: {stats['train_monotonic']}", flush=True)
        print(f"Validation MAPE increase starts at degree: {stats['val_increase_start']}", flush=True)
        print(
            f"Degree 5: train={stats['deg5_train']:.6g}, val={stats['deg5_val']:.6g}",
            flush=True,
        )
        print(
            f"At max degree {stats['max_degree']}: train={stats['train_at_max']:.6g}, "
            f"val={stats['val_at_max']:.6g}, gap={stats['gap_at_max']:.6g}",
            flush=True,
        )
        if stats["unstable_degrees"]:
            print(f"Unstable degrees: {stats['unstable_degrees']}", flush=True)
        print(f"Saved: {args.degree_output}", flush=True)
        if not args.skip_plot:
            print(f"Saved: {args.degree_figure}", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\nTotal wall-clock: {elapsed:.1f}s ({elapsed/60:.2f} min)", flush=True)


if __name__ == "__main__":
    main()
