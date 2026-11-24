#!/usr/bin/env python
"""Run joint degree x neighbors ablation grid for CALPHAD.

Also supports extended k-sweep over model families (polynomial degrees 1-15,
linear, ridge, PySR) for ``figures/k_sweep_corrected.png``.

NOTE: This script replaces the univariate ablation evidence files
`results/ablation_degree.csv` and `results/ablation_neighbors.csv` as primary
ablation support. Old files remain for record/history only.
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
from typing import Any, Iterator

import numpy as np
import pandas as pd
import sympy as sp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from baselines.symbolic_pysr import train_pysr
from gwr.core import GradientWeightedRegressor, PipelineGradientWeightedRegressor
from gwr.models import (
    fit_linear_regression,
    fit_polynomial_regression,
    fit_ridge_regression,
    model_to_sympy_equation,
)
from gwr.utils import compute_all_metrics, load_dataset, train_val_test_split_df


DEGREES = [2, 3, 4, 5, 6]
NEIGHBORS = [1, 2, 3, 4, 5]
DEFAULT_RANDOM_STATES = [0, 1, 2, 3, 4]

K_SWEEP_DEGREES = list(range(1, 16))
K_SWEEP_NEIGHBORS = list(range(1, 11))
K_SWEEP_SEEDS = list(range(10))
LOCKED_K = 1

PYSR_PARAMS = {
    "niterations": 50,
    "population_size": 20,
    "populations": 10,
    "random_state": 42,
}

K_SWEEP_CHECKPOINT_FIELDS = [
    "base_model",
    "degree",
    "k",
    "random_state",
    "eval_split",
    "mape",
    "mae",
    "rmse",
    "r2",
]


def _parse_seeds(seeds_arg: str | None) -> list[int]:
    if seeds_arg is None:
        return list(DEFAULT_RANDOM_STATES)
    return [int(s.strip()) for s in seeds_arg.split(",") if s.strip()]


def _parse_int_range(spec: str | None, default: list[int]) -> list[int]:
    if spec is None:
        return list(default)
    values: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            values.extend(range(int(start_s), int(end_s) + 1))
        else:
            values.append(int(part))
    return sorted(set(values))


def _pysr_to_sympy(model: Any, n_features: int) -> tuple[sp.Expr, list[sp.Symbol]]:
    equation = model.sympy()
    x_symbols = list(sp.symbols([f"x{i}" for i in range(n_features)]))
    return equation, x_symbols


def _fit_sympy_gwr(
    sympy_eq: sp.Expr,
    x_symbols: list[sp.Symbol],
    X_train: np.ndarray,
    y_train: np.ndarray,
    relative: bool,
) -> GradientWeightedRegressor:
    gwr = GradientWeightedRegressor(
        sympy_equation=sympy_eq,
        x_symbols=x_symbols,
        n_neighbors=1,
        relative=relative,
    )
    gwr.fit(X_train, y_train)
    return gwr


def _fit_pipeline_gwr(
    pipeline: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    relative: bool,
) -> PipelineGradientWeightedRegressor:
    gwr = PipelineGradientWeightedRegressor(
        pipeline=pipeline,
        n_neighbors=1,
        relative=relative,
    )
    gwr.fit(X_train, y_train)
    return gwr


def _eval_gwr_at_k(
    gwr: GradientWeightedRegressor | PipelineGradientWeightedRegressor,
    k: int,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> dict[str, float]:
    gwr.n_neighbors = k
    y_pred = gwr.predict(X_eval)
    metrics = compute_all_metrics(y_eval, y_pred)
    return {
        "mape": float(metrics["mape_mean"]),
        "mae": float(metrics["mae_mean"]),
        "rmse": float(metrics["rmse"]),
        "r2": float(metrics["r2"]),
    }


def _load_k_sweep_done_keys(checkpoint_path: str | None) -> set[tuple[str, int, int, int, str]]:
    done: set[tuple[str, int, int, int, str]] = set()
    if checkpoint_path and os.path.exists(checkpoint_path):
        existing = pd.read_csv(checkpoint_path, usecols=K_SWEEP_CHECKPOINT_FIELDS)
        for row in existing.itertuples(index=False):
            done.add(
                (
                    str(row.base_model),
                    int(row.degree),
                    int(row.k),
                    int(row.random_state),
                    str(row.eval_split),
                )
            )
        print(f"Resuming k-sweep from {len(done)} cached cells in {checkpoint_path}", flush=True)
    return done


def _append_k_sweep_row(checkpoint_path: str | None, row: dict[str, Any]) -> None:
    if not checkpoint_path:
        return
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    write_header = not os.path.exists(checkpoint_path) or os.path.getsize(checkpoint_path) == 0
    with open(checkpoint_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=K_SWEEP_CHECKPOINT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def _summarize_k_sweep(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(["base_model", "degree", "k", "eval_split"], as_index=False)
        .agg(
            n=("mape", "count"),
            mape_mean=("mape", "mean"),
            mape_std=("mape", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
        )
        .fillna(0.0)
    )


def plot_k_sweep_families(summary: pd.DataFrame, out_png: str, locked_k: int = LOCKED_K) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.lines import Line2D

    val = summary[summary["eval_split"] == "val"].copy()
    fig, ax = plt.subplots(figsize=(7.5, 9.0))

    poly_degrees = sorted(val.loc[val["base_model"] == "polynomial", "degree"].unique())
    cmap = cm.get_cmap("viridis", max(len(poly_degrees), 1))
    for idx, degree in enumerate(poly_degrees):
        sub = val[(val["base_model"] == "polynomial") & (val["degree"] == degree)].sort_values("k")
        ax.plot(
            sub["k"],
            sub["mape_mean"],
            color=cmap(idx),
            linewidth=1.6,
            label=f"Polynomial Degree {int(degree)}",
            zorder=2,
        )

    special_styles = {
        "ridge": {"color": "#d62728", "linestyle": ":", "marker": "^", "label": "Ridge Regression"},
        "linear": {"color": "#1f77b4", "linestyle": "--", "marker": "o", "label": "Linear Regression"},
        "pysr": {"color": "#2ca02c", "linestyle": "--", "marker": "o", "label": "PySR"},
    }
    for base_model, style in special_styles.items():
        sub = val[val["base_model"] == base_model].sort_values("k")
        if sub.empty:
            continue
        ax.plot(
            sub["k"],
            sub["mape_mean"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=2.0,
            markersize=5,
            label=style["label"],
            zorder=3,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Number of Nearest Absolute Neighbors (K)", fontsize=11)
    ax.set_ylabel("Validation MAPE (log scale)", fontsize=11)
    ax.set_title(
        "MAPE vs. Number of Absolute Neighbors (SymGWR Local Approximation)\n"
        f"(global base fit on train, evaluate val, n={K_SWEEP_SEEDS[0]}..{K_SWEEP_SEEDS[-1]} seeds)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(K_SWEEP_NEIGHBORS)
    ax.grid(True, which="both", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    ax.axvspan(locked_k - 0.25, locked_k + 0.25, color="#fff59d", alpha=0.45, zorder=0)

    handles, labels = ax.get_legend_handles_labels()
    poly_count = len(poly_degrees)
    if poly_count:
        cmap_legend = cm.get_cmap("viridis", poly_count)
        poly_handles = [
            Line2D([0], [0], color=cmap_legend(i), linewidth=1.6, label=f"Polynomial Degree {int(d)}")
            for i, d in enumerate(poly_degrees)
        ]
        special_labels = {style["label"] for style in special_styles.values()}
        special_handles = [h for h, label in zip(handles, labels) if label in special_labels]
        ax.legend(
            handles=poly_handles + special_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=7.5,
            title="Models",
            framealpha=0.95,
        )
    else:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=8,
            title="Models",
            framealpha=0.95,
        )

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_k_sweep_families(
    df: pd.DataFrame,
    target_col: str,
    degrees: list[int],
    neighbors: list[int],
    random_states: list[int],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    relative: bool,
    checkpoint_path: str | None,
    include_pysr: bool = True,
) -> pd.DataFrame:
    done = _load_k_sweep_done_keys(checkpoint_path)

    for random_state in random_states:
        X_train, X_val, _, y_train, y_val, _ = train_val_test_split_df(
            df,
            target_col=target_col,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            random_state=random_state,
        )

        for degree in degrees:
            needed = {
                (random_state, degree, k)
                for k in neighbors
                if ("polynomial", degree, k, random_state, "val") not in done
            }
            if not needed:
                continue
            poly_model, _ = fit_polynomial_regression(
                X_train,
                y_train,
                X_test=None,
                degree=degree,
            )
            gwr = _fit_pipeline_gwr(poly_model, X_train, y_train, relative=relative)
            for k in neighbors:
                key = ("polynomial", degree, k, random_state, "val")
                if key in done:
                    continue
                metrics = _eval_gwr_at_k(gwr, k, X_val, y_val)
                row = {
                    "base_model": "polynomial",
                    "degree": degree,
                    "k": k,
                    "random_state": random_state,
                    "eval_split": "val",
                    **metrics,
                }
                _append_k_sweep_row(checkpoint_path, row)
                done.add(key)
                print(
                    f"k-sweep poly degree={degree} seed={random_state} k={k} "
                    f"val_mape={metrics['mape']:.6g}",
                    flush=True,
                )
            del gwr, poly_model
            gc.collect()

        for base_model, fit_fn, degree_marker in [
            ("linear", fit_linear_regression, -1),
            ("ridge", fit_ridge_regression, -1),
        ]:
            needed = {
                k
                for k in neighbors
                if (base_model, degree_marker, k, random_state, "val") not in done
            }
            if not needed:
                continue
            model, _ = fit_fn(X_train, y_train, X_test=None)
            sympy_eq, x_symbols = model_to_sympy_equation(model)
            gwr = _fit_sympy_gwr(sympy_eq, x_symbols, X_train, y_train, relative=relative)
            for k in neighbors:
                key = (base_model, degree_marker, k, random_state, "val")
                if key in done:
                    continue
                metrics = _eval_gwr_at_k(gwr, k, X_val, y_val)
                row = {
                    "base_model": base_model,
                    "degree": degree_marker,
                    "k": k,
                    "random_state": random_state,
                    "eval_split": "val",
                    **metrics,
                }
                _append_k_sweep_row(checkpoint_path, row)
                done.add(key)
                print(
                    f"k-sweep {base_model} seed={random_state} k={k} val_mape={metrics['mape']:.6g}",
                    flush=True,
                )
            del gwr, model
            gc.collect()

        if include_pysr:
            needed = {
                k
                for k in neighbors
                if ("pysr", -1, k, random_state, "val") not in done
            }
            if needed:
                pysr_model, _ = train_pysr(X_train, y_train, **PYSR_PARAMS)
                sympy_eq, x_symbols = _pysr_to_sympy(pysr_model, X_train.shape[1])
                gwr = _fit_sympy_gwr(sympy_eq, x_symbols, X_train, y_train, relative=relative)
                for k in neighbors:
                    key = ("pysr", -1, k, random_state, "val")
                    if key in done:
                        continue
                    metrics = _eval_gwr_at_k(gwr, k, X_val, y_val)
                    row = {
                        "base_model": "pysr",
                        "degree": -1,
                        "k": k,
                        "random_state": random_state,
                        "eval_split": "val",
                        **metrics,
                    }
                    _append_k_sweep_row(checkpoint_path, row)
                    done.add(key)
                    print(
                        f"k-sweep pysr seed={random_state} k={k} val_mape={metrics['mape']:.6g}",
                        flush=True,
                    )
                del gwr, pysr_model
                gc.collect()

        gc.collect()

    if checkpoint_path and os.path.exists(checkpoint_path):
        return pd.read_csv(checkpoint_path)
    return pd.DataFrame(columns=K_SWEEP_CHECKPOINT_FIELDS)


def run_joint_grid(
    df: pd.DataFrame,
    target_col: str,
    degrees: list[int],
    neighbors: list[int],
    random_states: list[int],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    relative: bool,
) -> pd.DataFrame:
    raw_rows: list[dict[str, float | int | str]] = []

    for degree in degrees:
        for k in neighbors:
            for random_state in random_states:
                X_train, X_val, _, y_train, y_val, _ = train_val_test_split_df(
                    df,
                    target_col=target_col,
                    train_frac=train_frac,
                    val_frac=val_frac,
                    test_frac=test_frac,
                    random_state=random_state,
                )
                base_model, _ = fit_polynomial_regression(
                    X_train,
                    y_train,
                    X_test=None,
                    degree=degree,
                )
                sympy_eq, x_symbols = model_to_sympy_equation(base_model)
                gwr = GradientWeightedRegressor(
                    sympy_equation=sympy_eq,
                    x_symbols=x_symbols,
                    n_neighbors=k,
                    relative=relative,
                )
                gwr.fit(X_train, y_train)
                y_pred = gwr.predict(X_val)
                metrics = compute_all_metrics(y_val, y_pred)
                raw_rows.append(
                    {
                        "base_model": "polynomial",
                        "degree": degree,
                        "k": k,
                        "random_state": random_state,
                        "eval_split": "val",
                        "mape": float(metrics["mape_mean"]),
                        "mae": float(metrics["mae_mean"]),
                        "rmse": float(metrics["rmse"]),
                        "r2": float(metrics["r2"]),
                    }
                )
            print(f"done degree={degree} k={k}")

    return pd.DataFrame(raw_rows)


def _write_joint_grid_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    grouped = raw_df.groupby(["degree", "k"], as_index=False)
    summary_df = grouped.agg(
        n_repeats=("mape", "count"),
        mape_mean=("mape", "mean"),
        mape_std=("mape", "std"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        r2_mean=("r2", "mean"),
        r2_std=("r2", "std"),
    )
    summary_df = summary_df.fillna(0.0)

    best_idx = summary_df["mape_mean"].idxmin()
    best_row = summary_df.loc[best_idx]
    best_mape = float(best_row["mape_mean"])
    best_std = float(best_row["mape_std"])
    summary_df["in_indistinguishable_band"] = (
        (summary_df["mape_mean"] - best_mape).abs() <= (summary_df["mape_std"] + best_std)
    )

    band_rows = summary_df[summary_df["in_indistinguishable_band"]].copy()
    band_rows = band_rows.sort_values(["degree", "k"])
    band_label = ";".join([f"(d={int(r.degree)},k={int(r.k)})" for r in band_rows.itertuples()])
    summary_df["indistinguishable_set"] = band_label

    degree4k2_in_band = bool(((band_rows["degree"] == 4) & (band_rows["k"] == 2)).any())
    summary_df["degree4_k2_in_indistinguishable_set"] = degree4k2_in_band
    return summary_df.sort_values(["degree", "k"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Joint grid ablation over degree and k")
    parser.add_argument("--data", type=str, default="data/raw/718_all.csv")
    parser.add_argument("--target", type=str, default="Gibbs_energy")
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--relative", action="store_true")
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated random seeds (default: 0,1,2,3,4 for grid; 0-9 for k-sweep)",
    )
    parser.add_argument("--raw-output", type=str, default="results/joint_grid_raw.csv")
    parser.add_argument("--summary-output", type=str, default="results/joint_grid_summary.csv")
    parser.add_argument("--no-scale", action="store_true")
    parser.add_argument(
        "--k-sweep",
        action="store_true",
        help="Run extended k-sweep over model families (polynomial 1-15, linear, ridge, PySR)",
    )
    parser.add_argument(
        "--k-sweep-raw-output",
        type=str,
        default="results/k_sweep_families_n10.csv",
    )
    parser.add_argument(
        "--k-sweep-summary-output",
        type=str,
        default="results/k_sweep_families_summary.csv",
    )
    parser.add_argument(
        "--k-sweep-figure",
        type=str,
        default="figures/k_sweep_corrected.png",
    )
    parser.add_argument(
        "--degrees",
        type=str,
        default=None,
        help="Polynomial degrees for k-sweep, e.g. '1-15' (default: 1-15)",
    )
    parser.add_argument(
        "--neighbors",
        type=str,
        default=None,
        help="Neighbor counts for k-sweep, e.g. '1-10' (default: 1-10)",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate k-sweep figure/summary from existing CSV",
    )
    parser.add_argument(
        "--skip-pysr",
        action="store_true",
        help="Skip PySR family in k-sweep",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()

    if args.plot_only:
        raw_df = pd.read_csv(args.k_sweep_raw_output)
        summary_df = _summarize_k_sweep(raw_df)
        os.makedirs(os.path.dirname(args.k_sweep_summary_output) or ".", exist_ok=True)
        summary_df.to_csv(args.k_sweep_summary_output, index=False)
        plot_k_sweep_families(summary_df, args.k_sweep_figure)
        elapsed = time.perf_counter() - t0
        print(f"Saved k-sweep summary: {args.k_sweep_summary_output}", flush=True)
        print(f"Saved k-sweep figure: {args.k_sweep_figure}", flush=True)
        print(f"Total wall-clock time: {elapsed:.1f}s ({elapsed/60:.2f} min)", flush=True)
        return

    if args.k_sweep:
        random_states = _parse_int_range(args.seeds, K_SWEEP_SEEDS)
        degrees = _parse_int_range(args.degrees, K_SWEEP_DEGREES)
        neighbors = _parse_int_range(args.neighbors, K_SWEEP_NEIGHBORS)
        _, _, df = load_dataset(args.data, args.target, scale=not args.no_scale)
        print(
            f"K-sweep families: seeds={random_states} degrees={degrees} "
            f"neighbors={neighbors} include_pysr={not args.skip_pysr}",
            flush=True,
        )
        raw_df = run_k_sweep_families(
            df,
            target_col=args.target,
            degrees=degrees,
            neighbors=neighbors,
            random_states=random_states,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            relative=args.relative,
            checkpoint_path=args.k_sweep_raw_output,
            include_pysr=not args.skip_pysr,
        )
        summary_df = _summarize_k_sweep(raw_df)
        os.makedirs(os.path.dirname(args.k_sweep_raw_output) or ".", exist_ok=True)
        raw_df.to_csv(args.k_sweep_raw_output, index=False)
        os.makedirs(os.path.dirname(args.k_sweep_summary_output) or ".", exist_ok=True)
        summary_df.to_csv(args.k_sweep_summary_output, index=False)
        plot_k_sweep_families(summary_df, args.k_sweep_figure)
        elapsed = time.perf_counter() - t0
        print(f"Saved k-sweep raw: {args.k_sweep_raw_output}", flush=True)
        print(f"Saved k-sweep summary: {args.k_sweep_summary_output}", flush=True)
        print(f"Saved k-sweep figure: {args.k_sweep_figure}", flush=True)
        print(f"Total wall-clock time: {elapsed:.1f}s ({elapsed/60:.2f} min)", flush=True)
        return

    random_states = _parse_seeds(args.seeds)
    _, _, df = load_dataset(args.data, args.target, scale=not args.no_scale)
    raw_df = run_joint_grid(
        df,
        target_col=args.target,
        degrees=DEGREES,
        neighbors=NEIGHBORS,
        random_states=random_states,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        relative=args.relative,
    )
    summary_df = _write_joint_grid_summary(raw_df)

    os.makedirs(os.path.dirname(args.raw_output) or ".", exist_ok=True)
    raw_df.to_csv(args.raw_output, index=False)
    os.makedirs(os.path.dirname(args.summary_output) or ".", exist_ok=True)
    summary_df.to_csv(args.summary_output, index=False)
    elapsed = time.perf_counter() - t0
    print(f"Saved raw grid rows: {args.raw_output}")
    print(f"Saved summary grid rows: {args.summary_output}")
    print(f"Total wall-clock time: {elapsed:.1f}s ({elapsed/60:.2f} min)", flush=True)


if __name__ == "__main__":
    main()
