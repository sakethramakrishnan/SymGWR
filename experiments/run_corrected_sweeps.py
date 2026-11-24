#!/usr/bin/env python
"""Corrected k-sweep and degree-sweep for locked SymGWR config selection.

Protocol (matches ``run_joint_grid.py`` / ``joint_grid_n10_final.csv``):
- split 0.6 / 0.2 / 0.2
- fit base polynomial + SymGWR on ``X_train`` only
- k-sweep: evaluate on ``X_val`` (selection split)
- degree-sweep: train MAPE = per-element closed-form polynomial on train split
  (original Figure 6 "Avg Training MAPE (Element-wise)"); test MAPE = full SymGWR
  on held-out test split
- seeds 0..9 (n=10)
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gwr.core import GradientWeightedRegressor, PipelineGradientWeightedRegressor
from gwr.models import fit_polynomial_regression
from gwr.utils import compute_all_metrics, load_dataset

SEEDS = list(range(10))
LOCKED_DEGREE = 5
LOCKED_K = 1
K_VALUES = list(range(1, 11))
DEGREE_VALUES = list(range(1, 16))
PAIRED_T_STAT = 3.4502686358628867
PAIRED_T_P = 0.007271628607374566

ELEMENTS = {
    "nb": "Mass_fraction_Nb",
    "al": "Mass_fraction_Al",
    "ti": "Mass_fraction_Ti",
    "mo": "Mass_fraction_Mo",
    "fe": "Mass_fraction_Fe",
    "cr": "Mass_fraction_Cr",
}


@dataclass(frozen=True)
class SeedSplit:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    element_train: dict[str, tuple[np.ndarray, np.ndarray]]


CHECKPOINT_FIELDS = ["degree", "k", "seed", "split", "mape"]
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POLY_K_SWEEP_CSV = os.path.join(REPO_ROOT, "results/gwr_polynomial_evaluation_all_degrees.csv")
OTHER_K_SWEEP_CSV = os.path.join(REPO_ROOT, "results/other_models_sweep_results.csv")
# ACM 0.48\textwidth ~ 3.15 in; portrait-ish for two-column figures.
PORTRAIT_FIGSIZE = (3.15, 4.50)
FIG_DPI = 300


def _arrays_from_df(df: pd.DataFrame, target_col: str) -> tuple[np.ndarray, np.ndarray]:
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).values
    y = df[target_col].values
    return X, y


def _element_train_arrays(
    df_train: pd.DataFrame,
    target_col: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    element_train: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for element, mass_col in ELEMENTS.items():
        subset = df_train[df_train["temp_class"] == element]
        if subset.empty:
            continue
        X = subset[["Temperature_[K]", mass_col]].to_numpy(dtype=np.float64)
        y = subset[target_col].to_numpy(dtype=np.float64)
        element_train[element] = (X, y)
    return element_train


def _split_seed(df: pd.DataFrame, target_col: str, seed: int) -> SeedSplit:
    df_train, df_temp = train_test_split(df, train_size=0.6, random_state=seed)
    df_val, df_test = train_test_split(df_temp, train_size=0.5, random_state=seed)
    element_train = _element_train_arrays(df_train, target_col)
    X_train, y_train = _arrays_from_df(df_train, target_col)
    X_val, y_val = _arrays_from_df(df_val, target_col)
    X_test, y_test = _arrays_from_df(df_test, target_col)
    return SeedSplit(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        element_train=element_train,
    )


def _iter_seed_splits(
    df: pd.DataFrame,
    target_col: str,
    seeds: list[int],
) -> Iterator[tuple[int, SeedSplit]]:
    for seed in seeds:
        yield seed, _split_seed(df, target_col, seed)
def _fit_gwr(
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


def _eval_mape(
    gwr: GradientWeightedRegressor | PipelineGradientWeightedRegressor,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> float:
    y_pred = gwr.predict(X_eval)
    return float(compute_all_metrics(y_eval, y_pred)["mape_mean"])


def _eval_element_train_mape(element_train: dict[str, tuple[np.ndarray, np.ndarray]], degree: int) -> float:
    """Per-element closed-form polynomial train MAPE (no SymGWR neighbor lookup)."""
    mapes: list[float] = []
    for X, y in element_train.values():
        model = Pipeline(
            [
                ("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),
                ("linear_model", LinearRegression()),
            ]
        )
        model.fit(X, y)
        y_pred = model.predict(X)
        mapes.append(float(compute_all_metrics(y, y_pred)["mape_mean"]))
    if not mapes:
        raise ValueError("No element subsets found in training split.")
    return float(np.mean(mapes))


def _purge_corrupted_degree_rows(checkpoint_path: str, max_mape: float = 1e3) -> int:
    """Drop checkpoint rows with numerically exploded MAPE (bad dual-fit runs)."""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return 0
    raw = pd.read_csv(checkpoint_path)
    if raw.empty:
        return 0
    bad = raw["mape"].abs() > max_mape
    if not bad.any():
        return 0
    kept = raw.loc[~bad].copy()
    removed = int(bad.sum())
    kept.to_csv(checkpoint_path, index=False)
    print(
        f"Purged {removed} corrupted degree-sweep rows with |mape|>{max_mape}",
        flush=True,
    )
    return removed


def _k_keys_for_seeds(seeds: list[int]) -> set[tuple[int, int, str, int]]:
    return {(seed, LOCKED_DEGREE, "val", k) for seed in seeds for k in K_VALUES}


def _degree_keys_for_seeds(seeds: list[int], degrees: list[int]) -> set[tuple[int, int, str, int]]:
    keys: set[tuple[int, int, str, int]] = set()
    for seed in seeds:
        for degree in degrees:
            keys.add((seed, degree, "train", LOCKED_K))
            keys.add((seed, degree, "test", LOCKED_K))
    return keys


def _load_done_keys(checkpoint_path: str | None) -> set[tuple[int, int, str, int]]:
    done: set[tuple[int, int, str, int]] = set()
    if checkpoint_path and os.path.exists(checkpoint_path):
        existing = pd.read_csv(checkpoint_path, usecols=CHECKPOINT_FIELDS)
        for r in existing.itertuples(index=False):
            done.add((int(r.seed), int(r.degree), str(r.split), int(r.k)))
        print(f"Resuming from {len(done)} cached cells in {checkpoint_path}", flush=True)
    return done


def _append_checkpoint_row(checkpoint_path: str | None, row: dict[str, float | int | str]) -> None:
    if not checkpoint_path:
        return
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    write_header = not os.path.exists(checkpoint_path) or os.path.getsize(checkpoint_path) == 0
    with open(checkpoint_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CHECKPOINT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def _release_heavy_objects(*objs: object) -> None:
    for obj in objs:
        del obj
    gc.collect()
def run_k_sweep(
    df: pd.DataFrame,
    target_col: str,
    seeds: list[int],
    checkpoint_path: str | None = None,
) -> pd.DataFrame:
    done = _load_done_keys(checkpoint_path)
    needed = _k_keys_for_seeds(seeds)
    if needed.issubset(done):
        print("k-sweep already complete; skipping recompute", flush=True)
        if checkpoint_path and os.path.exists(checkpoint_path):
            return pd.read_csv(checkpoint_path)
        return pd.DataFrame(columns=CHECKPOINT_FIELDS)
    for seed, split in _iter_seed_splits(df, target_col, seeds):
        base_model, _ = fit_polynomial_regression(
            split.X_train,
            split.y_train,
            X_test=None,
            degree=LOCKED_DEGREE,
        )
        for k in K_VALUES:
            key = (seed, LOCKED_DEGREE, "val", k)
            if key in done:
                continue
            gwr = PipelineGradientWeightedRegressor(
                pipeline=base_model,
                n_neighbors=k,
                relative=False,
            )
            gwr.fit(split.X_train, split.y_train)
            mape = _eval_mape(gwr, split.X_val, split.y_val)
            row = {
                "degree": LOCKED_DEGREE,
                "k": k,
                "seed": seed,
                "split": "val",
                "mape": mape,
            }
            _append_checkpoint_row(checkpoint_path, row)
            done.add(key)
            print(f"k-sweep seed={seed} k={k} val_mape={mape:.6g}", flush=True)
            _release_heavy_objects(gwr)
        _release_heavy_objects(base_model, split)
        gc.collect()
    if checkpoint_path and os.path.exists(checkpoint_path):
        return pd.read_csv(checkpoint_path)
    return pd.DataFrame(columns=CHECKPOINT_FIELDS)


def run_degree_sweep(
    df: pd.DataFrame,
    target_col: str,
    seeds: list[int],
    degrees: list[int],
    checkpoint_path: str | None = None,
) -> pd.DataFrame:
    done = _load_done_keys(checkpoint_path)
    needed = _degree_keys_for_seeds(seeds, degrees)
    if needed.issubset(done):
        print("degree-sweep already complete; skipping recompute", flush=True)
        if checkpoint_path and os.path.exists(checkpoint_path):
            return pd.read_csv(checkpoint_path)
        return pd.DataFrame(columns=CHECKPOINT_FIELDS)
    for seed, split in _iter_seed_splits(df, target_col, seeds):
        for degree in degrees:
            if (seed, degree, "train", LOCKED_K) in done and (
                seed,
                degree,
                "test",
                LOCKED_K,
            ) in done:
                continue
            train_mape = _eval_element_train_mape(split.element_train, degree)
            gwr = _fit_gwr(split.X_train, split.y_train, degree, LOCKED_K)
            test_mape = _eval_mape(gwr, split.X_test, split.y_test)
            for split_name, mape in [("train", train_mape), ("test", test_mape)]:
                key = (seed, degree, split_name, LOCKED_K)
                if key in done:
                    continue
                row = {
                    "degree": degree,
                    "k": LOCKED_K,
                    "seed": seed,
                    "split": split_name,
                    "mape": mape,
                }
                _append_checkpoint_row(checkpoint_path, row)
                done.add(key)
            print(
                f"degree-sweep seed={seed} degree={degree} "
                f"train_mape={train_mape:.6g} test_mape={test_mape:.6g}",
                flush=True,
            )
            _release_heavy_objects(gwr)
        _release_heavy_objects(split)
        del split
        gc.collect()
    if checkpoint_path and os.path.exists(checkpoint_path):
        return pd.read_csv(checkpoint_path)
    return pd.DataFrame(columns=CHECKPOINT_FIELDS)


def _summarize(raw: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        raw.groupby(group_cols, as_index=False)
        .agg(
            n=("mape", "count"),
            mape_mean=("mape", "mean"),
            mape_std=("mape", "std"),
            mape_stderr=("mape", lambda s: float(np.std(s, ddof=0) / np.sqrt(len(s)))),
        )
        .fillna(0.0)
    )


def _png_render_line(figsize: tuple[float, float], dpi: int = FIG_DPI) -> str:
    w_in, h_in = figsize
    return (
        f"Render: figsize=({w_in:.2f}, {h_in:.2f}) in @ {dpi} dpi = "
        f"{int(round(w_in * dpi))} x {int(round(h_in * dpi))} px"
    )


def plot_k_sweep(
    summary: pd.DataFrame,
    out_png: str,
    out_txt: str | None = None,
    raw_csv: str | None = None,
) -> tuple[float, float]:
    """Val MAPE vs k from ``k_sweep_corrected_n10.csv`` only (degree=5, n=10 seeds).

  Single line with mean +/- std; k=1 marked as selected. Matplotlib default figsize.
    """
    del summary  # call-site compatibility; plot reads raw CSV only
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_path = raw_csv or os.path.join(REPO_ROOT, "results/k_sweep_corrected_n10.csv")
    raw = pd.read_csv(csv_path)
    agg = (
        raw.groupby("k", as_index=False)
        .agg(mape_mean=("mape", "mean"), mape_std=("mape", "std"))
        .sort_values("k")
    )
    ks = agg["k"].to_numpy()
    mean = agg["mape_mean"].to_numpy()
    std = agg["mape_std"].fillna(0.0).to_numpy()

    fig, ax = plt.subplots()
    figsize = fig.get_size_inches()
    ax.fill_between(ks, np.maximum(mean - std, 1e-12), mean + std, color="#1f77b4", alpha=0.22, zorder=1)
    ax.errorbar(
        ks,
        mean,
        yerr=std,
        color="#1f77b4",
        linewidth=2.0,
        marker="o",
        markersize=6,
        capsize=4,
        capthick=1.2,
        elinewidth=1.2,
        label=f"Polynomial degree {LOCKED_DEGREE} (mean +/- std, n={len(SEEDS)})",
        zorder=2,
    )
    sel = agg[agg["k"] == LOCKED_K].iloc[0]
    ax.scatter(
        [LOCKED_K],
        [sel["mape_mean"]],
        s=120,
        facecolors="none",
        edgecolors="#d62728",
        linewidths=2.5,
        zorder=4,
        label=f"Selected k={LOCKED_K}",
    )
    ax.annotate(
        f"k={LOCKED_K} (selected)",
        xy=(LOCKED_K, sel["mape_mean"]),
        xytext=(LOCKED_K + 1.2, sel["mape_mean"] * 2.2),
        arrowprops={"arrowstyle": "->", "color": "#d62728", "lw": 1.2},
        fontsize=10,
        fontweight="bold",
        color="#d62728",
        ha="left",
        zorder=5,
    )

    ax.set_yscale("log")
    ax.set_xlabel("Number of absolute neighbors k", fontsize=11, fontweight="bold")
    ax.set_ylabel("Validation MAPE (log scale)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"SymGWR validation MAPE vs. k (degree={LOCKED_DEGREE}, n={len(SEEDS)} seeds)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(list(range(1, 11)))
    ax.set_xlim(0.7, 10.3)
    ax.tick_params(labelsize=10)
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.12)
    ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out_png, dpi=FIG_DPI)
    plt.close(fig)

    if out_txt:
        _write_txt(
            out_txt,
            [
                f"Source: {os.path.relpath(csv_path, REPO_ROOT)}",
                "Columns used: degree, k, seed, split, mape",
                f"Filters: degree={LOCKED_DEGREE}, k=1..10, split=val, n={len(SEEDS)} seeds",
                "Aggregation: mean +/- std across seeds per k",
                f"Selected point: k={LOCKED_K}",
                "Script: experiments/run_corrected_sweeps.py --plot-only (plot_k_sweep)",
                _png_render_line(tuple(figsize), FIG_DPI),
                "Figure size: matplotlib default (no forced aspect crop)",
            ],
        )
    return float(figsize[0]), float(figsize[1])


def plot_degree_sweep(summary: pd.DataFrame, out_png: str, out_txt: str | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = summary[summary["split"] == "train"].sort_values("degree")
    test = summary[summary["split"] == "test"].sort_values("degree")
    common_degrees = sorted(set(train["degree"]).intersection(test["degree"]))
    train = train[train["degree"].isin(common_degrees)]
    test = test[test["degree"].isin(common_degrees)]
    degrees = train["degree"].to_numpy()
    train_mape = train["mape_mean"].to_numpy()
    test_mape = test["mape_mean"].to_numpy()

    fig, ax = plt.subplots(figsize=PORTRAIT_FIGSIZE)
    ax.plot(
        degrees, train_mape, "-", color="#1f77b4",
        linewidth=1.7, marker="o", markersize=4.0,
        label="Training MAPE", zorder=3,
    )
    ax.plot(
        degrees, test_mape, "--", color="#d62728",
        linewidth=1.7, marker="s", markersize=4.0,
        label="Test MAPE", zorder=3,
    )

    ax.set_yscale("log")
    ymin = min(train_mape.min(), test_mape.min()) * 0.40
    ymax = max(train_mape.max(), test_mape.max()) * 8.0
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(0.5, 15.5)
    ax.set_xlabel("Polynomial degree", fontsize=8, fontweight="bold")
    ax.set_ylabel("MAPE (log scale)", fontsize=8, fontweight="bold")
    # Title must not use LaTeX-style "vs.\ " — matplotlib keeps the backslash.
    ax.set_title(
        "Overfitting behavior in polynomial regression\n"
        f"(degrees 1–15, locked k={LOCKED_K}, n={len(SEEDS)} seeds)",
        fontsize=8.5, fontweight="bold",
    )
    ax.set_xticks(list(degrees))
    ax.tick_params(labelsize=7)
    ax.grid(True, which="major", alpha=0.30)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper left", fontsize=7.5, framealpha=0.95,
        handlelength=2.4, borderpad=0.28, labelspacing=0.22,
        handletextpad=0.4, borderaxespad=0.28,
    )

    test_7 = float(test.loc[test["degree"] == 7, "mape_mean"].iloc[0])
    test_8 = float(test.loc[test["degree"] == 8, "mape_mean"].iloc[0])
    jump = test_8 / test_7
    ax.annotate(
        f"Test MAPE diverges at degree 8\n({jump:.0f}x jump from degree 7)",
        xy=(8, test_8),
        xycoords="data",
        xytext=(0.42, 0.18),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#8b0000", "lw": 1.0,
                    "connectionstyle": "arc3,rad=-0.15"},
        fontsize=7, fontweight="bold", color="#8b0000",
        ha="left", va="center",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "#fff0f0",
              "edgecolor": "#8b0000", "alpha": 0.95},
        zorder=6,
    )

    fig.tight_layout(pad=0.28)
    fig.savefig(out_png, dpi=FIG_DPI)
    plt.close(fig)

    if out_txt:
        deg5_train = float(
            train.loc[train["degree"] == LOCKED_DEGREE, "mape_mean"].iloc[0]
        )
        deg5_test = float(
            test.loc[test["degree"] == LOCKED_DEGREE, "mape_mean"].iloc[0]
        )
        deg1_train = float(train.loc[train["degree"] == 1, "mape_mean"].iloc[0])
        best_test_degree = int(test.sort_values("mape_mean").iloc[0]["degree"])
        _write_txt(
            out_txt,
            [
                "Source: results/degree_sweep_corrected_n10.csv",
                "Columns used: degree, k, seed, split, mape",
                f"Filters: k={LOCKED_K}, degree={min(DEGREE_VALUES)}..{max(DEGREE_VALUES)}",
                f"n: {len(SEEDS)} seeds (0..9)",
                "Protocol: train MAPE = per-element closed-form polynomial on X_train; "
                "test MAPE = full SymGWR (k=1) on X_test",
                f"degree=1 mean train MAPE (element-wise): {deg1_train}",
                f"degree={LOCKED_DEGREE}: train MAPE={deg5_train}, test MAPE={deg5_test}",
                f"Lowest mean test MAPE at degree={best_test_degree}",
                f"Callout: test MAPE diverges at degree 8 ({jump:.0f}x jump from degree 7)",
                "Script: experiments/run_corrected_sweeps.py --plot-only (plot_degree_sweep)",
                _png_render_line(PORTRAIT_FIGSIZE),
                "Paper: 0.48 textwidth (~3.15 in wide), portrait-ish",
            ],
        )


def _write_txt(path: str, lines: list[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _parse_degrees(degree_arg: str | None) -> list[int]:
    if not degree_arg:
        return DEGREE_VALUES
    degrees: list[int] = []
    for part in degree_arg.split(","):
        part = part.strip()
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            degrees.extend(range(int(start_s), int(end_s) + 1))
        else:
            degrees.append(int(part))
    return sorted(set(degrees))


def _parse_seeds(seed_arg: str | None) -> list[int]:
    if not seed_arg:
        return SEEDS
    seeds: list[int] = []
    for part in seed_arg.split(","):
        part = part.strip()
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            seeds.extend(range(int(start_s), int(end_s) + 1))
        else:
            seeds.append(int(part))
    return sorted(set(seeds))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run corrected k/degree sweeps")
    parser.add_argument("--data", type=str, default="data/raw/718_all.csv")
    parser.add_argument("--target", type=str, default="Gibbs_energy")
    parser.add_argument("--k-raw-output", type=str, default="results/k_sweep_corrected_n10.csv")
    parser.add_argument("--k-summary-output", type=str, default="results/k_sweep_corrected_summary.csv")
    parser.add_argument("--degree-raw-output", type=str, default="results/degree_sweep_corrected_n10.csv")
    parser.add_argument(
        "--degree-summary-output",
        type=str,
        default="results/degree_sweep_corrected_summary.csv",
    )
    parser.add_argument("--k-figure", type=str, default="figures/k_sweep_corrected.png")
    parser.add_argument("--degree-figure", type=str, default="figures/degree_sweep_corrected.png")
    parser.add_argument("--k-figure-txt", type=str, default="figures/k_sweep_corrected.txt")
    parser.add_argument(
        "--degree-figure-txt",
        type=str,
        default="figures/degree_sweep_corrected.txt",
    )
    parser.add_argument("--no-scale", action="store_true")
    parser.add_argument("--k-only", action="store_true")
    parser.add_argument("--degree-only", action="store_true")
    parser.add_argument("--plot-only", action="store_true", help="Regenerate figures/summaries from existing CSVs")
    parser.add_argument(
        "--degrees",
        type=str,
        default=None,
        help="Comma-separated degrees or ranges, e.g. '12' or '1-15' (default: all)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds or ranges, e.g. '0,1,2' or '0-4' (default: all 0..9)",
    )
    args = parser.parse_args()
    seeds = _parse_seeds(args.seeds)
    degrees = _parse_degrees(args.degrees)

    t0 = time.perf_counter()

    if args.plot_only:
        plot_k_sweep(pd.DataFrame(), args.k_figure, args.k_figure_txt)
        print(f"Saved k-sweep figure: {args.k_figure}", flush=True)
        if os.path.exists(args.degree_raw_output):
            degree_raw = pd.read_csv(args.degree_raw_output)
            degree_summary = _summarize(degree_raw, ["degree", "k", "split"])
            plot_degree_sweep(degree_summary, args.degree_figure, args.degree_figure_txt)
            print(f"Saved degree-sweep figure: {args.degree_figure}", flush=True)
        elapsed = time.perf_counter() - t0
        print(f"Total wall-clock time: {elapsed:.1f}s ({elapsed/60:.2f} min)", flush=True)
        return

    _, _, df = load_dataset(args.data, args.target, scale=not args.no_scale)
    print(f"Loaded standardized data; seeds={seeds} degrees={degrees}", flush=True)

    run_k = not args.degree_only
    run_degree = not args.k_only

    k_raw = pd.DataFrame()
    k_summary = pd.DataFrame()
    if run_k:
        k_raw = run_k_sweep(df, args.target, seeds, checkpoint_path=args.k_raw_output)
        k_summary = _summarize(k_raw, ["degree", "k", "split"])

    degree_raw = pd.DataFrame()
    degree_summary = pd.DataFrame()
    if run_degree:
        if os.path.exists(args.degree_raw_output):
            existing = pd.read_csv(args.degree_raw_output)
            train_rows = existing[existing["split"] == "train"]
            if not train_rows.empty and (train_rows["mape"] == 0.0).all():
                print(
                    "Removing stale degree-sweep checkpoint with train_mape=0.0 bug",
                    flush=True,
                )
                os.remove(args.degree_raw_output)
            else:
                _purge_corrupted_degree_rows(args.degree_raw_output)
        degree_raw = run_degree_sweep(
            df,
            args.target,
            seeds,
            degrees,
            checkpoint_path=args.degree_raw_output,
        )
        degree_summary = _summarize(degree_raw, ["degree", "k", "split"])

    if run_k:
        os.makedirs(os.path.dirname(args.k_raw_output) or ".", exist_ok=True)
        k_raw.to_csv(args.k_raw_output, index=False)
        k_summary.to_csv(args.k_summary_output, index=False)
        plot_k_sweep(k_summary, args.k_figure, args.k_figure_txt)
        print(f"Saved k-sweep raw: {args.k_raw_output}", flush=True)
        print(f"Saved k-sweep summary: {args.k_summary_output}", flush=True)
        print(f"Saved figure: {args.k_figure}", flush=True)

    if run_degree:
        os.makedirs(os.path.dirname(args.degree_raw_output) or ".", exist_ok=True)
        degree_raw.to_csv(args.degree_raw_output, index=False)
        degree_summary.to_csv(args.degree_summary_output, index=False)
        plot_degree_sweep(degree_summary, args.degree_figure, args.degree_figure_txt)
        print(f"Saved degree-sweep raw: {args.degree_raw_output}", flush=True)
        print(f"Saved degree-sweep summary: {args.degree_summary_output}", flush=True)
        print(f"Saved figure: {args.degree_figure}", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"Total wall-clock time: {elapsed:.1f}s ({elapsed/60:.2f} min)", flush=True)


if __name__ == "__main__":
    main()
