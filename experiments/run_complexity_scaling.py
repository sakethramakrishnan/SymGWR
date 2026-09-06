#!/usr/bin/env python
"""Empirical complexity scaling: locked GWR vs GP fit-time stand-in."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gwr.core import GradientWeightedRegressor
from gwr.models import fit_polynomial_regression, model_to_sympy_equation
from gwr.utils import load_dataset, train_val_test_split_df


@dataclass
class CellStats:
    n: int
    method: str
    measurement_type: str
    mean: float
    std: float
    n_repeats: int


def _parse_n_values(text: str) -> list[int]:
    vals = []
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        vals.append(int(token))
    if not vals:
        raise ValueError("At least one n value required.")
    return sorted(set(vals))


def _build_gp_model(random_state: int) -> Pipeline:
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e0))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        random_state=random_state,
        # Disable hyperparameter search for tractable large-n scaling sweep.
        # Cholesky factorization of the kernel matrix still reflects the
        # O(n^3)-class correlation-matrix bottleneck we want to measure.
        optimizer=None,
    )
    return Pipeline([("scaler", StandardScaler()), ("gp", gp)])


def _fit_locked_gwr(X_train: np.ndarray, y_train: np.ndarray) -> GradientWeightedRegressor:
    base_model, _ = fit_polynomial_regression(X_train, y_train, X_test=None, degree=5)
    sympy_eq, x_symbols = model_to_sympy_equation(base_model)
    gwr = GradientWeightedRegressor(
        sympy_equation=sympy_eq,
        x_symbols=x_symbols,
        n_neighbors=1,
        relative=False,
    )
    gwr.fit(X_train, y_train)
    return gwr


def _single_query_timing_seconds(
    gwr: GradientWeightedRegressor,
    query_pool: np.ndarray,
    rng: np.random.Generator,
    n_queries: int,
) -> np.ndarray:
    # Warm-up query for fairer single-query latency sampling.
    _ = gwr.predict(query_pool[0:1])
    idx = rng.choice(query_pool.shape[0], size=n_queries, replace=True)
    samples = np.zeros(n_queries, dtype=np.float64)
    for i, q_idx in enumerate(idx):
        t0 = time.perf_counter_ns()
        _ = gwr.predict(query_pool[q_idx : q_idx + 1])
        samples[i] = (time.perf_counter_ns() - t0) / 1e9
    return samples


def _trend_exponent(series_df: pd.DataFrame) -> float:
    x = series_df["n"].to_numpy(dtype=np.float64)
    y = series_df["mean"].to_numpy(dtype=np.float64)
    mask = (x > 0) & (y > 0)
    if np.sum(mask) < 2:
        return float("nan")
    return float(np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)[0])


def _write_companion(
    path: str,
    results_df: pd.DataFrame,
    split_protocol: str,
    repeats: int,
    n_queries: int,
    gp_runtime_note: str,
    max_train_full_n: int,
    locked_compare_note: str,
    methodology_reconciliation_note: str,
    self_consistency_note: str,
) -> None:
    lines: list[str] = []
    lines.append("Source: results/complexity_scaling.csv")
    lines.append("Columns used: n, method, measurement_type, mean, std, n_repeats")
    lines.append(f"Split protocol: {split_protocol}")
    lines.append(f"Locked GWR config: degree=5, k=1, relative=False")
    lines.append(f"Seed convention: 0..{repeats - 1} (n={repeats})")
    lines.append(f"Single-query methodology: warmed-up, then {n_queries} one-by-one queries with replacement")
    lines.append(f"Maximum train+val pool size under split protocol: {max_train_full_n}")
    lines.append(gp_runtime_note)
    lines.append(methodology_reconciliation_note)
    lines.append(
        "Important substitution: GaussianProcess fit time is used as empirical stand-in for GEK-style "
        "correlation-matrix cost; this is NOT a direct GEK benchmark."
    )
    lines.append(locked_compare_note)
    lines.append(self_consistency_note)
    lines.append("")
    lines.append("Empirical trend characterization (log-log slope exponent):")

    for (method, measurement_type), group in results_df.groupby(["method", "measurement_type"]):
        exponent = _trend_exponent(group.sort_values("n"))
        lines.append(f"- {method} / {measurement_type}: slope ~ {exponent:.3f}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _plot_results(df: pd.DataFrame, out_png: str) -> None:
    plt.figure(figsize=(9, 5.5))
    styles = {
        ("GWR", "fit_time_s"): ("o-", "GWR fit"),
        ("GWR", "query_time_s"): ("s-", "GWR single-query"),
        ("GaussianProcess", "fit_time_s"): ("^-", "GP fit (GEK stand-in)"),
    }
    for key, (style, label) in styles.items():
        sub = df[(df["method"] == key[0]) & (df["measurement_type"] == key[1])].sort_values("n")
        if sub.empty:
            continue
        plt.errorbar(
            sub["n"],
            sub["mean"],
            yerr=sub["std"],
            fmt=style,
            capsize=3,
            linewidth=1.5,
            markersize=5,
            label=label,
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Training set size n")
    plt.ylabel("Time (seconds, log scale)")
    plt.title("Empirical complexity scaling (locked GWR vs GP fit stand-in)")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Complexity scaling benchmark")
    parser.add_argument("--data", type=str, default="data/raw/718_all.csv")
    parser.add_argument("--target", type=str, default="Gibbs_energy")
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--n-values", type=str, default="500,1000,2000,4000,6000,8181")
    parser.add_argument("--n-queries", type=int, default=10000)
    parser.add_argument("--output-csv", type=str, default="results/complexity_scaling.csv")
    parser.add_argument("--output-figure", type=str, default="figures/complexity_scaling.png")
    parser.add_argument("--output-txt", type=str, default="figures/complexity_scaling.txt")
    parser.add_argument(
        "--output-report",
        type=str,
        default="results/task1_complexity_summary_report.txt",
    )
    parser.add_argument("--no-scale", action="store_true")
    parser.add_argument(
        "--locked-timing-csv",
        type=str,
        default="results/gwr_timing_detail_v2_locked.csv",
    )
    args = parser.parse_args()

    _, _, df = load_dataset(args.data, args.target, scale=not args.no_scale)
    X_all = df.drop(columns=[args.target]).select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    y_all = df[args.target].to_numpy(dtype=np.float64)
    n_values = _parse_n_values(args.n_values)

    split_protocol = f"{args.train_frac:.1f}/{args.val_frac:.1f}/{args.test_frac:.1f}"
    max_train_full_n: int | None = None
    max_dataset_n = int(X_all.shape[0])

    gwr_fit_by_n: dict[int, list[float]] = {n: [] for n in n_values}
    gwr_query_by_n: dict[int, list[float]] = {n: [] for n in n_values}
    gp_fit_by_n: dict[int, list[float]] = {n: [] for n in n_values}

    for seed in range(args.repeats):
        X_train, X_val, X_test, y_train, y_val, _ = train_val_test_split_df(
            df,
            target_col=args.target,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            random_state=seed,
        )
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.concatenate([y_train, y_val])
        if max_train_full_n is None:
            max_train_full_n = int(X_train_full.shape[0])
        max_train_full_n = min(max_train_full_n, int(X_train_full.shape[0]))

        rng = np.random.default_rng(seed)
        for n in n_values:
            if n > max_dataset_n:
                raise ValueError(f"Requested n={n} exceeds full dataset size {max_dataset_n}.")

            if n <= X_train_full.shape[0]:
                idx = rng.choice(X_train_full.shape[0], size=n, replace=False)
                X_sub = X_train_full[idx]
                y_sub = y_train_full[idx]
                train_pool = "train_plus_val_split_pool"
            else:
                idx = rng.choice(X_all.shape[0], size=n, replace=False)
                X_sub = X_all[idx]
                y_sub = y_all[idx]
                train_pool = "full_dataset_pool"

            t0 = time.perf_counter()
            gwr = _fit_locked_gwr(X_sub, y_sub)
            gwr_fit_by_n[n].append(time.perf_counter() - t0)

            q_samples = _single_query_timing_seconds(gwr, X_test, rng, n_queries=args.n_queries)
            gwr_query_by_n[n].append(float(np.mean(q_samples)))

            gp_model = _build_gp_model(random_state=seed)
            t1 = time.perf_counter()
            gp_model.fit(X_sub, y_sub)
            gp_fit_by_n[n].append(time.perf_counter() - t1)

            print(
                f"seed={seed} n={n} "
                f"gwr_fit={gwr_fit_by_n[n][-1]:.4f}s gwr_query_mean={gwr_query_by_n[n][-1]*1e6:.2f}us "
                f"gp_fit={gp_fit_by_n[n][-1]:.4f}s train_pool={train_pool}"
            )

    rows: list[CellStats] = []
    for n in n_values:
        rows.append(
            CellStats(
                n=n,
                method="GWR",
                measurement_type="fit_time_s",
                mean=float(np.mean(gwr_fit_by_n[n])),
                std=float(np.std(gwr_fit_by_n[n])),
                n_repeats=args.repeats,
            )
        )
        rows.append(
            CellStats(
                n=n,
                method="GWR",
                measurement_type="query_time_s",
                mean=float(np.mean(gwr_query_by_n[n])),
                std=float(np.std(gwr_query_by_n[n])),
                n_repeats=args.repeats,
            )
        )
        rows.append(
            CellStats(
                n=n,
                method="GaussianProcess",
                measurement_type="fit_time_s",
                mean=float(np.mean(gp_fit_by_n[n])),
                std=float(np.std(gp_fit_by_n[n])),
                n_repeats=args.repeats,
            )
        )

    out_df = pd.DataFrame([vars(r) for r in rows]).sort_values(["method", "measurement_type", "n"])
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    os.makedirs(os.path.dirname(args.output_figure) or ".", exist_ok=True)
    _plot_results(out_df, args.output_figure)

    locked_df = pd.read_csv(args.locked_timing_csv)
    locked_row = locked_df.loc[locked_df["mode"] == "single_query"].iloc[0]
    locked_mean_us = float(locked_row["mean_us"])
    locked_min_us = float(locked_row["min_us"])
    locked_max_us = float(locked_row["max_us"])
    locked_p50_us = float(locked_row["p50_us"])
    locked_p95_us = float(locked_row["p95_us"])
    locked_p99_us = float(locked_row["p99_us"])
    max_n = max(n_values)
    row = out_df[
        (out_df["method"] == "GWR")
        & (out_df["measurement_type"] == "query_time_s")
        & (out_df["n"] == max_n)
    ]
    ours_mean_us = float(row["mean"].iloc[0]) * 1e6
    delta_us = ours_mean_us - locked_mean_us
    rel_pct = (delta_us / locked_mean_us) * 100.0 if locked_mean_us != 0 else float("nan")
    within_spread = locked_min_us <= ours_mean_us <= locked_max_us
    within_p50_p99 = locked_p50_us <= ours_mean_us <= locked_p99_us
    locked_compare_note = (
        f"Cross-check vs locked v2 reference `{args.locked_timing_csv}:mean_us`: "
        f"ours_at_n={max_n}={ours_mean_us:.6f} us, v2_mean={locked_mean_us:.6f} us, "
        f"delta={delta_us:.6f} us ({rel_pct:.2f}%)."
    )
    self_consistency_note = (
        "Same-environment self-consistency check against v2 canonical distribution: "
        f"v2_range=[{locked_min_us:.6f}, {locked_max_us:.6f}] us, "
        f"v2_p50/p95/p99=[{locked_p50_us:.6f}, {locked_p95_us:.6f}, {locked_p99_us:.6f}] us, "
        f"ours_at_n={max_n} within_range={within_spread}, within_p50_to_p99={within_p50_p99}."
    )

    gp_runtime_note = (
        "Runtime convention: n=5 repeats used (instead of n=10) because GP fit includes "
        "correlation-matrix construction/inversion and grows sharply with n."
    )
    methodology_reconciliation_note = (
        "Methodology reconciliation vs locked timing: this sweep now uses 10,000 single-point queries "
        "per (n, seed) cell with explicit warmup and with-replacement sampling from same run's split test pool, "
        "matching v2 metadata (`sampling=with_replacement_from_test_split_same_run`)."
    )
    os.makedirs(os.path.dirname(args.output_txt) or ".", exist_ok=True)
    _write_companion(
        path=args.output_txt,
        results_df=out_df,
        split_protocol=split_protocol,
        repeats=args.repeats,
        n_queries=args.n_queries,
        gp_runtime_note=gp_runtime_note,
        max_train_full_n=int(max_train_full_n or 0),
        locked_compare_note=locked_compare_note,
        methodology_reconciliation_note=methodology_reconciliation_note,
        self_consistency_note=self_consistency_note,
    )

    report_lines = [
        "Task 1 summary report",
        "Method reconciliation:",
        "- Current sweep uses 10000 warmed single queries per (n, seed) cell (with replacement).",
        "- Locked v2 file uses a single canonical warmed 10,000-query distribution for quantiles/center.",
        "- Independent multi-process runs are retained as noise characterization, not quantile source.",
        "Measured quantities:",
        "- GWR fit time: results/complexity_scaling.csv, columns [n, mean, std], filter method=GWR, measurement_type=fit_time_s",
        "- GWR query time: results/complexity_scaling.csv, columns [n, mean, std], filter method=GWR, measurement_type=query_time_s",
        "- GP fit time (GEK stand-in): results/complexity_scaling.csv, columns [n, mean, std], filter method=GaussianProcess, measurement_type=fit_time_s",
        f"- Locked timing cross-check source: {args.locked_timing_csv}, mode=single_query",
        "Notes:",
        f"- Split protocol used: {split_protocol}",
        f"- n values used: {n_values}",
        f"- Repeat convention: n={args.repeats}",
        f"- Max train+val split pool size: {int(max_train_full_n or 0)}; full dataset size used for larger n: {max_dataset_n}",
        f"- {locked_compare_note}",
        f"- {self_consistency_note}",
        "- GEK substitution note is explicit in figures/complexity_scaling.txt.",
    ]
    os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"Saved: {args.output_csv}")
    print(f"Saved: {args.output_figure}")
    print(f"Saved: {args.output_txt}")
    print(f"Saved: {args.output_report}")


if __name__ == "__main__":
    main()
