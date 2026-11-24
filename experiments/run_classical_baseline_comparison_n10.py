#!/usr/bin/env python
"""Classical local-regression baseline comparison at locked config (degree=5, k=1).

Extends the n=5 ``run_full_baseline_comparison`` classical baselines to n=10 seeds.
Seeds 0-4 reuse prior fitted results from ``results/full_baseline_raw.csv`` when
available; seeds 5-9 are fit and cached under ``results/model_cache/``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from baselines.gaussian_process import train_gaussian_process
from baselines.kernel_regression import train_kernel_regression
from baselines.knn import train_knn
from baselines.loess import train_loess
from gwr.core import GradientWeightedRegressor
from gwr.models import fit_polynomial_regression, model_to_sympy_equation
from gwr.utils import compute_all_metrics, load_dataset, train_val_test_split_df


RANDOM_STATES = list(range(10))
PRIOR_SEEDS = list(range(5))
LOCKED_DEGREE = 5
LOCKED_K = 1
CACHE_VERSION = "v1"

CLASSICAL_MODELS = {
    "GaussianProcess": train_gaussian_process,
    "LOESS": train_loess,
    "KNN": train_knn,
    "KernelRegression": train_kernel_regression,
}
SymGWR_MODEL_NAME = "Polynomial GWR"

BASELINE_PARAM_GRID: dict[str, list[dict[str, Any]]] = {
    "GaussianProcess": [{"alpha": 1e-6}, {"alpha": 1e-4}],
    "LOESS": [{"frac": 0.2}, {"frac": 0.3}, {"frac": 0.4}],
    "KNN": [{"n_neighbors": 3}, {"n_neighbors": 5}, {"n_neighbors": 10}],
    "KernelRegression": [
        {"bandwidth": 0.5},
        {"bandwidth": 1.0},
        {"bandwidth": 2.0},
    ],
}


def _extract_prediction(result: Any) -> np.ndarray | None:
    if isinstance(result, tuple):
        if len(result) >= 2:
            return result[1]
        return None
    return result


def _select_hparams(
    train_fn: Any,
    param_grid: list[dict[str, Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, Any]:
    best = param_grid[0]
    best_mse = float("inf")
    for params in param_grid:
        y_val_pred = _extract_prediction(train_fn(X_train, y_train, X_val, **params))
        val_mse = float(np.mean((y_val - y_val_pred) ** 2))
        if val_mse < best_mse:
            best_mse = val_mse
            best = params
    return best


def _subsample_training_data(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_train_samples: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    expensive = {"GaussianProcess", "KernelRegression", "LOESS"}
    if model_name not in expensive or len(X_train) <= max_train_samples:
        return X_train, y_train
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X_train), size=max_train_samples, replace=False)
    return X_train[idx], y_train[idx]


def _baseline_cache_key(
    model_name: str,
    seed: int,
    data_path: str,
    target: str,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    scale: bool,
    max_train_samples: int,
) -> str:
    payload = (
        f"{CACHE_VERSION}|model={model_name}|seed={seed}|"
        f"data={os.path.abspath(data_path)}|target={target}|"
        f"split={train_frac:.3f}/{val_frac:.3f}/{test_frac:.3f}|"
        f"scale={scale}|max_train={max_train_samples}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    safe_name = model_name.lower().replace(" ", "_")
    return f"baseline_{safe_name}_seed{seed}_{digest}.pkl"


def _gwr_cache_key(
    degree: int,
    k: int,
    seed: int,
    data_path: str,
    target: str,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    scale: bool,
) -> str:
    payload = (
        f"{CACHE_VERSION}|deg={degree}|k={k}|seed={seed}|"
        f"data={os.path.abspath(data_path)}|target={target}|"
        f"split={train_frac:.3f}/{val_frac:.3f}/{test_frac:.3f}|scale={scale}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"gwr_deg{degree}_k{k}_seed{seed}_{digest}.pkl"


def _train_and_predict_with_timing(
    train_fn: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    params: dict[str, Any],
) -> tuple[np.ndarray, float, float, Any, Any | None]:
    t0 = time.perf_counter()
    result = train_fn(X_train, y_train, None, **params)
    train_time = time.perf_counter() - t0
    if isinstance(result, tuple):
        model = result[0]
        scaler = result[2] if len(result) >= 3 else None
    else:
        model = result
        scaler = None

    X_eval = scaler.transform(X_test) if scaler is not None else X_test
    t1 = time.perf_counter()
    y_pred = model.predict(X_eval)
    infer_time = time.perf_counter() - t1
    return y_pred, float(train_time), float(infer_time), model, scaler


def _fit_locked_gwr(
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    degree: int,
    neighbors: int,
) -> GradientWeightedRegressor:
    base_model, _ = fit_polynomial_regression(
        X_train_full, y_train_full, X_test=None, degree=degree
    )
    sympy_eq, x_symbols = model_to_sympy_equation(base_model)
    gwr = GradientWeightedRegressor(
        sympy_equation=sympy_eq,
        x_symbols=x_symbols,
        n_neighbors=neighbors,
        relative=False,
    )
    gwr.fit(X_train_full, y_train_full)
    return gwr


def _load_or_fit_gwr(
    cache_dir: str,
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    degree: int,
    k: int,
    seed: int,
    data_path: str,
    target: str,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    scale: bool,
) -> tuple[GradientWeightedRegressor, bool, float, float]:
    os.makedirs(cache_dir, exist_ok=True)
    key = _gwr_cache_key(
        degree, k, seed, data_path, target, train_frac, val_frac, test_frac, scale
    )
    path = os.path.join(cache_dir, key)
    if os.path.isfile(path):
        with open(path, "rb") as fh:
            gwr = pickle.load(fh)
        return gwr, True, 0.0, 0.0

    t0 = time.perf_counter()
    gwr = _fit_locked_gwr(X_train_full, y_train_full, degree=degree, neighbors=k)
    train_time = time.perf_counter() - t0
    with open(path, "wb") as fh:
        pickle.dump(gwr, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return gwr, False, float(train_time), 0.0


def _load_prior_rows(prior_raw_path: str) -> pd.DataFrame:
    if not os.path.isfile(prior_raw_path):
        raise FileNotFoundError(f"Missing prior n=5 raw results: {prior_raw_path}")
    prior = pd.read_csv(prior_raw_path)
    wanted = set(CLASSICAL_MODELS) | {SymGWR_MODEL_NAME}
    prior = prior[prior["model"].isin(wanted)]
    prior = prior[prior["random_state"].isin(PRIOR_SEEDS)]
    if prior.empty:
        raise RuntimeError(f"No reusable rows found in {prior_raw_path}")
    return prior


def _population_summary(values: np.ndarray) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classical baseline comparison at locked degree=5, k=1 (n=10 seeds)"
    )
    parser.add_argument("--data", type=str, default="data/raw/718_all.csv")
    parser.add_argument("--target", type=str, default="Gibbs_energy")
    parser.add_argument("--degree", type=int, default=LOCKED_DEGREE)
    parser.add_argument("--neighbors", type=int, default=LOCKED_K)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--max-train-samples", type=int, default=2500)
    parser.add_argument(
        "--prior-raw",
        type=str,
        default="results/full_baseline_raw.csv",
        help="n=5 raw results to reuse for seeds 0-4",
    )
    parser.add_argument("--cache-dir", type=str, default="results/model_cache")
    parser.add_argument(
        "--output",
        type=str,
        default="results/classical_baseline_comparison_n10.csv",
    )
    parser.add_argument(
        "--gp-seed-time-limit-s",
        type=float,
        default=900.0,
        help="Abort if a single GP seed fit exceeds this many seconds (default 15 min)",
    )
    parser.add_argument("--no-scale", action="store_true")
    args = parser.parse_args()

    wall_t0 = time.perf_counter()
    prior_df = _load_prior_rows(args.prior_raw)
    _, _, df = load_dataset(args.data, args.target, scale=not args.no_scale)
    per_seed_rows: list[dict[str, Any]] = []

    for seed in RANDOM_STATES:
        if seed in PRIOR_SEEDS:
            prior_seed = prior_df[prior_df["random_state"] == seed]
            for _, row in prior_seed.iterrows():
                per_seed_rows.append(
                    {
                        "record_type": "per_seed",
                        "model": row["model"],
                        "seed": int(seed),
                        "degree": args.degree,
                        "k": args.neighbors,
                        "split": "test",
                        "mae": float(row["mae"]),
                        "rmse": float(row["rmse"]),
                        "mape": float(row["mape"]),
                        "training_time_s": float(row["training_time_s"]),
                        "inference_time_s": float(row["inference_time_s"]),
                        "result_source": "prior_n5_raw",
                        "model_from_cache": False,
                    }
                )
            print(
                f"seed={seed} reused {len(prior_seed)} rows from {args.prior_raw}",
                flush=True,
            )
            continue

        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split_df(
            df,
            target_col=args.target,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            random_state=seed,
        )
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.concatenate([y_train, y_val])

        gwr, gwr_cached, gwr_train_t, _ = _load_or_fit_gwr(
            cache_dir=args.cache_dir,
            X_train_full=X_train_full,
            y_train_full=y_train_full,
            degree=args.degree,
            k=args.neighbors,
            seed=seed,
            data_path=args.data,
            target=args.target,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            scale=not args.no_scale,
        )
        t1 = time.perf_counter()
        y_pred = gwr.predict(X_test)
        gwr_infer_t = time.perf_counter() - t1
        gwr_metrics = compute_all_metrics(y_test, y_pred)
        per_seed_rows.append(
            {
                "record_type": "per_seed",
                "model": SymGWR_MODEL_NAME,
                "seed": seed,
                "degree": args.degree,
                "k": args.neighbors,
                "split": "test",
                "mae": float(gwr_metrics["mae_mean"]),
                "rmse": float(gwr_metrics["rmse"]),
                "mape": float(gwr_metrics["mape_mean"]),
                "training_time_s": gwr_train_t,
                "inference_time_s": float(gwr_infer_t),
                "result_source": "cache" if gwr_cached else "fit",
                "model_from_cache": gwr_cached,
            }
        )
        print(
            f"seed={seed} model={SymGWR_MODEL_NAME} cache={'hit' if gwr_cached else 'miss'} "
            f"mape={gwr_metrics['mape_mean']:.6g}",
            flush=True,
        )

        for model_name, train_fn in CLASSICAL_MODELS.items():
            cache_key = _baseline_cache_key(
                model_name,
                seed,
                args.data,
                args.target,
                args.train_frac,
                args.val_frac,
                args.test_frac,
                not args.no_scale,
                args.max_train_samples,
            )
            cache_path = os.path.join(args.cache_dir, cache_key)

            if os.path.isfile(cache_path):
                with open(cache_path, "rb") as fh:
                    cached = pickle.load(fh)
                per_seed_rows.append(
                    {
                        "record_type": "per_seed",
                        "model": model_name,
                        "seed": seed,
                        "degree": args.degree,
                        "k": args.neighbors,
                        "split": "test",
                        "mae": float(cached["metrics"]["mae"]),
                        "rmse": float(cached["metrics"]["rmse"]),
                        "mape": float(cached["metrics"]["mape"]),
                        "training_time_s": float(cached["training_time_s"]),
                        "inference_time_s": float(cached["inference_time_s"]),
                        "result_source": "cache",
                        "model_from_cache": True,
                    }
                )
                print(
                    f"seed={seed} model={model_name} cache=hit mape={cached['metrics']['mape']:.6g}",
                    flush=True,
                )
                continue

            X_train_model, y_train_model = _subsample_training_data(
                model_name,
                X_train,
                y_train,
                args.max_train_samples,
                seed,
            )
            best_params = _select_hparams(
                train_fn,
                BASELINE_PARAM_GRID[model_name],
                X_train_model,
                y_train_model,
                X_val,
                y_val,
            )
            X_fit = np.vstack([X_train_model, X_val])
            y_fit = np.concatenate([y_train_model, y_val])

            if model_name == "GaussianProcess":
                gp_t0 = time.perf_counter()
                y_pred, train_t, infer_t, model, scaler = _train_and_predict_with_timing(
                    train_fn, X_fit, y_fit, X_test, best_params
                )
                gp_elapsed = time.perf_counter() - gp_t0
                if gp_elapsed > args.gp_seed_time_limit_s:
                    partial_df = pd.DataFrame(per_seed_rows)
                    partial_path = args.output.replace(".csv", "_partial.csv")
                    partial_df.to_csv(partial_path, index=False)
                    raise RuntimeError(
                        f"GP seed={seed} exceeded time limit "
                        f"({gp_elapsed:.1f}s > {args.gp_seed_time_limit_s:.1f}s). "
                        f"Partial progress saved to {partial_path}. "
                        "GP is the bottleneck; inspect before continuing."
                    )
            else:
                y_pred, train_t, infer_t, model, scaler = _train_and_predict_with_timing(
                    train_fn, X_fit, y_fit, X_test, best_params
                )

            metrics = compute_all_metrics(y_test, y_pred)
            cache_payload = {
                "model_name": model_name,
                "seed": seed,
                "best_params": best_params,
                "fitted_model": model,
                "scaler": scaler,
                "metrics": {
                    "mae": float(metrics["mae_mean"]),
                    "rmse": float(metrics["rmse"]),
                    "mape": float(metrics["mape_mean"]),
                },
                "training_time_s": train_t,
                "inference_time_s": infer_t,
            }
            with open(cache_path, "wb") as fh:
                pickle.dump(cache_payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

            per_seed_rows.append(
                {
                    "record_type": "per_seed",
                    "model": model_name,
                    "seed": seed,
                    "degree": args.degree,
                    "k": args.neighbors,
                    "split": "test",
                    "mae": float(metrics["mae_mean"]),
                    "rmse": float(metrics["rmse"]),
                    "mape": float(metrics["mape_mean"]),
                    "training_time_s": train_t,
                    "inference_time_s": infer_t,
                    "result_source": "fit",
                    "model_from_cache": False,
                }
            )
            print(
                f"seed={seed} model={model_name} cache=miss mape={metrics['mape_mean']:.6g} "
                f"train_s={train_t:.1f}",
                flush=True,
            )

    raw_df = pd.DataFrame(per_seed_rows)
    models = [SymGWR_MODEL_NAME, *CLASSICAL_MODELS.keys()]

    summary_rows: list[dict[str, Any]] = []
    for model_name in models:
        sub = raw_df[raw_df["model"] == model_name]
        summary: dict[str, Any] = {
            "record_type": "summary",
            "model": model_name,
            "seed": np.nan,
            "degree": args.degree,
            "k": args.neighbors,
            "split": "test",
            "n_repeats": float(len(sub)),
        }
        for metric in ("mae", "rmse", "mape"):
            mean, std = _population_summary(sub[metric].to_numpy(dtype=np.float64))
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
        summary_rows.append(summary)

    gwr_summary = next(r for r in summary_rows if r["model"] == SymGWR_MODEL_NAME)

    improvement_rows: list[dict[str, Any]] = []
    significance_rows: list[dict[str, Any]] = []
    for model_name in CLASSICAL_MODELS:
        gwr_by_seed = raw_df.loc[raw_df["model"] == SymGWR_MODEL_NAME].sort_values("seed")
        base_by_seed = raw_df.loc[raw_df["model"] == model_name].sort_values("seed")
        gwr_paired = gwr_by_seed["mape"].to_numpy(dtype=np.float64)
        base_paired = base_by_seed["mape"].to_numpy(dtype=np.float64)

        base_summary = next(r for r in summary_rows if r["model"] == model_name)
        gwr_mean = float(gwr_summary["mape_mean"])
        base_mean = float(base_summary["mape_mean"])
        abs_improvement = base_mean - gwr_mean
        pct_improvement = (abs_improvement / abs(base_mean)) * 100.0 if base_mean else np.nan

        improvement_rows.append(
            {
                "record_type": "improvement",
                "model": model_name,
                "metric": "mape",
                "gwr_mean": gwr_mean,
                "baseline_mean": base_mean,
                "absolute_improvement": abs_improvement,
                "percent_improvement": pct_improvement,
                "degree": args.degree,
                "k": args.neighbors,
                "n_repeats": len(RANDOM_STATES),
            }
        )

        paired_t = stats.ttest_rel(base_paired, gwr_paired)
        wilcoxon = stats.wilcoxon(base_paired, gwr_paired, zero_method="wilcox")
        for test_name, result in (
            ("paired_t_test", paired_t),
            ("wilcoxon_signed_rank", wilcoxon),
        ):
            significance_rows.append(
                {
                    "record_type": "significance_test",
                    "model": model_name,
                    "metric": "mape",
                    "test_name": test_name,
                    "statistic": float(result.statistic),
                    "p_value_two_sided": float(result.pvalue),
                    "n_pairs": len(RANDOM_STATES),
                    "degree": args.degree,
                    "k": args.neighbors,
                    "split": "test",
                    "split_protocol": f"{args.train_frac:.1f}/{args.val_frac:.1f}/{args.test_frac:.1f}",
                    "seed_convention": "0..9",
                }
            )

    meta_rows = [
        {
            "record_type": "run_meta",
            "model": "ALL",
            "wall_clock_s": time.perf_counter() - wall_t0,
            "prior_raw": args.prior_raw,
            "output": args.output,
            "degree": args.degree,
            "k": args.neighbors,
            "n_repeats": len(RANDOM_STATES),
        }
    ]

    out_df = pd.concat(
        [
            raw_df,
            pd.DataFrame(summary_rows),
            pd.DataFrame(improvement_rows),
            pd.DataFrame(significance_rows),
            pd.DataFrame(meta_rows),
        ],
        ignore_index=True,
        sort=False,
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Saved n=10 classical baseline comparison: {args.output}", flush=True)
    print(f"Wall clock: {meta_rows[0]['wall_clock_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
