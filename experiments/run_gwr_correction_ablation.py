#!/usr/bin/env python
"""Ablation: full SymGWR vs correction-zeroed (1-NN) at locked config (degree=5, k=1).

Compares Taylor gradient correction against returning the nearest reference
target only, using the same fitted model, reference set, and test split as
the headline n=10 CALPHAD repeats.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gwr.core import GradientWeightedRegressor
from gwr.models import fit_polynomial_regression, model_to_sympy_equation
from gwr.utils import compute_all_metrics, load_dataset, train_val_test_split_df


RANDOM_STATES = list(range(10))
LOCKED_DEGREE = 5
LOCKED_K = 1
CACHE_VERSION = "v1"


def _cache_key(
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


def _cache_path(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, key)


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
) -> tuple[GradientWeightedRegressor, bool]:
    os.makedirs(cache_dir, exist_ok=True)
    key = _cache_key(
        degree, k, seed, data_path, target, train_frac, val_frac, test_frac, scale
    )
    path = _cache_path(cache_dir, key)
    if os.path.isfile(path):
        with open(path, "rb") as fh:
            gwr = pickle.load(fh)
        return gwr, True

    gwr = _fit_locked_gwr(X_train_full, y_train_full, degree=degree, neighbors=k)
    with open(path, "wb") as fh:
        pickle.dump(gwr, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return gwr, False


def _predict_full_gwr(gwr: GradientWeightedRegressor, X: np.ndarray) -> np.ndarray:
    return np.asarray(gwr.predict(X), dtype=np.float64)


def _predict_nn_only(gwr: GradientWeightedRegressor, X: np.ndarray) -> np.ndarray:
    """Nearest-neighbor target only: same lookup/weights, zero Taylor correction."""
    X = np.asarray(X, dtype=np.float64)
    _, _, _, targets, weights = gwr._predict_components(X)
    if gwr.weighted:
        return np.sum(targets * weights, axis=1)
    return targets.mean(axis=1)


def _population_summary(values: np.ndarray) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SymGWR correction-term ablation at locked degree=5, k=1"
    )
    parser.add_argument("--data", type=str, default="data/raw/718_all.csv")
    parser.add_argument("--target", type=str, default="Gibbs_energy")
    parser.add_argument("--degree", type=int, default=LOCKED_DEGREE)
    parser.add_argument("--neighbors", type=int, default=LOCKED_K)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--cache-dir", type=str, default="results/model_cache")
    parser.add_argument(
        "--output",
        type=str,
        default="results/gwr_correction_ablation_n10.csv",
    )
    parser.add_argument("--no-scale", action="store_true")
    args = parser.parse_args()

    _, _, df = load_dataset(args.data, args.target, scale=not args.no_scale)
    per_seed_rows: list[dict[str, Any]] = []

    for seed in RANDOM_STATES:
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

        gwr, from_cache = _load_or_fit_gwr(
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

        for variant, y_pred in (
            ("full_gwr", _predict_full_gwr(gwr, X_test)),
            ("nn_only", _predict_nn_only(gwr, X_test)),
        ):
            metrics = compute_all_metrics(y_test, y_pred)
            per_seed_rows.append(
                {
                    "record_type": "per_seed",
                    "variant": variant,
                    "seed": seed,
                    "degree": args.degree,
                    "k": args.neighbors,
                    "split": "test",
                    "mae": float(metrics["mae_mean"]),
                    "rmse": float(metrics["rmse"]),
                    "mape": float(metrics["mape_mean"]),
                    "model_from_cache": from_cache,
                }
            )
        print(
            f"seed={seed} cache={'hit' if from_cache else 'miss'} "
            f"full_mape={per_seed_rows[-2]['mape']:.6g} "
            f"nn_mape={per_seed_rows[-1]['mape']:.6g}"
        )

    raw_df = pd.DataFrame(per_seed_rows)
    summary_rows: list[dict[str, Any]] = []
    for variant in ("full_gwr", "nn_only"):
        sub = raw_df[raw_df["variant"] == variant]
        summary: dict[str, Any] = {
            "record_type": "summary",
            "variant": variant,
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

    full_mapes = raw_df.loc[raw_df["variant"] == "full_gwr", "mape"].to_numpy(dtype=np.float64)
    nn_mapes = raw_df.loc[raw_df["variant"] == "nn_only", "mape"].to_numpy(dtype=np.float64)
    paired_t = stats.ttest_rel(full_mapes, nn_mapes)
    wilcoxon = stats.wilcoxon(full_mapes, nn_mapes, zero_method="wilcox")

    significance_rows = [
        {
            "record_type": "significance_test",
            "variant": "full_gwr_vs_nn_only",
            "metric": "mape",
            "test_name": "paired_t_test",
            "statistic": float(paired_t.statistic),
            "p_value_two_sided": float(paired_t.pvalue),
            "n_pairs": len(RANDOM_STATES),
            "degree": args.degree,
            "k": args.neighbors,
            "split": "test",
            "split_protocol": f"{args.train_frac:.1f}/{args.val_frac:.1f}/{args.test_frac:.1f}",
            "seed_convention": "0..9",
        },
        {
            "record_type": "significance_test",
            "variant": "full_gwr_vs_nn_only",
            "metric": "mape",
            "test_name": "wilcoxon_signed_rank",
            "statistic": float(wilcoxon.statistic),
            "p_value_two_sided": float(wilcoxon.pvalue),
            "n_pairs": len(RANDOM_STATES),
            "degree": args.degree,
            "k": args.neighbors,
            "split": "test",
            "split_protocol": f"{args.train_frac:.1f}/{args.val_frac:.1f}/{args.test_frac:.1f}",
            "seed_convention": "0..9",
        },
    ]

    out_df = pd.concat(
        [raw_df, pd.DataFrame(summary_rows), pd.DataFrame(significance_rows)],
        ignore_index=True,
        sort=False,
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Saved ablation results: {args.output}")


if __name__ == "__main__":
    main()
