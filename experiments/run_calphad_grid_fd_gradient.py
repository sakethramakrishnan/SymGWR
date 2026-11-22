#!/usr/bin/env python
"""Grid finite-difference validation of SymGWR's analytic dG/dT at locked config (degree=5, k=1).

Uses the real CALPHAD grid data (data/raw/718_all.csv), NOT a synthetic function.
The grid is regular in temperature per fixed composition (10 K steps), so a
central-difference estimate of dG/dT is available in closed form at every
interior grid point of every composition series:

    dG/dT |_i ~= (G(T_i + dT) - G(T_i - dT)) / ((T_i + dT) - (T_i - dT))

That finite-difference field is compared against SymGWR's own analytic
gradient (the dG/dT component of the fitted polynomial's gradient at the
nearest reference point, per the GWR k=1 rule) evaluated at the same points,
using the locked model config (degree=5, k=1) trained on the project's
standard single-seed split convention (seed=0, train/val/test=0.6/0.2/0.2,
scaled features -- matching run_gwr_correction_ablation.py /
run_synthetic_loop_stability.py).

NOTE ON CONVENTION: prior derivative work in this repo (derivative_evaluation.py)
used a k-NN local-regression finite-difference *proxy* because no true grid
derivative was available. This script is a new analysis type: it uses actual
grid neighbours instead of a proxy, and it evaluates over the *entire*
FD-eligible grid (not the model's held-out test split), since the finite
difference itself only depends on the raw CALPHAD table, not on the GWR
train/test partition. This is flagged explicitly in the summary output.

Caching: the FD-eligible point set (and its finite-difference values) is
content-hashed and cached independently of the GWR model; the locked GWR
model is fit/cached via the same key scheme as run_gwr_correction_ablation.py,
so a matching cached model is reused rather than retrained.
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gwr.core import GradientWeightedRegressor
from gwr.models import fit_polynomial_regression, model_to_sympy_equation
from gwr.utils import load_dataset, train_val_test_split_df

LOCKED_DEGREE = 5
LOCKED_K = 1
LOCKED_SEED = 0
CACHE_VERSION = "v1"
GRID_CACHE_VERSION = "v1"


# ----------------------------------------------------------------------
# Locked-GWR cache (identical scheme to run_gwr_correction_ablation.py so a
# matching cached model is reused).
# ----------------------------------------------------------------------

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
    key = _gwr_cache_key(
        degree, k, seed, data_path, target, train_frac, val_frac, test_frac, scale
    )
    path = os.path.join(cache_dir, key)
    if os.path.isfile(path):
        with open(path, "rb") as fh:
            gwr = pickle.load(fh)
        return gwr, True

    gwr = _fit_locked_gwr(X_train_full, y_train_full, degree=degree, neighbors=k)
    with open(path, "wb") as fh:
        pickle.dump(gwr, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return gwr, False


# ----------------------------------------------------------------------
# Grid finite-difference point set (cached independently, keyed on raw data
# content + dedup/eligibility rule -- not on the GWR model config).
# ----------------------------------------------------------------------

def _file_content_hash(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


def _grid_cache_key(data_path: str, target: str, temp_col: str, comp_cols: list[str]) -> str:
    payload = (
        f"{GRID_CACHE_VERSION}|data={os.path.abspath(data_path)}|"
        f"content={_file_content_hash(data_path)}|target={target}|temp_col={temp_col}|"
        f"comp_cols={','.join(comp_cols)}|dedup=comp_temp_keep_first"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"calphad_grid_fd_points_{digest}.pkl"


def _build_fd_eligible_points(
    df_raw: pd.DataFrame, target: str, temp_col: str, comp_cols: list[str]
) -> pd.DataFrame:
    """Central-difference dG/dT at every interior, regularly-spaced grid point.

    A point is eligible when its same-composition neighbours at the previous
    and next grid rows are exactly one (equal) temperature step away on both
    sides -- i.e. a genuine adjacent-step pair with no gap or duplicate.
    """
    key_cols = comp_cols + [temp_col]
    df_dedup = df_raw.drop_duplicates(subset=key_cols, keep="first")

    rows: list[dict[str, Any]] = []
    for comp_vals, group in df_dedup.groupby(comp_cols, sort=False):
        group = group.sort_values(temp_col)
        T = group[temp_col].to_numpy(dtype=np.float64)
        G = group[target].to_numpy(dtype=np.float64)
        idx = group.index.to_numpy()
        n = len(T)
        if n < 3:
            continue
        step_prev = T[1:-1] - T[:-2]
        step_next = T[2:] - T[1:-1]
        eligible = np.isclose(step_prev, step_next) & (step_prev > 0)
        comp_tuple = comp_vals if isinstance(comp_vals, tuple) else (comp_vals,)
        for j in np.nonzero(eligible)[0]:
            i = j + 1  # interior index in this group
            row = {c: v for c, v in zip(comp_cols, comp_tuple)}
            row.update(
                {
                    "row_index": int(idx[i]),
                    temp_col: T[i],
                    "T_prev": T[i - 1],
                    "T_next": T[i + 1],
                    "G_prev": G[i - 1],
                    "G_mid": G[i],
                    "G_next": G[i + 1],
                    "fd_dGdT": (G[i + 1] - G[i - 1]) / (T[i + 1] - T[i - 1]),
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


def _load_or_build_fd_points(
    cache_dir: str, data_path: str, target: str, temp_col: str, comp_cols: list[str], df_raw: pd.DataFrame
) -> tuple[pd.DataFrame, bool]:
    os.makedirs(cache_dir, exist_ok=True)
    key = _grid_cache_key(data_path, target, temp_col, comp_cols)
    path = os.path.join(cache_dir, key)
    if os.path.isfile(path):
        with open(path, "rb") as fh:
            return pickle.load(fh), True

    points = _build_fd_eligible_points(df_raw, target=target, temp_col=temp_col, comp_cols=comp_cols)
    with open(path, "wb") as fh:
        pickle.dump(points, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return points, False


# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid FD validation of SymGWR analytic dG/dT at locked degree=5, k=1"
    )
    parser.add_argument("--data", type=str, default="data/raw/718_all.csv")
    parser.add_argument("--target", type=str, default="Gibbs_energy")
    parser.add_argument("--temp-col", type=str, default="Temperature_[K]")
    parser.add_argument("--degree", type=int, default=LOCKED_DEGREE)
    parser.add_argument("--neighbors", type=int, default=LOCKED_K)
    parser.add_argument("--seed", type=int, default=LOCKED_SEED)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--cache-dir", type=str, default="results/model_cache")
    parser.add_argument("--output", type=str, default="results/calphad_grid_fd_gradient_points.csv")
    parser.add_argument(
        "--summary-output", type=str, default="results/calphad_grid_fd_gradient_summary.csv"
    )
    parser.add_argument("--no-scale", action="store_true")
    args = parser.parse_args()
    scale = not args.no_scale

    # ---- raw (unscaled) grid, for finding FD-eligible neighbours -------
    try:
        df_raw = pd.read_csv(args.data, sep=None, engine="python")
    except Exception:
        df_raw = pd.read_csv(args.data, delim_whitespace=True)

    comp_cols = [
        c
        for c in df_raw.select_dtypes(include=[np.number]).columns
        if c not in (args.target, args.temp_col)
    ]

    fd_points, fd_from_cache = _load_or_build_fd_points(
        args.cache_dir, args.data, args.target, args.temp_col, comp_cols, df_raw
    )
    n_sample = len(fd_points)
    if n_sample == 0:
        raise RuntimeError("No finite-difference-eligible grid points found.")

    # ---- locked GWR model, standard project split convention -----------
    feature_scaler, target_scaler, df_scaled = load_dataset(args.data, args.target, scale=scale)
    numeric_feature_cols = [c for c in df_raw.select_dtypes(include=[np.number]).columns if c != args.target]
    t_idx = numeric_feature_cols.index(args.temp_col)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split_df(
        df_scaled,
        target_col=args.target,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        random_state=args.seed,
    )
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    gwr, gwr_from_cache = _load_or_fit_gwr(
        cache_dir=args.cache_dir,
        X_train_full=X_train_full,
        y_train_full=y_train_full,
        degree=args.degree,
        k=args.neighbors,
        seed=args.seed,
        data_path=args.data,
        target=args.target,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        scale=scale,
    )

    # ---- query SymGWR's analytic gradient at the same FD-eligible points
    X_query_raw = fd_points[numeric_feature_cols].to_numpy(dtype=np.float64)
    if scale:
        X_query = feature_scaler.transform(X_query_raw)
        conv_factor = float(target_scaler.scale_[0]) / float(feature_scaler.scale_[t_idx])
    else:
        X_query = X_query_raw
        conv_factor = 1.0

    _, local_grad_scaled = gwr.predict_with_local_gradient(X_query)
    analytic_dGdT = local_grad_scaled[:, t_idx] * conv_factor

    fd_points = fd_points.copy()
    fd_points["analytic_dGdT"] = analytic_dGdT
    fd_points["abs_error"] = np.abs(fd_points["analytic_dGdT"] - fd_points["fd_dGdT"])

    mae = float(fd_points["abs_error"].mean())
    typical_magnitude = float(fd_points["fd_dGdT"].abs().mean())
    normalized_error = mae / typical_magnitude if typical_magnitude > 0 else float("nan")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fd_points.to_csv(args.output, index=False)

    summary = {
        "n_sample": n_sample,
        "mae": mae,
        "typical_abs_dGdT": typical_magnitude,
        "normalized_error": normalized_error,
        "degree": args.degree,
        "n_neighbors": args.neighbors,
        "seed": args.seed,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "scale": scale,
        "data": args.data,
        "fd_points_from_cache": fd_from_cache,
        "gwr_model_from_cache": gwr_from_cache,
        "split_convention_note": (
            "GWR trained on the project's standard locked single-seed split "
            f"(seed={args.seed}, train/val/test={args.train_frac}/{args.val_frac}/{args.test_frac}, "
            "scale=True), matching run_gwr_correction_ablation.py / "
            "run_synthetic_loop_stability.py. The finite-difference sample itself is drawn "
            "from the FULL raw grid (not restricted to the GWR test split), since central "
            "differences depend only on the CALPHAD table's regular T-spacing, not on the "
            "model's train/test partition. This is a NEW analysis type not covered by the "
            "existing derivative_evaluation.py (which used a k-NN local-regression proxy, "
            "not true grid finite differences)."
        ),
    }
    os.makedirs(os.path.dirname(args.summary_output) or ".", exist_ok=True)
    pd.DataFrame([summary]).to_csv(args.summary_output, index=False)

    print(f"FD-eligible sample size: {n_sample} (cache hit: {fd_from_cache})")
    print(f"GWR model cache hit: {gwr_from_cache}")
    print(f"MAE(analytic, finite-difference) dG/dT: {mae:.6g}")
    print(f"Typical |dG/dT| (mean |fd|): {typical_magnitude:.6g}")
    print(f"Normalized error (MAE / typical |dG/dT|): {normalized_error:.6g}")
    print(f"Saved per-point results to {args.output}")
    print(f"Saved summary to {args.summary_output}")


if __name__ == "__main__":
    main()
