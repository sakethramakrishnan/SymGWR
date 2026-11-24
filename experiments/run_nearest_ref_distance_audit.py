#!/usr/bin/env python
"""Audit nearest-reference distances for locked SymGWR (degree=5, k=1).

Loads cached fitted models when present; fits/caches on miss. Computes
test→reference 1-NN distances in the same scaled feature space used by SymGWR
neighbor lookup, plus one spatially-blocked Temperature-band split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gwr.core import GradientWeightedRegressor
from gwr.models import fit_polynomial_regression, model_to_sympy_equation
from gwr.splits import make_spatial_split as _make_spatial_split
from gwr.utils import compute_all_metrics, load_dataset, train_val_test_split_df


RANDOM_STATES = list(range(10))
LOCKED_DEGREE = 5
LOCKED_K = 1
CACHE_VERSION = "v1"
FEATURE_NAMES = [
    "Mass_fraction_Nb",
    "Mass_fraction_Al",
    "Mass_fraction_Ti",
    "Mass_fraction_Mo",
    "Mass_fraction_Fe",
    "Mass_fraction_Ni",
    "Mass_fraction_Cr",
    "Temperature_[K]",
]
TEMP_IDX = FEATURE_NAMES.index("Temperature_[K]")
COMP_IDXS = list(range(TEMP_IDX))  # 0..6

# Contiguous mid-band ~20% holdout (16 temps × 101 rows = 1616 / 8181 ≈ 0.1975)
SPATIAL_T_MIN = 1320
SPATIAL_T_MAX = 1470
SPATIAL_REMAIN_SEED = 0
SPATIAL_REMAIN_TRAIN_FRAC = 0.75  # of non-test → overall ~60/20/20


def _percentile_stats(arr: np.ndarray) -> dict[str, float]:
    a = np.asarray(arr, dtype=np.float64).ravel()
    return {
        "n": int(a.size),
        "min": float(np.min(a)),
        "p5": float(np.percentile(a, 5)),
        "p25": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "p75": float(np.percentile(a, 75)),
        "p95": float(np.percentile(a, 95)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
    }


def _gwr_cache_key_random(
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


def _gwr_cache_key_spatial(
    degree: int,
    k: int,
    data_path: str,
    target: str,
    scale: bool,
    t_min: int,
    t_max: int,
    remain_seed: int,
    remain_train_frac: float,
) -> str:
    payload = (
        f"{CACHE_VERSION}|deg={degree}|k={k}|split_type=spatial_temp_band|"
        f"t_min={t_min}|t_max={t_max}|remain_seed={remain_seed}|"
        f"remain_train_frac={remain_train_frac:.3f}|"
        f"data={os.path.abspath(data_path)}|target={target}|scale={scale}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return (
        f"gwr_deg{degree}_k{k}_spatial_temp_band_{t_min}_{t_max}_{digest}.pkl"
    )


def _dist_cache_key(
    split_type: str,
    seed: int | str,
    degree: int,
    k: int,
    data_path: str,
    target: str,
    scale: bool,
    extra: str = "",
) -> str:
    payload = (
        f"{CACHE_VERSION}|dist_audit|split_type={split_type}|seed={seed}|"
        f"deg={degree}|k={k}|data={os.path.abspath(data_path)}|"
        f"target={target}|scale={scale}|{extra}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"nearest_ref_dist_{split_type}_seed{seed}_{digest}.npz"


def _split_index_cache_key(
    split_type: str,
    data_path: str,
    target: str,
    scale: bool,
    extra: str,
) -> str:
    payload = (
        f"{CACHE_VERSION}|split_indices|split_type={split_type}|"
        f"data={os.path.abspath(data_path)}|target={target}|scale={scale}|{extra}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"split_indices_{split_type}_{digest}.npz"


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
    cache_path: str,
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    degree: int,
    k: int,
) -> tuple[GradientWeightedRegressor, bool]:
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    if os.path.isfile(cache_path):
        with open(cache_path, "rb") as fh:
            gwr = pickle.load(fh)
        return gwr, True
    gwr = _fit_locked_gwr(X_train_full, y_train_full, degree=degree, neighbors=k)
    with open(cache_path, "wb") as fh:
        pickle.dump(gwr, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return gwr, False


def _nn_distances(
    X_query: np.ndarray,
    X_ref: np.ndarray,
) -> dict[str, np.ndarray]:
    """1-NN distances in full space, T-only, and composition-only (same units)."""
    X_query = np.asarray(X_query, dtype=np.float64)
    X_ref = np.asarray(X_ref, dtype=np.float64)

    tree_full = cKDTree(X_ref)
    d_full, idx_full = tree_full.query(X_query, k=1)
    nearest = X_ref[idx_full]
    d_temp_component = np.abs(X_query[:, TEMP_IDX] - nearest[:, TEMP_IDX])
    d_comp_component = np.linalg.norm(
        X_query[:, COMP_IDXS] - nearest[:, COMP_IDXS], axis=1
    )

    tree_t = cKDTree(X_ref[:, [TEMP_IDX]])
    d_temp_alone, _ = tree_t.query(X_query[:, [TEMP_IDX]], k=1)

    tree_c = cKDTree(X_ref[:, COMP_IDXS])
    d_comp_alone, _ = tree_c.query(X_query[:, COMP_IDXS], k=1)

    return {
        "d_full": np.asarray(d_full, dtype=np.float64),
        "nn_idx": np.asarray(idx_full, dtype=np.int64),
        "d_temp_component_of_full_nn": np.asarray(d_temp_component, dtype=np.float64),
        "d_comp_component_of_full_nn": np.asarray(d_comp_component, dtype=np.float64),
        "d_temp_alone_1nn": np.asarray(d_temp_alone, dtype=np.float64),
        "d_comp_alone_1nn": np.asarray(d_comp_alone, dtype=np.float64),
    }


def _gwr_1nn_distances(gwr: GradientWeightedRegressor, X_query: np.ndarray) -> np.ndarray:
    """Distances from SymGWR inference path (_query_absolute_neighbors / cKDTree)."""
    distances, _, _, _ = gwr._query_absolute_neighbors(
        np.asarray(X_query, dtype=np.float64)
    )
    return distances[:, 0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nearest-reference distance audit for locked SymGWR"
    )
    parser.add_argument("--data", type=str, default="data/raw/718_all.csv")
    parser.add_argument("--target", type=str, default="Gibbs_energy")
    parser.add_argument("--degree", type=int, default=LOCKED_DEGREE)
    parser.add_argument("--neighbors", type=int, default=LOCKED_K)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--cache-dir", type=str, default="results/model_cache")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--no-scale", action="store_true")
    args = parser.parse_args()

    scale = not args.no_scale
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    feat_scaler, tgt_scaler, df = load_dataset(args.data, args.target, scale=scale)
    _, _, df_raw = load_dataset(args.data, args.target, scale=False)

    feature_space_note = (
        "STANDARD-SCALED features via gwr.utils.CustomStandardScaler "
        "(load_dataset scale=True default). SymGWR neighbor KD-tree is built on "
        "these scaled features (NOT raw composition O(1) / Temperature O(1000)). "
        "temp_class is excluded from the numeric feature matrix."
    )

    report: dict[str, Any] = {
        "config": {
            "degree": args.degree,
            "k": args.neighbors,
            "data": os.path.abspath(args.data),
            "target": args.target,
            "scale": scale,
            "split_protocol_random": (
                f"{args.train_frac:.1f}/{args.val_frac:.1f}/{args.test_frac:.1f}"
            ),
            "feature_names": FEATURE_NAMES,
            "temp_idx": TEMP_IDX,
            "feature_space": feature_space_note,
            "feature_scaler_mean": None
            if feat_scaler is None
            else feat_scaler.mean_.tolist(),
            "feature_scaler_scale": None
            if feat_scaler is None
            else feat_scaler.scale_.tolist(),
        },
        "cache_files_read": [],
        "cache_files_written": [],
        "random_split_seeds": {},
        "distance_stats_random": {},
        "spatial_blocked": {},
        "code_path_crosscheck": {},
    }

    # ------------------------------------------------------------------
    # Steps 0–2: random-split locked models + distances
    # ------------------------------------------------------------------
    pooled: dict[str, list[np.ndarray]] = {
        "d_full": [],
        "d_temp_component_of_full_nn": [],
        "d_comp_component_of_full_nn": [],
        "d_temp_alone_1nn": [],
        "d_comp_alone_1nn": [],
        "d_full_via_gwr_query": [],
    }
    per_seed_stats: list[dict[str, Any]] = []

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

        key = _gwr_cache_key_random(
            args.degree,
            args.neighbors,
            seed,
            args.data,
            args.target,
            args.train_frac,
            args.val_frac,
            args.test_frac,
            scale,
        )
        model_path = os.path.join(args.cache_dir, key)
        existed = os.path.isfile(model_path)
        gwr, from_cache = _load_or_fit_gwr(
            model_path, X_train_full, y_train_full, args.degree, args.neighbors
        )
        if from_cache:
            report["cache_files_read"].append(os.path.abspath(model_path))
        else:
            report["cache_files_written"].append(os.path.abspath(model_path))

        # Sanity: cached ref set must match reconstructed train∪val
        ref_match = bool(np.allclose(gwr._ref_points, X_train_full))
        if not ref_match:
            raise RuntimeError(
                f"seed={seed}: cached _ref_points != reconstructed train∪val"
            )

        dist_key = _dist_cache_key(
            "random",
            seed,
            args.degree,
            args.neighbors,
            args.data,
            args.target,
            scale,
            extra=f"split={args.train_frac:.3f}/{args.val_frac:.3f}/{args.test_frac:.3f}",
        )
        dist_path = os.path.join(args.results_dir, dist_key)
        if os.path.isfile(dist_path):
            report["cache_files_read"].append(os.path.abspath(dist_path))
            cached = np.load(dist_path)
            dists = {k: cached[k] for k in cached.files}
        else:
            dists = _nn_distances(X_test, gwr._ref_points)
            dists["d_full_via_gwr_query"] = _gwr_1nn_distances(gwr, X_test)
            # confirm SymGWR path == manual cKDTree on same ref set
            if not np.allclose(dists["d_full"], dists["d_full_via_gwr_query"]):
                raise RuntimeError(
                    f"seed={seed}: manual 1-NN distances != SymGWR _query_absolute_neighbors"
                )
            np.savez_compressed(dist_path, **dists)
            report["cache_files_written"].append(os.path.abspath(dist_path))

        seed_entry = {
            "seed": seed,
            "model_cache_path": os.path.abspath(model_path),
            "model_cache_hit": bool(from_cache),
            "model_cache_existed_before_run": bool(existed),
            "dist_cache_path": os.path.abspath(dist_path),
            "ref_n": int(gwr._ref_points.shape[0]),
            "test_n": int(X_test.shape[0]),
            "ref_equals_train_union_val": ref_match,
            "stats_d_full": _percentile_stats(dists["d_full"]),
            "stats_d_temp_component_of_full_nn": _percentile_stats(
                dists["d_temp_component_of_full_nn"]
            ),
            "stats_d_comp_component_of_full_nn": _percentile_stats(
                dists["d_comp_component_of_full_nn"]
            ),
            "stats_d_temp_alone_1nn": _percentile_stats(dists["d_temp_alone_1nn"]),
            "stats_d_comp_alone_1nn": _percentile_stats(dists["d_comp_alone_1nn"]),
        }
        report["random_split_seeds"][str(seed)] = seed_entry
        per_seed_stats.append(seed_entry)
        for k in pooled:
            pooled[k].append(np.asarray(dists[k], dtype=np.float64))

        print(
            f"seed={seed} model_cache={'HIT' if from_cache else 'MISS'} "
            f"path={model_path} d_full_median={seed_entry['stats_d_full']['median']:.6g}"
        )

    pooled_stats = {
        name: _percentile_stats(np.concatenate(arrs)) for name, arrs in pooled.items()
    }
    report["distance_stats_random"] = {
        "pooled_all_seeds": pooled_stats,
        "per_seed": {
            str(e["seed"]): {
                "d_full": e["stats_d_full"],
                "d_temp_component_of_full_nn": e["stats_d_temp_component_of_full_nn"],
                "d_comp_component_of_full_nn": e["stats_d_comp_component_of_full_nn"],
                "d_temp_alone_1nn": e["stats_d_temp_alone_1nn"],
                "d_comp_alone_1nn": e["stats_d_comp_alone_1nn"],
            }
            for e in per_seed_stats
        },
    }

    # ------------------------------------------------------------------
    # Step 3: code-path cross-check (static paths + runtime identity)
    # ------------------------------------------------------------------
    report["code_path_crosscheck"] = {
        "deployed_gwr_lookup": {
            "file": "gwr/core.py",
            "methods": [
                "GradientWeightedRegressor._predict_components",
                "GradientWeightedRegressor._query_absolute_neighbors",
            ],
            "mechanism": "scipy.spatial.cKDTree built on self._ref_points; "
            "query(query_points, k=n_neighbors); Euclidean (default p=2)",
            "feature_scaling": "same scaled X passed to fit(); no extra scaling inside SymGWR",
            "snippet_query_absolute_neighbors": (
                "distances, indices = self._kdtree.query(query_points, k=k)\n"
                "closest_pts = self._ref_points[indices]"
            ),
        },
        "section_5_3_correction_zeroed_1nn": {
            "file": "experiments/run_gwr_correction_ablation.py",
            "function": "_predict_nn_only",
            "mechanism": "calls gwr._predict_components(X) then returns "
            "sum(targets * weights) / mean(targets) with ZERO Taylor correction; "
            "neighbor lookup is identical to full SymGWR predict path",
            "snippet_predict_nn_only": (
                "_, _, _, targets, weights = gwr._predict_components(X)\n"
                "if gwr.weighted:\n"
                "    return np.sum(targets * weights, axis=1)\n"
                "return targets.mean(axis=1)"
            ),
            "same_reference_set": (
                "Yes: both use the same fitted GradientWeightedRegressor "
                "loaded from results/model_cache/gwr_deg5_k1_seed{N}_*.pkl "
                "(_ref_points = train∪val for that seed)."
            ),
            "same_distance_metric": "Yes: both go through _query_absolute_neighbors → cKDTree Euclidean",
            "same_feature_scaling": "Yes: load_dataset(..., scale=True) then train_val_test_split_df",
        },
        "runtime_identity_check": {
            "description": (
                "For each seed, manual cKDTree(gwr._ref_points).query(X_test, k=1) "
                "distances were asserted allclose to "
                "gwr._query_absolute_neighbors(X_test) distances."
            ),
            "result": "PASS for all seeds 0-9",
        },
    }

    # ------------------------------------------------------------------
    # Step 4: spatially-blocked Temperature-band split
    # ------------------------------------------------------------------
    spatial = _make_spatial_split(
        df,
        df_raw,
        args.target,
        SPATIAL_T_MIN,
        SPATIAL_T_MAX,
        SPATIAL_REMAIN_SEED,
        SPATIAL_REMAIN_TRAIN_FRAC,
    )
    split_extra = (
        f"t_min={SPATIAL_T_MIN}|t_max={SPATIAL_T_MAX}|"
        f"remain_seed={SPATIAL_REMAIN_SEED}|"
        f"remain_train_frac={SPATIAL_REMAIN_TRAIN_FRAC:.3f}"
    )
    split_key = _split_index_cache_key(
        "spatial_temp_band", args.data, args.target, scale, split_extra
    )
    split_path = os.path.join(args.results_dir, split_key)
    if os.path.isfile(split_path):
        report["cache_files_read"].append(os.path.abspath(split_path))
        split_cached = np.load(split_path)
        # prefer cached indices; rebuild arrays from df
        train_idx = split_cached["train_idx"]
        val_idx = split_cached["val_idx"]
        test_idx = split_cached["test_idx"]
        X_all = (
            df.drop(columns=[args.target]).select_dtypes(include=[np.number]).values
        )
        y_all = df[args.target].values
        X_train = X_all[train_idx]
        X_val = X_all[val_idx]
        X_test = X_all[test_idx]
        y_train = y_all[train_idx]
        y_val = y_all[val_idx]
        y_test = y_all[test_idx]
    else:
        train_idx = spatial["train_idx"]
        val_idx = spatial["val_idx"]
        test_idx = spatial["test_idx"]
        X_train = spatial["X_train"]
        X_val = spatial["X_val"]
        X_test = spatial["X_test"]
        y_train = spatial["y_train"]
        y_val = spatial["y_val"]
        y_test = spatial["y_test"]
        np.savez_compressed(
            split_path,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            t_min=np.array([SPATIAL_T_MIN]),
            t_max=np.array([SPATIAL_T_MAX]),
            remain_seed=np.array([SPATIAL_REMAIN_SEED]),
            remain_train_frac=np.array([SPATIAL_REMAIN_TRAIN_FRAC]),
        )
        report["cache_files_written"].append(os.path.abspath(split_path))

    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    spatial_model_key = _gwr_cache_key_spatial(
        args.degree,
        args.neighbors,
        args.data,
        args.target,
        scale,
        SPATIAL_T_MIN,
        SPATIAL_T_MAX,
        SPATIAL_REMAIN_SEED,
        SPATIAL_REMAIN_TRAIN_FRAC,
    )
    spatial_model_path = os.path.join(args.cache_dir, spatial_model_key)
    spatial_existed = os.path.isfile(spatial_model_path)
    gwr_s, spatial_from_cache = _load_or_fit_gwr(
        spatial_model_path, X_train_full, y_train_full, args.degree, args.neighbors
    )
    if spatial_from_cache:
        report["cache_files_read"].append(os.path.abspath(spatial_model_path))
    else:
        report["cache_files_written"].append(os.path.abspath(spatial_model_path))

    y_pred = np.asarray(gwr_s.predict(X_test), dtype=np.float64)
    metrics = compute_all_metrics(y_test, y_pred)

    spatial_dist_key = _dist_cache_key(
        "spatial_temp_band",
        f"{SPATIAL_T_MIN}_{SPATIAL_T_MAX}",
        args.degree,
        args.neighbors,
        args.data,
        args.target,
        scale,
        extra=split_extra,
    )
    spatial_dist_path = os.path.join(args.results_dir, spatial_dist_key)
    if os.path.isfile(spatial_dist_path):
        report["cache_files_read"].append(os.path.abspath(spatial_dist_path))
        sc = np.load(spatial_dist_path)
        sdists = {k: sc[k] for k in sc.files}
    else:
        sdists = _nn_distances(X_test, gwr_s._ref_points)
        sdists["d_full_via_gwr_query"] = _gwr_1nn_distances(gwr_s, X_test)
        if not np.allclose(sdists["d_full"], sdists["d_full_via_gwr_query"]):
            raise RuntimeError("spatial: manual 1-NN != SymGWR query distances")
        np.savez_compressed(spatial_dist_path, **sdists)
        report["cache_files_written"].append(os.path.abspath(spatial_dist_path))

    report["spatial_blocked"] = {
        "split_type": "spatial_temp_band",
        "t_min_K_raw": SPATIAL_T_MIN,
        "t_max_K_raw": SPATIAL_T_MAX,
        "remain_seed": SPATIAL_REMAIN_SEED,
        "remain_train_frac": SPATIAL_REMAIN_TRAIN_FRAC,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "n_total": int(len(df)),
        "test_frac": float(len(test_idx) / len(df)),
        "split_indices_path": os.path.abspath(split_path),
        "model_cache_path": os.path.abspath(spatial_model_path),
        "model_cache_hit": bool(spatial_from_cache),
        "model_cache_existed_before_run": bool(spatial_existed),
        "dist_cache_path": os.path.abspath(spatial_dist_path),
        "metrics_test": {
            "mape": float(metrics["mape_mean"]),
            "mae": float(metrics["mae_mean"]),
            "rmse": float(metrics["rmse"]),
            "r2": float(metrics["r2"]),
        },
        "distance_stats": {
            "d_full": _percentile_stats(sdists["d_full"]),
            "d_temp_component_of_full_nn": _percentile_stats(
                sdists["d_temp_component_of_full_nn"]
            ),
            "d_comp_component_of_full_nn": _percentile_stats(
                sdists["d_comp_component_of_full_nn"]
            ),
            "d_temp_alone_1nn": _percentile_stats(sdists["d_temp_alone_1nn"]),
            "d_comp_alone_1nn": _percentile_stats(sdists["d_comp_alone_1nn"]),
        },
    }

    # Deduplicate cache lists while preserving order
    def _dedupe(xs: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    report["cache_files_read"] = _dedupe(report["cache_files_read"])
    report["cache_files_written"] = _dedupe(report["cache_files_written"])
    report["confirmation_random_locked_models_cached"] = {
        "all_seeds_0_to_9_present_in_model_cache": all(
            report["random_split_seeds"][str(s)]["model_cache_hit"]
            or os.path.isfile(report["random_split_seeds"][str(s)]["model_cache_path"])
            for s in RANDOM_STATES
        ),
        "per_seed_status": {
            str(s): (
                "loaded_from_cache"
                if report["random_split_seeds"][str(s)]["model_cache_hit"]
                else "freshly_fit_and_saved"
            )
            for s in RANDOM_STATES
        },
    }

    out_json = os.path.join(args.results_dir, "nearest_ref_distance_audit.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    report["cache_files_written"] = _dedupe(
        report["cache_files_written"] + [os.path.abspath(out_json)]
    )
    # rewrite with final written list including this json
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    # flat CSV for distance stats
    rows = []
    for scope, block in (
        ("pooled_random", pooled_stats),
        (
            "spatial_temp_band",
            report["spatial_blocked"]["distance_stats"],
        ),
    ):
        for metric_name, stats in block.items():
            if scope == "pooled_random" and metric_name == "d_full_via_gwr_query":
                continue
            row = {"scope": scope, "distance": metric_name, **stats}
            rows.append(row)
    for e in per_seed_stats:
        for metric_name, key in (
            ("d_full", "stats_d_full"),
            ("d_temp_component_of_full_nn", "stats_d_temp_component_of_full_nn"),
            ("d_comp_component_of_full_nn", "stats_d_comp_component_of_full_nn"),
            ("d_temp_alone_1nn", "stats_d_temp_alone_1nn"),
            ("d_comp_alone_1nn", "stats_d_comp_alone_1nn"),
        ):
            row = {
                "scope": f"random_seed{e['seed']}",
                "distance": metric_name,
                **e[key],
            }
            rows.append(row)
    out_csv = os.path.join(args.results_dir, "nearest_ref_distance_audit.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    if os.path.abspath(out_csv) not in report["cache_files_written"]:
        report["cache_files_written"].append(os.path.abspath(out_csv))
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)

    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")
    print(
        "spatial metrics:",
        report["spatial_blocked"]["metrics_test"],
    )


if __name__ == "__main__":
    main()
