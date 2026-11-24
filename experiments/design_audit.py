#!/usr/bin/env python
"""Design audit for the camera-ready CALPHAD splits.

Writes results/camera_ready/design_audit.json with:
  * row counts (raw / deduped), distinct compositions, temperatures
  * rank + condition number of the degree-5 polynomial design matrix on the
    training split (per split kind, seeds 0-9; headline = seed 0)
  * count of exact-duplicate feature vectors shared between train and test
    under the OLD random split (non-deduped, train_val_test_split_df)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gwr.splits import DEFAULT_DATA, DEFAULT_TARGET, KINDS, TEMP_COL, get_split, load_raw
from gwr.utils import train_val_test_split_df

DEGREE = 5
SEEDS = list(range(10))


def _design_stats(X_train: np.ndarray, degree: int = DEGREE) -> dict:
    A = PolynomialFeatures(degree=degree).fit_transform(X_train)
    s = np.linalg.svd(A, compute_uv=False)
    tol = s.max() * max(A.shape) * np.finfo(A.dtype).eps  # numpy matrix_rank default
    rank = int((s > tol).sum())
    return {
        "shape": list(A.shape),
        "rank": rank,
        "rank_deficiency": int(min(A.shape) - rank),
        "sv_max": float(s[0]),
        "sv_min": float(s[-1]),
        "sv_rank_th": float(s[rank - 1]),
        "sv_first_below_tol": float(s[rank]) if rank < len(s) else None,
        "rank_tol": float(tol),
        # cond over the full spectrum (huge when rank-deficient) ...
        "cond_full": float(s[0] / s[-1]) if s[-1] > 0 else float("inf"),
        # ... and over the numerically nonzero part (sv_max / sv_rank).
        "cond_effective": float(s[0] / s[rank - 1]),
    }


def _row_key(X: np.ndarray) -> list[bytes]:
    return [r.tobytes() for r in np.ascontiguousarray(X, dtype=np.float64)]


def _old_split_duplicates(path: str, target: str, seed: int) -> dict:
    """Legacy random split: no dedupe, load_dataset-style frame, 60/20/20."""
    df, feats = load_raw(path, target)
    df_num = df[feats + [target]]
    X_tr, X_va, X_te, _, _, _ = train_val_test_split_df(
        df_num, target_col=target, random_state=seed
    )
    k_tr, k_va, k_te = _row_key(X_tr), _row_key(X_va), _row_key(X_te)
    s_tr, s_trva = set(k_tr), set(k_tr) | set(k_va)
    in_tr = sum(k in s_tr for k in k_te)
    in_trva = sum(k in s_trva for k in k_te)
    return {
        "seed": seed,
        "n_train": len(k_tr),
        "n_val": len(k_va),
        "n_test": len(k_te),
        "test_rows_with_exact_dup_in_train": int(in_tr),
        "test_rows_with_exact_dup_in_train_or_val": int(in_trva),
        "frac_test_dup_in_train": in_tr / len(k_te),
        "frac_test_dup_in_train_or_val": in_trva / len(k_te),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--target", default=DEFAULT_TARGET)
    ap.add_argument("--output", default="results/camera_ready/design_audit.json")
    args = ap.parse_args()

    df_raw, feats = load_raw(args.data, args.target)
    comp_cols = [c for c in feats if c != TEMP_COL]
    df_dd = df_raw.drop_duplicates(subset=feats)

    per_comp_raw = df_raw.groupby(comp_cols).size()
    per_comp_dd = df_dd.groupby(comp_cols).size()
    const_cols = [c for c in feats if df_raw[c].nunique() == 1]
    uniq_comp = df_raw[comp_cols].drop_duplicates()
    base = uniq_comp.mode().iloc[0]
    n_diff = (uniq_comp != base).sum(axis=1)

    out: dict = {
        "data": args.data,
        "feature_columns": feats,
        "constant_feature_columns": const_cols,
        "composition_structure": {
            "base_composition": {k: float(v) for k, v in base.items()},
            "n_elements_differing_from_base__count_of_compositions": {
                str(k): int(v) for k, v in n_diff.value_counts().sort_index().items()
            },
            "note": "OFAT star design if all non-base compositions differ in exactly 1 element",
        },
        "row_counts": {
            "raw": int(len(df_raw)),
            "after_dedupe_on_8_features": int(len(df_dd)),
            "duplicate_rows_removed": int(len(df_raw) - len(df_dd)),
            "n_distinct_compositions": int(len(per_comp_raw)),
            "n_distinct_temperatures": int(df_raw[TEMP_COL].nunique()),
            "rows_per_composition_raw": {
                str(k): int(v) for k, v in per_comp_raw.value_counts().sort_index().items()
            },
            "rows_per_composition_deduped": {
                str(k): int(v) for k, v in per_comp_dd.value_counts().sort_index().items()
            },
            "dedupe_ever_changes_target": bool(
                (df_raw.groupby(feats)[args.target].nunique() > 1).any()
            ),
        },
    }

    # ---- degree-5 design matrix on the training split -------------------
    design: dict = {"degree": DEGREE, "note": (
        "PolynomialFeatures(degree=5) incl. bias on the train-split-scaled 8 features; "
        "rank = SVD count above numpy matrix_rank tolerance; cond_effective = "
        "sv_max/sv_rank; cond_full = sv_max/sv_min."
    )}
    for kind in KINDS:
        runs = []
        for seed in SEEDS:
            sp = get_split(kind, seed, dedupe=True, data_path=args.data,
                           target=args.target, verbose=False)
            st = _design_stats(sp.X_train)
            st.update(seed=seed, n_train=int(len(sp.train_idx)))
            runs.append(st)
        design[f"{kind}_dedupe"] = {"seed0": runs[0], "all_seeds": runs}
    # Legacy-style: random split without dedupe (training split only).
    sp = get_split("random", 0, dedupe=False, data_path=args.data,
                   target=args.target, verbose=False)
    st = _design_stats(sp.X_train)
    st.update(seed=0, n_train=int(len(sp.train_idx)))
    design["random_no_dedupe"] = {"seed0": st}
    out["design_matrix"] = design

    # ---- exact duplicates train/test under the OLD random split ----------
    old = [_old_split_duplicates(args.data, args.target, s) for s in SEEDS]
    new = []
    for s in SEEDS:
        sp = get_split("random", s, dedupe=True, data_path=args.data,
                       target=args.target, verbose=False)
        Xr = sp.df[sp.feature_cols].to_numpy()
        tr_keys = set(_row_key(Xr[sp.train_idx]) + _row_key(Xr[sp.val_idx]))
        new.append(int(sum(k in tr_keys for k in _row_key(Xr[sp.test_idx]))))
    arr = np.array([r["test_rows_with_exact_dup_in_train"] for r in old])
    arr2 = np.array([r["test_rows_with_exact_dup_in_train_or_val"] for r in old])
    out["old_random_split_duplicates"] = {
        "split": "train_val_test_split_df, 60/20/20, no dedupe, seeds 0-9",
        "per_seed": old,
        "test_dup_in_train": {"mean": float(arr.mean()), "min": int(arr.min()), "max": int(arr.max())},
        "test_dup_in_train_or_val": {"mean": float(arr2.mean()), "min": int(arr2.min()), "max": int(arr2.max())},
        "new_deduped_random_split_test_dup_in_train_or_val_per_seed": new,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
