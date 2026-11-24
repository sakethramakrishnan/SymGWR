"""
gwr.splits
==========

Train/val/test splits for the CALPHAD dataset.

``get_split(kind, seed, dedupe=True)`` is the single entry point.  Kinds:

``random``
    60/20/20 random row split.
``temp_band``
    Hold out every row with 1320 <= T <= 1470 K as test; split the remainder
    75/25 into train/val (seed selects the remainder split).
``composition_group``
    GroupKFold over the distinct composition vectors.  Whole compositions are
    held out (all their temperatures).  ``seed`` shuffles the fold assignment;
    test = fold ``seed % n_folds``, val = fold ``(seed + 1) % n_folds``,
    train = the remaining folds (~60/20/20 by composition).

Pipeline order: load raw -> optional dedupe on the feature columns -> split
on raw rows -> fit scalers on the *training* split only -> transform
val/test with those scalers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split

from gwr.utils import CustomStandardScaler

DEFAULT_DATA = "data/raw/718_all.csv"
DEFAULT_TARGET = "Gibbs_energy"
TEMP_COL = "Temperature_[K]"

TEMP_BAND = (1320, 1470)  # inclusive, K
TEMP_BAND_TRAIN_FRAC = 0.75  # of non-test rows
RANDOM_FRACS = (0.6, 0.2, 0.2)
N_GROUP_FOLDS = 5

KINDS = ("random", "temp_band", "composition_group")


@dataclass
class Split:
    """Scaled arrays (scalers fit on train only) plus raw-unit targets."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_scaler: CustomStandardScaler
    target_scaler: CustomStandardScaler
    # row indices into the (deduped) frame ``df``; raw frame in ``df``
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    df: pd.DataFrame
    feature_cols: list[str]
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def y_train_raw(self) -> np.ndarray:
        return self.df[self._target].to_numpy()[self.train_idx]

    @property
    def y_val_raw(self) -> np.ndarray:
        return self.df[self._target].to_numpy()[self.val_idx]

    @property
    def y_test_raw(self) -> np.ndarray:
        return self.df[self._target].to_numpy()[self.test_idx]

    @property
    def X_train_raw(self) -> np.ndarray:
        return self.df[self.feature_cols].to_numpy()[self.train_idx]

    @property
    def X_test_raw(self) -> np.ndarray:
        return self.df[self.feature_cols].to_numpy()[self.test_idx]

    _target: str = DEFAULT_TARGET


# ----------------------------------------------------------------------
#  Data
# ----------------------------------------------------------------------

def load_raw(path: str = DEFAULT_DATA, target: str = DEFAULT_TARGET) -> tuple[pd.DataFrame, list[str]]:
    """Read the raw CSV; return (frame, numeric feature column names)."""
    df = pd.read_csv(path, sep=None, engine="python")
    feats = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
    return df, feats


def dedupe_rows(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Drop rows whose feature vector (all ``feature_cols``) is a repeat; keep first."""
    return df.drop_duplicates(subset=feature_cols, keep="first").reset_index(drop=True)


def composition_ids(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Integer id per distinct composition vector (all feature cols except T)."""
    comp = [c for c in feature_cols if c != TEMP_COL]
    _, ids = np.unique(df[comp].to_numpy(), axis=0, return_inverse=True)
    return np.asarray(ids).reshape(-1)


# ----------------------------------------------------------------------
#  Index-level splitters
# ----------------------------------------------------------------------

def random_indices(n: int, seed: int, fracs: tuple[float, float, float] = RANDOM_FRACS):
    tr, va, te = fracs
    idx = np.arange(n)
    train_idx, rest = train_test_split(idx, train_size=tr, random_state=seed)
    val_idx, test_idx = train_test_split(rest, train_size=va / (va + te), random_state=seed)
    return train_idx, val_idx, test_idx


def temp_band_indices(
    temp: np.ndarray,
    seed: int,
    t_min: int = TEMP_BAND[0],
    t_max: int = TEMP_BAND[1],
    remain_train_frac: float = TEMP_BAND_TRAIN_FRAC,
):
    mask = (temp >= t_min) & (temp <= t_max)
    test_idx = np.where(mask)[0]
    remain_idx = np.where(~mask)[0]
    train_idx, val_idx = train_test_split(
        remain_idx, train_size=remain_train_frac, random_state=seed
    )
    return np.asarray(train_idx), np.asarray(val_idx), test_idx


def composition_group_indices(groups: np.ndarray, seed: int, n_folds: int = N_GROUP_FOLDS):
    n_groups = len(np.unique(groups))
    if n_groups < n_folds:
        raise ValueError(f"need >= {n_folds} compositions, got {n_groups}")
    gkf = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    dummy = np.zeros((len(groups), 1))
    folds = [test for _, test in gkf.split(dummy, groups=groups)]
    test_f = seed % n_folds
    val_f = (seed + 1) % n_folds
    test_idx = np.sort(folds[test_f])
    val_idx = np.sort(folds[val_f])
    train_idx = np.sort(
        np.concatenate([f for i, f in enumerate(folds) if i not in (test_f, val_f)])
    )
    return train_idx, val_idx, test_idx


# ----------------------------------------------------------------------
#  Public API
# ----------------------------------------------------------------------

def get_split(
    kind: str,
    seed: int,
    dedupe: bool = True,
    data_path: str = DEFAULT_DATA,
    target: str = DEFAULT_TARGET,
    verbose: bool = True,
) -> Split:
    """Return a scaled train/val/test split (scalers fit on train only)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")

    df, feats = load_raw(data_path, target)
    n_before = len(df)
    if dedupe:
        df = dedupe_rows(df, feats)
    n_after = len(df)
    if verbose:
        print(
            f"[splits] {kind} seed={seed}: rows before dedupe={n_before}, "
            f"after={n_after} (dedupe={'on' if dedupe else 'off'})"
        )

    X_raw = df[feats].to_numpy(dtype=np.float64)
    y_raw = df[target].to_numpy(dtype=np.float64)

    groups: Optional[np.ndarray] = None
    if kind == "random":
        tr, va, te = random_indices(n_after, seed)
    elif kind == "temp_band":
        tr, va, te = temp_band_indices(df[TEMP_COL].to_numpy(), seed)
    else:
        groups = composition_ids(df, feats)
        tr, va, te = composition_group_indices(groups, seed)

    # Scalers: train split only.
    fscaler = CustomStandardScaler().fit(X_raw[tr])
    tscaler = CustomStandardScaler().fit(y_raw[tr].reshape(-1, 1))

    def fx(i):
        return fscaler.transform(X_raw[i])

    def fy(i):
        return tscaler.transform(y_raw[i].reshape(-1, 1)).ravel()

    info: dict[str, Any] = {
        "kind": kind,
        "seed": seed,
        "dedupe": dedupe,
        "n_rows_before_dedupe": n_before,
        "n_rows_after_dedupe": n_after,
        "n_train": len(tr),
        "n_val": len(va),
        "n_test": len(te),
    }
    if groups is not None:
        info["n_compositions"] = int(len(np.unique(groups)))
        info["n_comp_train"] = int(len(np.unique(groups[tr])))
        info["n_comp_val"] = int(len(np.unique(groups[va])))
        info["n_comp_test"] = int(len(np.unique(groups[te])))
    if verbose:
        print(f"[splits] sizes train/val/test = {len(tr)}/{len(va)}/{len(te)}")

    return Split(
        X_train=fx(tr), X_val=fx(va), X_test=fx(te),
        y_train=fy(tr), y_val=fy(va), y_test=fy(te),
        feature_scaler=fscaler, target_scaler=tscaler,
        train_idx=tr, val_idx=va, test_idx=te,
        df=df, feature_cols=feats, info=info, _target=target,
    )


# ----------------------------------------------------------------------
#  Legacy spatial split (moved from experiments/run_nearest_ref_distance_audit.py)
# ----------------------------------------------------------------------

def make_spatial_split(
    df_scaled: pd.DataFrame,
    df_raw: pd.DataFrame,
    target: str,
    t_min: int,
    t_max: int,
    remain_seed: int,
    remain_train_frac: float,
) -> dict[str, Any]:
    """Hold out contiguous raw Temperature band; split remainder into train/val.

    Behaviour-preserving move of the old ``_make_spatial_split``: operates on
    a pre-scaled frame (scaler fit on all rows), so it reproduces the cached
    legacy indices.  New code should use ``get_split("temp_band", seed)``.
    """
    train_idx, val_idx, test_idx = temp_band_indices(
        df_raw[TEMP_COL].to_numpy(), remain_seed, t_min, t_max, remain_train_frac
    )
    X_all = df_scaled.drop(columns=[target]).select_dtypes(include=[np.number]).values
    y_all = df_scaled[target].values
    return {
        "X_train": X_all[train_idx],
        "X_val": X_all[val_idx],
        "X_test": X_all[test_idx],
        "y_train": y_all[train_idx],
        "y_val": y_all[val_idx],
        "y_test": y_all[test_idx],
        "train_idx": train_idx.astype(np.int64),
        "val_idx": val_idx.astype(np.int64),
        "test_idx": test_idx.astype(np.int64),
        "t_min": t_min,
        "t_max": t_max,
        "n_test": int(len(test_idx)),
        "n_total": int(len(df_scaled)),
        "test_frac": float(len(test_idx) / len(df_scaled)),
    }
