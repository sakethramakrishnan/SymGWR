"""
gwr.utils
=========

Shared utilities: data loading / scaling, evaluation metrics, and other
helper functions used across the GWR pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional, Tuple


# ======================================================================
#  Custom Standard Scaler (deterministic, no sklearn state)
# ======================================================================

class CustomStandardScaler:
    """Minimal standard-scaler that stores mean / std explicitly."""

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "CustomStandardScaler":
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        # Constant columns: np.std of a subset can be ~1e-17 rather than exactly 0
        # (float rounding in the mean), which would blow noise up to +-1 after scaling.
        tol = 1e-12 * np.maximum(1.0, np.abs(self.mean_))
        self.scale_ = np.where(self.scale_ <= tol, 1.0, self.scale_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        return X * self.scale_ + self.mean_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ======================================================================
#  Data loading
# ======================================================================

def load_dataset(
    file_path: str,
    target_col: str,
    scale: bool = True,
) -> Tuple[Optional[CustomStandardScaler], Optional[CustomStandardScaler], pd.DataFrame]:
    """Load a tabular dataset and optionally standard-scale features and target.

    Parameters
    ----------
    file_path : str
        Path to a ``.csv``, ``.txt`` (whitespace-delimited), or ``.xlsx`` file.
    target_col : str
        Name of the target column.
    scale : bool, default True
        Whether to apply standard scaling.

    Returns
    -------
    feature_scaler : CustomStandardScaler or None
    target_scaler  : CustomStandardScaler or None
    scaled_df      : pd.DataFrame
        DataFrame with (optionally) scaled features and target column.
    """
    if file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    elif file_path.endswith(".txt") or file_path.endswith(".csv"):
        # Auto-detect delimiter (comma, tab, or whitespace)
        try:
            df = pd.read_csv(file_path, sep=None, engine="python")
        except Exception:
            df = pd.read_csv(file_path, delim_whitespace=True)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    if not scale:
        X[target_col] = y
        return None, None, X

    feature_scaler = CustomStandardScaler()
    numeric = X.select_dtypes(include=[np.number])
    feature_scaler.fit(numeric.values)
    X_scaled = pd.DataFrame(
        feature_scaler.transform(numeric.values),
        columns=numeric.columns,
        index=X.index,
    )
    non_numeric = X.drop(columns=numeric.columns)
    X_scaled = pd.concat([X_scaled, non_numeric], axis=1)

    y_2d = y.values.reshape(-1, 1)
    target_scaler = CustomStandardScaler().fit(y_2d)
    X_scaled[target_col] = target_scaler.transform(y_2d).flatten()

    return feature_scaler, target_scaler, X_scaled


def train_test_split_df(
    df: pd.DataFrame,
    target_col: str,
    train_frac: float = 0.8,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a DataFrame into train/test arrays.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).values
    y = df[target_col].values
    return train_test_split(X, y, train_size=train_frac, random_state=random_state)


def train_val_test_split_df(
    df: pd.DataFrame,
    target_col: str,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a DataFrame into train/validation/test arrays.

    Fractions must sum to 1.0.
    """
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0.")

    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).values
    y = df[target_col].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        train_size=train_frac,
        random_state=random_state,
    )
    rel_val = val_frac / (val_frac + test_frac)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        train_size=rel_val,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ======================================================================
#  Metrics
# ======================================================================

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Return (MAE, MAE_std)."""
    ae = np.abs(y_true - y_pred)
    return float(np.mean(ae)), float(np.std(ae))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Return (MAPE, MAPE_std), ignoring samples where y_true == 0."""
    mask = y_true != 0
    ape = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    return float(np.mean(ape)), float(np.std(ape))


def symmetric_mape(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Return (SMAPE, SMAPE_std)."""
    denom = np.abs(y_true) + np.abs(y_pred)
    smape = 2.0 * np.abs(y_true - y_pred) / (denom + 1e-12)
    return float(np.mean(smape)), float(np.std(smape))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))


def compute_all_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute a standard set of regression metrics.

    Returns
    -------
    dict with keys: ``mse``, ``rmse``, ``r2``, ``mae_mean``, ``mae_std``,
    ``mape_mean``, ``mape_std``, ``smape_mean``, ``smape_std``.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mae_mean, mae_std = mean_absolute_error(y_true, y_pred)
    mape_mean, mape_std = mean_absolute_percentage_error(y_true, y_pred)
    smape_mean, smape_std = symmetric_mape(y_true, y_pred)

    return {
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "mae_mean": mae_mean,
        "mae_std": mae_std,
        "mape_mean": mape_mean,
        "mape_std": mape_std,
        "smape_mean": smape_mean,
        "smape_std": smape_std,
    }


def aggregate_metric_dicts(metric_rows: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate repeated metric rows into mean/std columns."""
    if not metric_rows:
        return {}

    numeric_keys = [
        key for key in metric_rows[0]
        if isinstance(metric_rows[0][key], (int, float, np.floating))
    ]
    out: Dict[str, float] = {}
    for key in numeric_keys:
        values = np.array([row[key] for row in metric_rows], dtype=np.float64)
        out[f"{key}_mean"] = float(np.mean(values))
        out[f"{key}_std"] = float(np.std(values))
    out["n_repeats"] = float(len(metric_rows))
    return out
