"""
baselines.kernel_regression
===========================

Nadaraya-Watson kernel regression baseline with Gaussian kernel.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple

from sklearn.preprocessing import StandardScaler

from gwr.utils import compute_all_metrics


class NadarayaWatsonRegressor:
    """Gaussian-kernel Nadaraya-Watson regressor."""

    def __init__(self, bandwidth: float = 1.0):
        self.bandwidth = bandwidth
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NadarayaWatsonRegressor":
        self._X_train = np.asarray(X, dtype=np.float64)
        self._y_train = np.asarray(y, dtype=np.float64)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._X_train is None or self._y_train is None:
            raise RuntimeError("NadarayaWatsonRegressor not fitted.")

        X = np.asarray(X, dtype=np.float64)
        sq_dist = np.sum(
            (X[:, np.newaxis, :] - self._X_train[np.newaxis, :, :]) ** 2,
            axis=2,
        )
        bw2 = max(self.bandwidth**2, 1e-12)
        w = np.exp(-0.5 * sq_dist / bw2)
        w_sum = np.sum(w, axis=1, keepdims=True) + 1e-12
        return np.sum(w * self._y_train[np.newaxis, :], axis=1) / w_sum.squeeze(1)


def train_kernel_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    bandwidth: float = 1.0,
    scale: bool = True,
) -> Tuple[NadarayaWatsonRegressor, Optional[np.ndarray], Optional[StandardScaler]]:
    """Train kernel regression and optionally return predictions."""
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        if X_test is not None:
            X_test = scaler.transform(X_test)

    model = NadarayaWatsonRegressor(bandwidth=bandwidth).fit(X_train, y_train)
    y_pred = model.predict(X_test) if X_test is not None else None
    return model, y_pred, scaler


def evaluate_kernel_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    **kwargs,
) -> Dict[str, float]:
    """Train kernel regression and return evaluation metrics."""
    _, y_pred, _ = train_kernel_regression(X_train, y_train, X_test, **kwargs)
    return compute_all_metrics(y_test, y_pred)
