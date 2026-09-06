"""
baselines.loess
===============

Multivariate LOESS/LOWESS baseline via local weighted linear regression with
tricube distance kernel.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from gwr.utils import compute_all_metrics


class LoessRegressor:
    """Simple multivariate LOESS with local linear fits."""

    def __init__(self, frac: float = 0.3, reg: float = 1e-8):
        self.frac = frac
        self.reg = reg
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._nn: Optional[NearestNeighbors] = None
        self._k: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LoessRegressor":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self._X_train = X
        self._y_train = y
        n_samples = X.shape[0]
        self._k = max(2, int(np.ceil(self.frac * n_samples)))
        self._nn = NearestNeighbors(n_neighbors=self._k, algorithm="auto")
        self._nn.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._nn is None or self._X_train is None or self._y_train is None:
            raise RuntimeError("LoessRegressor not fitted.")

        X = np.asarray(X, dtype=np.float64)
        distances, indices = self._nn.kneighbors(X, return_distance=True)

        y_pred = np.zeros(X.shape[0], dtype=np.float64)
        for i in range(X.shape[0]):
            nn_idx = indices[i]
            X_local = self._X_train[nn_idx]
            y_local = self._y_train[nn_idx]
            d_local = distances[i]

            d_max = float(np.max(d_local)) + 1e-12
            u = d_local / d_max
            w = np.where(u < 1.0, (1.0 - u**3) ** 3, 0.0)
            w = np.maximum(w, 1e-12)

            X_design = np.hstack(
                [np.ones((X_local.shape[0], 1)), X_local - X[i][np.newaxis, :]]
            )
            W = np.diag(w)
            XtWX = X_design.T @ W @ X_design
            XtWy = X_design.T @ W @ y_local
            beta = np.linalg.solve(
                XtWX + self.reg * np.eye(XtWX.shape[0], dtype=np.float64),
                XtWy,
            )
            # Centering at x_i makes intercept equal local prediction at x_i.
            y_pred[i] = beta[0]
        return y_pred


def train_loess(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    frac: float = 0.3,
    reg: float = 1e-8,
    scale: bool = True,
) -> Tuple[LoessRegressor, Optional[np.ndarray], Optional[StandardScaler]]:
    """Train LOESS baseline and optionally return predictions."""
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        if X_test is not None:
            X_test = scaler.transform(X_test)

    model = LoessRegressor(frac=frac, reg=reg).fit(X_train, y_train)
    y_pred = model.predict(X_test) if X_test is not None else None
    return model, y_pred, scaler


def evaluate_loess(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    **kwargs,
) -> Dict[str, float]:
    """Train LOESS and return evaluation metrics."""
    _, y_pred, _ = train_loess(X_train, y_train, X_test, **kwargs)
    return compute_all_metrics(y_test, y_pred)
