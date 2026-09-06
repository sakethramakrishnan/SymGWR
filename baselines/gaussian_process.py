"""
baselines.gaussian_process
==========================

Gaussian Process Regression baseline.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

from gwr.utils import compute_all_metrics


def train_gaussian_process(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    alpha: float = 1e-6,
    random_state: int = 42,
    scale: bool = True,
) -> Tuple[GaussianProcessRegressor, Optional[np.ndarray], Optional[StandardScaler]]:
    """Train Gaussian Process regressor."""
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        if X_test is not None:
            X_test = scaler.transform(X_test)

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
        length_scale=1.0,
        length_scale_bounds=(1e-3, 1e3),
    ) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e0))
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=True,
        random_state=random_state,
        n_restarts_optimizer=2,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test) if X_test is not None else None
    return model, y_pred, scaler


def evaluate_gaussian_process(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    **kwargs,
) -> Dict[str, float]:
    """Train GPR and return evaluation metrics."""
    _, y_pred, _ = train_gaussian_process(X_train, y_train, X_test, **kwargs)
    return compute_all_metrics(y_test, y_pred)
