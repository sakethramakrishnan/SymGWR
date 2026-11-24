"""
baselines.knn
=============

K-Nearest Neighbours regressor baseline for comparison against SymGWR.

While SymGWR also uses nearest neighbours, the key difference is that SymGWR
extrapolates from neighbours using analytical gradient information,
whereas vanilla KNN simply averages (or distance-weights) neighbour targets.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from gwr.utils import compute_all_metrics


def train_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    n_neighbors: int = 5,
    weights: str = "distance",
    algorithm: str = "kd_tree",
    scale: bool = True,
) -> Tuple[KNeighborsRegressor, Optional[np.ndarray], Optional[StandardScaler]]:
    """Train a scikit-learn KNN regressor.

    Parameters
    ----------
    X_train, y_train : array-like
    X_test : array-like, optional
    n_neighbors : int
    weights : str
        ``'uniform'`` or ``'distance'``.
    algorithm : str
    scale : bool

    Returns
    -------
    model   : KNeighborsRegressor
    y_pred  : np.ndarray or None
    scaler  : StandardScaler or None
    """
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        if X_test is not None:
            X_test = scaler.transform(X_test)

    model = KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test) if X_test is not None else None
    return model, y_pred, scaler


def evaluate_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    **kwargs,
) -> Dict[str, float]:
    """Train KNN and return evaluation metrics."""
    _, y_pred, _ = train_knn(X_train, y_train, X_test, **kwargs)
    return compute_all_metrics(y_test, y_pred)
