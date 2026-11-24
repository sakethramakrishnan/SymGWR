"""
baselines.mlp
=============

Multi-Layer Perceptron (MLP) regressor baseline for comparison against SymGWR.

Wraps scikit-learn's ``MLPRegressor`` for a consistent interface with
the rest of the benchmarking pipeline.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from gwr.utils import compute_all_metrics


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    hidden_layer_sizes: Tuple[int, ...] = (256, 128, 64),
    activation: str = "relu",
    solver: str = "adam",
    max_iter: int = 1000,
    learning_rate_init: float = 1e-3,
    early_stopping: bool = True,
    random_state: int = 42,
    scale: bool = True,
) -> Tuple[MLPRegressor, Optional[np.ndarray], Optional[StandardScaler]]:
    """Train a scikit-learn MLP regressor.

    Parameters
    ----------
    X_train, y_train : array-like
        Training data.
    X_test : array-like, optional
        If provided, predictions are returned.
    hidden_layer_sizes : tuple[int, ...]
    activation : str
    solver : str
    max_iter : int
    learning_rate_init : float
    early_stopping : bool
    random_state : int
    scale : bool
        Whether to standard-scale the features before fitting.

    Returns
    -------
    model   : MLPRegressor
    y_pred  : np.ndarray or None
    scaler  : StandardScaler or None
    """
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        if X_test is not None:
            X_test = scaler.transform(X_test)

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        max_iter=max_iter,
        learning_rate_init=learning_rate_init,
        early_stopping=early_stopping,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test) if X_test is not None else None
    return model, y_pred, scaler


def evaluate_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    **kwargs,
) -> Dict[str, float]:
    """Train an MLP and return evaluation metrics.

    Returns
    -------
    dict
        Standard regression metrics (RMSE, R², MAE, MAPE, SMAPE).
    """
    _, y_pred, _ = train_mlp(X_train, y_train, X_test, **kwargs)
    return compute_all_metrics(y_test, y_pred)
