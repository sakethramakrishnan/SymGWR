"""
baselines.xgboost
=================

XGBoost regressor baseline for comparison against GWR.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

from gwr.utils import compute_all_metrics


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42,
    **kwargs,
) -> Tuple[XGBRegressor, Optional[np.ndarray]]:
    """Train an XGBoost regressor.

    Parameters
    ----------
    X_train, y_train : array-like
        Training data.
    X_test : array-like, optional
        If provided, predictions are returned.
    n_estimators : int
    max_depth : int
    learning_rate : float
    random_state : int

    Returns
    -------
    model : XGBRegressor
    y_pred : np.ndarray or None
    """
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        n_jobs=-1,
        **kwargs,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test) if X_test is not None else None
    return model, y_pred


def evaluate_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    **kwargs,
) -> Dict[str, float]:
    """Train XGBoost and return evaluation metrics.

    Returns
    -------
    dict
        Standard regression metrics (RMSE, R², MAE, MAPE, SMAPE).
    """
    _, y_pred = train_xgboost(X_train, y_train, X_test, **kwargs)
    return compute_all_metrics(y_test, y_pred)
