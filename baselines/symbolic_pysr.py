"""
baselines.symbolic_pysr
=======================

PySR (symbolic regression) baseline for comparison against GWR.

PySR discovers interpretable mathematical expressions from data using
evolutionary search.  This module wraps the ``PySRRegressor`` with
sensible defaults and a consistent evaluation interface.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple

from gwr.utils import compute_all_metrics


def train_pysr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    niterations: int = 100,
    binary_operators: Optional[list] = None,
    unary_operators: Optional[list] = None,
    populations: int = 31,
    population_size: int = 27,
    ncycles_per_iteration: int = 380,
    random_state: int = 42,
    **kwargs,
) -> Tuple:
    """Train a PySR symbolic regression model.

    Parameters
    ----------
    X_train, y_train : array-like
    X_test : array-like, optional
    niterations : int
    binary_operators : list[str], optional
    unary_operators : list[str], optional
    populations : int
    population_size : int
    ncycles_per_iteration : int
    random_state : int

    Returns
    -------
    model  : PySRRegressor
    y_pred : np.ndarray or None
    """
    from pysr import PySRRegressor

    if binary_operators is None:
        binary_operators = ["+", "-", "*", "/", "^"]

    model = PySRRegressor(
        niterations=niterations,
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        populations=populations,
        population_size=population_size,
        ncycles_per_iteration=ncycles_per_iteration,
        random_state=random_state,
        **kwargs,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test) if X_test is not None else None
    return model, y_pred


def evaluate_pysr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    **kwargs,
) -> Dict[str, float]:
    """Train PySR and return evaluation metrics."""
    _, y_pred = train_pysr(X_train, y_train, X_test, **kwargs)
    return compute_all_metrics(y_test, y_pred)
