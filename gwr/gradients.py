"""
gwr.gradients
=============

Symbolic gradient computation via SymPy.

Provides:
* ``symbolic_gradient`` – differentiate a SymPy expression w.r.t. each variable.
* ``evaluate_gradients`` – vectorised numerical evaluation of the gradient at
  an array of reference points.
"""

from __future__ import annotations

import numpy as np
import sympy as sp
from sklearn.pipeline import Pipeline
from typing import List


def symbolic_gradient(
    expression: sp.Expr, variables: List[sp.Symbol]
) -> List[sp.Expr]:
    """Compute the symbolic partial derivatives of *expression* w.r.t. each variable.

    Parameters
    ----------
    expression : sp.Expr
        A SymPy expression (the base model equation).
    variables : list[sp.Symbol]
        Ordered list of SymPy symbols corresponding to features.

    Returns
    -------
    list[sp.Expr]
        Partial derivatives ``[∂f/∂x0, ∂f/∂x1, …]``.
    """
    return [sp.diff(expression, var) for var in variables]


def evaluate_gradients(
    expression: sp.Expr,
    variables: List[sp.Symbol],
    points: np.ndarray,
) -> np.ndarray:
    """Numerically evaluate the gradient of *expression* at every row of *points*.

    This first computes the symbolic gradient, then lambdifies each partial
    derivative for fast NumPy evaluation over the entire array at once.

    Parameters
    ----------
    expression : sp.Expr
        A SymPy expression (the base model equation).
    variables : list[sp.Symbol]
        Ordered symbols matching the columns of *points*.
    points : np.ndarray, shape (n_samples, n_features)
        Array of reference points at which to evaluate.

    Returns
    -------
    gradients : np.ndarray, shape (n_samples, n_features)
        Each row is the gradient vector at the corresponding reference point.
    """
    gradient_exprs = symbolic_gradient(expression, variables)
    n_samples = points.shape[0]

    gradient_funcs = []
    for g in gradient_exprs:
        if isinstance(g, (sp.Float, sp.Integer, int, float)):
            # Constant partial derivative
            gradient_funcs.append(
                lambda *args, val=float(g): np.full(n_samples, val)
            )
        else:
            gradient_funcs.append(
                sp.lambdify(variables, g, modules="numpy")
            )

    # Evaluate each partial derivative over all points (columns = features)
    all_gradients = np.column_stack(
        [gf(*points.T) for gf in gradient_funcs]
    )

    return np.asarray(all_gradients, dtype=np.float64)


def evaluate_polynomial_pipeline_gradients(
    pipeline: Pipeline,
    points: np.ndarray,
) -> np.ndarray:
    """Evaluate polynomial-regression gradients without SymPy.

    Uses explicit monomial derivatives for modest term counts and switches to
    batched central-difference numeric gradients when the expanded basis is
    large (degree 10+ on 8 features), keeping both RAM and wall time bounded.
    """
    poly = pipeline.named_steps["poly_features"]
    n_terms = poly.n_output_features_
    if n_terms > 50_000:
        return _evaluate_polynomial_pipeline_gradients_numeric(pipeline, points)
    return _evaluate_polynomial_pipeline_gradients_explicit(pipeline, points)


def _evaluate_polynomial_pipeline_gradients_explicit(
    pipeline: Pipeline,
    points: np.ndarray,
) -> np.ndarray:
    poly = pipeline.named_steps["poly_features"]
    linear = pipeline.named_steps["linear_model"]
    coefs = np.asarray(linear.coef_, dtype=np.float64)
    powers = poly.powers_
    points = np.asarray(points, dtype=np.float64)
    n_samples, n_features = points.shape
    gradients = np.zeros((n_samples, n_features), dtype=np.float64)

    for coef, power in zip(coefs, powers):
        if coef == 0.0:
            continue
        for j in range(n_features):
            pj = power[j]
            if pj == 0:
                continue
            col = np.full(n_samples, coef * pj, dtype=np.float64)
            for k in range(n_features):
                pk = power[k]
                if pk == 0:
                    continue
                if k == j:
                    if pk > 1:
                        col *= points[:, k] ** (pk - 1)
                else:
                    col *= points[:, k] ** pk
            gradients[:, j] += col

    return gradients


def _predict_in_batches(pipeline: Pipeline, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    preds = np.empty(X.shape[0], dtype=np.float64)
    for start in range(0, X.shape[0], batch_size):
        stop = start + batch_size
        preds[start:stop] = pipeline.predict(X[start:stop])
    return preds


def _evaluate_polynomial_pipeline_gradients_numeric(
    pipeline: Pipeline,
    points: np.ndarray,
    step: float = 1e-5,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    n_samples, n_features = points.shape
    gradients = np.zeros((n_samples, n_features), dtype=np.float64)
    for j in range(n_features):
        perturbed_plus = points.copy()
        perturbed_minus = points.copy()
        perturbed_plus[:, j] += step
        perturbed_minus[:, j] -= step
        gradients[:, j] = (
            _predict_in_batches(pipeline, perturbed_plus)
            - _predict_in_batches(pipeline, perturbed_minus)
        ) / (2.0 * step)
    return gradients
