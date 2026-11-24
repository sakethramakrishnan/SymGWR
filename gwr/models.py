"""
gwr.models
==========

Wrappers around scikit-learn regression models that additionally expose
their fitted equations as SymPy symbolic expressions.

This is essential for SymGWR because the method requires the *analytical
gradient* of the base model, which is obtained by differentiating the
SymPy equation.

Supported base models
---------------------
* Linear Regression
* Ridge Regression
* Polynomial Regression (arbitrary degree)
* Lasso Regression
"""

from __future__ import annotations

import os

import numpy as np
import sympy as sp
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from typing import List, Optional, Tuple


# ======================================================================
#  Training helpers
# ======================================================================

def fit_linear_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
) -> Tuple[LinearRegression, Optional[np.ndarray]]:
    """Fit an ordinary least-squares linear regression.

    Returns
    -------
    model : LinearRegression
    y_pred : np.ndarray or None
        Predictions on *X_test* if provided, else ``None``.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test) if X_test is not None else None
    return model, y_pred


def fit_ridge_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    alpha: float = 1.0,
) -> Tuple[Ridge, Optional[np.ndarray]]:
    """Fit a Ridge regression.

    Returns
    -------
    model : Ridge
    y_pred : np.ndarray or None
    """
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test) if X_test is not None else None
    return model, y_pred


from typing import List, Optional, Tuple


_DENSE_POLY_MAX_BYTES = int(1.2e9)


def _dual_row_block(n_terms: int) -> int:
    """Block size for streaming poly transforms; caps peak RAM."""
    bytes_per_row = max(8 * n_terms, 1)
    return max(4, min(32, _DENSE_POLY_MAX_BYTES // bytes_per_row))


def _fit_polynomial_dual(
    X_train: np.ndarray,
    y_train: np.ndarray,
    poly: PolynomialFeatures,
) -> LinearRegression:
    """Kernel OLS matching sklearn centering + intercept; block-built."""
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    n_samples = X_train.shape[0]
    n_terms = poly.n_output_features_
    row_block = _dual_row_block(n_terms)

    phi_mean = np.zeros(n_terms, dtype=np.float64)
    for start in range(0, n_samples, row_block):
        stop = min(n_samples, start + row_block)
        phi_mean += poly.transform(X_train[start:stop]).sum(axis=0)
    phi_mean /= n_samples
    y_mean = float(y_train.mean())
    y_centered = y_train - y_mean

    kernel = np.zeros((n_samples, n_samples), dtype=np.float64)
    for i0 in range(0, n_samples, row_block):
        i1 = min(n_samples, i0 + row_block)
        Xi = poly.transform(X_train[i0:i1]) - phi_mean
        for j0 in range(i0, n_samples, row_block):
            j1 = min(n_samples, j0 + row_block)
            if j0 == i0:
                Xj = Xi
            else:
                Xj = poly.transform(X_train[j0:j1]) - phi_mean
            block = Xi @ Xj.T
            kernel[i0:i1, j0:j1] = block
            if j0 != i0:
                kernel[j0:j1, i0:i1] = block.T
        del Xi

    dual_coef, _, _, _ = np.linalg.lstsq(kernel, y_centered, rcond=None)

    coef = np.zeros(n_terms, dtype=np.float64)
    for start in range(0, n_samples, row_block):
        stop = min(n_samples, start + row_block)
        block = poly.transform(X_train[start:stop]) - phi_mean
        coef += block.T @ dual_coef[start:stop]

    intercept = y_mean - float(phi_mean @ coef)
    linear = LinearRegression()
    linear.coef_ = coef
    linear.intercept_ = intercept
    linear.n_features_in_ = n_terms
    return linear


def fit_polynomial_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    degree: int = 3,
    memory_safe: bool = True,
) -> Tuple[Pipeline, Optional[np.ndarray]]:
    """Fit a polynomial regression of the given degree.

    The returned model is a scikit-learn ``Pipeline`` containing
    ``PolynomialFeatures`` followed by ``LinearRegression``.

    When the expanded design matrix would exceed ~1.2GB, uses a blocked
    kernel (dual) solver instead of materializing ``poly.transform(X)``.

    Returns
    -------
    model : Pipeline
    y_pred : np.ndarray or None
    """
    poly = PolynomialFeatures(degree=degree)
    poly.fit(X_train)
    n_samples = X_train.shape[0]
    n_terms = poly.n_output_features_
    dense_bytes = n_samples * n_terms * 8

    if memory_safe and dense_bytes > _DENSE_POLY_MAX_BYTES:
        linear = _fit_polynomial_dual(X_train, y_train, poly)
        model = Pipeline(
            [
                ("poly_features", poly),
                ("linear_model", linear),
            ]
        )
    else:
        model = Pipeline(
            [
                ("poly_features", PolynomialFeatures(degree=degree)),
                ("linear_model", LinearRegression()),
            ]
        )
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test) if X_test is not None else None
    return model, y_pred


def fit_lasso_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    alpha: float = 1.0,
) -> Tuple[Lasso, Optional[np.ndarray]]:
    """Fit a Lasso regression.

    Returns
    -------
    model : Lasso
    y_pred : np.ndarray or None
    """
    model = Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test) if X_test is not None else None
    return model, y_pred


# ======================================================================
#  Symbolic conversion
# ======================================================================

def model_to_sympy_equation(
    model,
    feature_names: Optional[List[str]] = None,
) -> Tuple[sp.Expr, List[sp.Symbol]]:
    """Convert a fitted sklearn model to a SymPy symbolic expression.

    Supports ``LinearRegression``, ``Ridge``, ``Lasso``, and ``Pipeline``
    objects that include ``PolynomialFeatures``.

    Parameters
    ----------
    model : sklearn estimator or Pipeline
    feature_names : list[str], optional
        Names like ``['x0', 'x1', …]``.  Inferred if not provided.

    Returns
    -------
    expression : sp.Expr
        Symbolic equation of the fitted model.
    x_symbols  : list[sp.Symbol]
        Ordered list of SymPy symbols for the original features.
    """
    # Unpack Pipeline if necessary
    if isinstance(model, Pipeline):
        poly_transformer = model.named_steps.get("poly_features", None)
        linear_model = model.named_steps.get("linear_model", model)
    else:
        poly_transformer = None
        linear_model = model

    # Determine feature names
    if feature_names is None:
        if poly_transformer is not None and hasattr(poly_transformer, "n_features_in_"):
            feature_names = [f"x{i}" for i in range(poly_transformer.n_features_in_)]
        elif hasattr(linear_model, "coef_"):
            feature_names = [f"x{i}" for i in range(linear_model.coef_.shape[0])]
        else:
            raise ValueError("Cannot determine feature count; pass feature_names explicitly.")

    x_symbols = sp.symbols(feature_names)

    # Build transformed-feature symbols
    if poly_transformer is not None:
        transformed_names = poly_transformer.get_feature_names_out(feature_names)
    else:
        transformed_names = feature_names

    transformed_symbols = []
    for name in transformed_names:
        if name == "1":
            transformed_symbols.append(sp.Integer(1))
        else:
            factors = []
            for term in name.split(" "):
                term = term.strip()
                if not term:
                    continue
                if "^" in term:
                    var, exp_ = term.split("^")
                    factors.append(sp.symbols(var) ** int(exp_))
                else:
                    factors.append(sp.symbols(term))
            transformed_symbols.append(
                sp.Mul(*factors) if len(factors) > 1 else factors[0]
            )

    coefficients = linear_model.coef_
    intercept = linear_model.intercept_

    expression = intercept
    for coeff, sym in zip(coefficients, transformed_symbols):
        expression += coeff * sym

    return expression, list(x_symbols) if not isinstance(x_symbols, list) else x_symbols


def parse_equation_string(equation_str: str) -> Tuple[sp.Expr, List[sp.Symbol]]:
    """Parse a string equation (e.g. ``'1.5*x0 + 2.3*x1 - 0.7'``) into SymPy.

    Returns
    -------
    expression : sp.Expr
    variables  : list[sp.Symbol]
    """
    import re

    var_names = sorted(set(re.findall(r"x\d+", equation_str)))
    variables = sp.symbols(var_names)
    local_dict = {str(v): v for v in variables}
    expression = sp.sympify(equation_str, locals=local_dict)
    return expression, list(variables)
