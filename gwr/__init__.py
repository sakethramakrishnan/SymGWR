"""
Gradient-Weighted Regression (SymGWR)
===================================

A local regression method that uses analytical gradient information from
closed-form base models to improve predictions via directional derivative
extrapolation from nearest neighbours.

Modules
-------
core
    Main ``GradientWeightedRegressor`` class.
neighbors
    Absolute and relative (class-stratified) neighbour selection.
gradients
    Symbolic gradient computation via SymPy.
models
    Polynomial, linear, and ridge regression wrappers that expose
    their fitted equations as SymPy expressions.
utils
    Data loading, scaling, metrics, and miscellaneous helpers.
"""

from gwr.core import GradientWeightedRegressor
from gwr.neighbors import AbsoluteNeighborSelector, RelativeNeighborSelector
from gwr.gradients import symbolic_gradient, evaluate_gradients
from gwr.models import (
    fit_linear_regression,
    fit_ridge_regression,
    fit_polynomial_regression,
    model_to_sympy_equation,
)

__version__ = "0.1.0"

__all__ = [
    "GradientWeightedRegressor",
    "AbsoluteNeighborSelector",
    "RelativeNeighborSelector",
    "symbolic_gradient",
    "evaluate_gradients",
    "fit_linear_regression",
    "fit_ridge_regression",
    "fit_polynomial_regression",
    "model_to_sympy_equation",
]
