"""
gwr.core
========

Main Gradient-Weighted Regression (SymGWR) implementation.

The ``GradientWeightedRegressor`` takes a closed-form base model (expressed as
a SymPy equation), computes its analytical gradient at every reference point,
and predicts new targets by extrapolating from the *k* nearest neighbours
using first-order directional derivatives.
"""

from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.spatial import cKDTree
from typing import List, Optional

from sklearn.pipeline import Pipeline

from gwr.gradients import (
    evaluate_gradients,
    evaluate_polynomial_pipeline_gradients,
    symbolic_gradient,
)
from gwr.neighbors import AbsoluteNeighborSelector, RelativeNeighborSelector


class GradientWeightedRegressor:
    """Gradient-Weighted Regression (SymGWR) predictor.

    Parameters
    ----------
    sympy_equation : sp.Expr
        The symbolic expression of the base model (e.g. a polynomial).
    x_symbols : list[sp.Symbol]
        The ordered list of SymPy symbols that correspond to feature columns.
    n_neighbors : int, default=5
        Number of nearest neighbours to use for prediction.
    relative : bool, default=False
        If ``True``, select *n_neighbors* per class/group (stratified).
        If ``False``, select *n_neighbors* globally (absolute).
    weighted : bool, default=True
        Weight neighbour contributions by inverse distance.
    """

    def __init__(
        self,
        sympy_equation: sp.Expr,
        x_symbols: List[sp.Symbol],
        n_neighbors: int = 5,
        relative: bool = False,
        weighted: bool = True,
    ):
        self.sympy_equation = sympy_equation
        self.x_symbols = x_symbols
        self.n_neighbors = n_neighbors
        self.relative = relative
        self.weighted = weighted

        # Computed during fit
        self._ref_points: Optional[np.ndarray] = None
        self._ref_targets: Optional[np.ndarray] = None
        self._all_gradients: Optional[np.ndarray] = None
        self._kdtree: Optional[cKDTree] = None
        self._class_kdtrees: Optional[dict] = None
        self._ref_classes: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classes: Optional[np.ndarray] = None,
    ) -> "GradientWeightedRegressor":
        """Store reference data and pre-compute gradients and KD-trees.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Reference feature matrix.
        y : np.ndarray, shape (n_samples,)
            Reference target values.
        classes : np.ndarray, optional, shape (n_samples,)
            Group labels for relative neighbour selection.

        Returns
        -------
        self
        """
        self._ref_points = np.asarray(X, dtype=np.float64)
        self._ref_targets = np.asarray(y, dtype=np.float64)

        # Compute analytical gradients at every reference point
        self._all_gradients = evaluate_gradients(
            self.sympy_equation, self.x_symbols, self._ref_points
        )

        # Build KD-tree(s)
        if self.relative and classes is not None:
            self._ref_classes = np.asarray(classes)
            self._class_kdtrees = {}
            self._class_indices = {}
            for cls in np.unique(self._ref_classes):
                mask = self._ref_classes == cls
                self._class_indices[cls] = np.where(mask)[0]
                self._class_kdtrees[cls] = cKDTree(self._ref_points[mask])
        else:
            self._kdtree = cKDTree(self._ref_points)

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for query points.

        Parameters
        ----------
        X : np.ndarray, shape (n_query, n_features)
            Query feature matrix.

        Returns
        -------
        predictions : np.ndarray, shape (n_query,)
        """
        X = np.asarray(X, dtype=np.float64)

        _, closest_pts, grads, targets, weights = self._predict_components(X)
        predictions = self._directional_derivative_predict(
            X, closest_pts, grads, targets, weights
        )
        return predictions

    def predict_with_local_gradient(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict targets and return approximate local gradient.

        Local gradient is computed as the normalized weighted average of
        neighbour gradients used for prediction.
        """
        X = np.asarray(X, dtype=np.float64)
        distances, closest_pts, grads, targets, weights = self._predict_components(X)
        predictions = self._directional_derivative_predict(
            X, closest_pts, grads, targets, weights
        )
        local_grad = np.sum(grads * weights[:, :, np.newaxis], axis=1)
        return predictions, local_grad

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _predict_components(self, X: np.ndarray):
        if self.relative and self._class_kdtrees is not None:
            distances, closest_pts, grads, targets = self._query_relative_neighbors(X)
        else:
            distances, closest_pts, grads, targets = self._query_absolute_neighbors(X)
        weights = self._calculate_weights(distances)
        return distances, closest_pts, grads, targets, weights

    def _query_absolute_neighbors(self, query_points: np.ndarray):
        k = self.n_neighbors
        distances, indices = self._kdtree.query(query_points, k=k)
        if k == 1:
            distances = distances[:, np.newaxis]
            indices = indices[:, np.newaxis]
        closest_pts = self._ref_points[indices]
        grads = self._all_gradients[indices]
        targets = self._ref_targets[indices]
        return distances, closest_pts, grads, targets

    def _query_relative_neighbors(self, query_points: np.ndarray):
        k = self.n_neighbors
        all_dists, all_pts, all_grads, all_targets = [], [], [], []

        for q in query_points:
            d_list, p_list, g_list, t_list = [], [], [], []
            for cls, tree in self._class_kdtrees.items():
                idx_map = self._class_indices[cls]
                dists, local_idx = tree.query(q, k=k)
                if k == 1:
                    dists = np.array([dists])
                    local_idx = np.array([local_idx])
                global_idx = idx_map[local_idx]
                d_list.append(dists)
                p_list.append(self._ref_points[global_idx])
                g_list.append(self._all_gradients[global_idx])
                t_list.append(self._ref_targets[global_idx])
            all_dists.append(np.concatenate(d_list))
            all_pts.append(np.vstack(p_list))
            all_grads.append(np.vstack(g_list))
            all_targets.append(np.concatenate(t_list))

        return (
            np.array(all_dists),
            np.array(all_pts),
            np.array(all_grads),
            np.array(all_targets),
        )

    @staticmethod
    def _calculate_weights(distances: np.ndarray) -> np.ndarray:
        epsilon = 1e-6
        weights = 1.0 / (distances + epsilon)
        denom = weights.sum(axis=1, keepdims=True)
        # Guard against pathological all-inf/all-zero rows.
        denom = np.where(denom <= 0, 1.0, denom)
        weights = weights / denom
        return weights

    def _directional_derivative_predict(
        self,
        query_points: np.ndarray,
        closest_features: np.ndarray,
        gradients: np.ndarray,
        targets: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """Vectorised SymGWR prediction via first-order Taylor extrapolation."""
        # directions: (n_query, k, n_features)
        directions = query_points[:, np.newaxis, :] - closest_features

        # Dot product along feature axis → change in target
        directions_flat = directions.reshape(-1, directions.shape[-1])
        gradients_flat = gradients.reshape(-1, gradients.shape[-1])
        change_flat = np.einsum("ij,ij->i", directions_flat, gradients_flat)
        change = change_flat.reshape(directions.shape[0], directions.shape[1])

        pred_per_neighbor = targets + change

        if self.weighted:
            predictions = np.sum(pred_per_neighbor * weights, axis=1)
        else:
            predictions = pred_per_neighbor.mean(axis=1)

        return predictions

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"GradientWeightedRegressor("
            f"n_neighbors={self.n_neighbors}, "
            f"relative={self.relative}, "
            f"weighted={self.weighted})"
        )


class PipelineGradientWeightedRegressor:
    """SymGWR using a fitted sklearn polynomial pipeline (no SymPy).

    Same neighbour / Taylor extrapolation logic as ``GradientWeightedRegressor``,
    but gradients come from ``evaluate_polynomial_pipeline_gradients`` so
    high-degree sweeps stay memory-bounded.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        n_neighbors: int = 5,
        relative: bool = False,
        weighted: bool = True,
    ):
        self.pipeline = pipeline
        self.n_neighbors = n_neighbors
        self.relative = relative
        self.weighted = weighted
        self._ref_points: Optional[np.ndarray] = None
        self._ref_targets: Optional[np.ndarray] = None
        self._all_gradients: Optional[np.ndarray] = None
        self._kdtree: Optional[cKDTree] = None
        self._class_kdtrees: Optional[dict] = None
        self._ref_classes: Optional[np.ndarray] = None
        self._class_indices: Optional[dict] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classes: Optional[np.ndarray] = None,
    ) -> "PipelineGradientWeightedRegressor":
        self._ref_points = np.asarray(X, dtype=np.float64)
        self._ref_targets = np.asarray(y, dtype=np.float64)
        self._all_gradients = evaluate_polynomial_pipeline_gradients(
            self.pipeline, self._ref_points
        )

        if self.relative and classes is not None:
            self._ref_classes = np.asarray(classes)
            self._class_kdtrees = {}
            self._class_indices = {}
            for cls in np.unique(self._ref_classes):
                mask = self._ref_classes == cls
                self._class_indices[cls] = np.where(mask)[0]
                self._class_kdtrees[cls] = cKDTree(self._ref_points[mask])
        else:
            self._kdtree = cKDTree(self._ref_points)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        _, closest_pts, grads, targets, weights = self._predict_components(X)
        return GradientWeightedRegressor._directional_derivative_predict(
            self, X, closest_pts, grads, targets, weights
        )

    def _predict_components(self, X: np.ndarray):
        if self.relative and self._class_kdtrees is not None:
            distances, closest_pts, grads, targets = self._query_relative_neighbors(X)
        else:
            distances, closest_pts, grads, targets = self._query_absolute_neighbors(X)
        weights = GradientWeightedRegressor._calculate_weights(distances)
        return distances, closest_pts, grads, targets, weights

    def _query_absolute_neighbors(self, query_points: np.ndarray):
        k = self.n_neighbors
        distances, indices = self._kdtree.query(query_points, k=k)
        if k == 1:
            distances = distances[:, np.newaxis]
            indices = indices[:, np.newaxis]
        closest_pts = self._ref_points[indices]
        grads = self._all_gradients[indices]
        targets = self._ref_targets[indices]
        return distances, closest_pts, grads, targets

    def _query_relative_neighbors(self, query_points: np.ndarray):
        k = self.n_neighbors
        all_dists, all_pts, all_grads, all_targets = [], [], [], []
        for q in query_points:
            d_list, p_list, g_list, t_list = [], [], [], []
            for cls, tree in self._class_kdtrees.items():
                idx_map = self._class_indices[cls]
                dists, local_idx = tree.query(q, k=k)
                if k == 1:
                    dists = np.array([dists])
                    local_idx = np.array([local_idx])
                global_idx = idx_map[local_idx]
                d_list.append(dists)
                p_list.append(self._ref_points[global_idx])
                g_list.append(self._all_gradients[global_idx])
                t_list.append(self._ref_targets[global_idx])
            all_dists.append(np.concatenate(d_list))
            all_pts.append(np.vstack(p_list))
            all_grads.append(np.vstack(g_list))
            all_targets.append(np.concatenate(t_list))
        return (
            np.array(all_dists),
            np.array(all_pts),
            np.array(all_grads),
            np.array(all_targets),
        )

    def __repr__(self) -> str:
        return (
            f"PipelineGradientWeightedRegressor("
            f"n_neighbors={self.n_neighbors}, "
            f"relative={self.relative}, "
            f"weighted={self.weighted})"
        )
