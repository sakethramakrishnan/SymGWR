"""
gwr.neighbors
=============

Neighbour selection strategies for GWR.

* **AbsoluteNeighborSelector** – picks the *k* globally closest points.
* **RelativeNeighborSelector** – picks *k* closest points *per class/group*
  (e.g., per temperature class), then concatenates them.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from typing import Dict, Optional, Tuple


class AbsoluteNeighborSelector:
    """Select the *k* nearest neighbours globally using a single KD-tree.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbours to return.
    """

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self._tree: Optional[cKDTree] = None
        self._ref_points: Optional[np.ndarray] = None

    def fit(self, ref_points: np.ndarray) -> "AbsoluteNeighborSelector":
        """Build the KD-tree from reference points.

        Parameters
        ----------
        ref_points : np.ndarray, shape (n_ref, n_features)
        """
        self._ref_points = np.asarray(ref_points, dtype=np.float64)
        self._tree = cKDTree(self._ref_points)
        return self

    def query(
        self, query_points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return distances and indices of the *k* nearest neighbours.

        Parameters
        ----------
        query_points : np.ndarray, shape (n_query, n_features)

        Returns
        -------
        distances : np.ndarray, shape (n_query, k)
        indices   : np.ndarray, shape (n_query, k)
        """
        k = self.n_neighbors
        distances, indices = self._tree.query(query_points, k=k)
        if k == 1:
            distances = distances[:, np.newaxis]
            indices = indices[:, np.newaxis]
        return distances, indices


class RelativeNeighborSelector:
    """Select *k* nearest neighbours **per class**, then concatenate.

    This is useful when reference points belong to distinct groups
    (e.g., temperature classes) and you want balanced representation from
    each group in the neighbour set.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbours to return *per class*.
    """

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self._trees: Dict[str, cKDTree] = {}
        self._class_indices: Dict[str, np.ndarray] = {}
        self._ref_points: Optional[np.ndarray] = None

    def fit(
        self, ref_points: np.ndarray, classes: np.ndarray
    ) -> "RelativeNeighborSelector":
        """Build one KD-tree per unique class.

        Parameters
        ----------
        ref_points : np.ndarray, shape (n_ref, n_features)
        classes    : np.ndarray, shape (n_ref,)
        """
        self._ref_points = np.asarray(ref_points, dtype=np.float64)
        classes = np.asarray(classes)
        for cls in np.unique(classes):
            mask = classes == cls
            self._class_indices[cls] = np.where(mask)[0]
            self._trees[cls] = cKDTree(self._ref_points[mask])
        return self

    def query(
        self, query_points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return distances and *global* indices of neighbours across all classes.

        Parameters
        ----------
        query_points : np.ndarray, shape (n_query, n_features)

        Returns
        -------
        distances : np.ndarray, shape (n_query, k * n_classes)
        indices   : np.ndarray, shape (n_query, k * n_classes)
        """
        k = self.n_neighbors
        all_dists, all_indices = [], []

        for q in query_points:
            d_parts, i_parts = [], []
            for cls, tree in self._trees.items():
                dists, local_idx = tree.query(q, k=k)
                if k == 1:
                    dists = np.array([dists])
                    local_idx = np.array([local_idx])
                global_idx = self._class_indices[cls][local_idx]
                d_parts.append(dists)
                i_parts.append(global_idx)
            all_dists.append(np.concatenate(d_parts))
            all_indices.append(np.concatenate(i_parts))

        return np.array(all_dists), np.array(all_indices)
