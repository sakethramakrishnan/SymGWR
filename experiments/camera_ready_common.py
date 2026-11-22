"""Shared helpers for the camera-ready interpolation / continuity experiments.

Nothing here modifies gwr/*.  SymGWR = degree-5 polynomial fit on train+val
(minimum-norm least squares, via gwr.models.fit_polynomial_regression) with
k=1 nearest-reference Taylor extrapolation, all in the split's scaled space.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

import sys
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gwr.models import fit_polynomial_regression  # noqa: E402
from gwr.splits import KINDS, TEMP_COL, composition_ids, get_split  # noqa: E402
from gwr.utils import compute_calphad_metrics  # noqa: E402

SEEDS = list(range(42, 52))
SPLITS = list(KINDS)  # random, temp_band, composition_group
DEGREE = 5
OUT_DIR = "results/camera_ready"


# ----------------------------------------------------------------------
#  Vectorised polynomial value / gradient / Hessian from PolynomialFeatures
# ----------------------------------------------------------------------
class PolyModel:
    """Evaluate P(x), grad P(x), Hess P(x) for a fitted poly pipeline."""

    def __init__(self, pipeline):
        poly = pipeline.named_steps["poly_features"]
        lin = pipeline.named_steps["linear_model"]
        self.pipeline = pipeline
        self.powers = np.asarray(poly.powers_, dtype=np.int64)  # (T, d)
        self.coef = np.asarray(lin.coef_, dtype=np.float64).ravel()
        self.intercept = float(lin.intercept_)
        self.T, self.d = self.powers.shape
        self.deg = int(self.powers.sum(axis=1).max())

    # factor tables: f[n, T, d] = x^p, d1 = p x^(p-1), d2 = p(p-1) x^(p-2)
    def _factors(self, X):
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        pw = X[:, :, None] ** np.arange(self.deg + 1)[None, None, :]  # (n, d, deg+1)
        p = self.powers  # (T, d)
        jj = np.arange(self.d)[None, :]
        f = pw[:, jj, p]  # (n, T, d)
        pm1 = np.maximum(p - 1, 0)
        pm2 = np.maximum(p - 2, 0)
        d1 = p[None] * pw[:, jj, pm1]
        d2 = (p * np.maximum(p - 1, 0))[None] * pw[:, jj, pm2]
        return f, d1, d2

    def value(self, X, batch=400):
        out = []
        for s in range(0, len(X), batch):
            f, _, _ = self._factors(X[s:s + batch])
            out.append(f.prod(axis=2) @ self.coef + self.intercept)
        return np.concatenate(out)

    def grad(self, X, batch=400):
        X = np.asarray(X, dtype=np.float64)
        G = np.empty_like(X)
        for s in range(0, len(X), batch):
            f, d1, _ = self._factors(X[s:s + batch])
            for j in range(self.d):
                t = f.copy()
                t[:, :, j] = d1[:, :, j]
                G[s:s + batch, j] = t.prod(axis=2) @ self.coef
        return G

    def hess(self, X, batch=200):
        X = np.asarray(X, dtype=np.float64)
        H = np.empty((len(X), self.d, self.d))
        for s in range(0, len(X), batch):
            f, d1, d2 = self._factors(X[s:s + batch])
            for j in range(self.d):
                for l in range(j, self.d):
                    t = f.copy()
                    if j == l:
                        t[:, :, j] = d2[:, :, j]
                    else:
                        t[:, :, j] = d1[:, :, j]
                        t[:, :, l] = d1[:, :, l]
                    v = t.prod(axis=2) @ self.coef
                    H[s:s + batch, j, l] = v
                    H[s:s + batch, l, j] = v
        return H


# ----------------------------------------------------------------------
#  SymGWR fit bundle
# ----------------------------------------------------------------------
@dataclass
class Fit:
    split: object
    kind: str
    seed: int
    ref_X: np.ndarray  # train+val, scaled
    ref_y: np.ndarray  # train+val, z-scored target
    ref_idx: np.ndarray  # rows into split.df
    model: PolyModel
    tree: cKDTree
    ref_grad: np.ndarray  # grad P at every reference
    ref_P: np.ndarray  # P at every reference


def fit_symgwr(kind: str, seed: int, degree: int = DEGREE) -> Fit:
    sp = get_split(kind, seed, verbose=False)
    ref_X = np.vstack([sp.X_train, sp.X_val])
    ref_y = np.concatenate([sp.y_train, sp.y_val])
    ref_idx = np.concatenate([sp.train_idx, sp.val_idx])
    pipe, _ = fit_polynomial_regression(ref_X, ref_y, X_test=None, degree=degree)
    pm = PolyModel(pipe)
    return Fit(
        split=sp, kind=kind, seed=seed, ref_X=ref_X, ref_y=ref_y, ref_idx=ref_idx,
        model=pm, tree=cKDTree(ref_X), ref_grad=pm.grad(ref_X), ref_P=pm.value(ref_X),
    )


def predict_symgwr(fit: Fit, Q: np.ndarray, nn=None) -> np.ndarray:
    """Ghat(q) = G_i + grad P(r_i) . (q - r_i), i = nearest reference (k=1)."""
    if nn is None:
        _, nn = fit.tree.query(Q, k=1)
    d = Q - fit.ref_X[nn]
    return fit.ref_y[nn] + np.einsum("ij,ij->i", fit.ref_grad[nn], d)


def predict_residual_anchored(fit: Fit, Q: np.ndarray, nn=None) -> np.ndarray:
    """Ghat(q) = P(q) + (G_i - P(r_i))."""
    if nn is None:
        _, nn = fit.tree.query(Q, k=1)
    return fit.model.value(Q) + (fit.ref_y[nn] - fit.ref_P[nn])


def predict_second_order(fit: Fit, Q: np.ndarray, nn=None) -> np.ndarray:
    """Ghat(q) = G_i + grad P(r_i).d + 0.5 d^T H(r_i) d, d = q - r_i."""
    if nn is None:
        _, nn = fit.tree.query(Q, k=1)
    d = Q - fit.ref_X[nn]
    first = np.einsum("ij,ij->i", fit.ref_grad[nn], d)
    H = fit.model.hess(fit.ref_X[nn])
    second = 0.5 * np.einsum("ni,nij,nj->n", d, H, d)
    return fit.ref_y[nn] + first + second


def metrics_raw(fit: Fit, pred_z: np.ndarray) -> dict:
    sp = fit.split
    return compute_calphad_metrics(sp.y_test, pred_z, sp.target_scaler)


__all__ = [
    "SEEDS", "SPLITS", "DEGREE", "OUT_DIR", "TEMP_COL", "composition_ids",
    "PolyModel", "Fit", "fit_symgwr", "predict_symgwr", "predict_residual_anchored",
    "predict_second_order", "metrics_raw", "get_split", "compute_calphad_metrics",
]
