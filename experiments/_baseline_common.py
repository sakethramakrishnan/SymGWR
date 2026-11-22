"""Shared helpers for the camera-ready baseline harness.

Copied (not imported) from ``run_locked_baseline_refresh.py`` and
``run_classical_baseline_comparison_n10.py``, whose module-level imports are
broken or pull in unrelated machinery.  Changes vs. the originals:

* one model registry + one PARAM_GRID dict covering every baseline;
* ``random_state`` of stochastic models is set to the run seed (the originals
  hard-coded 42 for every seed);
* tuning is by validation MSE on z-scored targets, as before;
* the 2,500-row training cap for GP / kernel regression / LOESS is kept and
  reported (``train_cap`` / ``n_train_used``);
* PINN: the library's physics loss penalises dG/d(feature 0).  In this dataset
  feature 0 is Mass_fraction_Nb and temperature is the LAST column, so the
  temperature column is moved to position 0 before calling ``train_pinn``.

Nothing here reads or writes any cache.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
from sklearn.tree import DecisionTreeRegressor

TRAIN_CAP = 2500
GP_TUNE_ROWS = 1000  # GP: tune on a train subsample this size, then refit once
CAPPED_MODELS = {"GP", "Kernel regression", "LOESS"}
SYMGWR_DEGREE = 5
SYMGWR_K = 1

# Display order within a split (PySR last, alone).
MODEL_ORDER = [
    "SymGWR", "Linear", "Ridge", "Polynomial", "k-NN", "Decision tree",
    "XGBoost", "MLP", "Kernel regression", "LOESS", "PINN", "GP", "PySR",
]

# Grids.  ``{}`` entry = single fixed config (nothing to tune).
PARAM_GRID: dict[str, list[dict[str, Any]]] = {
    "Linear": [{}],
    "Ridge": [{"alpha": a} for a in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)],
    "Polynomial": [{"degree": d} for d in (2, 3, 4, 5)],
    # --- copied from run_classical_baseline_comparison_n10.BASELINE_PARAM_GRID
    "GP": [{"alpha": 1e-6}, {"alpha": 1e-4}],
    "LOESS": [{"frac": 0.2}, {"frac": 0.3}, {"frac": 0.4}],
    "k-NN": [{"n_neighbors": 3}, {"n_neighbors": 5}, {"n_neighbors": 10}],
    "Kernel regression": [{"bandwidth": 0.5}, {"bandwidth": 1.0}, {"bandwidth": 2.0}],
    # --- copied from run_locked_baseline_refresh.PARAM_GRID
    "XGBoost": [
        {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05},
        {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.1},
    ],
    "MLP": [
        {"hidden_layer_sizes": (256, 128, 64), "max_iter": 250},
        {"hidden_layer_sizes": (512, 256, 128), "max_iter": 300},
    ],
    "Decision tree": [
        {"max_depth": None, "min_samples_leaf": 1},
        {"max_depth": 20, "min_samples_leaf": 2},
        {"max_depth": 40, "min_samples_leaf": 1},
    ],
    "PySR": [
        {"niterations": 50, "population_size": 20, "populations": 10},
    ],
    "PINN": [
        {"hidden_sizes": (128, 128, 64), "epochs": 400, "lambda_physics": 0.1},
        {"hidden_sizes": (256, 128, 64), "epochs": 500, "lambda_physics": 0.1},
    ],
}

# Free-text protocol notes written into every row (and the summary).
_CAP = f"train rows capped at {TRAIN_CAP} (random subsample, seed-dependent); val rows added on refit"
MODEL_NOTES: dict[str, str] = {
    "PINN": ("temperature column moved to index 0 before train_pinn: the library physics loss "
             "penalises dG/d(feature 0), and feature 0 in the data is Mass_fraction_Nb"),
    "GP": (f"GP protocol: alpha tuned on a {GP_TUNE_ROWS}-row train subsample (val MSE), then ONE "
           f"refit on train (capped {TRAIN_CAP}) + val; 25 min timeout"),
    "Kernel regression": _CAP,
    "LOESS": _CAP,
}

# Models whose constructor takes a random_state that should follow the run seed.
_SEEDED = {"GP", "XGBoost", "MLP", "Decision tree", "PySR", "PINN"}


def _load_train_fn(model_name: str):
    """Import lazily so a child process only pays for what it uses."""
    if model_name == "GP":
        from baselines.gaussian_process import train_gaussian_process as f
    elif model_name == "LOESS":
        from baselines.loess import train_loess as f
    elif model_name == "k-NN":
        from baselines.knn import train_knn as f
    elif model_name == "Kernel regression":
        from baselines.kernel_regression import train_kernel_regression as f
    elif model_name == "XGBoost":
        from baselines.xgboost import train_xgboost as f
    elif model_name == "MLP":
        from baselines.mlp import train_mlp as f
    elif model_name == "PINN":
        from baselines.pinn import train_pinn as f
    elif model_name == "PySR":
        from baselines.symbolic_pysr import train_pysr as f
    else:
        raise KeyError(model_name)
    return f


def _extract_prediction(result: Any) -> np.ndarray:
    if isinstance(result, tuple):
        if len(result) >= 2 and result[1] is not None:
            return np.asarray(result[1], dtype=np.float64)
        raise RuntimeError("Baseline returned tuple without prediction array.")
    return np.asarray(result, dtype=np.float64)


def _with_seed(model_name: str, params: dict[str, Any], seed: int) -> dict[str, Any]:
    out = dict(params)
    if model_name in _SEEDED:
        out["random_state"] = seed
    return out


def predict_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    params: dict[str, Any],
    seed: int,
    temp_col: int,
) -> np.ndarray:
    """Fit ``model_name`` with ``params`` on (X_train, y_train); predict X_eval."""
    p = _with_seed(model_name, params, seed)
    if model_name == "Linear":
        from gwr.models import fit_linear_regression
        return np.asarray(fit_linear_regression(X_train, y_train, X_eval)[1], dtype=np.float64)
    if model_name == "Ridge":
        from gwr.models import fit_ridge_regression
        return np.asarray(fit_ridge_regression(X_train, y_train, X_eval, **p)[1], dtype=np.float64)
    if model_name == "Polynomial":
        # min-norm least squares (sklearn LinearRegression / dual lstsq);
        # the degree-5 design is rank-deficient, so only that solution is defined.
        from gwr.models import fit_polynomial_regression
        return np.asarray(
            fit_polynomial_regression(X_train, y_train, X_eval, **p)[1], dtype=np.float64
        )
    if model_name == "Decision tree":
        m = DecisionTreeRegressor(**p)
        m.fit(X_train, y_train)
        return m.predict(X_eval)
    if model_name == "PINN":
        order = [temp_col] + [i for i in range(X_train.shape[1]) if i != temp_col]
        X_train, X_eval = X_train[:, order], X_eval[:, order]
    if model_name == "PySR":
        p.setdefault("verbosity", 0)
        p.setdefault("progress", False)
        p.setdefault("temp_equation_file", True)  # no hall-of-fame files in cwd
    fn = _load_train_fn(model_name)
    return _extract_prediction(fn(X_train, y_train, X_eval, **p))


def select_hparams(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    temp_col: int,
) -> dict[str, Any]:
    """Pick the grid entry with the lowest validation MSE (z-scored targets)."""
    grid = PARAM_GRID[model_name]
    if len(grid) == 1:
        return grid[0]
    best, best_mse = grid[0], float("inf")
    for params in grid:
        pred = predict_model(model_name, X_train, y_train, X_val, params, seed, temp_col)
        mse = float(np.mean((y_val - pred) ** 2))
        if not np.isfinite(mse):
            continue
        if mse < best_mse:
            best, best_mse = params, mse
    return best


def subsample_train(
    model_name: str, X: np.ndarray, y: np.ndarray, seed: int, cap: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Existing cap: only GP / kernel regression / LOESS, train rows only."""
    cap = TRAIN_CAP if cap is None else cap
    if model_name not in CAPPED_MODELS or len(X) <= cap:
        return X, y
    idx = np.random.default_rng(seed).choice(len(X), size=cap, replace=False)
    return X[idx], y[idx]


def fit_predict_symgwr(
    X_fit: np.ndarray, y_fit: np.ndarray, X_test: np.ndarray
) -> np.ndarray:
    """SymGWR, fixed config (degree 5, k=1, absolute neighbours), no tuning."""
    from gwr.core import GradientWeightedRegressor
    from gwr.models import fit_polynomial_regression, model_to_sympy_equation

    base, _ = fit_polynomial_regression(X_fit, y_fit, X_test=None, degree=SYMGWR_DEGREE)
    eq, syms = model_to_sympy_equation(base)
    gwr = GradientWeightedRegressor(
        sympy_equation=eq, x_symbols=syms, n_neighbors=SYMGWR_K, relative=False
    )
    gwr.fit(X_fit, y_fit)
    return np.asarray(gwr.predict(X_test), dtype=np.float64)


def run_one(model_name: str, split: Any, seed: int) -> dict[str, Any]:
    """Tune on val, refit on train+val, predict test once.  Returns a result dict.

    ``split`` is a ``gwr.splits.Split``.  All arrays are z-scored with
    train-only scalers (``split.target_scaler`` maps predictions back).
    """
    from gwr.utils import compute_calphad_metrics

    temp_col = split.feature_cols.index("Temperature_[K]")
    t0 = time.perf_counter()
    n_avail = len(split.X_train) + len(split.X_val)

    if model_name == "SymGWR":
        X_fit = np.vstack([split.X_train, split.X_val])
        y_fit = np.concatenate([split.y_train, split.y_val])
        hp: Any = f"fixed(degree={SYMGWR_DEGREE},k={SYMGWR_K})"
        y_pred = fit_predict_symgwr(X_fit, y_fit, split.X_test)
        n_used, cap = len(X_fit), None
    else:
        Xtr, ytr = subsample_train(model_name, split.X_train, split.y_train, seed)
        if model_name == "GP":
            # One expensive fit per seed: tune alpha on a small train subsample,
            # then refit once on (capped train + val).
            Xs, ys = subsample_train(model_name, Xtr, ytr, seed, cap=GP_TUNE_ROWS)
            hp = select_hparams(model_name, Xs, ys, split.X_val, split.y_val, seed, temp_col)
        else:
            hp = select_hparams(model_name, Xtr, ytr, split.X_val, split.y_val, seed, temp_col)
        X_fit = np.vstack([Xtr, split.X_val])
        y_fit = np.concatenate([ytr, split.y_val])
        y_pred = predict_model(model_name, X_fit, y_fit, split.X_test, hp, seed, temp_col)
        n_used = len(X_fit)
        cap = TRAIN_CAP if model_name in CAPPED_MODELS else None

    metrics = compute_calphad_metrics(split.y_test, y_pred, split.target_scaler)
    return {
        "status": "ok",
        "chosen_hparams": str(hp),
        "n_train_used": int(n_used),
        "n_train_available": int(n_avail),
        "train_cap": cap,
        "notes": MODEL_NOTES.get(model_name, ""),
        **metrics,
        "runtime_s": time.perf_counter() - t0,
    }


def set_thread_env(n: int = 4) -> None:
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = str(n)
