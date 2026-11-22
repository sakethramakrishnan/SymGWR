#!/usr/bin/env python
"""Grid finite-difference validation of SymGWR's analytic gradient (degree=5).

Real CALPHAD grid (data/raw/718_all.csv, deduped), NOT a synthetic function.

The data is a one-factor-at-a-time (OFAT) star design: one base composition and,
for each element e in {Nb, Al, Ti, Mo, Fe, Cr}, a sweep of c_e with every other
element at its base value, each at 81 temperatures (1000..1800 K, 10 K).  Ni is
constant.  So the data fills six (T, c_e) planes that share the base T-line.

Finite differences (central, from the RAW table, J/mol per unit):
  * T axis      : along T on every composition line (needs equal T steps).
  * c_e axes    : along c_e at fixed T on element e's sweep, only where the
                  neighbours on both sides are equally spaced (Al and Fe grids
                  are irregular near the base value, so some points drop out).

Compared with the analytic gradient of the fitted degree-5 polynomial
(sklearn Pipeline = PolynomialFeatures + LinearRegression, minimum-norm
solution because the design matrix is rank-deficient), evaluated at the same
points and converted to raw units by the chain rule

    dG/dx_raw = dG_z/dx_z * target_scale / feature_scale[x].

Two estimators are reported:
  poly   analytic polynomial gradient at the query point itself
  gwr_k1 gradient of the polynomial at the nearest fit (train+val) reference
         point in scaled feature space (the GWR k=1 rule; the convention of the
         older version of this check)

Two point sets per seed (random split, seeds 42..51, polynomial fit on train+val):
  all_grid   every FD-eligible grid point (optimistic: most points were fit)
  test_only  FD-eligible points whose centre row is NOT in the fit data

Identifiability: SVD of the centred, bias-free fit design (the matrix the
minimum-norm solve sees) gives the null space.  For a derivative functional
J(x) (vector of monomial derivatives), the part of J(x) in the null space is
NOT determined by the data and is set to zero by the minimum-norm solution:
null_frac = ||P_null J(x)|| / ||J(x)||.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gwr.gradients import evaluate_polynomial_pipeline_gradients
from gwr.models import fit_polynomial_regression
from gwr.splits import DEFAULT_DATA, DEFAULT_TARGET, TEMP_COL, get_split

DEGREE = 5
SEEDS = list(range(42, 52))
ELEMENTS = ["Nb", "Al", "Ti", "Mo", "Fe", "Cr"]  # Ni skipped: constant (0.53)
STEP_RTOL, STEP_ATOL = 1e-6, 1e-9  # "equally spaced" test


def col(el: str) -> str:
    return f"Mass_fraction_{el}"


# ----------------------------------------------------------------------
# FD point sets (depend only on the deduped raw table, not on the seed)
# ----------------------------------------------------------------------

def _central(x: np.ndarray, g: np.ndarray):
    """Interior points with equal spacing on both sides + central difference."""
    prev, nxt = x[1:-1] - x[:-2], x[2:] - x[1:-1]
    ok = np.isclose(prev, nxt, rtol=STEP_RTOL, atol=STEP_ATOL) & (prev > 0)
    j = np.nonzero(ok)[0]  # index into interior -> interior i = j + 1
    i = j + 1
    fd = (g[i + 1] - g[i - 1]) / (x[i + 1] - x[i - 1])
    return i, i - 1, i + 1, fd


def build_T_points(df: pd.DataFrame, comp_cols: list[str]) -> pd.DataFrame:
    rows = []
    for _, grp in df.groupby(comp_cols, sort=False):
        grp = grp.sort_values(TEMP_COL)
        T = grp[TEMP_COL].to_numpy(float)
        G = grp[DEFAULT_TARGET].to_numpy(float)
        ridx = grp.index.to_numpy()
        if len(T) < 3:
            continue
        i, lo, hi, fd = _central(T, G)
        rows.append(pd.DataFrame({"row": ridx[i], "lo_row": ridx[lo], "hi_row": ridx[hi], "fd": fd}))
    out = pd.concat(rows, ignore_index=True)
    out["axis"] = "T"
    return out


def build_comp_points(df: pd.DataFrame, comp_cols: list[str], base: pd.Series) -> pd.DataFrame:
    out = []
    for el in ELEMENTS:
        others = [c for c in comp_cols if c != col(el)]
        on_line = np.all(np.isclose(df[others].to_numpy(), base[others].to_numpy()), axis=1)
        line = df[on_line]
        for _, grp in line.groupby(TEMP_COL, sort=False):
            grp = grp.sort_values(col(el))
            c = grp[col(el)].to_numpy(float)
            G = grp[DEFAULT_TARGET].to_numpy(float)
            ridx = grp.index.to_numpy()
            if len(c) < 3:
                continue
            i, lo, hi, fd = _central(c, G)
            out.append(
                pd.DataFrame(
                    {"row": ridx[i], "lo_row": ridx[lo], "hi_row": ridx[hi], "fd": fd, "axis": el}
                )
            )
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------
# Polynomial helpers
# ----------------------------------------------------------------------

def deriv_matrix(powers: np.ndarray, X: np.ndarray, d: dict[int, int]) -> np.ndarray:
    """(n, n_terms): d-th mixed partial of every monomial; d = {feature: order}."""
    n = X.shape[0]
    M = np.ones((n, powers.shape[0]))
    for f in range(powers.shape[1]):
        p = powers[:, f].astype(float)
        o = d.get(f, 0)
        coef = np.ones_like(p)
        for k in range(o):
            coef = coef * np.maximum(p - k, 0.0)
        e = np.maximum(p - o, 0.0)
        M *= coef[None, :] * X[:, [f]] ** e[None, :]
    return M


def null_frac(J: np.ndarray, Vr: np.ndarray) -> np.ndarray:
    """||P_null J|| / ||J|| per row; Vr = (r, n_terms) orthonormal row space."""
    nrm = np.linalg.norm(J, axis=1)
    proj = J @ Vr.T @ Vr
    nul = np.linalg.norm(J - proj, axis=1)
    return np.where(nrm > 0, nul / np.where(nrm > 0, nrm, 1.0), np.nan)


def metrics(fd: np.ndarray, est: np.ndarray) -> dict:
    err = np.abs(est - fd)
    typ = float(np.mean(np.abs(fd)))
    mae = float(err.mean())
    r = float(np.corrcoef(fd, est)[0, 1]) if len(fd) > 2 and np.std(est) > 0 and np.std(fd) > 0 else np.nan
    return {"n": int(len(fd)), "mae": mae, "typical_abs": typ,
            "normalized_error": mae / typ if typ > 0 else np.nan, "pearson": r}


# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out-dir", default="results/camera_ready")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--degree", type=int, default=DEGREE)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sp0 = get_split("random", args.seeds[0], data_path=args.data, verbose=False)
    df, feats = sp0.df, sp0.feature_cols
    comp_cols = [c for c in feats if c != TEMP_COL]
    t_idx = feats.index(TEMP_COL)
    fidx = {"T": t_idx, **{el: feats.index(col(el)) for el in ELEMENTS}}
    base = df[comp_cols].mode().iloc[0]

    pts = pd.concat(
        [build_T_points(df, comp_cols), build_comp_points(df, comp_cols, base)], ignore_index=True
    )
    axes = ["T"] + ELEMENTS
    print(f"rows={len(df)}  compositions={df.groupby(comp_cols).ngroups}")
    print("FD-eligible points per axis:", pts.groupby("axis").size().reindex(axes).to_dict())
    n_comp = df.groupby(comp_cols).ngroups
    print(f"dG/dT expected n = {n_comp} x {df[TEMP_COL].nunique() - 2} = "
          f"{n_comp * (df[TEMP_COL].nunique() - 2)}; got {(pts.axis == 'T').sum()}")

    X_raw_all = df[feats].to_numpy(float)
    rows, ident_rows = [], []
    grad_store = {a: [] for a in axes}  # raw-unit poly gradient at all-grid FD points, per seed

    for seed in args.seeds:
        t0 = time.time()
        sp = get_split("random", seed, data_path=args.data, verbose=False)
        fit_idx = np.concatenate([sp.train_idx, sp.val_idx])
        X_fit = np.vstack([sp.X_train, sp.X_val])
        y_fit = np.concatenate([sp.y_train, sp.y_val])
        model, _ = fit_polynomial_regression(X_fit, y_fit, degree=args.degree)
        poly = model.named_steps["poly_features"]

        # scaled-space gradient at every row -> raw units
        X_all = sp.feature_scaler.transform(X_raw_all)
        g_scaled = evaluate_polynomial_pipeline_gradients(model, X_all)
        t_scale = float(sp.target_scaler.scale_[0])
        f_scale = np.asarray(sp.feature_scaler.scale_)
        g_raw = g_scaled * (t_scale / f_scale)[None, :]

        # GWR k=1: gradient at nearest fit reference point (scaled space)
        tree = cKDTree(X_fit)
        _, nn = tree.query(X_all, k=1)
        g_raw_k1 = g_raw[fit_idx[nn]]

        # model values (raw) for the stencil-truncation diagnostic
        G_poly = sp.target_scaler.inverse_transform(model.predict(X_all).reshape(-1, 1)).ravel()

        # design null space (centred, bias-free = what min-norm solve sees)
        Phi = poly.transform(X_fit)[:, 1:]
        Phi = Phi - Phi.mean(axis=0)
        s = np.linalg.svd(Phi, compute_uv=False)
        tol = s.max() * max(Phi.shape) * np.finfo(Phi.dtype).eps
        rank = int((s > tol).sum())
        _, _, Vh = np.linalg.svd(Phi, full_matrices=False)
        Vr = Vh[:rank]
        powers = poly.powers_[1:]

        in_fit = np.zeros(len(df), bool)
        in_fit[fit_idx] = True

        for axis in axes:
            sub = pts[pts.axis == axis]
            r, lo, hi = sub.row.to_numpy(), sub.lo_row.to_numpy(), sub.hi_row.to_numpy()
            fd = sub.fd.to_numpy()
            j = fidx[axis]
            # raw-unit analytic gradient component (poly) and stencil version of the poly itself
            gp = g_raw[r, j]
            gk = g_raw_k1[r, j]
            h2 = X_raw_all[hi, j] - X_raw_all[lo, j]
            stencil = (G_poly[hi] - G_poly[lo]) / h2
            grad_store[axis].append(pd.Series(gp, index=r))

            # null fraction of d/d(axis) at these points (scaled space, in-plane)
            nf = null_frac(deriv_matrix(powers, X_all[r], {j: 1}), Vr)

            for subset, mask in (("all_grid", np.ones(len(r), bool)), ("test_only", ~in_fit[r])):
                for est, g in (("poly", gp), ("gwr_k1", gk)):
                    m = metrics(fd[mask], g[mask])
                    rows.append({
                        "seed": seed, "axis": axis, "subset": subset, "estimator": est, **m,
                        "stencil_trunc_mae": float(np.mean(np.abs(gp[mask] - stencil[mask]))),
                        "fd_vs_polystencil_mae": float(np.mean(np.abs(stencil[mask] - fd[mask]))),
                        "null_frac_mean": float(np.nanmean(nf[mask])) if est == "poly" else np.nan,
                        "fit_rank": rank, "n_fit_rows": len(fit_idx),
                    })

        # ---- identifiability table (poly, all FD points of the plane) ----
        def add(label, order, where, J):
            nf = null_frac(J, Vr)
            ident_rows.append({
                "seed": seed, "derivative": label, "order": order, "evaluated_at": where,
                "n_points": int(len(J)), "null_frac_mean": float(np.nanmean(nf)),
                "null_frac_max": float(np.nanmax(nf)), "fit_rank": rank,
                "n_terms": int(Phi.shape[1]), "sv_rank_th": float(s[rank - 1]),
                "sv_next": float(s[rank]) if rank < len(s) else np.nan,
            })

        ni = feats.index(col("Ni"))
        for el in ELEMENTS:
            plane_rows = np.unique(pts[pts.axis == el].row.to_numpy())
            Xp = X_all[plane_rows]
            off_base = np.abs(X_raw_all[plane_rows, fidx[el]] - base[col(el)]) > 1e-9
            je, jt = fidx[el], t_idx
            add(f"d/d{el}", 1, f"{el}-plane", deriv_matrix(powers, Xp, {je: 1}))
            add(f"d/dT", 1, f"{el}-plane", deriv_matrix(powers, Xp, {jt: 1}))
            add(f"d2/d{el}2", 2, f"{el}-plane", deriv_matrix(powers, Xp, {je: 2}))
            add(f"d2/d{el}dT", 2, f"{el}-plane", deriv_matrix(powers, Xp, {je: 1, jt: 1}))
            add("d/dNi", 1, f"{el}-plane", deriv_matrix(powers, Xp, {ni: 1}))
            for other in ELEMENTS:
                if other == el:
                    continue
                jo = fidx[other]
                add(f"d/d{other}", 1, f"{el}-plane, {el}!=base", deriv_matrix(powers, Xp[off_base], {jo: 1}))
                add(f"d2/d{el}d{other}", 2, f"{el}-plane, {el}!=base",
                    deriv_matrix(powers, Xp[off_base], {je: 1, jo: 1}))

        print(f"seed {seed}: rank={rank}/{Phi.shape[1]} fit_rows={len(fit_idx)} ({time.time()-t0:.1f}s)")

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(os.path.join(args.out_dir, "derivative_fidelity.csv"), index=False)
    pd.DataFrame(ident_rows).to_csv(os.path.join(args.out_dir, "derivative_identifiability.csv"), index=False)

    # ---- cross-seed pointwise spread of the poly gradient (all-grid points) ----
    spread = {}
    for axis in axes:
        G = pd.concat(grad_store[axis], axis=1)  # points x seeds (index = row; unique per axis)
        fd_typ = float(np.mean(np.abs(pts[pts.axis == axis].fd)))
        spread[axis] = float(G.std(axis=1, ddof=0).mean() / fd_typ)

    # ---- summary: mean +/- std over seeds ----
    metric_cols = ["n", "mae", "typical_abs", "normalized_error", "pearson",
                   "stencil_trunc_mae", "fd_vs_polystencil_mae", "null_frac_mean"]
    g = per_seed.groupby(["axis", "subset", "estimator"], sort=False)
    mean, std = g[metric_cols].mean(), g[metric_cols].std(ddof=0)
    summ = mean.add_suffix("_mean").join(std.add_suffix("_std")).reset_index()
    summ["n_seeds"] = g.size().to_numpy()
    summ["grad_seed_std_over_typical"] = summ.apply(
        lambda r: spread[r["axis"]] if (r["subset"] == "all_grid" and r["estimator"] == "poly") else np.nan, axis=1)
    summ["axis"] = pd.Categorical(summ["axis"], axes, ordered=True)
    summ = summ.sort_values(["subset", "estimator", "axis"]).reset_index(drop=True)
    summ.to_csv(os.path.join(args.out_dir, "derivative_fidelity_summary.csv"), index=False)
    print(summ[["axis", "subset", "estimator", "n_mean", "mae_mean", "mae_std", "typical_abs_mean",
                "normalized_error_mean", "normalized_error_std", "pearson_mean", "pearson_std"]]
          .to_string(float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
