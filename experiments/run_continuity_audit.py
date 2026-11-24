#!/usr/bin/env python
"""Continuity audit of the k=1 nearest-reference surrogates (tasks 2, 3, 4).

Surrogates (all in the split's scaled-feature / z-scored-target space; fit on
train+val, degree-5 min-norm polynomial P, nearest reference i = argmin ||q-r||):

  SymGWR              Ghat(q) = G_i + gradP(r_i) . (q - r_i)       grad = gradP(r_i)
  residual_anchored   Ghat(q) = P(q) + (G_i - P(r_i))              grad = gradP(q)
  second_order        Ghat(q) = G_i + gradP(r_i).d + .5 d'H(r_i)d   (accuracy only)

Segments: 5,000 long (between two random test queries) and 5,000 short (start at
a test query, random direction in the non-constant feature subspace, length
0.1-1.0 x nearest-reference distance) per split-seed.  Switches of the nearest
reference are found on a grid and located by bisection to adjacent doubles; the
surrogate is evaluated on both sides (q-, cell a) and (q+, cell b).
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pandas as pd

from camera_ready_common import (
    OUT_DIR, SEEDS, SPLITS, TEMP_COL, composition_ids, fit_symgwr, metrics_raw,
    predict_residual_anchored, predict_second_order, predict_symgwr,
)
from gwr.core import PipelineGradientWeightedRegressor

KD_WORKERS = int(os.environ.get("OMP_NUM_THREADS", "4"))
N_SEG = 5000
GRID_LONG = 2048
GRID_SHORT = 64
BISECT_ITERS = 52
VARIANTS = ["SymGWR", "residual_anchored"]


# ----------------------------------------------------------------------
def surrogate(fit, variant, Q, idx):
    if variant == "SymGWR":
        return fit.ref_y[idx] + np.einsum("ij,ij->i", fit.ref_grad[idx], Q - fit.ref_X[idx])
    return fit.model.value(Q) + (fit.ref_y[idx] - fit.ref_P[idx])


def surrogate_grad(fit, variant, Q, idx):
    if variant == "SymGWR":
        return fit.ref_grad[idx]
    return fit.model.grad(Q)


def sample_segments(fit, rng, kind):
    X = fit.split.X_test
    n = len(X)
    if kind == "long":
        i = rng.integers(0, n, N_SEG)
        j = (i + rng.integers(1, n, N_SEG)) % n  # distinct pair
        return X[i], X[j]
    i = rng.integers(0, n, N_SEG)
    A = X[i]
    d_nn, _ = fit.tree.query(A, k=1)
    free = fit.ref_X.std(axis=0) > 0  # non-constant features only (Ni is constant)
    u = rng.normal(size=(N_SEG, X.shape[1])) * free
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    s = rng.uniform(0.1, 1.0, N_SEG) * d_nn
    return A, A + u * s[:, None]


TIE_TOL = 1e-9  # |d(q,r_a) - d(q,r_b)| below this at both neighbouring grid points => segment lies on a Voronoi face


def find_switches(fit, A, B, grid, chunk=250):
    """Locate nearest-reference switches along each segment A[s] -> B[s].

    Returns (n_sw, n_tie, sw): per-segment count of genuine (transversal) switches,
    per-segment count of tie-plane switches, and the genuine-switch records.
    A switch is a *tie-plane* switch when the two cells are equidistant (within TIE_TOL) at both
    grid points bracketing it: the segment then runs along the shared Voronoi face (exact on this
    lattice-like design, e.g. Nb=0.04 vs 0.06 around a missing Nb=0.05) and which cell is 'nearest'
    is decided by float noise.  Those switches are real discontinuities of the surrogate but their
    number is arbitrary, so they are excluded from the jump statistics and counted separately.
    """
    t = np.linspace(0.0, 1.0, grid + 1)
    n_sw = np.zeros(len(A), dtype=np.int64)
    n_tie = np.zeros(len(A), dtype=np.int64)
    Qm, Qp, Ia, Ib, Sid = [], [], [], [], []
    for s0 in range(0, len(A), chunk):
        a, b = A[s0:s0 + chunk], B[s0:s0 + chunk]
        D = b - a
        pts = a[:, None, :] + t[None, :, None] * D[:, None, :]
        _, idx = fit.tree.query(pts.reshape(-1, a.shape[1]), k=1, workers=KD_WORKERS)
        idx = idx.reshape(len(a), grid + 1)
        chg = idx[:, 1:] != idx[:, :-1]
        s, i = np.nonzero(chg)
        if len(s) == 0:
            continue
        lo, hi = t[i].copy(), t[i + 1].copy()
        cell = idx[s, i]
        Ax, Dx = a[s], D[s]
        for _ in range(BISECT_ITERS):
            mid = 0.5 * (lo + hi)
            _, im = fit.tree.query(Ax + mid[:, None] * Dx, k=1, workers=KD_WORKERS)
            same = im == cell
            lo = np.where(same, mid, lo)
            hi = np.where(same, hi, mid)
        qm, qp = Ax + lo[:, None] * Dx, Ax + hi[:, None] * Dx
        _, ib = fit.tree.query(qp, k=1, workers=KD_WORKERS)
        assert (ib != cell).all()
        ra, rb = fit.ref_X[cell], fit.ref_X[ib]
        tie = np.ones(len(s), bool)
        for tg in (t[i], t[i + 1]):
            pg = Ax + tg[:, None] * Dx
            gap = np.abs(np.linalg.norm(pg - ra, axis=1) - np.linalg.norm(pg - rb, axis=1))
            tie &= gap < TIE_TOL
        n_tie[s0:s0 + chunk] = np.bincount(s[tie], minlength=len(a))
        keep = ~tie
        n_sw[s0:s0 + chunk] = np.bincount(s[keep], minlength=len(a))
        Qm.append(qm[keep]); Qp.append(qp[keep]); Ia.append(cell[keep]); Ib.append(ib[keep]); Sid.append(s[keep] + s0)
    if not Qm:
        d = A.shape[1]
        return n_sw, n_tie, dict(qm=np.empty((0, d)), qp=np.empty((0, d)), a=np.empty(0, int), b=np.empty(0, int), seg=np.empty(0, int))
    return n_sw, n_tie, dict(qm=np.vstack(Qm), qp=np.vstack(Qp), a=np.concatenate(Ia), b=np.concatenate(Ib), seg=np.concatenate(Sid))


def quant(x):
    if len(x) == 0:
        return dict(median=np.nan, p95=np.nan, max=np.nan)
    return dict(median=float(np.median(x)), p95=float(np.quantile(x, 0.95)), max=float(np.max(x)))


def in_cell_second_derivative(fit, variant, rng, n=1500):
    """Central second difference along a random unit direction, both points in the same cell."""
    X = fit.split.X_test
    i = rng.integers(0, len(X), n)
    q = X[i]
    d_nn, idx = fit.tree.query(q, k=1)
    free = fit.ref_X.std(axis=0) > 0
    u = rng.normal(size=q.shape) * free
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    h = 0.05 * d_nn
    qp, qm = q + h[:, None] * u, q - h[:, None] * u
    _, ip = fit.tree.query(qp, k=1)
    _, im = fit.tree.query(qm, k=1)
    ok = (ip == idx) & (im == idx)
    q, qp, qm, u, h, idx = q[ok], qp[ok], qm[ok], u[ok], h[ok], idx[ok]
    f0, fp, fm = (surrogate(fit, variant, z, idx) for z in (q, qp, qm))
    d2 = (fp - 2 * f0 + fm) / h ** 2
    out = dict(n_in_cell=int(ok.sum()), d2_fd_abs_max=float(np.abs(d2).max()), d2_fd_abs_median=float(np.median(np.abs(d2))))
    if variant == "residual_anchored":
        H = fit.model.hess(q)
        an = np.einsum("ni,nij,nj->n", u, H, u)
        out["d2_analytic_abs_median"] = float(np.median(np.abs(an)))
        out["d2_fd_vs_analytic_max_abs_err"] = float(np.abs(d2 - an).max())
    return out


def hessian_stats(fit):
    H = fit.model.hess(fit.split.X_test)
    sv = np.linalg.svd(H, compute_uv=False)  # (n, d)
    rk_default = (sv > (sv.max(axis=1, keepdims=True) * H.shape[1] * np.finfo(float).eps)).sum(1)
    rk_rel1e8 = (sv > 1e-8 * sv.max(axis=1, keepdims=True)).sum(1)
    g = fit.model.grad(fit.split.X_test)
    gn = np.linalg.norm(g, axis=1)
    return dict(
        hess_rank_default_min=int(rk_default.min()), hess_rank_default_median=float(np.median(rk_default)),
        hess_rank_default_max=int(rk_default.max()),
        hess_rank_rel1e8_min=int(rk_rel1e8.min()), hess_rank_rel1e8_median=float(np.median(rk_rel1e8)),
        hess_rank_rel1e8_max=int(rk_rel1e8.max()),
        hess_spectral_norm_median=float(np.median(sv[:, 0])), hess_spectral_norm_p95=float(np.quantile(sv[:, 0], 0.95)),
        hess_spectral_norm_max=float(sv[:, 0].max()),
        hess_frobenius_median=float(np.median(np.linalg.norm(H, axis=(1, 2)))),
        hess_abs_entry_median=float(np.median(np.abs(H))),
        hess_abs_entry_max=float(np.abs(H).max()),
        grad_norm_median_at_test=float(np.median(gn)),
        hess_spec_over_grad_norm_median=float(np.median(sv[:, 0] / gn)),
        hess_zero_row_features=int(((np.abs(H).max(axis=(0, 1)) == 0)).sum()),
        hess_sv7_over_sv1_median=float(np.median(sv[:, 6] / sv[:, 0])),
        hess_sv7_over_sv1_min=float(np.min(sv[:, 6] / sv[:, 0])),
        hess_sv8_over_sv1_median=float(np.median(sv[:, 7] / sv[:, 0])),
        hess_sv8_over_sv1_max=float(np.max(sv[:, 7] / sv[:, 0])),
    )


def run(splits, seeds, out_dir, max_poly_switches):
    os.makedirs(out_dir, exist_ok=True)
    cont_rows, acc_rows, hess_rows, chk_rows, d2_rows = [], [], [], [], []
    for kind in splits:
        for seed in seeds:
            fit = fit_symgwr(kind, seed)
            sp = fit.split
            rng = np.random.default_rng(10_000 + seed)
            Q = sp.X_test
            scale = float(sp.target_scaler.scale_[0])
            y_ref_raw = sp.target_scaler.inverse_transform(fit.ref_y.reshape(-1, 1)).ravel()
            std_G = float(np.std(y_ref_raw))
            gnorm_model = float(np.median(np.linalg.norm(fit.ref_grad, axis=1)))
            # data-based gradient scale: |dG/dT| central FD on the reference table (scaled-T units)
            df = sp.df
            gid = composition_ids(df, sp.feature_cols)[fit.ref_idx]
            Tr = df[TEMP_COL].to_numpy(float)[fit.ref_idx]
            tscale = float(sp.feature_scaler.scale_[list(sp.feature_cols).index(TEMP_COL)])
            o = np.lexsort((Tr, gid))
            g_o, T_o, y_o = gid[o], Tr[o], fit.ref_y[o]
            ok = (g_o[2:] == g_o[:-2]) & (T_o[2:] - T_o[:-2] == 20)
            gT = np.abs((y_o[2:] - y_o[:-2])[ok] / (T_o[2:] - T_o[:-2])[ok] * tscale)
            gnorm_dataT = float(np.median(gT))

            # accuracy (tasks 3, 4) + check against the repo's k=1 regressor
            preds = {"SymGWR": predict_symgwr(fit, Q), "residual_anchored": predict_residual_anchored(fit, Q),
                     "second_order": predict_second_order(fit, Q)}
            repo = PipelineGradientWeightedRegressor(fit.model.pipeline, n_neighbors=1).fit(fit.ref_X, fit.ref_y).predict(Q)
            chk_rows.append(dict(split=kind, seed=seed, max_abs_diff_vs_repo_k1=float(np.abs(repo - preds["SymGWR"]).max())))
            mets = {}
            for m, p in preds.items():
                r = metrics_raw(fit, p)
                mets[m] = r
                row = dict(split=kind, seed=seed, method=m, mape_raw=r["mape_raw"])
                if m != "second_order":  # task 4 is MAPE only
                    row.update(mae_raw=r["mae_raw"], rmse=r["rmse"], r2=r["r2"])
                acc_rows.append(row)
            hs = hessian_stats(fit)
            hs.update(split=kind, seed=seed)
            hess_rows.append(hs)

            # continuity
            for seg_kind, grid in (("long", GRID_LONG), ("short", GRID_SHORT)):
                A, B = sample_segments(fit, rng, seg_kind)
                n_sw, n_tie, sw = find_switches(fit, A, B, grid)
                # grid-resolution check: 4x finer grid on 300 segments must not find more switches
                n_sw_fine, _, _ = find_switches(fit, A[:300], B[:300], 4 * grid)
                missed = int(n_sw_fine.sum() - n_sw[:300].sum())
                n_total = len(sw["a"])
                for variant in VARIANTS:
                    sel = np.arange(n_total)
                    if variant == "residual_anchored" and n_total > max_poly_switches:
                        sel = np.sort(rng.choice(n_total, max_poly_switches, replace=False))
                    qm, qp, a, b = sw["qm"][sel], sw["qp"][sel], sw["a"][sel], sw["b"][sel]
                    if len(sel):
                        gm = surrogate(fit, variant, qm, a)
                        gp = surrogate(fit, variant, qp, b)
                        vj_z = np.abs(gp - gm)
                        dgm = surrogate_grad(fit, variant, qm, a)
                        dgp = surrogate_grad(fit, variant, qp, b)
                        gj = np.linalg.norm(dgp - dgm, axis=1)
                    else:
                        vj_z, gj = np.empty(0), np.empty(0)
                    vj_raw = vj_z * scale
                    row = dict(variant=variant, split=kind, seed=seed, segment_type=seg_kind,
                               n_segments=N_SEG, grid=grid,
                               frac_segments_crossing=float((n_sw > 0).mean()),
                               switches_per_segment_mean=float(n_sw.mean()), n_switches_total=int(n_total),
                               n_switches_evaluated=int(len(sel)),
                               n_tie_plane_switches_excluded=int(n_tie.sum()),
                               n_segments_with_tie_plane=int((n_tie > 0).sum()), switches_missed_in_grid_check_of_300_seg=missed,
                               test_mae_raw=mets[variant]["mae_raw"], std_G_raw=std_G,
                               gradnorm_median_model=gnorm_model, gradnorm_median_dataT=gnorm_dataT)
                    for nm, arr in (("value_jump_over_stdG", vj_raw / std_G), ("value_jump_raw", vj_raw),
                                    ("grad_jump_over_gradnorm", gj / gnorm_model),
                                    ("grad_jump_over_gradnorm_dataT", gj / gnorm_dataT), ("grad_jump_abs", gj)):
                        for qn, qv in quant(arr).items():
                            row[f"{nm}_{qn}"] = qv
                    row["value_jump_median_over_test_mae"] = row["value_jump_raw_median"] / row["test_mae_raw"]
                    row["value_jump_p95_over_test_mae"] = row["value_jump_raw_p95"] / row["test_mae_raw"]
                    row["value_jump_max_over_test_mae"] = row["value_jump_raw_max"] / row["test_mae_raw"]
                    if variant == "residual_anchored" and len(sel):
                        # analytic prediction of the value jump: |(G_b-P(r_b)) - (G_a-P(r_a))|
                        res = fit.ref_y - fit.ref_P
                        an = np.abs(res[b] - res[a])
                        row["resid_value_jump_minus_analytic_max_abs_z"] = float(np.abs(vj_z - an).max())
                        row["resid_analytic_value_jump_raw_median"] = float(np.median(an) * scale)
                    cont_rows.append(row)
                print(f"[{kind} seed={seed} {seg_kind}] switches={n_total} crossing={float((n_sw>0).mean()):.3f} "
                      f"missed_in_check={missed} tie_excluded={int(n_tie.sum())}", flush=True)

            # second derivative inside a cell
            for variant in VARIANTS:
                d2 = in_cell_second_derivative(fit, variant, rng)
                d2.update(variant=variant, split=kind, seed=seed)
                d2_rows.append(d2)

    cont = pd.DataFrame(cont_rows)
    acc = pd.DataFrame(acc_rows)
    pd.DataFrame(hess_rows).to_csv(os.path.join(out_dir, "hessian_P_at_test_queries.csv"), index=False)
    pd.DataFrame(d2_rows).to_csv(os.path.join(out_dir, "in_cell_second_derivative.csv"), index=False)
    cont.to_csv(os.path.join(out_dir, "continuity_per_seed.csv"), index=False)
    acc.to_csv(os.path.join(out_dir, "taylor_variants_accuracy_per_seed.csv"), index=False)
    pd.DataFrame(chk_rows).to_csv(os.path.join(out_dir, "continuity_check_vs_repo_k1.csv"), index=False)
    return cont, acc


def summarize(cont, acc, out_dir):
    """Across-seed summary: mean of per-seed median/p95, max of per-seed max (descriptive)."""
    g = cont.groupby(["variant", "split", "segment_type"])
    rows = []
    for key, d in g:
        r = dict(variant=key[0], split=key[1], segment_type=key[2], n_seeds=len(d),
                 frac_segments_crossing_mean=d.frac_segments_crossing.mean(),
                 switches_per_segment_mean=d.switches_per_segment_mean.mean(),
                 test_mae_raw_mean=d.test_mae_raw.mean())
        for nm in ("value_jump_over_stdG", "value_jump_raw", "grad_jump_over_gradnorm", "grad_jump_over_gradnorm_dataT"):
            r[f"{nm}_median(mean over seeds)"] = d[f"{nm}_median"].mean()
            r[f"{nm}_p95(mean over seeds)"] = d[f"{nm}_p95"].mean()
            r[f"{nm}_max(max over seeds)"] = d[f"{nm}_max"].max()
        r["value_jump_median_over_test_mae(mean)"] = d.value_jump_median_over_test_mae.mean()
        r["value_jump_p95_over_test_mae(mean)"] = d.value_jump_p95_over_test_mae.mean()
        r["value_jump_max_over_test_mae(max)"] = d.value_jump_max_over_test_mae.max()
        rows.append(r)
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(out_dir, "continuity_summary.csv"), index=False)
    a = acc.groupby(["split", "method"]).agg(
        mape_raw_mean=("mape_raw", "mean"), mape_raw_std=("mape_raw", lambda x: x.std(ddof=1)),
        mae_raw_mean=("mae_raw", "mean"), rmse_mean=("rmse", "mean"), r2_mean=("r2", "mean")).reset_index()
    a.to_csv(os.path.join(out_dir, "taylor_variants_accuracy_summary.csv"), index=False)
    return s, a


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--splits", nargs="*", default=SPLITS)
    ap.add_argument("--grid-long", type=int, default=GRID_LONG, help="grid points per long segment")
    ap.add_argument("--merge", nargs="*", default=None,
                    help="skip running; concatenate per-split output dirs into --out-dir and summarise")
    ap.add_argument("--max-poly-switches", type=int, default=20000,
                    help="cap on switches evaluated per segment set for the residual-anchored variant "
                         "(needs P(q), gradP(q) at each point; SymGWR always uses all switches)")
    args = ap.parse_args()
    globals()["GRID_LONG"] = args.grid_long
    if args.merge:
        os.makedirs(args.out_dir, exist_ok=True)
        for name in ("continuity_per_seed", "taylor_variants_accuracy_per_seed", "hessian_P_at_test_queries",
                     "in_cell_second_derivative", "continuity_check_vs_repo_k1"):
            m = pd.concat([pd.read_csv(os.path.join(d, name + ".csv")) for d in args.merge])
            keys = [k for k in ("split", "seed", "variant", "segment_type", "method") if k in m.columns]
            # later dirs override earlier ones for the same (split, seed, ...) rows (used for re-runs)
            m = m.drop_duplicates(subset=keys, keep="last").sort_values(keys, kind="stable")
            m.to_csv(os.path.join(args.out_dir, name + ".csv"), index=False)
        cont = pd.read_csv(os.path.join(args.out_dir, "continuity_per_seed.csv"))
        acc = pd.read_csv(os.path.join(args.out_dir, "taylor_variants_accuracy_per_seed.csv"))
    else:
        cont, acc = run(args.splits, args.seeds, args.out_dir, args.max_poly_switches)
    s, a = summarize(cont, acc, args.out_dir)
    pd.set_option("display.width", 250, "display.max_columns", 50, "display.float_format", lambda v: f"{v:.3e}")
    print(a.to_string())
    print(s.T.to_string())


if __name__ == "__main__":
    main()
