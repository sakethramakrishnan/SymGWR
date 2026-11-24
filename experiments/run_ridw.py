#!/usr/bin/env python
"""SymGWR-R-IDW: residual-anchored with an inverse-distance blend of k residuals.

    Ghat(q) = P(q) + sum_{i in kNN(q)} w_i(q) (G_i - P(r_i)),
    w_i = s_i / sum_j s_j,  s_i = 1/(d_i + eps)^2,  eps = 1e-12  (exact at references up to ~1e-24).

Outputs (results/camera_ready/):
  ridw_accuracy_per_seed.csv       raw MAPE/MAE/RMSE/R2 per split-seed: plain_polynomial, SymGWR-T,
                                   SymGWR-R, SymGWR-R-IDW k=2,4,8
  ridw_continuity_per_seed.csv     short-segment audit (value / gradient jumps) per k and event type
  ridw_continuity_summary.csv      across-seed summary of the above
  ridw_hessian_per_seed.csv        Hessian of the full surrogate at test queries
  ridw_gradient_flatness.csv       |grad(blend term)| near references
  ridw_paired_tests.csv            seed-level paired t-test / Wilcoxon (R-IDW k=4 vs others)
  ridw_combined_table.csv          RMSE-first combined table (also printed)
  interp_summary.csv / _long.csv   one row per new method APPENDED (existing rows untouched)

Event types for the continuity audit (short segments, 5,000 per split-seed, same tie-plane
handling as run_continuity_audit.py):
  nearest_ref_switch  the nearest reference changes (the audit of the earlier tasks)
  kset_change         the SET of the k nearest references changes (k-th <-> (k+1)-th swap); this is
                      where a truncated k-NN blend can jump, because the dropped neighbour has
                      non-zero weight.
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from scipy import stats

import run_continuity_audit as ca
from camera_ready_common import (
    OUT_DIR, SEEDS, SPLITS, composition_ids, fit_symgwr, metrics_raw, predict_residual_anchored,
    predict_symgwr,
)

KS = [2, 4, 8]
EPS = 1e-12
IDW_POWER = 2.0
MAX_EVENTS = 5000


# ----------------------------------------------------------------------
#  surrogate
# ----------------------------------------------------------------------
def blend(fit, Q, k, want_grad=False, idx=None):
    """Return (B, gradB) for B(q) = sum w_i res_i over the k nearest references.

    ``idx`` (n, k) fixes the neighbour set (used for Hessians by finite differences, so the
    difference never straddles a k-set change)."""
    if idx is None:
        d, idx = fit.tree.query(Q, k=k, workers=ca.KD_WORKERS)
        if k == 1:
            d, idx = d[:, None], idx[:, None]
    else:
        d = np.linalg.norm(Q[:, None, :] - fit.ref_X[idx], axis=2)
    res = (fit.ref_y - fit.ref_P)[idx]  # (n, k)
    s = 1.0 / (d + EPS) ** IDW_POWER
    W = s.sum(1)
    B = (s * res).sum(1) / W
    if not want_grad:
        return B, None
    diff = Q[:, None, :] - fit.ref_X[idx]  # (n, k, dim)
    dsafe = np.where(d > 0, d, 1.0)
    ds = (-IDW_POWER * (d + EPS) ** (-IDW_POWER - 1) / dsafe)[:, :, None] * diff  # grad s_i
    ds = np.where((d > 0)[:, :, None], ds, 0.0)
    gB = (ds * (res - B[:, None])[:, :, None]).sum(1) / W[:, None]
    return B, gB


def ridw_predict(fit, Q, k):
    return fit.model.value(Q) + blend(fit, Q, k)[0]


def ridw_value_grad(fit, Q, k):
    B, gB = blend(fit, Q, k, want_grad=True)
    return fit.model.value(Q) + B, fit.model.grad(Q) + gB


# ----------------------------------------------------------------------
#  k-set change detection (same grid + bisection + tie-plane logic as nearest-ref switches)
# ----------------------------------------------------------------------
def find_kset_switches(fit, k, A, B, grid, chunk=250):
    t = np.linspace(0.0, 1.0, grid + 1)
    n_sw = np.zeros(len(A), dtype=np.int64)
    n_tie = np.zeros(len(A), dtype=np.int64)
    out = dict(qm=[], qp=[], seg=[])
    W = ca.KD_WORKERS
    for s0 in range(0, len(A), chunk):
        a, b = A[s0:s0 + chunk], B[s0:s0 + chunk]
        D = b - a
        pts = a[:, None, :] + t[None, :, None] * D[:, None, :]
        dd, idx = fit.tree.query(pts.reshape(-1, a.shape[1]), k=k + 1, workers=W)
        dd = dd.reshape(len(a), grid + 1, k + 1)
        key = np.sort(idx.reshape(len(a), grid + 1, k + 1)[..., :k], axis=-1)
        chg = (key[:, 1:] != key[:, :-1]).any(-1)
        s, i = np.nonzero(chg)
        if len(s) == 0:
            continue
        cell = key[s, i]
        lo, hi = t[i].copy(), t[i + 1].copy()
        Ax, Dx = a[s], D[s]
        for _ in range(ca.BISECT_ITERS):
            mid = 0.5 * (lo + hi)
            _, im = fit.tree.query(Ax + mid[:, None] * Dx, k=k, workers=W)
            same = (np.sort(im.reshape(len(s), k), axis=-1) == cell).all(1)
            lo = np.where(same, mid, lo)
            hi = np.where(same, hi, mid)
        qm, qp = Ax + lo[:, None] * Dx, Ax + hi[:, None] * Dx
        _, ip = fit.tree.query(qp, k=k, workers=W)
        assert (np.sort(ip.reshape(len(s), k), axis=-1) != cell).any(1).all()
        gap_lo = np.abs(dd[s, i, k - 1] - dd[s, i, k])
        gap_hi = np.abs(dd[s, i + 1, k - 1] - dd[s, i + 1, k])
        tie = (gap_lo < ca.TIE_TOL) & (gap_hi < ca.TIE_TOL)
        n_tie[s0:s0 + chunk] = np.bincount(s[tie], minlength=len(a))
        keep = ~tie
        n_sw[s0:s0 + chunk] = np.bincount(s[keep], minlength=len(a))
        out["qm"].append(qm[keep]); out["qp"].append(qp[keep]); out["seg"].append(s[keep] + s0)
    d_ = A.shape[1]
    if not out["qm"]:
        return n_sw, n_tie, dict(qm=np.empty((0, d_)), qp=np.empty((0, d_)), seg=np.empty(0, int))
    return n_sw, n_tie, {kk: np.concatenate(v) if kk == "seg" else np.vstack(v) for kk, v in out.items()}


def short_segments(fit, rng):
    X = fit.split.X_test
    i = rng.integers(0, len(X), ca.N_SEG)
    A = X[i]
    d_nn, _ = fit.tree.query(A, k=1)
    free = fit.ref_X.std(axis=0) > 0
    u = rng.normal(size=(ca.N_SEG, X.shape[1])) * free
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return A, A + u * (rng.uniform(0.1, 1.0, ca.N_SEG) * d_nn)[:, None]


# ----------------------------------------------------------------------
def pair_cache(fit, qm, qp, rng):
    """P and grad P at both sides of (a subsample of) the events; independent of k."""
    n_tot = len(qm)
    sel = np.arange(n_tot)
    if n_tot > MAX_EVENTS:
        sel = np.sort(rng.choice(n_tot, MAX_EVENTS, replace=False))
    c = dict(sel=sel, n_tot=n_tot)
    if len(sel):
        c.update(Pm=fit.model.value(qm[sel]), Pp=fit.model.value(qp[sel]),
                 gPm=fit.model.grad(qm[sel]), gPp=fit.model.grad(qp[sel]))
    return c


def jump_row(fit, k, event, n_sw, n_tie, qm, qp, cache, std_G, scale, gnorm, mae_k):
    sel = cache["sel"]
    row = dict(k=k, event=event, n_segments=ca.N_SEG, frac_segments_with_event=float((n_sw > 0).mean()),
               n_events_total=int(cache["n_tot"]), n_events_evaluated=int(len(sel)),
               n_tie_plane_excluded=int(n_tie.sum()), test_mae_raw=mae_k, std_G_raw=std_G, gradnorm_median_model=gnorm)
    if len(sel):
        Bm, gBm = blend(fit, qm[sel], k, True)
        Bp, gBp = blend(fit, qp[sel], k, True)
        vj = np.abs((cache["Pp"] + Bp) - (cache["Pm"] + Bm)) * scale
        gj = np.linalg.norm((cache["gPp"] + gBp) - (cache["gPm"] + gBm), axis=1)
    else:
        vj, gj = np.empty(0), np.empty(0)
    for nm, arr in (("value_jump_raw", vj), ("value_jump_over_stdG", vj / std_G),
                    ("value_jump_over_test_mae", vj / mae_k), ("grad_jump_abs", gj),
                    ("grad_jump_over_gradnorm", gj / gnorm)):
        for qn, qv in ca.quant(arr).items():
            row[f"{nm}_{qn}"] = qv
    if len(sel) and k >= 2:
        _, im = fit.tree.query(qm[sel], k=k, workers=ca.KD_WORKERS)
        _, ip = fit.tree.query(qp[sel], k=k, workers=ca.KD_WORKERS)
        kchg = (np.sort(im.reshape(len(sel), k), 1) != np.sort(ip.reshape(len(sel), k), 1)).any(1)
        big = vj > 1e-6
        row["frac_events_with_kset_change"] = float(kchg.mean())
        row["frac_big_jump_events_with_kset_change"] = float(kchg[big].mean()) if big.any() else np.nan
        row["frac_kset_change_events_with_big_jump"] = float(big[kchg].mean()) if kchg.any() else np.nan
    row["frac_events_value_jump_gt_1e-6_Jmol"] = float((vj > 1e-6).mean()) if len(vj) else np.nan
    row["frac_events_grad_jump_gt_1e-9_rel"] = float((gj / gnorm > 1e-9).mean()) if len(gj) else np.nan
    return row


def kset_prediction_check(fit, k, qm, qp, rng, n=2000):
    """At k-set changes: measured value jump vs analytic w_dropped * |res_added - res_dropped|."""
    if len(qm) == 0:
        return np.nan, np.nan
    sel = rng.choice(len(qm), min(n, len(qm)), replace=False)
    qm, qp = qm[sel], qp[sel]
    dm, im = fit.tree.query(qm, k=k, workers=ca.KD_WORKERS)
    _, ip = fit.tree.query(qp, k=k, workers=ca.KD_WORKERS)
    res = fit.ref_y - fit.ref_P
    s = 1.0 / (dm + EPS) ** IDW_POWER
    w = s / s.sum(1, keepdims=True)
    meas = np.abs((blend(fit, qp, k)[0]) - blend(fit, qm, k)[0])
    pred = np.empty(len(sel))
    for r in range(len(sel)):
        drop = [j for j in im[r] if j not in set(ip[r])]
        add = [j for j in ip[r] if j not in set(im[r])]
        if len(drop) != 1 or len(add) != 1:  # simultaneous swaps: skip
            pred[r] = np.nan
            continue
        pos = int(np.where(im[r] == drop[0])[0][0])
        pred[r] = w[r, pos] * abs(res[add[0]] - res[drop[0]])
    ok = np.isfinite(pred)
    return float(np.abs(meas[ok] - pred[ok]).max()) if ok.any() else np.nan, float(ok.mean())


def hessian_rows(fit, k):
    Q = fit.split.X_test
    HP = fit.model.hess(Q)
    h = 1e-6
    HB = np.empty_like(HP)
    _, nb = fit.tree.query(Q, k=k, workers=ca.KD_WORKERS)
    nb = nb.reshape(len(Q), k)
    for j in range(Q.shape[1]):
        e = np.zeros(Q.shape[1]); e[j] = h
        HB[:, :, j] = (blend(fit, Q + e, k, True, nb)[1] - blend(fit, Q - e, k, True, nb)[1]) / (2 * h)
    HB = 0.5 * (HB + HB.transpose(0, 2, 1))
    H = HP + HB
    sv = np.linalg.svd(H, compute_uv=False)
    fro = lambda M: np.linalg.norm(M, axis=(1, 2))
    return dict(
        k=k, hess_P_frobenius_median=float(np.median(fro(HP))),
        hess_blend_frobenius_median=float(np.median(fro(HB))), hess_blend_frobenius_p95=float(np.quantile(fro(HB), .95)),
        hess_blend_frobenius_max=float(fro(HB).max()),
        hess_total_frobenius_median=float(np.median(fro(H))),
        hess_blend_over_P_frobenius_median=float(np.median(fro(HB) / fro(HP))),
        hess_total_rank_rel1e8_min=int((sv > 1e-8 * sv[:, :1]).sum(1).min()),
        hess_total_rank_rel1e8_max=int((sv > 1e-8 * sv[:, :1]).sum(1).max()),
        frac_queries_blend_hessian_nonzero=float((fro(HB) > 1e-10).mean()),
    )


def flatness_row(fit, k, rng, n=300, off=1e-5):
    """|grad B| at a tiny offset from reference points (expect ~0: gradient reverts to grad P at references)."""
    j = rng.choice(len(fit.ref_X), n, replace=False)
    u = rng.normal(size=(n, fit.ref_X.shape[1])) * (fit.ref_X.std(0) > 0)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    gB = blend(fit, fit.ref_X[j] + off * u, k, True)[1]
    gP = np.linalg.norm(fit.ref_grad[j], axis=1)
    r = np.linalg.norm(gB, axis=1) / gP
    return dict(k=k, offset=off, gradB_over_gradP_median=float(np.median(r)), gradB_over_gradP_max=float(r.max()))


# ----------------------------------------------------------------------
def run(splits, seeds, out_dir):
    acc, cont, hess, flat, kchk = [], [], [], [], []
    for kind in splits:
        for seed in seeds:
            fit = fit_symgwr(kind, seed)
            sp = fit.split
            Q = sp.X_test
            preds = {"plain_polynomial": fit.model.value(Q), "SymGWR-T": predict_symgwr(fit, Q),
                     "SymGWR-R": predict_residual_anchored(fit, Q)}
            for k in KS:
                preds[f"SymGWR-R-IDW_k{k}"] = ridw_predict(fit, Q, k)
            mae = {}
            for m, p in preds.items():
                r = metrics_raw(fit, p)
                mae[m] = r["mae_raw"]
                acc.append(dict(split=kind, seed=seed, method=m, mape_raw=r["mape_raw"], mae_raw=r["mae_raw"],
                                rmse=r["rmse"], r2=r["r2"]))
            # exactness at references
            ref_err = [float(np.abs(ridw_predict(fit, fit.ref_X[:500], k) - fit.ref_y[:500]).max()) for k in KS]
            kchk.append(dict(split=kind, seed=seed, **{f"max_abs_err_at_refs_k{k}": e for k, e in zip(KS, ref_err)}))

            rng = np.random.default_rng(20_000 + seed)
            scale = float(sp.target_scaler.scale_[0])
            std_G = float(np.std(sp.target_scaler.inverse_transform(fit.ref_y.reshape(-1, 1)).ravel()))
            gnorm = float(np.median(np.linalg.norm(fit.ref_grad, axis=1)))
            A, B = short_segments(fit, rng)
            n_sw, n_tie, sw = ca.find_switches(fit, A, B, ca.GRID_SHORT)
            cache = pair_cache(fit, sw["qm"], sw["qp"], rng)
            for k in [1] + KS:
                mk = mae["SymGWR-R"] if k == 1 else mae[f"SymGWR-R-IDW_k{k}"]
                row = jump_row(fit, k, "nearest_ref_switch", n_sw, n_tie, sw["qm"], sw["qp"], cache, std_G, scale, gnorm, mk)
                cont.append(dict(split=kind, seed=seed, **row))
            for k in KS:
                n_k, nt_k, swk = find_kset_switches(fit, k, A, B, ca.GRID_SHORT)
                n_k4, _, _ = find_kset_switches(fit, k, A[:300], B[:300], 4 * ca.GRID_SHORT)
                row = jump_row(fit, k, "kset_change", n_k, nt_k, swk["qm"], swk["qp"],
                               pair_cache(fit, swk["qm"], swk["qp"], rng), std_G, scale, gnorm, mae[f"SymGWR-R-IDW_k{k}"])
                dev, frac = kset_prediction_check(fit, k, swk["qm"], swk["qp"], rng)
                row.update(analytic_max_abs_dev_z=dev, analytic_checked_frac=frac,
                           missed_in_4x_grid_check_of_300_seg=int(n_k4.sum() - n_k[:300].sum()))
                cont.append(dict(split=kind, seed=seed, **row))
                hr = hessian_rows(fit, k); hr.update(split=kind, seed=seed); hess.append(hr)
                fr = flatness_row(fit, k, rng); fr.update(split=kind, seed=seed); flat.append(fr)
            print(f"[{kind} seed={seed}] done", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(acc).to_csv(os.path.join(out_dir, "ridw_accuracy_per_seed.csv"), index=False)
    pd.DataFrame(cont).to_csv(os.path.join(out_dir, "ridw_continuity_per_seed.csv"), index=False)
    pd.DataFrame(hess).to_csv(os.path.join(out_dir, "ridw_hessian_per_seed.csv"), index=False)
    pd.DataFrame(flat).to_csv(os.path.join(out_dir, "ridw_gradient_flatness.csv"), index=False)
    pd.DataFrame(kchk).to_csv(os.path.join(out_dir, "ridw_exactness_at_references.csv"), index=False)


# ----------------------------------------------------------------------
#  tables
# ----------------------------------------------------------------------
def ms(x, d=3):
    x = np.asarray(x, float)
    return f"{np.nanmean(x):.{d}e} ± {np.nanstd(x, ddof=1):.{d}e}" if np.isfinite(x).all() else "n/a"


def build_tables(out_dir):
    acc = pd.read_csv(os.path.join(out_dir, "ridw_accuracy_per_seed.csv"))
    base = pd.read_csv(os.path.join(out_dir, "interp_baselines_per_seed.csv"))
    cub = base[((base.split != "composition_group") & (base.method == "b_cubic_T")) |
               ((base.split == "composition_group") & (base.method == "e_cubic_arm"))].copy()
    cub["method"] = "cubic_T_or_arm"
    cub = cub[["split", "seed", "method", "mape_raw", "mae_raw", "rmse", "r2", "n_eval", "n_test"]]
    cov = cub.groupby("split").apply(lambda d: d.n_eval.sum() / d.n_test.sum(), include_groups=False)
    allm = pd.concat([acc, cub.drop(columns=["n_eval", "n_test"])], ignore_index=True)
    # cross-check against the first-pass numbers
    chk = base[base.method.isin(["SymGWR", "residual_anchored"])].merge(
        acc[acc.method.isin(["SymGWR-T", "SymGWR-R"])].assign(method=lambda d: d.method.map({"SymGWR-T": "SymGWR", "SymGWR-R": "residual_anchored"})),
        on=["split", "seed", "method"], suffixes=("", "_new"))
    print("max |MAPE new - MAPE interp_baselines| for SymGWR-T/R:", float((chk.mape_raw - chk.mape_raw_new).abs().max()))

    order = ["plain_polynomial", "SymGWR-T", "SymGWR-R", "SymGWR-R-IDW_k2", "SymGWR-R-IDW_k4", "SymGWR-R-IDW_k8", "cubic_T_or_arm"]
    rows = []
    for kind in SPLITS:
        for m in order:
            d = allm[(allm.split == kind) & (allm.method == m)].sort_values("seed")
            if d.empty or d.rmse.isna().all():
                rows.append(dict(split=kind, method=m, rmse_J_per_mol="n/a", mae_J_per_mol="n/a", mape_raw="n/a", r2="n/a"))
                continue
            note = ""
            if m == "cubic_T_or_arm" and cov.get(kind, 1) < 0.999:
                note = f" [{cov[kind]:.0%} of rows]"
            rows.append(dict(split=kind, method=m, rmse_J_per_mol=ms(d.rmse, 3), mae_J_per_mol=ms(d.mae_raw, 3),
                             mape_raw=ms(d.mape_raw, 3) + note, r2=f"{d.r2.mean():.10f} ± {d.r2.std(ddof=1):.1e}"))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out_dir, "ridw_combined_table.csv"), index=False)

    # paired tests: R-IDW k=4 vs others (same seeds; cubic compared on its covered rows only via SymGWR-R-IDW MAPE on all rows -> flagged)
    tests = []
    for kind in SPLITS:
        r4 = acc[(acc.split == kind) & (acc.method == "SymGWR-R-IDW_k4")].sort_values("seed")
        for other in ["plain_polynomial", "SymGWR-T", "SymGWR-R", "SymGWR-R-IDW_k2", "SymGWR-R-IDW_k8"]:
            o = acc[(acc.split == kind) & (acc.method == other)].sort_values("seed")
            for metric in ("rmse", "mape_raw"):
                x, y = r4[metric].to_numpy(), o[metric].to_numpy()
                t, pt = stats.ttest_rel(x, y)
                try:
                    w, pw = stats.wilcoxon(x, y)
                except ValueError:
                    w, pw = np.nan, np.nan
                tests.append(dict(split=kind, ridw_k4_vs=other, metric=metric, mean_ridw_k4=x.mean(), mean_other=y.mean(),
                                  ratio_ridw_over_other=x.mean() / y.mean(), n_seeds_ridw_better=int((x < y).sum()),
                                  t_p=pt, wilcoxon_p=pw))
    pd.DataFrame(tests).to_csv(os.path.join(out_dir, "ridw_paired_tests.csv"), index=False)

    # append to interp_summary.csv / _long.csv (one row per new method; existing rows untouched)
    sp_path = os.path.join(out_dir, "interp_summary.csv")
    summ = pd.read_csv(sp_path, index_col="method", keep_default_na=False)
    new_names = {"plain_polynomial": "plain_polynomial", "SymGWR-R-IDW_k2": "SymGWR-R-IDW_k2",
                 "SymGWR-R-IDW_k4": "SymGWR-R-IDW_k4", "SymGWR-R-IDW_k8": "SymGWR-R-IDW_k8"}
    summ = summ.drop(index=[n for n in new_names.values() if n in summ.index])
    new_rows = {}
    for src, name in new_names.items():
        new_rows[name] = {kind: ms(acc[(acc.split == kind) & (acc.method == src)].mape_raw, 3) for kind in SPLITS}
    summ = pd.concat([summ, pd.DataFrame(new_rows).T[SPLITS].rename_axis("method")])
    summ.to_csv(sp_path)
    lp = os.path.join(out_dir, "interp_summary_long.csv")
    lg = pd.read_csv(lp)
    lg = lg[~lg.method.isin(new_names.values())]
    add = []
    for src, name in new_names.items():
        for kind in SPLITS:
            v = acc[(acc.split == kind) & (acc.method == src)].mape_raw.to_numpy()
            add.append(dict(method=name, split=kind, mape_raw_mean=v.mean(), mape_raw_std_ddof1=v.std(ddof=1),
                            n_seeds=len(v), row_coverage=1.0))
    pd.concat([lg, pd.DataFrame(add)], ignore_index=True).to_csv(lp, index=False)
    return tab, pd.DataFrame(tests)


def continuity_summary(out_dir):
    c = pd.read_csv(os.path.join(out_dir, "ridw_continuity_per_seed.csv"))
    rows = []
    for key, d in c.groupby(["split", "k", "event"]):
        r = dict(split=key[0], k=key[1], event=key[2], n_seeds=len(d),
                 frac_segments_with_event_mean=d.frac_segments_with_event.mean(),
                 n_events_total_mean=d.n_events_total.mean(), tie_plane_excluded_total=int(d.n_tie_plane_excluded.sum()),
                 test_mae_raw_mean=d.test_mae_raw.mean())
        for nm in ("value_jump_raw", "value_jump_over_stdG", "value_jump_over_test_mae", "grad_jump_over_gradnorm"):
            r[f"{nm}_median(mean)"] = d[f"{nm}_median"].mean()
            r[f"{nm}_p95(mean)"] = d[f"{nm}_p95"].mean()
            r[f"{nm}_max(max)"] = d[f"{nm}_max"].max()
        if "analytic_max_abs_dev_z" in d:
            r["analytic_max_abs_dev_z(max)"] = d.analytic_max_abs_dev_z.max()
        rows.append(r)
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(out_dir, "ridw_continuity_summary.csv"), index=False)
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--splits", nargs="*", default=SPLITS)
    ap.add_argument("--tables-only", action="store_true")
    args = ap.parse_args()
    if not args.tables_only:
        run(args.splits, args.seeds, args.out_dir)
    tab, tests = build_tables(args.out_dir)
    s = continuity_summary(args.out_dir)
    pd.set_option("display.width", 300, "display.max_columns", 50, "display.max_colwidth", 60)
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
