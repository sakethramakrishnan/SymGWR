#!/usr/bin/env python
"""SymGWR-R-CS (compactly supported) vs SymGWR-R-IDW, k in {2,4,8}.

  IDW : Ghat(q) = P(q) + sum_{i in kNN(q)} w_i res_i,  w_i ~ 1/(d_i+eps)^2          (run_ridw.py)
  CS  : Ghat(q) = P(q) + sum_{i in kNN(q)} w_i res_i,  w_i ~ max(0, 1/d_i^2 - 1/d_{k+1}^2)
        (support radius = (k+1)-th nearest distance; the swapping neighbour has weight exactly 0)
        res_i = G_i - P(r_i).  Weights are normalised.  If all raw weights vanish (the k+1 nearest
        references are exactly equidistant) the blend is 0/0; we use the uniform mean of the k
        residuals there and COUNT those queries (n_cs_degenerate).

Event types audited on 5,000 short segments per split-seed (identical segments for IDW and CS,
same tie-plane exclusion as run_continuity_audit.py):
  nearest_ref_switch    nearest reference changes
  kset_change           the set of the k nearest references changes (k-th <-> (k+1)-th swap)
  support_ref_change    (CS only) the identity of the (k+1)-th reference changes with the k-set
                        unchanged: the support radius d_{k+1} stays continuous but its gradient jumps

Outputs: results/camera_ready/ridwcs_*.csv, appended rows in interp_summary.csv / _long.csv.
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
import run_ridw as rw
from camera_ready_common import (
    OUT_DIR, SEEDS, SPLITS, fit_symgwr, metrics_raw, predict_residual_anchored, predict_symgwr,
)

KS = [2, 4, 8]
KINDS = ["idw", "cs"]
D_FLOOR = 1e-150


# ----------------------------------------------------------------------
def cs_blend(fit, Q, k, want_grad=False, idx=None):
    """B(q) = sum_i w_i res_i with compact-support weights; also grad B and a degenerate flag."""
    if idx is None:
        d, idx = fit.tree.query(Q, k=k + 1, workers=ca.KD_WORKERS)
    else:
        d = np.linalg.norm(Q[:, None, :] - fit.ref_X[idx], axis=2)
    res = (fit.ref_y - fit.ref_P)[idx[:, :k]]
    ds = np.maximum(d, D_FLOOR)
    inv2 = 1.0 / ds ** 2
    raw = np.maximum(0.0, inv2[:, :k] - inv2[:, k:k + 1])
    W = raw.sum(1)
    degen = W <= 1e-300
    Ws = np.where(degen, 1.0, W)
    B = np.where(degen, res.mean(1), (raw * res).sum(1) / Ws)
    if not want_grad:
        return B, None, degen
    diff = Q[:, None, :] - fit.ref_X[idx]
    gi = -2.0 * diff[:, :k, :] / (ds[:, :k, None] ** 4)
    gm = 2.0 * diff[:, k, :] / (ds[:, k, None] ** 4)
    graw = np.where((raw > 0)[:, :, None] & (d[:, :k, None] > D_FLOOR), gi + gm[:, None, :], 0.0)
    gB = (graw * (res - B[:, None])[:, :, None]).sum(1) / Ws[:, None]
    gB = np.where(degen[:, None], 0.0, gB)
    return B, gB, degen


def blend_any(fit, Q, k, kind, want_grad=False, idx=None):
    if kind == "idw":
        B, g = rw.blend(fit, Q, k, want_grad, idx)
        return B, g
    B, g, _ = cs_blend(fit, Q, k, want_grad, idx)
    return B, g


def nb_count(kind, k):
    return k if kind == "idw" else k + 1


# ----------------------------------------------------------------------
def find_events(fit, k, A, B, grid, mode="set", chunk=250):
    """Grid + bisection detection of changes of the k-set (mode='set') or of the identity of the
    (k+1)-th reference (mode='rank'); tie-plane switches (adjacent-rank gap < TIE_TOL at both bracketing
    grid points) are counted separately and excluded."""
    t = np.linspace(0.0, 1.0, grid + 1)
    n_sw = np.zeros(len(A), dtype=np.int64)
    n_tie = np.zeros(len(A), dtype=np.int64)
    out = dict(qm=[], qp=[], seg=[])
    W = ca.KD_WORKERS
    nq = k + 2
    g0, g1 = (k - 1, k) if mode == "set" else (k, k + 1)

    def key_of(idx):
        return np.sort(idx[..., :k], axis=-1) if mode == "set" else idx[..., k:k + 1]

    for s0 in range(0, len(A), chunk):
        a, b = A[s0:s0 + chunk], B[s0:s0 + chunk]
        D = b - a
        pts = a[:, None, :] + t[None, :, None] * D[:, None, :]
        dd, idx = fit.tree.query(pts.reshape(-1, a.shape[1]), k=nq, workers=W)
        dd = dd.reshape(len(a), grid + 1, nq)
        key = key_of(idx.reshape(len(a), grid + 1, nq))
        chg = (key[:, 1:] != key[:, :-1]).any(-1)
        s, i = np.nonzero(chg)
        if len(s) == 0:
            continue
        cell = key[s, i]
        lo, hi = t[i].copy(), t[i + 1].copy()
        Ax, Dx = a[s], D[s]
        for _ in range(ca.BISECT_ITERS):
            mid = 0.5 * (lo + hi)
            _, im = fit.tree.query(Ax + mid[:, None] * Dx, k=nq, workers=W)
            same = (key_of(im.reshape(len(s), nq)) == cell).all(1)
            lo = np.where(same, mid, lo)
            hi = np.where(same, hi, mid)
        qm, qp = Ax + lo[:, None] * Dx, Ax + hi[:, None] * Dx
        tie = (np.abs(dd[s, i, g0] - dd[s, i, g1]) < ca.TIE_TOL) & (np.abs(dd[s, i + 1, g0] - dd[s, i + 1, g1]) < ca.TIE_TOL)
        n_tie[s0:s0 + chunk] = np.bincount(s[tie], minlength=len(a))
        keep = ~tie
        n_sw[s0:s0 + chunk] = np.bincount(s[keep], minlength=len(a))
        out["qm"].append(qm[keep]); out["qp"].append(qp[keep]); out["seg"].append(s[keep] + s0)
    d_ = A.shape[1]
    if not out["qm"]:
        return n_sw, n_tie, dict(qm=np.empty((0, d_)), qp=np.empty((0, d_)), seg=np.empty(0, int))
    return n_sw, n_tie, {kk: np.concatenate(v) if kk == "seg" else np.vstack(v) for kk, v in out.items()}


def kset_unchanged(fit, k, qm, qp):
    _, im = fit.tree.query(qm, k=k, workers=ca.KD_WORKERS)
    _, ip = fit.tree.query(qp, k=k, workers=ca.KD_WORKERS)
    return (np.sort(im.reshape(len(qm), k), 1) == np.sort(ip.reshape(len(qm), k), 1)).all(1)


# ----------------------------------------------------------------------
def jump_row(fit, kind, k, event, n_sw, n_tie, qm, qp, cache, std_G, scale, gnorm, mae_k):
    sel = cache["sel"]
    row = dict(kind=kind, k=k, event=event, n_segments=ca.N_SEG, frac_segments_with_event=float((n_sw > 0).mean()),
               n_events_total=int(cache["n_tot"]), n_events_evaluated=int(len(sel)),
               n_tie_plane_excluded=int(n_tie.sum()), test_mae_raw=mae_k, std_G_raw=std_G, gradnorm_median_model=gnorm)
    if len(sel):
        Bm, gBm = blend_any(fit, qm[sel], k, kind, True)
        Bp, gBp = blend_any(fit, qp[sel], k, kind, True)
        vj_z = np.abs((cache["Pp"] + Bp) - (cache["Pm"] + Bm))
        vj = vj_z * scale
        gj = np.linalg.norm((cache["gPp"] + gBp) - (cache["gPm"] + gBm), axis=1)
    else:
        vj_z, vj, gj = np.empty(0), np.empty(0), np.empty(0)
    for nm, arr in (("value_jump_raw", vj), ("value_jump_z", vj_z), ("value_jump_over_test_mae", vj / mae_k),
                    ("grad_jump_over_gradnorm", gj / gnorm)):
        for qn, qv in ca.quant(arr).items():
            row[f"{nm}_{qn}"] = qv
    row["frac_events_value_jump_gt_1e-9_Jmol"] = float((vj > 1e-9).mean()) if len(vj) else np.nan
    row["frac_events_grad_jump_gt_1e-9_rel"] = float((gj / gnorm > 1e-9).mean()) if len(gj) else np.nan
    if len(sel) and event == "nearest_ref_switch":
        unch = kset_unchanged(fit, k, qm[sel], qp[sel])
        row["frac_events_with_kset_change"] = float(1 - unch.mean())
    return row


def hessian_row(fit, kind, k):
    Q = fit.split.X_test
    HP = fit.model.hess(Q)
    _, nb = fit.tree.query(Q, k=nb_count(kind, k), workers=ca.KD_WORKERS)
    nb = nb.reshape(len(Q), -1)
    h = 1e-6
    HB = np.empty_like(HP)
    for j in range(Q.shape[1]):
        e = np.zeros(Q.shape[1]); e[j] = h
        HB[:, :, j] = (blend_any(fit, Q + e, k, kind, True, nb)[1] - blend_any(fit, Q - e, k, kind, True, nb)[1]) / (2 * h)
    HB = 0.5 * (HB + HB.transpose(0, 2, 1))
    fro = lambda M: np.linalg.norm(M, axis=(1, 2))
    sv = np.linalg.svd(HP + HB, compute_uv=False)
    return dict(kind=kind, k=k, hess_P_frobenius_median=float(np.median(fro(HP))),
                hess_blend_frobenius_median=float(np.median(fro(HB))), hess_blend_frobenius_p95=float(np.quantile(fro(HB), .95)),
                hess_blend_over_P_frobenius_median=float(np.median(fro(HB) / fro(HP))),
                frac_queries_blend_hessian_nonzero=float((fro(HB) > 1e-10).mean()),
                hess_total_rank_rel1e8_min=int((sv > 1e-8 * sv[:, :1]).sum(1).min()),
                hess_total_rank_rel1e8_max=int((sv > 1e-8 * sv[:, :1]).sum(1).max()))


def gradient_check_row(fit, kind, k, rng, n=300):
    """grad B at tiny offsets from references (nodes), and the size of grad B at test queries."""
    j = rng.choice(len(fit.ref_X), n, replace=False)
    u = rng.normal(size=(n, fit.ref_X.shape[1])) * (fit.ref_X.std(0) > 0)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    row = dict(kind=kind, k=k)
    for off in (1e-3, 1e-5):
        gB = blend_any(fit, fit.ref_X[j] + off * u, k, kind, True)[1]
        r = np.linalg.norm(gB, axis=1) / np.linalg.norm(fit.ref_grad[j], axis=1)
        row[f"gradB_over_gradP_at_node_offset{off:g}_median"] = float(np.median(r))
        row[f"gradB_over_gradP_at_node_offset{off:g}_max"] = float(r.max())
    gBt = blend_any(fit, fit.split.X_test, k, kind, True)[1]
    gP = fit.ref_grad[fit.tree.query(fit.split.X_test, k=1)[1]]
    row["gradB_over_gradP_at_test_median"] = float(np.median(np.linalg.norm(gBt, axis=1) / np.linalg.norm(gP, axis=1)))
    return row


# ----------------------------------------------------------------------
def run(splits, seeds, out_dir):
    acc, cont, hess, grad = [], [], [], []
    for kind_split in splits:
        for seed in seeds:
            fit = fit_symgwr(kind_split, seed)
            sp = fit.split
            Q = sp.X_test
            mae = {}
            for kind in KINDS:
                for k in KS:
                    if kind == "idw":
                        p = fit.model.value(Q) + rw.blend(fit, Q, k)[0]
                        extra = {}
                    else:
                        B, _, degen = cs_blend(fit, Q, k)
                        p = fit.model.value(Q) + B
                        extra = dict(n_cs_degenerate_test_queries=int(degen.sum()))
                    r = metrics_raw(fit, p)
                    name = f"SymGWR-R-{kind.upper()}_k{k}"
                    mae[(kind, k)] = r["mae_raw"]
                    acc.append(dict(split=kind_split, seed=seed, method=name, mape_raw=r["mape_raw"], mae_raw=r["mae_raw"],
                                    rmse=r["rmse"], r2=r["r2"], **extra))
            mae1 = metrics_raw(fit, predict_residual_anchored(fit, Q))["mae_raw"]
            # exactness at references
            for k in KS:
                e = float(np.abs(fit.model.value(fit.ref_X[:500]) + cs_blend(fit, fit.ref_X[:500], k)[0] - fit.ref_y[:500]).max())
                acc.append(dict(split=kind_split, seed=seed, method=f"_check_CS_k{k}_max_abs_err_at_refs_z", mape_raw=e))

            rng = np.random.default_rng(30_000 + seed)
            scale = float(sp.target_scaler.scale_[0])
            std_G = float(np.std(sp.target_scaler.inverse_transform(fit.ref_y.reshape(-1, 1)).ravel()))
            gnorm = float(np.median(np.linalg.norm(fit.ref_grad, axis=1)))
            A, B = rw.short_segments(fit, rng)
            n_sw, n_tie, sw = ca.find_switches(fit, A, B, ca.GRID_SHORT)
            cache = rw.pair_cache(fit, sw["qm"], sw["qp"], rng)
            for kind in KINDS:
                for k in KS:
                    cont.append(dict(split=kind_split, seed=seed, **jump_row(
                        fit, kind, k, "nearest_ref_switch", n_sw, n_tie, sw["qm"], sw["qp"], cache, std_G, scale, gnorm, mae[(kind, k)])))
            for k in KS:
                n_k, nt_k, swk = find_events(fit, k, A, B, ca.GRID_SHORT, "set")
                ck = rw.pair_cache(fit, swk["qm"], swk["qp"], rng)
                for kind in KINDS:
                    cont.append(dict(split=kind_split, seed=seed, **jump_row(
                        fit, kind, k, "kset_change", n_k, nt_k, swk["qm"], swk["qp"], ck, std_G, scale, gnorm, mae[(kind, k)])))
                # CS-only: identity of the (k+1)-th reference changes, k-set unchanged
                n_r, nt_r, swr = find_events(fit, k, A, B, ca.GRID_SHORT, "rank")
                if len(swr["qm"]):
                    keep = kset_unchanged(fit, k, swr["qm"], swr["qp"])
                    swr = dict(qm=swr["qm"][keep], qp=swr["qp"][keep], seg=swr["seg"][keep])
                    n_r = np.bincount(swr["seg"], minlength=len(A))
                cr = rw.pair_cache(fit, swr["qm"], swr["qp"], rng)
                cont.append(dict(split=kind_split, seed=seed, **jump_row(
                    fit, "cs", k, "support_ref_change", n_r, nt_r, swr["qm"], swr["qp"], cr, std_G, scale, gnorm, mae[("cs", k)])))
                for kind in KINDS:
                    hess.append(dict(split=kind_split, seed=seed, **hessian_row(fit, kind, k)))
                    grad.append(dict(split=kind_split, seed=seed, **gradient_check_row(fit, kind, k, rng)))
            print(f"[{kind_split} seed={seed}] done", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(acc).to_csv(os.path.join(out_dir, "ridwcs_accuracy_per_seed.csv"), index=False)
    pd.DataFrame(cont).to_csv(os.path.join(out_dir, "ridwcs_continuity_per_seed.csv"), index=False)
    pd.DataFrame(hess).to_csv(os.path.join(out_dir, "ridwcs_hessian_per_seed.csv"), index=False)
    pd.DataFrame(grad).to_csv(os.path.join(out_dir, "ridwcs_gradient_behaviour.csv"), index=False)


# ----------------------------------------------------------------------
def ms(x, d=3):
    x = np.asarray(x, float)
    return f"{np.nanmean(x):.{d}e} ± {np.nanstd(x, ddof=1):.{d}e}" if np.isfinite(x).all() and len(x) > 1 else "n/a"


def build_tables(out_dir):
    new = pd.read_csv(os.path.join(out_dir, "ridwcs_accuracy_per_seed.csv"))
    new = new[~new.method.str.startswith("_check")]
    old = pd.read_csv(os.path.join(out_dir, "ridw_accuracy_per_seed.csv"))
    base = pd.read_csv(os.path.join(out_dir, "interp_baselines_per_seed.csv"))
    cub = base[((base.split != "composition_group") & (base.method == "b_cubic_T")) |
               ((base.split == "composition_group") & (base.method == "e_cubic_arm"))].copy()
    cov = cub.groupby("split").apply(lambda d: d.n_eval.sum() / d.n_test.sum(), include_groups=False)
    cub = cub.assign(method="cubic_T_or_arm")[["split", "seed", "method", "mape_raw", "mae_raw", "rmse", "r2"]]
    # IDW rows: recomputed here must equal run_ridw.py's
    chk = old[old.method.str.startswith("SymGWR-R-IDW")].merge(new, on=["split", "seed", "method"], suffixes=("_old", "_new"))
    print("max |RMSE(IDW) run_ridw - run_ridw_cs|:", float((chk.rmse_old - chk.rmse_new).abs().max()))
    allm = pd.concat([old[old.method.isin(["plain_polynomial", "SymGWR-T", "SymGWR-R"])], new, cub], ignore_index=True)
    order = ["plain_polynomial", "SymGWR-T", "SymGWR-R"] + [f"SymGWR-R-{a}_k{k}" for a in ("IDW", "CS") for k in KS] + ["cubic_T_or_arm"]
    rows = []
    for ks in SPLITS:
        for m in order:
            d = allm[(allm.split == ks) & (allm.method == m)].sort_values("seed")
            if d.empty or d.rmse.isna().all():
                rows.append(dict(split=ks, method=m, rmse_J_per_mol="n/a", mae_J_per_mol="n/a", mape_raw="n/a", r2="n/a"))
                continue
            note = f" [{cov[ks]:.0%} of rows]" if m == "cubic_T_or_arm" and cov.get(ks, 1) < 0.999 else ""
            rows.append(dict(split=ks, method=m, rmse_J_per_mol=ms(d.rmse), mae_J_per_mol=ms(d.mae_raw),
                             mape_raw=ms(d.mape_raw) + note, r2=f"{d.r2.mean():.10f} ± {d.r2.std(ddof=1):.1e}"))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out_dir, "ridwcs_combined_table.csv"), index=False)

    tests = []
    for ks in SPLITS:
        r4 = new[(new.split == ks) & (new.method == "SymGWR-R-CS_k4")].sort_values("seed")
        for other in ["SymGWR-R-IDW_k4", "SymGWR-R", "plain_polynomial", "SymGWR-T"]:
            src = new if other.startswith("SymGWR-R-IDW") else old
            o = src[(src.split == ks) & (src.method == other)].sort_values("seed")
            for metric in ("rmse", "mape_raw"):
                x, y = r4[metric].to_numpy(), o[metric].to_numpy()
                t, pt = stats.ttest_rel(x, y)
                try:
                    w, pw = stats.wilcoxon(x, y)
                except ValueError:
                    w, pw = np.nan, np.nan
                tests.append(dict(split=ks, cs_k4_vs=other, metric=metric, mean_cs_k4=x.mean(), mean_other=y.mean(),
                                  ratio=x.mean() / y.mean(), n_seeds_cs_better=int((x < y).sum()), t_p=pt, wilcoxon_p=pw))
    pd.DataFrame(tests).to_csv(os.path.join(out_dir, "ridwcs_paired_tests.csv"), index=False)

    # append CS rows to interp_summary.csv / _long.csv (existing rows untouched)
    sp_path = os.path.join(out_dir, "interp_summary.csv")
    summ = pd.read_csv(sp_path, index_col="method", keep_default_na=False)
    names = [f"SymGWR-R-CS_k{k}" for k in KS]
    summ = summ.drop(index=[n for n in names if n in summ.index])
    add = {n: {ks: ms(new[(new.split == ks) & (new.method == n)].mape_raw) for ks in SPLITS} for n in names}
    pd.concat([summ, pd.DataFrame(add).T[SPLITS].rename_axis("method")]).to_csv(sp_path)
    lp = os.path.join(out_dir, "interp_summary_long.csv")
    lg = pd.read_csv(lp)
    lg = lg[~lg.method.isin(names)]
    rows_l = []
    for n in names:
        for ks in SPLITS:
            v = new[(new.split == ks) & (new.method == n)].mape_raw.to_numpy()
            rows_l.append(dict(method=n, split=ks, mape_raw_mean=v.mean(), mape_raw_std_ddof1=v.std(ddof=1), n_seeds=len(v), row_coverage=1.0))
    pd.concat([lg, pd.DataFrame(rows_l)], ignore_index=True).to_csv(lp, index=False)
    return tab


def continuity_summary(out_dir):
    c = pd.read_csv(os.path.join(out_dir, "ridwcs_continuity_per_seed.csv"))
    rows = []
    for key, d in c.groupby(["split", "kind", "k", "event"]):
        r = dict(split=key[0], kind=key[1], k=key[2], event=key[3], n_seeds=len(d),
                 frac_segments_with_event_mean=d.frac_segments_with_event.mean(), n_events_total_mean=d.n_events_total.mean(),
                 tie_plane_excluded_total=int(d.n_tie_plane_excluded.sum()), test_mae_raw_mean=d.test_mae_raw.mean(),
                 frac_events_value_jump_gt_1e9_Jmol_mean=d["frac_events_value_jump_gt_1e-9_Jmol"].mean(),
                 frac_events_grad_jump_gt_1e9_rel_mean=d["frac_events_grad_jump_gt_1e-9_rel"].mean())
        for nm in ("value_jump_raw", "value_jump_z", "value_jump_over_test_mae", "grad_jump_over_gradnorm"):
            r[f"{nm}_median(mean)"] = d[f"{nm}_median"].mean()
            r[f"{nm}_p95(mean)"] = d[f"{nm}_p95"].mean()
            r[f"{nm}_max(max)"] = d[f"{nm}_max"].max()
        rows.append(r)
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(out_dir, "ridwcs_continuity_summary.csv"), index=False)
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
    tab = build_tables(args.out_dir)
    continuity_summary(args.out_dir)
    pd.set_option("display.width", 300, "display.max_columns", 50, "display.max_colwidth", 60)
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
