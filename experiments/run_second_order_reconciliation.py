#!/usr/bin/env python
"""Reconcile the d2G/dT2 sweep (~11% normalised error, r ~ 0.64 vs raw FD) with the Newton test (~3% Jacobian error).

Read-only with respect to gwr/ and to every existing output; writes only
  results/camera_ready/second_order_reconciliation.md
  results/camera_ready/second_order_reconciliation.csv   (every number in the note)

Same polynomial as both experiments (degree 5, protocol "random", seeds 42-51, fit on train+val,
chain rule through the scalers).  Two references for d2G/dT2 (raw units):
  FD      central 3-point second difference of the RAW G table (run_calphad_second_derivative_fd.build_TT)
  spline  analytic G'' of the degree-4 smoothing spline used as truth in run_newton_test (PPolyEval.smooth_truth)
Cp = -T d2G/dT2.  "Jacobian error" = the Newton metric |R'_P - R'_ref| / R'_ref with R' = Cp + a,
a = dt*kappa, kappa read from the reference trajectory of each composition (calibrate_a).

Point sets
  grid        all 7742 FD-eligible (composition, T) rows of the sweep (T 1010..1790 K, 10 K step)
  grid_T1050  the same, restricted to T >= 1050 K (the range the Newton reference path covers)
  path        the Newton reference trajectory of every composition (T_n, 1800 -> 1050 K, off the T grid),
              spline reference only (FD needs table rows)
Normalisation (sweep): mean|P'' - ref| / mean|ref| over the point set.  Newton: per-point
|R'_P - R'_ref| / R'_ref, then the mean.  Both are reported on every set.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from camera_ready_common import SEEDS, TEMP_COL, composition_ids, fit_symgwr  # noqa: E402
from run_calphad_second_derivative_fd import G_ROUND_SIGMA, build_TT  # noqa: E402
from run_newton_test import (  # noqa: E402
    DT, N_STEPS, SIGMA_ROUND, T_HI, T_LO, PPolyEval, calibrate_a, trajectory,
)

Q_BIG, Q_SMALL, BIG_ABS = 1.0, 0.01, 1e5  # table quantum (J/mol) for |G| >= / < 1e5; asserted on the data below
MAX_M = 5  # widest stencil half-width, in grid steps
T_PATH_MIN = 1050.0


def quantum(g):
    return np.where(np.abs(g) >= BIG_ABS, Q_BIG, Q_SMALL)


def stats(p2, ref2, T, a=None):
    """Normalised errors of the polynomial's d2G/dT2 (p2) against a reference (ref2) at temperatures T."""
    e = np.abs(p2 - ref2)
    cp_p, cp_r = -T * p2, -T * ref2
    out = {
        "n": len(p2),
        "mean_abs_ref_g2": float(np.mean(np.abs(ref2))),
        "norm_err_g2": float(e.mean() / np.mean(np.abs(ref2))),
        "pearson_g2": float(np.corrcoef(p2, ref2)[0, 1]),
        "norm_err_cp": float(np.mean(np.abs(cp_p - cp_r)) / np.mean(np.abs(cp_r))),
        "mean_abs_ref_cp": float(np.mean(np.abs(cp_r))),
        "frac_ref_cp_nonpositive": float(np.mean(cp_r <= 0)),
    }
    den = np.abs(cp_r)
    out["relerr_cp_mean"] = float(np.mean(np.abs(cp_p - cp_r) / den))
    out["relerr_cp_median"] = float(np.median(np.abs(cp_p - cp_r) / den))
    if a is not None:
        j = np.abs(cp_p - cp_r) / np.abs(cp_r + a)
        out["jac_relerr_mean"] = float(np.mean(j))
        out["jac_relerr_median"] = float(np.median(j))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results/camera_ready")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()

    f0 = fit_symgwr("random", args.seeds[0])
    sp0 = f0.split
    df, feats = sp0.df, sp0.feature_cols
    assert (df.index.to_numpy() == np.arange(len(df))).all()
    kT = feats.index(TEMP_COL)
    comp_cols = [c for c in feats if c != TEMP_COL]
    cid = composition_ids(df, feats)
    comps = np.unique(cid)
    Tall = df[TEMP_COL].to_numpy(float)
    Gall = df[sp0._target].to_numpy(float)
    X_raw = df[feats].to_numpy(float)

    # table quantisation: the premise of the noise model, checked on the data
    big = np.abs(Gall) >= BIG_ABS
    assert np.allclose(Gall[big], np.round(Gall[big], 0)), "|G| >= 1e5 rows are not integer-valued"
    assert np.allclose(Gall[~big] * 100, np.round(Gall[~big] * 100)), "|G| < 1e5 rows are not multiples of 0.01"
    share_big = float(big.mean())

    # ---- spline truth, Newton reference path, kappa --------------------------------------
    S0 = np.full(len(df), np.nan)
    S2 = np.full(len(df), np.nan)
    kappa_row = np.full(len(df), np.nan)
    path_rows, path_T, path_a, path_S2 = [], [], [], []
    n_knots = []
    for c in comps:
        r = np.where(cid == c)[0]
        o = np.argsort(Tall[r])
        r = r[o]
        ref_ev, u = PPolyEval.smooth_truth(Tall[r], Gall[r])
        n_knots.append(len(u.get_knots()) - 2)
        g = np.array([ref_ev.gtt(t) for t in Tall[r]])
        S0[r], S2[r] = g[:, 0], g[:, 2]
        a = calibrate_a(ref_ev)
        Tref, _ = trajectory(ref_ev, a, ref_ev)
        keep = (Tref >= T_LO) & (Tref <= T_HI)
        path_rows += [r[0]] * int(keep.sum())
        path_T += list(Tref[keep])
        path_a += [a] * int(keep.sum())
        path_S2 += [ref_ev.gtt(t)[2] for t in Tref[keep]]
        kappa_row[r] = a
    path_rows, path_T, path_a, S2_path = (np.asarray(x) for x in (path_rows, path_T, path_a, path_S2))
    # consistency with the stored Newton run (reference rows carry kappa per composition)
    kappa_check = None
    nr_path = os.path.join(args.out_dir, "newton_runs.csv")
    if os.path.exists(nr_path):
        nr = pd.read_csv(nr_path)
        nr = nr[nr.surrogate == "reference"].set_index("comp_id").kappa
        mine = pd.Series({int(c): kappa_row[np.where(cid == c)[0][0]] / DT for c in comps})
        kappa_check = float(np.max(np.abs(mine - nr.loc[mine.index]) / nr.loc[mine.index]))

    # ---- FD references ----------------------------------------------------------------
    tt = build_TT(df, comp_cols)  # centre row, lo/hi rows, fd, stencil weights
    r = tt.row.to_numpy().astype(np.int64)
    lo, hi = tt.lo.to_numpy().astype(np.int64), tt.hi.to_numpy().astype(np.int64)
    fd = tt.fd.to_numpy()
    T_g = Tall[r]
    pred_noise = np.sqrt((tt.w_lo.to_numpy() * quantum(Gall[lo])) ** 2 + (tt.w_c.to_numpy() * quantum(Gall[r])) ** 2
                         + (tt.w_hi.to_numpy() * quantum(Gall[hi])) ** 2) / np.sqrt(12.0)
    old_noise = tt.round_rms.to_numpy()  # what run_calphad_second_derivative_fd assumed (0.01 for every value)
    fine = (Gall[[lo, r, hi]] ** 2 < BIG_ABS ** 2).all(axis=0)  # all three stencil values on the 0.01 quantum
    s2_g = S2[r]
    a_g = kappa_row[r]

    # wide stencils on the complete (composition x T) lattice
    order = np.lexsort((Tall, cid))
    n_T = df[TEMP_COL].nunique()
    assert len(order) == len(comps) * n_T, "grid is not complete"
    Tm = Tall[order].reshape(len(comps), n_T)
    assert np.allclose(Tm, Tm[0]) and np.allclose(np.diff(Tm[0]), np.diff(Tm[0])[0])
    h = float(np.diff(Tm[0])[0])
    Rm = order.reshape(len(comps), n_T)
    idx = np.arange(MAX_M, n_T - MAX_M)  # centre columns valid for every half-width m
    wide = {}
    for m in range(1, MAX_M + 1):
        Rc, Rl, Rh = Rm[:, idx], Rm[:, idx - m], Rm[:, idx + m]
        d = (m * h) ** 2
        wide[m] = dict(
            r=Rc.ravel(), fd=((Gall[Rl] - 2 * Gall[Rc] + Gall[Rh]) / d).ravel(),
            noise=(np.sqrt(quantum(Gall[Rl]) ** 2 + 4 * quantum(Gall[Rc]) ** 2 + quantum(Gall[Rh]) ** 2)
                   / np.sqrt(12.0) / d).ravel(),
            trunc_spline=(((S0[Rl] - 2 * S0[Rc] + S0[Rh]) / d) - S2[Rc]).ravel(),
        )

    # ---- polynomial d2G/dT2 per seed -----------------------------------------------------
    recs = []  # (seed, set, ref, metric, value)

    def add(seed, set_, ref, d):
        for k, v in d.items():
            recs.append((seed, set_, ref, k, v))

    for seed in args.seeds:
        fit = fit_symgwr("random", seed)
        sp, pm = fit.split, fit.model
        t_scale, f_scale = float(sp.target_scaler.scale_[0]), np.asarray(sp.feature_scaler.scale_)
        conv = t_scale / f_scale[kT] ** 2

        def p2_at(Xr):
            Xs = sp.feature_scaler.transform(Xr)
            return pm.hess(Xs)[:, kT, kT] * conv

        all_rows = np.arange(len(df))
        P2 = np.empty(len(df))
        P2[all_rows] = p2_at(X_raw)
        Xq = X_raw[path_rows].copy()
        Xq[:, kT] = path_T
        P2_path = p2_at(Xq)

        p2_g = P2[r]
        for name, mask in (("grid", np.ones(len(r), bool)), ("grid_T1050", T_g >= T_PATH_MIN)):
            add(seed, name, "FD", stats(p2_g[mask], fd[mask], T_g[mask], a_g[mask]))
            add(seed, name, "spline", stats(p2_g[mask], s2_g[mask], T_g[mask], a_g[mask]))
        add(seed, "path", "spline", stats(P2_path, S2_path, path_T, path_a))
        for name, mask in (("grid_fine_stencil", fine), ("grid_coarse_stencil", ~fine)):
            add(seed, name, "FD", stats(p2_g[mask], fd[mask], T_g[mask], a_g[mask]))
            add(seed, name, "spline", stats(p2_g[mask], s2_g[mask], T_g[mask], a_g[mask]))

        # error decomposition on the grid: e_PF = P - FD = (P - S) + (S - FD)
        e_ps, e_sf, e_pf = p2_g - s2_g, s2_g - fd, p2_g - fd
        add(seed, "grid", "decomp", {
            "rms_P_minus_FD": float(np.sqrt(np.mean(e_pf ** 2))),
            "rms_P_minus_spline": float(np.sqrt(np.mean(e_ps ** 2))),
            "rms_spline_minus_FD": float(np.sqrt(np.mean(e_sf ** 2))),
            "corr_Pminusspline_splineminusFD": float(np.corrcoef(e_ps, e_sf)[0, 1]),
            "rms_pred_rounding_noise": float(np.sqrt(np.mean(pred_noise ** 2))),
            "rms_sweep_assumed_rounding_noise": float(np.sqrt(np.mean(old_noise ** 2))),
            "mean_abs_FD": float(np.mean(np.abs(fd))), "mean_abs_spline": float(np.mean(np.abs(s2_g))),
            "norm_err_spline_vs_FD": float(np.mean(np.abs(e_sf)) / np.mean(np.abs(fd))),
        })
        # widening the stencil averages the table noise down (FD noise ~ 1/h^2) at the cost of truncation
        for m, w in wide.items():
            rr = w["r"]
            p2w = P2[rr]
            add(seed, f"wide_m{m}", "FD", {
                "norm_err_poly_vs_FD": float(np.mean(np.abs(p2w - w["fd"])) / np.mean(np.abs(w["fd"]))),
                "norm_err_poly_vs_spline": float(np.mean(np.abs(p2w - S2[rr])) / np.mean(np.abs(S2[rr]))),
                "norm_err_spline_vs_FD": float(np.mean(np.abs(S2[rr] - w["fd"])) / np.mean(np.abs(w["fd"]))),
                "pearson_poly_FD": float(np.corrcoef(p2w, w["fd"])[0, 1]),
                "rms_pred_rounding_noise_over_mean_abs_FD": float(np.sqrt(np.mean(w["noise"] ** 2))
                                                                  / np.mean(np.abs(w["fd"]))),
                "trunc_spline_over_mean_abs_spline": float(np.mean(np.abs(w["trunc_spline"])) / np.mean(np.abs(S2[rr]))),
                "n": len(rr),
            })
        print(f"seed {seed} done", flush=True)

    long = pd.DataFrame(recs, columns=["seed", "set", "ref", "metric", "value"])
    g = long.groupby(["set", "ref", "metric"], sort=False).value
    summ = pd.concat([g.mean().rename("mean"), g.std(ddof=1).rename("std")], axis=1).reset_index()
    summ["n_seeds"] = len(args.seeds)
    os.makedirs(args.out_dir, exist_ok=True)
    summ.to_csv(os.path.join(args.out_dir, "second_order_reconciliation.csv"), index=False)

    ns = pd.read_csv(os.path.join(args.out_dir, "newton_summary.csv"))
    newton_poly_jac = float(ns[(ns.surrogate == "Polynomial") & (ns.protocol == "random")].jac_relerr_mean.iloc[0])
    context = dict(newton_poly_jac=newton_poly_jac,
        n_comps=len(comps), n_T=n_T, h=h, share_big=share_big, kappa_check=kappa_check,
        max_knots=int(max(n_knots)), g_round_sigma=float(G_ROUND_SIGMA), spline_sigma=SIGMA_ROUND,
        n_grid=len(r), n_path=len(path_T), seeds=args.seeds,
        share_fine=float(fine.mean()), mean_a=float(np.mean(path_a)),
        mean_abs_cp_spline_path=float(np.mean(path_T * np.abs(S2_path))),
    )
    write_markdown(args.out_dir, summ, context)
    print(summ.to_string(float_format=lambda v: f"{v:.4g}"))


def write_markdown(out_dir, summ, ctx):
    S = summ.set_index(["set", "ref", "metric"])

    def v(set_, ref, metric, f=".3g"):
        m, s = S.loc[(set_, ref, metric), "mean"], S.loc[(set_, ref, metric), "std"]
        return f"{m:{f}}" if not np.isfinite(s) or s == 0 else f"{m:{f}} ± {s:.2g}"

    def m(set_, ref, metric):
        return float(S.loc[(set_, ref, metric), "mean"])

    L = ["# d2G/dT2 reconciliation: sweep vs Newton Jacobian", "",
         "Generated by `experiments/run_second_order_reconciliation.py`; every number comes from "
         "`second_order_reconciliation.csv` (mean ± std over seeds "
         f"{ctx['seeds'][0]}-{ctx['seeds'][-1]}, n = {len(ctx['seeds'])}; protocol `random`; degree-5 polynomial). "
         "Nothing in `gwr/` or in existing outputs was changed.", "",
         "## 1. Same polynomial, two references, same normalisation", "",
         "Normalised error = mean|P'' - ref| / mean|ref| (the sweep's definition). "
         f"FD = central 3-point second difference of raw G ({ctx['h']:.0f} K step). "
         f"Spline = degree-4 smoothing spline (Newton truth). Cp = -T d2G/dT2. "
         f"Jacobian error = Newton metric |R'_P - R'_ref| / R'_ref, R' = Cp + a (mean a on the path = {ctx['mean_a']:.3f} J/mol/K).", "",
         "| point set | n | ref | d2G/dT2 norm. err | Pearson r | Cp norm. err | Cp per-point rel. err (mean / median) | Jacobian err (mean / median) | share ref Cp <= 0 |",
         "|---|---|---|---|---|---|---|---|---|"]
    labels = {"grid": "FD-eligible grid (sweep set)", "grid_T1050": "grid, T >= 1050 K",
              "path": "Newton reference path (off-grid)"}
    for set_, refs in (("grid", ("FD", "spline")), ("grid_T1050", ("FD", "spline")), ("path", ("spline",))):
        for ref in refs:
            L.append(f"| {labels[set_]} | {m(set_, ref, 'n'):.0f} | {ref} | {v(set_, ref, 'norm_err_g2')} | "
                     f"{v(set_, ref, 'pearson_g2', '.4f')} | {v(set_, ref, 'norm_err_cp')} | "
                     f"{m(set_, ref, 'relerr_cp_mean'):.3g} / {m(set_, ref, 'relerr_cp_median'):.3g} | "
                     f"{m(set_, ref, 'jac_relerr_mean'):.3g} / {m(set_, ref, 'jac_relerr_median'):.3g} | "
                     f"{m(set_, ref, 'frac_ref_cp_nonpositive'):.3g} |")
    L += ["",
          f"Check: the grid/FD row is the sweep's own number (`derivative_second_order_summary.csv`, d2/dT2, poly, all_grid). "
          + (f"kappa recomputed here matches `newton_runs.csv` reference rows to a max relative difference of {ctx['kappa_check']:.1e}."
             if ctx["kappa_check"] is not None else "`newton_runs.csv` not found; kappa check skipped."), "",
          "## 2. How noisy is the FD reference?", "",
          f"The table is quantised: |G| >= 1e5 J/mol rows ({ctx['share_big']:.1%} of the table) are integers (quantum {Q_BIG:g} J/mol), "
          f"the rest are multiples of {Q_SMALL:g} J/mol (both asserted on the data). "
          f"`run_calphad_second_derivative_fd.py` assumes {Q_SMALL:g} J/mol for every value (G_ROUND_SIGMA = {ctx['g_round_sigma']:.2g} J/mol).", "",
          "| quantity (d2G/dT2, J/(mol K^2), grid) | value |", "|---|---|",
          f"| mean abs FD | {v('grid', 'decomp', 'mean_abs_FD')} |",
          f"| mean abs spline | {v('grid', 'decomp', 'mean_abs_spline')} |",
          f"| rms rounding noise of FD assumed by the sweep (`fd_rounding_rms`) | {v('grid', 'decomp', 'rms_sweep_assumed_rounding_noise')} |",
          f"| rms rounding noise of FD with the actual per-value quantum | {v('grid', 'decomp', 'rms_pred_rounding_noise')} |",
          f"| rms (spline - FD), observed | {v('grid', 'decomp', 'rms_spline_minus_FD')} |",
          f"| rms (P - spline) | {v('grid', 'decomp', 'rms_P_minus_spline')} |",
          f"| rms (P - FD) | {v('grid', 'decomp', 'rms_P_minus_FD')} |",
          f"| corr(P - spline, spline - FD) | {v('grid', 'decomp', 'corr_Pminusspline_splineminusFD', '.3f')} |",
          f"| norm. err of the spline itself vs FD | {v('grid', 'decomp', 'norm_err_spline_vs_FD')} |", "",
          "Split by the quantum of the three stencil values:", "",
          "| stencil | share of grid | n | ref | d2G/dT2 norm. err | Pearson r |", "|---|---|---|---|---|---|"]
    for set_, name, share in (("grid_fine_stencil", "all three |G| < 1e5 (0.01 quantum)", ctx["share_fine"]),
                              ("grid_coarse_stencil", "at least one |G| >= 1e5 (1 quantum)", 1 - ctx["share_fine"])):
        for ref in ("FD", "spline"):
            L.append(f"| {name} | {share:.1%} | {m(set_, ref, 'n'):.0f} | {ref} | {v(set_, ref, 'norm_err_g2')} | "
                     f"{v(set_, ref, 'pearson_g2', '.4f')} |")
    L += ["", "## 3. Widening the FD stencil averages the table noise down (noise ~ 1/h^2)", "",
          f"Half-width m x {ctx['h']:.0f} K, same centre points for every m (n = {m('wide_m1', 'FD', 'n'):.0f}). "
          "Truncation proxy = stencil applied to the spline minus its analytic G'', relative to mean abs spline G''.", "",
          "| m | poly vs FD | poly vs spline | spline vs FD | Pearson(P, FD) | predicted rounding noise / mean abs FD | truncation proxy / mean abs spline |",
          "|---|---|---|---|---|---|---|"]
    for k in range(1, MAX_M + 1):
        s = f"wide_m{k}"
        L.append(f"| {k} | {v(s, 'FD', 'norm_err_poly_vs_FD')} | {v(s, 'FD', 'norm_err_poly_vs_spline')} | "
                 f"{m(s, 'FD', 'norm_err_spline_vs_FD'):.3g} | {v(s, 'FD', 'pearson_poly_FD', '.4f')} | "
                 f"{m(s, 'FD', 'rms_pred_rounding_noise_over_mean_abs_FD'):.3g} | "
                 f"{m(s, 'FD', 'trunc_spline_over_mean_abs_spline'):.3g} |")
    L += [""]
    L += explanation(S, ctx, m)
    with open(os.path.join(out_dir, "second_order_reconciliation.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


def explanation(S, ctx, m):
    """Prose; every number is read from the summary table."""
    g = lambda s, r, k: m(s, r, k)  # noqa: E731
    sq = lambda k: g("grid", "decomp", k) ** 2  # noqa: E731
    noise_share = sq("rms_spline_minus_FD") / sq("rms_P_minus_FD")
    return [
        "## Explanation", "",
        f"1. **The 11% is mostly noise in the FD reference, not polynomial error.** On the sweep's own points the same "
        f"polynomial has normalised error {g('grid', 'FD', 'norm_err_g2'):.1%} (r = {g('grid', 'FD', 'pearson_g2'):.2f}) against raw FD "
        f"and {g('grid', 'spline', 'norm_err_g2'):.1%} (r = {g('grid', 'spline', 'pearson_g2'):.2f}) against the smoothing spline. "
        f"The FD reference is itself {g('grid', 'decomp', 'norm_err_spline_vs_FD'):.1%} off the spline (same normalisation).",
        f"2. **Cause: table quantisation.** {ctx['share_big']:.0%} of the G values are rounded to {Q_BIG:g} J/mol "
        f"(|G| >= 1e5). A 3-point second difference with a {ctx['h']:.0f} K step multiplies that by about 1/h^2: predicted rms noise "
        f"{g('grid', 'decomp', 'rms_pred_rounding_noise'):.2g} J/(mol K^2) with the true per-value quantum, observed rms(spline - FD) "
        f"{g('grid', 'decomp', 'rms_spline_minus_FD'):.2g}, against mean abs d2G/dT2 of {g('grid', 'decomp', 'mean_abs_FD'):.2g}. "
        f"Squared errors add: rms(P - FD)^2 = rms(P - spline)^2 + rms(spline - FD)^2 "
        f"(cross-correlation {g('grid', 'decomp', 'corr_Pminusspline_splineminusFD'):.3f}); the FD noise carries "
        f"{noise_share:.0%} of the squared P-vs-FD error.",
        f"3. **Dose-response.** Stencils whose three values are all on the {Q_SMALL:g} J/mol quantum ({ctx['share_fine']:.0%} of points): "
        f"{g('grid_fine_stencil', 'FD', 'norm_err_g2'):.1%} vs FD, {g('grid_fine_stencil', 'spline', 'norm_err_g2'):.1%} vs spline. "
        f"Stencils touching a {Q_BIG:g} J/mol value: {g('grid_coarse_stencil', 'FD', 'norm_err_g2'):.1%} vs FD "
        f"(r = {g('grid_coarse_stencil', 'FD', 'pearson_g2'):.2f}) but {g('grid_coarse_stencil', 'spline', 'norm_err_g2'):.1%} vs spline. "
        f"Widening the stencil to m = {MAX_M} ({MAX_M * ctx['h']:.0f} K half-width) takes P-vs-FD from "
        f"{g('wide_m1', 'FD', 'norm_err_poly_vs_FD'):.1%} to {g('wide_m%d' % MAX_M, 'FD', 'norm_err_poly_vs_FD'):.1%}, "
        f"which is the P-vs-spline value ({g('wide_m%d' % MAX_M, 'FD', 'norm_err_poly_vs_spline'):.1%}) within the spline-vs-FD residual "
        f"({g('wide_m%d' % MAX_M, 'FD', 'norm_err_spline_vs_FD'):.1%}); the truncation proxy stays at "
        f"{g('wide_m%d' % MAX_M, 'FD', 'trunc_spline_over_mean_abs_spline'):.2%} of mean abs G''. "
        "This second route uses raw data only, so the ~3% is not an artefact of the smoothing spline's own bias.",
        f"4. **Normalisation and point set are minor.** Mean-abs normalisation vs the Newton per-point relative error: "
        f"{g('grid', 'spline', 'norm_err_g2'):.1%} vs {g('grid', 'spline', 'relerr_cp_mean'):.1%} (Cp, spline, grid). "
        f"d2G/dT2 vs Cp: {g('grid', 'spline', 'norm_err_g2'):.1%} vs {g('grid', 'spline', 'norm_err_cp'):.1%}. "
        f"Adding a = dt*kappa to Cp in the denominator (Jacobian) changes the grid/spline mean from {g('grid', 'spline', 'relerr_cp_mean'):.2%} "
        f"to {g('grid', 'spline', 'jac_relerr_mean'):.2%} (Cp ~ {g('grid', 'spline', 'mean_abs_ref_cp'):.0f} J/mol/K dwarfs a). "
        f"Moving from the grid to the Newton reference path (off-grid, 1800 -> 1050 K) gives "
        f"{g('path', 'spline', 'jac_relerr_mean'):.2%} Jacobian error against the Newton script's stored "
        f"{ctx['newton_poly_jac']:.2%} (`newton_summary.csv`, Polynomial, protocol random). The path is off the "
        f"{ctx['h']:.0f} K grid and weights temperatures by where the reference trajectory spends its steps, so it is a different "
        f"average from the grid (grid T >= {T_PATH_MIN:.0f} K, spline: {g('grid_T1050', 'spline', 'norm_err_g2'):.1%}; path: "
        f"{g('path', 'spline', 'norm_err_g2'):.1%}); this residual difference was not decomposed further.",
        "",
        "## Bug in the sweep's error budget (existing outputs not modified)", "",
        f"`run_calphad_second_derivative_fd.py` sets G_ROUND_SIGMA = {Q_SMALL:g}/sqrt(12) for every value, so `fd_rounding_rms` "
        f"in `derivative_second_order*.csv` is {g('grid', 'decomp', 'rms_sweep_assumed_rounding_noise'):.2g} for d2/dT2, "
        f"{g('grid', 'decomp', 'rms_pred_rounding_noise') / g('grid', 'decomp', 'rms_sweep_assumed_rounding_noise'):.0f}x below the quantum-aware "
        f"{g('grid', 'decomp', 'rms_pred_rounding_noise'):.2g}. Its `rounding_over_mae` / `flag_fd_limited` therefore under-report, and the "
        "same `0.01` assumption is likely to affect the d2/dc2 and d2/dTdc rows too (not checked here). "
        "The FD d2G/dT2 error of ~11% is a measurement floor of the reference, not a model deficiency; "
        "the paper should report the spline- or wide-stencil-referenced value (or both) next to it.",
        "",
        "## Caveats", "",
        "- The spline truth is a per-composition degree-4 smoothing spline (interior knots at most "
        f"{ctx['max_knots']}); agreement with the polynomial may partly reflect shared smoothness. The wide-stencil FD in section 3 "
        "is the independent check.",
        "- Seeds vary only the polynomial fit; the references and point sets are seed independent, so the ± is fit-to-fit variation only.",
        "- Only the `random` protocol and the all-grid point set are used (the sweep's held-out-only subset is not re-run here; "
        "see `derivative_second_order_summary.csv`); the Newton `temp_band` Jacobian error is not reproduced here.",
    ]


if __name__ == "__main__":
    main()
