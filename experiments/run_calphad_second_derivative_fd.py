#!/usr/bin/env python
"""Second-derivative fidelity of SymGWR on the real CALPHAD OFAT grid (degree 5).

Companion of run_calphad_grid_fd_gradient.py (same data, seeds 42-51, random split,
polynomial fit on train+val, chain rule through the scalers).  Reference values are
second finite differences of the RAW Gibbs table, in raw units:

  d2G/dT2      every composition line (needs equally spaced T on both sides)
  d2G/dc_e2    own-axis, fixed T, on element e's sweep (e in Nb Al Ti Mo Fe Cr; Ni constant),
               only where the c_e neighbours on both sides are equally spaced
  d2G/dT dc_e  central 4-corner stencil on element e's (T, c_e) plane, where the T and c_e
               neighbours are equally spaced and all four corners exist

    d2G/dxdy ~ [G(x+,y+) - G(x+,y-) - G(x-,y+) + G(x-,y-)] / [(x+ - x-)(y+ - y-)]

Estimators (all raw units, Hessian component at the query point):
  poly       analytic Hessian of the fitted degree-5 polynomial P at the query point,
             H_raw[i,j] = H_z[i,j] * target_scale / (feature_scale[i] * feature_scale[j])
  symgwr_t   SymGWR-T (k=1 first-order Taylor): the surrogate is piecewise affine, so its
             Hessian is identically 0 inside a Voronoi cell  -> constant 0
  taylor2    second-order Taylor variant, k=1: Ghat(q) = G_i + gradP(r_i).d + .5 d'H(r_i)d,
             whose Hessian inside the cell is H_P at the nearest fit reference r_i
             (nearest in scaled feature space, as in camera_ready_common.predict_second_order)

FD error budget, per axis:
  stencil_trunc_mae   mean |analytic H_P - same FD stencil applied to P's own values|
                      (truncation error of the stencil, estimated on the polynomial;
                      this is a proxy: the true G may have larger higher derivatives)
  noise_rms (= fd_rounding_rms)
                      rms error of the stencil from the rounding of the G table, using the
                      ACTUAL quantum of every value in that stencil: 1 J/mol where |G| >= 1e5
                      (38% of rows), 0.01 J/mol elsewhere; uniform rounding, variance q^2/12.
                      (The first version of this script assumed 0.01 for every value: the old
                      number is kept as fd_rounding_rms_legacy_assumed.)
  noise_over_mae, noise_over_typical, noise_sq_share, implied_model_rms ...
                      see run_calphad_grid_fd_gradient.ref_metrics
  fd_vs_polystencil   mean |FD(P) - FD(G)|: model-vs-data error with the stencil applied to
                      both (truncation largely cancels)

Second reference (derivative_second_order_refs*.csv): a ladder of wider central stencils on the
common set of centres where every ladder entry exists (same points for narrow and wide), and for
d2G/dT2 also the degree-4 smoothing spline of the Newton test.  Wide stencils average the table noise
down (~1/h^2) at the price of truncation; the "headline" wide entry minimises the predicted
sqrt(noise^2 + trunc^2) without looking at the polynomial-vs-FD error.

Identifiability: null-space share of the evaluated second-derivative functionals (SVD of the
centred bias-free fit design, as in the gradient script).  Cross-element mixed terms are NOT
evaluated; they are summarised from derivative_identifiability.csv.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from camera_ready_common import fit_symgwr  # noqa: E402
from gwr.splits import DEFAULT_TARGET, TEMP_COL  # noqa: E402
from run_calphad_grid_fd_gradient import (  # noqa: E402
    ELEMENTS, FLAG_RATIO, MAX_MC, MAX_MT, NOISE_COLS, STEP_ATOL, STEP_RTOL, attribution, axis_lines,
    build_ladder, check_quantisation, col, deriv_matrix, null_frac, pick_headline, plane_grid, ref_metrics,
    ref_rows, spline_ref, spline_T_reference, summarise_refs, pc, pm, ref_label,
    NOISE_DOMINATES, Q_BIG, Q_SMALL, BIG_ABS, MC_DRAWS, SPLINE_DRAWS, MIN_COMMON_SHARE,
)

SEEDS = list(range(42, 52))
G_ROUND_SIGMA = 0.01 / np.sqrt(12.0)  # LEGACY: 0.01 J/mol for every value (wrong for |G| >= 1e5); only the legacy
#                                       column and run_second_order_reconciliation.py still use it
ESTIMATORS = ["poly", "symgwr_t", "taylor2"]


def term_name(kind: str, el: str | None) -> str:
    return {"TT": "d2/dT2", "cc": f"d2/d{el}2", "Tc": f"d2/dTd{el}"}[kind]


def equal_step(a, b):
    return np.isclose(a, b, rtol=STEP_RTOL, atol=STEP_ATOL) & (a > 0)


# ----------------------------------------------------------------------
# LEGACY narrow-stencil builder (assumes 0.01 J/mol everywhere).  Kept because
# run_second_order_reconciliation.py imports build_TT and G_ROUND_SIGMA; the sweep itself now uses
# run_calphad_grid_fd_gradient.build_ladder (identical narrow stencils, per-value quantum).
# ----------------------------------------------------------------------

def second_diff_points(x, g, ridx):
    """Interior points with equal spacing both sides; exact 3-point formula."""
    h1, h2 = x[1:-1] - x[:-2], x[2:] - x[1:-1]
    j = np.nonzero(equal_step(h1, h2))[0]
    i = j + 1
    h1, h2 = x[i] - x[i - 1], x[i + 1] - x[i]
    w_lo, w_c, w_hi = 2 / ((h1 + h2) * h1), -2 / (h1 * h2), 2 / ((h1 + h2) * h2)
    fd = w_lo * g[i - 1] + w_c * g[i] + w_hi * g[i + 1]
    rms = np.sqrt(w_lo**2 + w_c**2 + w_hi**2)
    return pd.DataFrame({"row": ridx[i], "lo": ridx[i - 1], "hi": ridx[i + 1], "fd": fd,
                         "w_lo": w_lo, "w_c": w_c, "w_hi": w_hi, "round_rms": rms * G_ROUND_SIGMA})


def build_TT(df, comp_cols):
    out = []
    for _, grp in df.groupby(comp_cols, sort=False):
        grp = grp.sort_values(TEMP_COL)
        if len(grp) >= 3:
            out.append(second_diff_points(grp[TEMP_COL].to_numpy(float),
                                          grp[DEFAULT_TARGET].to_numpy(float), grp.index.to_numpy()))
    out = pd.concat(out, ignore_index=True)
    out["kind"], out["el"] = "TT", ""
    return out


# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results/camera_ready")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--max-mc", type=int, default=MAX_MC, help="widest half-width on a composition axis (grid steps)")
    ap.add_argument("--max-mt", type=int, default=MAX_MT, help="widest half-width along T (grid steps)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    f0 = fit_symgwr("random", args.seeds[0])
    df, feats = f0.split.df, f0.split.feature_cols
    assert (df.index.to_numpy() == np.arange(len(df))).all()
    comp_cols = [c for c in feats if c != TEMP_COL]
    t_idx = feats.index(TEMP_COL)
    fidx = {el: feats.index(col(el)) for el in ELEMENTS}
    base = df[comp_cols].mode().iloc[0]
    X_raw = df[feats].to_numpy(float)
    G = df[DEFAULT_TARGET].to_numpy(float)
    share_big = check_quantisation(G)
    n_comp = df.groupby(comp_cols).ngroups
    n_T = df[TEMP_COL].nunique()
    terms = [("TT", "")] + [("cc", e) for e in ELEMENTS] + [("Tc", e) for e in ELEMENTS]

    # ---- references (seed independent) ----
    narrow_all, ladders, infos = {}, {}, {}
    for kind, el in terms:
        t = term_name(kind, el)
        if kind == "TT":
            src = build_ladder(G, "d2", axis_lines(df, comp_cols, base, "T"), args.max_mt, args.max_mt)
        elif kind == "cc":
            src = build_ladder(G, "d2", axis_lines(df, comp_cols, base, el), args.max_mc, args.max_mt)
        else:
            src = build_ladder(G, "mixed", plane_grid(df, comp_cols, base, el), args.max_mc, args.max_mt)
        narrow_all[t], ladders[t], infos[t] = src
    counts = {t: infos[t]["n_narrow"] for t in infos}
    print(f"rows={len(df)} compositions={n_comp} temperatures={n_T}; {share_big:.1%} of G on the 1 J/mol quantum")
    print(f"d2G/dT2 expected n = {n_comp} x {n_T - 2} = {n_comp * (n_T - 2)}; got {counts['d2/dT2']}")
    print("FD-eligible points per term:", counts)
    print("ladder (cap, |common set|):", {t: (infos[t]["cap"], infos[t]["n_common"]) for t in infos})
    _, S2, _, N2 = spline_T_reference(df, comp_cols)
    spl = spline_ref(ladders["d2/dT2"]["narrow"].centre, S2, N2)

    rows, ref_out = [], []
    for seed in args.seeds:
        t0 = time.time()
        fit = fit_symgwr("random", seed)
        sp, pm = fit.split, fit.model
        X_all = sp.feature_scaler.transform(X_raw)
        t_scale = float(sp.target_scaler.scale_[0])
        f_scale = np.asarray(sp.feature_scaler.scale_)
        H_z = pm.hess(X_all)  # (n, d, d), scaled space, every row
        G_poly = sp.target_scaler.inverse_transform(pm.value(X_all).reshape(-1, 1)).ravel()

        _, nn_pos = fit.tree.query(X_all, k=1)
        nn_row = fit.ref_idx[nn_pos]  # nearest fit reference (row in df) for every row
        in_fit = np.zeros(len(df), bool)
        in_fit[fit.ref_idx] = True

        # design null space
        poly_tf = pm.pipeline.named_steps["poly_features"]
        Phi = poly_tf.transform(fit.ref_X)[:, 1:]
        Phi = Phi - Phi.mean(axis=0)
        s = np.linalg.svd(Phi, compute_uv=False)
        tol = s.max() * max(Phi.shape) * np.finfo(Phi.dtype).eps
        rank = int((s > tol).sum())
        Vr = np.linalg.svd(Phi, full_matrices=False)[2][:rank]
        powers = poly_tf.powers_[1:]

        for kind, el in terms:
            t = term_name(kind, el)
            na = narrow_all[t]
            r = na.centre
            if kind == "TT":
                i, j = t_idx, t_idx
            elif kind == "cc":
                i, j = fidx[el], fidx[el]
            else:
                i, j = t_idx, fidx[el]
            conv = t_scale / (f_scale[i] * f_scale[j])
            est_all = {"poly": H_z[:, i, j] * conv, "symgwr_t": np.zeros(len(df)), "taylor2": H_z[nn_row, i, j] * conv}
            ststc = na.stencil.apply(G_poly)
            legacy = np.sqrt((na.stencil.w ** 2).sum(axis=1)) * G_ROUND_SIGMA  # 0.01 J/mol for every value
            d = {i: 1, j: 1} if i != j else {i: 2}
            nf = null_frac(deriv_matrix(powers, X_all[r], d), Vr)

            for subset, mask in (("all_grid", np.ones(len(r), bool)), ("test_only", ~in_fit[r])):
                for est, e in est_all.items():
                    m = ref_metrics(na, e[r], est_all["poly"][r], G_poly, mask, na.val)
                    m["stencil_trunc_mae"] = m.pop("trunc_mae")
                    m.pop("ref_vs_narrow_mae")
                    rows.append({
                        "seed": seed, "term": t, "kind": kind, "element": el,
                        "subset": subset, "estimator": est, **m,
                        "fd_vs_polystencil_mae": float(np.mean(np.abs(ststc[mask] - na.val[mask]))),
                        "fd_rounding_rms": m["noise_rms"],  # = noise_rms, per-value quantum (corrected budget)
                        "fd_rounding_rms_legacy_assumed": float(np.sqrt(np.mean(legacy[mask] ** 2))),
                        "null_frac_mean": float(np.nanmean(nf[mask])),
                        "frac_taylor2_hess_is_own": float(np.mean(nn_row[r][mask] == r[mask])),
                        "fit_rank": rank, "n_fit_rows": len(fit.ref_idx),
                    })

            C = ladders[t]["narrow"].centre
            ref_out += ref_rows({"seed": seed, "term": t, "kind": kind, "element": el}, ladders[t],
                                spl if kind == "TT" else None,
                                {"values": {k: v[C] for k, v in est_all.items()}, "G_poly": G_poly}, in_fit)
        print(f"seed {seed}: rank={rank}/{Phi.shape[1]} ({time.time() - t0:.1f}s)", flush=True)

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(os.path.join(args.out_dir, "derivative_second_order.csv"), index=False)

    # ---- summary: mean +/- std (ddof=1) over seeds ----
    metric_cols = ["n", "mae", "typical_abs", "normalized_error", "pearson", "stencil_trunc_mae",
                   "fd_vs_polystencil_mae", "fd_rounding_rms", "fd_rounding_rms_legacy_assumed", *NOISE_COLS,
                   "null_frac_mean", "frac_taylor2_hess_is_own"]
    g = per_seed.groupby(["term", "kind", "element", "subset", "estimator"], sort=False)
    summ = g[metric_cols].mean().add_suffix("_mean").join(g[metric_cols].std(ddof=1).add_suffix("_std"))
    summ["n_seeds"] = g.size()
    summ = summ.reset_index()
    summ["trunc_over_mae"] = summ.stencil_trunc_mae_mean / summ.mae_mean
    summ["rounding_over_mae"] = summ.fd_rounding_rms_mean / summ.mae_mean  # corrected budget (== noise_over_mae)
    summ["flag_noise_limited"] = summ.rounding_over_mae >= FLAG_RATIO
    summ["flag_trunc_limited"] = summ.trunc_over_mae >= FLAG_RATIO
    summ["flag_fd_limited"] = summ.flag_noise_limited | summ.flag_trunc_limited
    order = {t: k for k, t in enumerate(dict.fromkeys(per_seed.term))}
    summ["_o"] = summ.term.map(order)
    summ["_e"] = summ.estimator.map({e: k for k, e in enumerate(ESTIMATORS)})
    summ = summ.sort_values(["subset", "_o", "_e"]).drop(columns=["_o", "_e"]).reset_index(drop=True)
    summ.to_csv(os.path.join(args.out_dir, "derivative_second_order_summary.csv"), index=False)

    # ---- second reference: wide-stencil ladder on the common set; smoothing spline for d2G/dT2 ----
    refs = pd.DataFrame(ref_out)
    refs.to_csv(os.path.join(args.out_dir, "derivative_second_order_refs.csv"), index=False)
    refs_summ = summarise_refs(refs, "term", list(order), ddof=1)
    headline = pick_headline(refs_summ, "term")
    refs_summ["headline"] = [headline.get(t) == r for t, r in zip(refs_summ["term"], refs_summ["ref"])]
    refs_summ.to_csv(os.path.join(args.out_dir, "derivative_second_order_refs_summary.csv"), index=False)

    ctx = {"counts": counts, "n_comp": n_comp, "n_T": n_T, "seeds": args.seeds, "share_big": share_big,
           "infos": infos, "headline": headline}
    write_markdown(args.out_dir, summ, refs_summ, ctx)
    S = summ[(summ.subset == "all_grid") & (summ.estimator == "poly")]
    print(S[["term", "n_mean", "mae_mean", "typical_abs_mean", "normalized_error_mean", "noise_rms_mean",
             "noise_over_mae_mean", "noise_over_typical_mean", "noise_sq_share_mean", "trunc_over_mae", "flag_fd_limited"]]
          .to_string(float_format=lambda v: f"{v:.4g}"))


# ----------------------------------------------------------------------
# Markdown (generated from the CSVs / summary frames; no hand-typed numbers)
# ----------------------------------------------------------------------

NAMED = [("d2/dFe2", "Fe pure d2/dc2"), ("d2/dCr2", "Cr pure d2/dc2"), ("d2/dTdTi", "Ti mixed d2/dTdc")]


def write_markdown(out_dir, summ, refs_summ, ctx):
    seeds, n_comp, n_T, counts, infos, head = (ctx[k] for k in ("seeds", "n_comp", "n_T", "counts", "infos", "headline"))
    terms = list(dict.fromkeys(summ.term))
    A = summ[(summ.subset == "all_grid") & (summ.estimator == "poly")].set_index("term", drop=False)
    K = refs_summ[(refs_summ.subset == "all_grid") & (refs_summ.estimator == "poly")]
    attr = {t: attribution(A.loc[t], K[K.term == t], head[t]) for t in terms}
    L = ["# Second-derivative fidelity (degree-5 SymGWR polynomial, real CALPHAD grid)", "",
         "Generated by `experiments/run_calphad_second_derivative_fd.py` "
         "(files: `derivative_second_order.csv`, `derivative_second_order_summary.csv`, `derivative_second_order_refs.csv`, "
         "`derivative_second_order_refs_summary.csv`). Previous outputs: `backup_pre_noise_budget/`.",
         f"Seeds {seeds[0]}-{seeds[-1]} (n = {len(seeds)}), random split, deduped data; fit on train+val. "
         "All values are mean ± std over seeds (ddof=1), raw units "
         "(d2/dT2: J/(mol K^2); d2/dc2: J/mol per unit mass fraction squared; d2/dTdc: J/(mol K) per unit mass fraction).",
         "Reference: second finite differences of the raw G table (narrow stencil, one grid step each side) unless stated. Estimators: `poly` = analytic "
         "Hessian of P; `SymGWR-T` = effective Hessian of the k=1 first-order Taylor surrogate = 0 inside a cell; "
         "`Taylor2` = second-order Taylor variant, Hessian = H_P at the nearest fit reference.", "",
         f"d2G/dT2: expected n = {n_comp} x {n_T - 2} = {n_comp * (n_T - 2)}; "
         f"got {counts['d2/dT2']} ({'confirmed' if counts['d2/dT2'] == n_comp * (n_T - 2) else 'MISMATCH'}).", "",
         "## Corrected rounding budget", "",
         f"The table is quantised: {ctx['share_big']:.1%} of the G values (|G| >= {BIG_ABS:g} J/mol) are integers (quantum {Q_BIG:g} J/mol), the rest are "
         f"multiples of {Q_SMALL:g} J/mol (both asserted on the data). The first version of this sweep assumed {Q_SMALL:g} J/mol for every value; the budget below "
         "uses the actual quantum of every value in each stencil (error ~ U(-q/2, q/2), variance q^2/12).", "",
         "| term | legacy assumed noise rms (0.01 J/mol everywhere) | actual noise rms | ratio |", "|---|---|---|---|"]
    for t in terms:
        r = A.loc[t]
        L.append(f"| {t} | {r.fd_rounding_rms_legacy_assumed_mean:.3g} | {r.noise_rms_mean:.3g} | {r.noise_rms_mean / r.fd_rounding_rms_legacy_assumed_mean:.1f}x |")
    L.append("")

    for subset, title in (("all_grid", "All FD-eligible grid points"),
                          ("test_only", "Held-out points only (centre row not in train+val; n is the seed mean)")):
        S = summ[summ.subset == subset]
        L += [f"## {title}", "",
              "| term | n | typical abs | poly MAE | poly norm. err | poly r | Taylor2 MAE | Taylor2 norm. err | Taylor2 r | "
              "SymGWR-T MAE (=typical abs) | SymGWR-T norm. err |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for t in terms:
            a = {e: S[(S.term == t) & (S.estimator == e)].iloc[0] for e in ESTIMATORS}
            p, q, z = a["poly"], a["taylor2"], a["symgwr_t"]
            L.append(f"| {t} | {p.n_mean:.0f} | {p.typical_abs_mean:.3g} | {pm(p.mae_mean, p.mae_std)} | "
                     f"{pm(p.normalized_error_mean, p.normalized_error_std)} | {pm(p.pearson_mean, p.pearson_std, '.4f')} | "
                     f"{pm(q.mae_mean, q.mae_std)} | {pm(q.normalized_error_mean, q.normalized_error_std)} | "
                     f"{pm(q.pearson_mean, q.pearson_std, '.4f')} | {pm(z.mae_mean, z.mae_std)} | "
                     f"{pm(z.normalized_error_mean, z.normalized_error_std)} |")
        L += [""]

    L += ["## FD error budget (all grid points, poly estimator, narrow FD)", "",
          "`noise rms` = rms over the points of the stencil's rounding error, sqrt(sum_k (w_k q_k)^2 / 12). `noise/MAE` = noise rms / poly MAE "
          f"(= the old `rounding_over_mae`); `E|noise|/MAE` uses the expected mean absolute noise (Monte Carlo, {MC_DRAWS} draws per stencil). "
          "`noise/typical` = noise rms / typical |d|. `noise share` = noise rms^2 / mean((P - FD)^2): the share of the squared error that rounding noise could explain "
          "if independent of the model error. `implied model rms err` = sqrt(max(mean((P - FD)^2) - noise rms^2, 0)) / typical, to compare with `poly rms err`. "
          "`trunc` = stencil truncation proxy (the stencil applied to P's own values vs its analytic Hessian), a proxy for the true G's truncation error. "
          f"Flag = noise/MAE or trunc/MAE >= {FLAG_RATIO:.0%}. `FD(P) vs FD(G)` applies the same stencil to P and to G, so truncation largely cancels.", "",
          "| term | poly MAE | noise rms | noise/MAE | E\\|noise\\|/MAE | noise/typical | noise share | poly rms err / typical | implied model rms err / typical | "
          "trunc (poly stencil) | trunc/MAE | FD(P) vs FD(G) / typical | null-space share of functional | flag |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in terms:
        r = A.loc[t]
        fl = ", ".join(n for n, f in (("**noise**", r.flag_noise_limited), ("**trunc**", r.flag_trunc_limited)) if f) or "ok"
        L.append(f"| {t} | {r.mae_mean:.3g} | {r.noise_rms_mean:.3g} | {r.noise_over_mae_mean:.3g} | {r.noise_mae_over_mae_mean:.3g} | "
                 f"{pc(r.noise_over_typical_mean, 2)} | {pc(r.noise_sq_share_mean, 2)} | {pc(r.rms_error_over_typical_mean, 1)} | "
                 f"{pc(r.implied_model_over_typical_mean, 1)} | {r.stencil_trunc_mae_mean:.3g} | {r.trunc_over_mae:.2g} | "
                 f"{r.fd_vs_polystencil_mae_mean / r.typical_abs_mean:.2g} | {r.null_frac_mean_mean:.1e} | {fl} |")
    fn = [t for t in terms if A.loc[t].flag_noise_limited]
    ft = [t for t in terms if A.loc[t].flag_trunc_limited]
    L += ["", "- Noise-limited terms (noise/MAE >= %.0f%%): %s." % (100 * FLAG_RATIO, ", ".join(
        f"{t} ({A.loc[t].noise_over_mae_mean:.2f}; noise share of squared error {pc(A.loc[t].noise_sq_share_mean, 0)})" for t in fn) or "none"),
          "- Truncation-limited terms (trunc/MAE >= %.0f%%): %s." % (100 * FLAG_RATIO, ", ".join(
              f"{t} ({A.loc[t].trunc_over_mae:.2f})" for t in ft) or "none"), ""]

    L += ["## Second reference: wide stencil / smoothing spline, side by side", "",
          "Central stencil of half-width m grid steps, symmetric about the centre (noise ~ 1/h^2, truncation ~ h^2); for mixed terms (m_c, m_T) are the half-widths "
          "along c and T. All columns use the SAME centres: the common set C where every ladder entry exists (the narrow stencil is the ladder entry m = 1; "
          f"the ladder is capped so that C keeps at least {MIN_COMMON_SHARE:.0%} of the narrow points). The headline wide stencil is the ladder entry other than the "
          "narrow one that minimises the predicted sqrt(noise rms^2 + trunc^2) (seed-mean); it never looks at the polynomial-vs-FD error. "
          f"For d2G/dT2 the degree-4 smoothing spline of the Newton test is added (noise floor = std over {SPLINE_DRAWS} refits of the table with fresh rounding errors). "
          "Normalized error = MAE / mean |reference|.", "",
          "| term | n (all grid) | poly vs narrow, all grid | n (C) | poly vs narrow, C | wide (headline) | poly vs wide | poly vs spline | "
          "noise floor / typical: narrow | wide | spline | wide trunc proxy / typical | predicted rss: wide vs narrow |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    rss = lambda r: float(np.hypot(r.noise_rms_mean, r.trunc_mae_mean))  # noqa: E731
    worse = []
    for t in terms:
        x = attr[t]
        rn = K[(K.term == t) & (K.ref == "narrow")].iloc[0]
        rw = K[(K.term == t) & (K.ref == head[t])].iloc[0] if head[t] else None
        if rw is not None and rss(rw) > rss(rn):
            worse.append(t)
        L.append(f"| {t} | {A.loc[t].n_mean:,.0f} | {pc(x['nerr_narrow'], 2)} | {x['n_common']:,} | {pc(x['nerr_narrow_common'], 2)} | "
                 f"{ref_label(head[t]) if head[t] else 'n/a'} | {pc(x['nerr_wide'], 2)} | {pc(x['nerr_spline'], 2)} | "
                 f"{pc(rn.noise_over_typical_mean, 2)} | {pc(x['wide_noise_over_typical'], 2)} | {pc(x['spline_noise_over_typical'], 2)} | "
                 f"{pc(x['wide_trunc_over_typical'], 2)} | {('%.3g vs %.3g' % (rss(rw), rss(rn))) if rw is not None else 'n/a'} |")
    L += ["", "- Terms where the headline wide stencil has a LARGER predicted error budget than the narrow one (narrow noise is already small, so widening trades "
          "a small noise reduction for more truncation; there the wide reference is a consistency check, not a better reference): " + (", ".join(worse) or "none") + ".", "",
          "Full ladder (poly estimator, all grid points of C):", "",
          "| term | reference | n | poly normalized error | Pearson r | noise/typical | noise share | trunc/typical | reference vs narrow FD (MAE / typical) | headline |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for t in terms:
        for _, r in K[K.term == t].iterrows():
            L.append(f"| {t} | {ref_label(r.ref)} | {r.n_mean:,.0f} | {pc(r.normalized_error_mean, 2)} | {r.pearson_mean:.4f} | "
                     f"{pc(r.noise_over_typical_mean, 2)} | {pc(r.noise_sq_share_mean, 2)} | "
                     f"{pc(r.trunc_mae_mean / r.typical_abs_mean, 2) if np.isfinite(r.trunc_mae_mean) else 'n/a'} | "
                     f"{pc(r.ref_vs_narrow_mae_mean / r.typical_abs_mean, 2)} | {'**yes**' if r.headline else ''} |")
    L.append("")

    L += ["## Model error or FD noise?", "",
          f"Rule (narrow FD, all grid points): noise share >= {NOISE_DOMINATES:.0%} -> FD noise; < {FLAG_RATIO:.0%} -> model error; between -> mixed. "
          "The noise share is the exact rounding budget above, not a fit. Cross-check: the error against the best second reference vs the narrow-FD error on "
          "the same points, with the gap between that reference and the narrow FD in units of the narrow FD's expected |noise| (a gap far above 1 means the two "
          "references differ by stencil truncation / smoothing, not by rounding noise).", "",
          "| term | poly vs narrow FD (MAE) | poly rms err / typical | noise/typical | noise share of sq. error | implied model rms err / typical | "
          "best second ref | its error | change vs narrow (C) | gap / narrow noise | verdict |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in terms:
        x = attr[t]
        L.append(f"| {t} | {pc(x['nerr_narrow'], 1)} | {pc(x['rms_over_typical'], 1)} | {pc(x['noise_over_typical'], 2)} | {pc(x['noise_sq_share'], 1)} | "
                 f"{pc(x['implied_model_over_typical'], 1)} | {ref_label(x['best_second_label']) if x['best_second_label'] else 'n/a'} | "
                 f"{pc(x['best_second_ref'], 1)} | {x['second_ref_change']:+.0%} | {x['gap_over_noise']:.1f} | **{x['verdict']}** |")
    L += [""]
    for t, name in NAMED:
        x = attr[t]
        L += [f"- **{name} ({t}): {pc(x['nerr_narrow'], 1)} normalized error vs narrow FD ({pc(x['rms_over_typical'], 1)} rms/typical) -> {x['verdict'].upper()}.** "
              f"Rounding noise of the narrow FD is {x['noise_rms']:.3g} = {pc(x['noise_over_typical'], 1)} of typical |d| ({x['typical_abs']:.3g}) and "
              f"{x['noise_over_mae']:.2f} of the MAE; it carries {pc(x['noise_sq_share'], 1)} of the squared error. Removing it in quadrature leaves a model rms error "
              f"of {pc(x['implied_model_over_typical'], 1)} of typical (vs {pc(x['rms_over_typical'], 1)} before). The best second reference "
              f"({ref_label(x['best_second_label'])}, noise floor {pc(x['wide_noise_over_typical'], 2)} of typical) gives {pc(x['best_second_ref'], 1)} vs "
              f"{pc(x['nerr_narrow_common'], 1)} for the narrow FD on the same {x['n_common']:,} points ({x['second_ref_change']:+.0%}; the two references differ by "
              f"{x['gap_over_noise']:.1f} x the narrow FD's expected noise). Truncation proxy of the narrow stencil: {x['trunc_over_mae']:.2f} of the MAE."]
    t = "d2/dT2"
    x = attr[t]
    L += [f"- For reference, **d2/dT2** ({pc(x['nerr_narrow'], 1)} vs narrow FD) -> {x['verdict'].upper()}: noise share {pc(x['noise_sq_share'], 0)} of the squared error; "
          f"against the smoothing spline {pc(x['nerr_spline'], 1)}, against the headline wide FD ({ref_label(x['head'])}) {pc(x['nerr_wide'], 1)}.", ""]

    ident_path = os.path.join(out_dir, "derivative_identifiability.csv")
    if os.path.exists(ident_path):
        idf = pd.read_csv(ident_path)
        labels = {f"d2/d{a}d{b}" for a in ELEMENTS for b in ELEMENTS if a != b}
        cross = idf[(idf.order == 2) & idf.derivative.isin(labels)]
        c = cross.groupby("derivative").agg(null_frac_mean_seed_mean=("null_frac_mean", "mean"),
                                            null_frac_mean_seed_min=("null_frac_mean", "min"),
                                            null_frac_max=("null_frac_max", "max"),
                                            n_points=("n_points", "first"), n_seeds=("seed", "nunique"))
        L += ["## Cross-element mixed second derivatives: not identifiable, not evaluated", "",
              "From `derivative_identifiability.csv` (evaluated on the off-base points of the first element's plane). "
              f"Across the {len(c)} cross-element terms the seed-mean null-space share ranges "
              f"{c.null_frac_mean_seed_mean.min():.3f} to {c.null_frac_mean_seed_mean.max():.3f} "
              f"(per-seed min {c.null_frac_mean_seed_min.min():.3f}, max over points {c.null_frac_max.max():.3f}). "
              "No finite-difference reference exists either: the OFAT design never varies two elements together.", "",
              "| term | null-space share (seed mean) | per-seed min | max over points | n points | seeds |",
              "|---|---|---|---|---|---|"]
        for t, r in c.iterrows():
            L.append(f"| {t} | {r.null_frac_mean_seed_mean:.3f} | {r.null_frac_mean_seed_min:.3f} | "
                     f"{r.null_frac_max:.3f} | {int(r.n_points)} | {int(r.n_seeds)} |")
        L += [""]
    L += ["## Caveats", "",
          "- Rounding noise is modelled as independent uniform errors per table value, a statistical budget rather than a per-point bound; the table may carry other "
          "upstream noise that it does not see. The d2G/dT2 prediction was checked against the spline in `second_order_reconciliation.md`; the other terms rely on the model.",
          "- The truncation proxy is the stencil applied to the polynomial itself, so it is only as good as the polynomial's higher derivatives (weakest on the "
          "poorly fitted terms, which are exactly the ones in question).",
          "- The smoothing spline (d2G/dT2 only) is the Newton-test truth: degree 4, smoothing level set for 0.3 J/mol noise on every row, so it does not adapt to the "
          "0.01 J/mol rows; agreement with the polynomial may partly reflect shared smoothness.",
          "- Composition-axis and mixed terms have short sweeps (11-26 nodes, uneven near the base value for Al and Fe), which caps the wide ladder; "
          "the common set C is smaller than the narrow-FD set there.",
          "- Seeds vary only the polynomial fit; the references and point sets are seed independent.", ""]
    with open(os.path.join(out_dir, "derivative_second_order_summary.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
