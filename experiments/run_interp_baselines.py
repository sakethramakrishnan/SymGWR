#!/usr/bin/env python
"""Interpolation baselines (no global polynomial) vs SymGWR, camera-ready protocol.

Protocol: reference set = train+val of ``get_split(kind, seed)``; evaluation on
test; targets handled in the split's z-scored space and reported in raw J/mol
through ``gwr.utils.compute_calphad_metrics``.  Seeds 42..51.

Baselines
  a  linear interpolation along T on the same-composition line
  b  cubic spline (not-a-knot) along T on the same-composition line
  c  FD-Taylor: nearest reference on the line + dG/dT from neighbouring table
     entries (central secant if both neighbours exist, else one-sided)
  d  Shepard / IDW, k=8, power 2, on the split's scaled features
  e  (composition_group only) interpolation along the composition arm at fixed
     T between reference compositions on the same single-element sweep
     (e_linear, e_cubic)
Variants (same SymGWR fit): residual_anchored, second_order (tasks 3/4).

Nothing is silently replaced: rows a method cannot predict are NaN with an
explicit status / count; extrapolated rows (query outside the reference range)
are counted per seed.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import CubicSpline

from camera_ready_common import (
    OUT_DIR, SEEDS, SPLITS, TEMP_COL, composition_ids, fit_symgwr, metrics_raw,
    predict_residual_anchored, predict_second_order, predict_symgwr,
)

FD_MAX_GAP_K = 30.0  # FD neighbour must be within 3 native 10 K steps, else treated as missing
IDW_K = 8
IDW_POWER = 2.0
MIN_CUBIC_POINTS = 4

ALONG_T = ["a_linear_T", "b_cubic_T", "c_fd_taylor"]
METHODS_ORDER = [
    "SymGWR", "residual_anchored", "second_order",
    "a_linear_T", "b_cubic_T", "c_fd_taylor", "d_idw_k8", "e_linear_arm", "e_cubic_arm",
]
METRIC_KEYS = ["mape_raw", "mae_raw", "rmse", "r2"]


# ----------------------------------------------------------------------
#  along-T baselines (a), (b), (c)
# ----------------------------------------------------------------------
def along_T_predictions(fit):
    """Return dict name -> (pred_z, extrap_mask), plus diagnostics for (c)."""
    sp = fit.split
    df = sp.df
    gid = composition_ids(df, sp.feature_cols)
    Tall = df[TEMP_COL].to_numpy(dtype=float)
    ref_g, ref_T, ref_y = gid[fit.ref_idx], Tall[fit.ref_idx], fit.ref_y
    te_g, te_T = gid[sp.test_idx], Tall[sp.test_idx]
    n = len(te_g)
    pa, pb, pc = (np.full(n, np.nan) for _ in range(3))
    ex_a = np.zeros(n, bool)
    ex_b = np.zeros(n, bool)
    c_info = []  # (T_test, T_nearest, T_lo, T_hi, mode)
    n_nostencil = 0

    for g in np.unique(te_g):
        m = ref_g == g
        if m.sum() < 2:
            continue
        o = np.argsort(ref_T[m])
        Tr, yr = ref_T[m][o], ref_y[m][o]
        rows = np.where(te_g == g)[0]
        Tq = te_T[rows]

        # (a) linear; outside the range -> linear extrapolation from the 2 end points (flagged)
        j = np.clip(np.searchsorted(Tr, Tq) - 1, 0, len(Tr) - 2)
        w = (Tq - Tr[j]) / (Tr[j + 1] - Tr[j])
        pa[rows] = yr[j] * (1 - w) + yr[j + 1] * w
        ex_a[rows] = (Tq < Tr[0]) | (Tq > Tr[-1])

        # (b) cubic spline (not-a-knot), extrapolate flagged
        cs = CubicSpline(Tr, yr)
        pb[rows] = cs(Tq)
        ex_b[rows] = ex_a[rows]

        # (c) FD-Taylor on the nearest reference (ties -> lower T)
        for r, tq in zip(rows, Tq):
            jn = int(np.argmin(np.abs(Tr - tq)))
            lo = jn - 1 if jn > 0 and Tr[jn] - Tr[jn - 1] <= FD_MAX_GAP_K else None
            hi = jn + 1 if jn < len(Tr) - 1 and Tr[jn + 1] - Tr[jn] <= FD_MAX_GAP_K else None
            if lo is not None and hi is not None:
                slope, mode = (yr[hi] - yr[lo]) / (Tr[hi] - Tr[lo]), "central"
            elif lo is not None:
                slope, mode = (yr[jn] - yr[lo]) / (Tr[jn] - Tr[lo]), "backward"
            elif hi is not None:
                slope, mode = (yr[hi] - yr[jn]) / (Tr[hi] - Tr[jn]), "forward"
            else:
                slope, mode = 0.0, "no_stencil"
                n_nostencil += 1
            pc[r] = yr[jn] + slope * (tq - Tr[jn])
            c_info.append((tq, Tr[jn], Tr[lo] if lo is not None else np.nan,
                           Tr[hi] if hi is not None else np.nan, mode,
                           Tr[max(np.searchsorted(Tr, tq) - 1, 0)],
                           Tr[min(np.searchsorted(Tr, tq), len(Tr) - 1)]))
    return pa, ex_a, pb, ex_b, pc, c_info, n_nostencil


# ----------------------------------------------------------------------
#  (d) Shepard / IDW
# ----------------------------------------------------------------------
def idw_predictions(fit, k=IDW_K, power=IDW_POWER):
    Q = fit.split.X_test
    d, idx = fit.tree.query(Q, k=k)
    w = 1.0 / np.maximum(d, 1e-12) ** power
    return (w * fit.ref_y[idx]).sum(1) / w.sum(1)


# ----------------------------------------------------------------------
#  (e) composition-arm interpolation (composition_group only)
# ----------------------------------------------------------------------
def star_structure(df, feature_cols):
    comp_cols = [c for c in feature_cols if c != TEMP_COL]
    gid = composition_ids(df, feature_cols)
    uniq = df.groupby(gid)[comp_cols].first()  # gid -> composition
    base_vec = uniq.mode().iloc[0]
    diff = (uniq != base_vec)
    n_diff = diff.sum(axis=1)
    assert (n_diff <= 1).all(), "not a one-factor-at-a-time star design"
    base_gid = int(uniq.index[n_diff == 0][0])
    arm = {}
    for g in uniq.index:
        if n_diff[g] == 0:
            arm[int(g)] = (None, float("nan"))
        else:
            e = diff.columns[diff.loc[g].to_numpy()][0]
            arm[int(g)] = (e, float(uniq.loc[g, e]))
    return gid, comp_cols, uniq, base_vec, base_gid, arm


def arm_predictions(fit):
    sp = fit.split
    df = sp.df
    gid, comp_cols, uniq, base_vec, base_gid, arm = star_structure(df, sp.feature_cols)
    Tall = df[TEMP_COL].to_numpy(dtype=float)
    temps = np.unique(Tall)
    tidx = np.searchsorted(temps, Tall)
    ref_g = set(np.unique(gid[fit.ref_idx]).tolist())
    te_g_all = np.unique(gid[sp.test_idx]).tolist()

    # per-composition G(T) rows over the ref set (z-scored)
    Ymat = {}
    pos = {int(r): i for i, r in enumerate(fit.ref_idx)}
    for g in ref_g:
        rows = fit.ref_idx[gid[fit.ref_idx] == g]
        yv = np.full(len(temps), np.nan)
        for r in rows:
            yv[tidx[r]] = fit.ref_y[pos[int(r)]]
        Ymat[g] = yv
    assert all(not np.isnan(v).any() for v in Ymat.values()), "ref composition missing temperatures"

    n = len(sp.test_idx)
    p_lin, p_cub = np.full(n, np.nan), np.full(n, np.nan)
    status_lin, status_cub = np.array([""] * n, dtype=object), np.array([""] * n, dtype=object)
    te_gid = gid[sp.test_idx]
    te_tidx = tidx[sp.test_idx]
    for g in te_g_all:
        rows = np.where(te_gid == g)[0]
        e, xq = arm[int(g)]
        if e is None:
            status_lin[rows] = status_cub[rows] = "n/a_star_center_in_test"
            continue
        members = [h for h in ref_g if arm[int(h)][0] == e or h == base_gid]
        xs = np.array([arm[int(h)][1] if h != base_gid else float(base_vec[e]) for h in members])
        o = np.argsort(xs)
        xs = xs[o]
        Y = np.stack([Ymat[members[i]] for i in o])  # (nx, nT)
        if xq < xs[0] or xq > xs[-1]:
            status_lin[rows] = status_cub[rows] = "n/a_outside_reference_range"
            continue
        j = min(max(np.searchsorted(xs, xq) - 1, 0), len(xs) - 2)
        w = (xq - xs[j]) / (xs[j + 1] - xs[j])
        yl = Y[j] * (1 - w) + Y[j + 1] * w
        p_lin[rows] = yl[te_tidx[rows]]
        status_lin[rows] = "ok"
        if len(xs) >= MIN_CUBIC_POINTS:
            yc = CubicSpline(xs, Y, axis=0)(xq)
            p_cub[rows] = yc[te_tidx[rows]]
            status_cub[rows] = "ok"
        else:
            status_cub[rows] = f"n/a_fewer_than_{MIN_CUBIC_POINTS}_points"

    gids_tr = set(np.unique(gid[sp.train_idx]).tolist())
    gids_va = set(np.unique(gid[sp.val_idx]).tolist())
    fold = {
        "base_in_test": base_gid in te_g_all,
        "base_in_val": base_gid in gids_va,
        "base_in_train": base_gid in gids_tr,
        "n_test_comps": len(te_g_all),
        "n_test_rows_base": int((te_gid == base_gid).sum()),
        "base_composition": {c: float(base_vec[c]) for c in comp_cols},
    }
    return p_lin, status_lin, p_cub, status_cub, fold


# ----------------------------------------------------------------------
def metric_row(fit, pred, mask, **extra):
    row = dict(extra)
    n = len(pred)
    ok = mask & np.isfinite(pred)
    row["n_test"] = n
    row["n_eval"] = int(ok.sum())
    row["n_na"] = int(n - ok.sum())
    if ok.sum() == 0:
        row.update({k: np.nan for k in METRIC_KEYS})
        return row
    sp = fit.split
    m = compute_metrics(sp, pred, ok)
    row.update({k: m[k] for k in METRIC_KEYS})
    return row


def compute_metrics(sp, pred_z, ok):
    from gwr.utils import compute_calphad_metrics
    return compute_calphad_metrics(sp.y_test[ok], pred_z[ok], sp.target_scaler)


def run(splits, seeds, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows, fold_rows, band_rows = [], [], []
    for kind in splits:
        for seed in seeds:
            fit = fit_symgwr(kind, seed)
            sp = fit.split
            n = len(sp.test_idx)
            allm = np.ones(n, bool)
            sym = predict_symgwr(fit, sp.X_test)
            ra = predict_residual_anchored(fit, sp.X_test)
            so = predict_second_order(fit, sp.X_test)
            base = dict(split=kind, seed=seed)

            def add(method, pred, mask, status="ok", **kw):
                r = metric_row(fit, pred, mask, method=method, status=status, **base, **kw)
                ok = mask & np.isfinite(pred)
                if method not in ("SymGWR", "residual_anchored", "second_order") and ok.sum():
                    s = compute_metrics(sp, sym, ok)
                    r["symgwr_mape_same_rows"] = s["mape_raw"]
                    r["symgwr_mae_same_rows"] = s["mae_raw"]
                rows.append(r)

            add("SymGWR", sym, allm)
            add("residual_anchored", ra, allm)
            add("second_order", so, allm)
            add("d_idw_k8", idw_predictions(fit), allm)

            if kind in ("random", "temp_band"):
                pa, ex_a, pb, ex_b, pc, c_info, n_nost = along_T_predictions(fit)
                add("a_linear_T", pa, allm, n_extrapolated=int(ex_a.sum()))
                add("b_cubic_T", pb, allm, n_extrapolated=int(ex_b.sum()))
                add("c_fd_taylor", pc, allm, n_fd_no_stencil=n_nost,
                    n_fd_central=sum(1 for c in c_info if c[4] == "central"),
                    n_fd_one_sided=sum(1 for c in c_info if c[4] in ("backward", "forward")))
                if kind == "temp_band":
                    for c in c_info:
                        band_rows.append(dict(seed=seed, T_test=c[0], nearest_ref_T=c[1],
                                              fd_lo_T=c[2], fd_hi_T=c[3], fd_mode=c[4],
                                              bracket_lo_T=c[5], bracket_hi_T=c[6]))
            else:
                for m in ALONG_T:
                    r = dict(split=kind, seed=seed, method=m, status="n/a", n_test=n, n_eval=0, n_na=n,
                             reason="every temperature of a test composition is held out; no same-composition reference line")
                    r.update({k: np.nan for k in METRIC_KEYS})
                    rows.append(r)
                p_lin, st_lin, p_cub, st_cub, fold = arm_predictions(fit)
                fold.update(split=kind, seed=seed, test_fold=seed % 5, val_fold=(seed + 1) % 5)
                for nm, p, st in (("e_linear_arm", p_lin, st_lin), ("e_cubic_arm", p_cub, st_cub)):
                    ok = np.isfinite(p)
                    status = "ok" if ok.all() else ("n/a" if not ok.any() else "partial")
                    add(nm, p, allm, status=status,
                        n_na_star_center=int((st == "n/a_star_center_in_test").sum()),
                        n_na_outside_range=int((st == "n/a_outside_reference_range").sum()),
                        n_na_few_points=int(sum(str(s).startswith("n/a_fewer") for s in st)))
                    fold[f"coverage_{nm}"] = float(ok.mean())
                fold["flag"] = "STAR_CENTER_IN_TEST" if fold["base_in_test"] else ""
                fold_rows.append(fold)
            print(f"[{kind} seed={seed}] done", flush=True)

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(os.path.join(out_dir, "interp_baselines_per_seed.csv"), index=False)
    folds = pd.DataFrame(fold_rows)
    if len(folds):
        folds["base_composition"] = folds["base_composition"].astype(str)
        folds.to_csv(os.path.join(out_dir, "interp_composition_group_folds.csv"), index=False)
    if band_rows:
        band = pd.DataFrame(band_rows)
        # reference points used: must be identical across seeds and compositions per T
        key = ["T_test", "nearest_ref_T", "fd_lo_T", "fd_hi_T", "fd_mode", "bracket_lo_T", "bracket_hi_T"]
        u = band[key].drop_duplicates().sort_values("T_test")
        u.to_csv(os.path.join(out_dir, "interp_temp_band_refs.csv"), index=False)
        multi = u.groupby("T_test").size()
        print("temp_band: distinct (T_test -> refs used) mappings per T:", multi.max())
    return per_seed, folds


def paired_tests(per_seed, out_dir):
    out = []
    for kind in per_seed["split"].unique():
        d = per_seed[per_seed.split == kind]
        for method in d.method.unique():
            if method in ("SymGWR",):
                continue
            b = d[d.method == method].sort_values("seed")
            if b["mape_raw"].isna().all():
                out.append(dict(split=kind, baseline=method, metric="mape_raw", note="n/a (no predictions)"))
                continue
            for metric, symcol in (("mape_raw", "symgwr_mape_same_rows"), ("mae_raw", "symgwr_mae_same_rows")):
                if method in ("residual_anchored", "second_order"):
                    s = d[d.method == "SymGWR"].sort_values("seed")[metric].to_numpy()
                else:
                    s = b[symcol].to_numpy()
                x = b[metric].to_numpy()
                if np.isnan(x).any() or np.isnan(s).any() or len(x) < 2:
                    continue
                diff = x - s
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    t, pt = stats.ttest_rel(s, x)
                    try:
                        w, pw = stats.wilcoxon(s, x)
                    except ValueError:
                        w, pw = np.nan, np.nan
                note = ""
                if np.allclose(diff, diff[0]) and np.allclose(diff, 0):
                    note = "identical per-seed values"
                elif np.isnan(pt):
                    note = "degenerate paired differences (zero variance)"
                out.append(dict(
                    split=kind, baseline=method, metric=metric, n_seeds=len(x),
                    mean_symgwr=s.mean(), mean_baseline=x.mean(),
                    mean_diff_baseline_minus_symgwr=diff.mean(),
                    ratio_baseline_over_symgwr=x.mean() / s.mean(),
                    n_seeds_baseline_better=int((x < s).sum()),
                    t_stat=t, t_p=pt, wilcoxon_stat=w, wilcoxon_p=pw, note=note,
                ))
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(out_dir, "interp_baselines_paired_tests.csv"), index=False)
    return df


def fmt(m, s):
    return f"{m:.3e} ± {s:.3e}"


def summarize(per_seed, tests, out_dir, folds=None):
    long_rows, wide = [], {}
    for method in METHODS_ORDER:
        wide[method] = {}
        for kind in SPLITS:
            d = per_seed[(per_seed.method == method) & (per_seed.split == kind)]
            v = d["mape_raw"].to_numpy(dtype=float) if len(d) else np.array([np.nan])
            if len(d) == 0 or np.isnan(v).all():
                wide[method][kind] = "n/a"
                continue
            m, s = np.nanmean(v), np.nanstd(v, ddof=1)
            cov = d["n_eval"].sum() / d["n_test"].sum()
            cell = fmt(m, s)
            if cov < 0.999:
                same = d["symgwr_mape_same_rows"].to_numpy(dtype=float)
                cell += f" [{cov:.0%} of test rows; SymGWR on same rows {np.nanmean(same):.3e}]"

            wide[method][kind] = cell
            long_rows.append(dict(method=method, split=kind, mape_raw_mean=m, mape_raw_std_ddof1=s,
                                  n_seeds=int(np.isfinite(v).sum()), row_coverage=cov))
    table = pd.DataFrame(wide).T[SPLITS]
    table.index.name = "method"
    table.to_csv(os.path.join(out_dir, "interp_summary.csv"))
    pd.DataFrame(long_rows).to_csv(os.path.join(out_dir, "interp_summary_long.csv"), index=False)

    # explicit conclusions (baseline vs SymGWR, same rows)
    lines = ["Interpolation baselines vs SymGWR (raw MAPE, mean over seeds 42-51, SymGWR on the SAME rows).",
             "Lower is better. 'beats' = baseline mean MAPE < SymGWR mean MAPE.", ""]
    for kind in SPLITS:
        lines.append(f"== {kind} ==")
        beats_any = False
        for method in ["a_linear_T", "b_cubic_T", "c_fd_taylor", "d_idw_k8", "e_linear_arm", "e_cubic_arm"]:
            t = tests[(tests.split == kind) & (tests.baseline == method) & (tests.metric == "mape_raw")]
            if t.empty or "mean_baseline" not in t or np.isnan(t.iloc[0].get("mean_baseline", np.nan)):
                lines.append(f"  {method}: n/a")
                continue
            r = t.iloc[0]
            beats = r.mean_baseline < r.mean_symgwr
            beats_any |= beats
            lines.append(
                f"  {method}: {'BEATS' if beats else 'does NOT beat'} SymGWR | baseline {r.mean_baseline:.3e} "
                f"vs SymGWR {r.mean_symgwr:.3e} | baseline/SymGWR = {r.ratio_baseline_over_symgwr:.3g}x | "
                f"better in {int(r.n_seeds_baseline_better)}/{int(r.n_seeds)} seeds | "
                f"t-test p={r.t_p:.3g}, Wilcoxon p={r.wilcoxon_p:.3g}"
                + (f" | {r.note}" if isinstance(r.note, str) and r.note else ""))
        lines.append(f"  -> any interpolation baseline beats SymGWR on {kind}: {'YES' if beats_any else 'NO'}")
        lines.append("")
    if folds is not None and len(folds):
        flagged = sorted(folds.loc[folds["base_in_test"].astype(str) == "True", "seed"].tolist())
        lines.append("== composition_group sensitivity: folds where the star-center (base) composition is in TEST ==")
        lines.append(f"  flagged seeds (star center in test): {flagged}; remaining {len(SEEDS) - len(flagged)} seeds below")
        d = per_seed[(per_seed.split == "composition_group") & (~per_seed.seed.isin(flagged))]
        for method in ["SymGWR", "residual_anchored", "second_order", "d_idw_k8", "e_linear_arm", "e_cubic_arm"]:
            dm = d[d.method == method]
            if method.startswith("e_"):
                lines.append(f"  {method}: MAPE {dm.mape_raw.mean():.3e} vs SymGWR same rows "
                             f"{dm.symgwr_mape_same_rows.mean():.3e} "
                             f"(ratio {dm.mape_raw.mean() / dm.symgwr_mape_same_rows.mean():.3g}x, n_seeds={len(dm)})")
            else:
                lines.append(f"  {method}: MAPE {dm.mape_raw.mean():.3e} (n_seeds={len(dm)})")
        lines.append("")
    lines.append("NOTES")
    lines.append("  * temp_band: the reference set (all non-band rows) and the test set are identical for every seed, so every "
                 "T-line baseline is deterministic across seeds; only SymGWR/IDW vary (via the train-only scaler). The across-seed "
                 "std and the paired p-values for temp_band are therefore NOT independent evidence (effectively n=1).")
    lines.append("  * composition_group: seeds 42/47, 43/48, 44/49, 45/50, 46/51 share a test fold index but use different shuffles; "
                 "the 10 seeds are 5 folds x 2 shuffles, not 10 independent splits.")
    lines.append("  * e_*: evaluated only on test rows whose composition is bracketed by reference compositions on its arm "
                 "(no extrapolation; star-center rows n/a); SymGWR is compared on the SAME rows.")
    with open(os.path.join(out_dir, "interp_conclusions.txt"), "w") as f:
        f.write("\n".join(lines))
    return table, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--splits", nargs="*", default=SPLITS)
    ap.add_argument("--summary-only", action="store_true", help="rebuild tests/summary from saved CSVs")
    args = ap.parse_args()
    if args.summary_only:
        per_seed = pd.read_csv(os.path.join(args.out_dir, "interp_baselines_per_seed.csv"))
        folds = pd.read_csv(os.path.join(args.out_dir, "interp_composition_group_folds.csv"))
    else:
        per_seed, folds = run(args.splits, args.seeds, args.out_dir)
    tests = paired_tests(per_seed, args.out_dir)
    table, text = summarize(per_seed, tests, args.out_dir, folds)
    pd.set_option("display.width", 250, "display.max_colwidth", 80)
    print(table.to_string())
    print()
    print(text)


if __name__ == "__main__":
    main()
