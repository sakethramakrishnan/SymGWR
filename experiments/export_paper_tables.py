#!/usr/bin/env python
"""Export camera-ready paper tables (Markdown + IEEE-style booktabs LaTeX).

Reads results/camera_ready/*.csv (+ design_audit.json, the only source of the
design audit) and writes results/camera_ready/paper_tables.{md,tex}.  Every
number is read or computed from those files; nothing is typed in by hand.

Tables
  1  main accuracy: methods x splits (RMSE, MAPE, 1-R2; mean +/- std over seeds)
  2  derivative fidelity per axis, with identifiability notes
  3  continuity of the reference-based surrogates (short segments; SymGWR-T/R, R-IDW and R-CS at k=4)
  4  design audit (rows, compositions, rank, condition number, duplicate leakage)
  5  scaling (rendered once results/camera_ready/scaling*.csv exists)

Quantities that are not in the inputs (a missing input CSV) are rendered as an
explicit "not available" / "pending" row, never invented.

LaTeX needs \\usepackage{booktabs}; wide tables use table* (IEEEtran two-column).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd

IN_DIR = "results/camera_ready"
SPLITS = ["random", "temp_band", "composition_group"]
SPLIT_LABEL = {"random": "Random", "temp_band": "Temp.-band", "composition_group": "Comp.-group"}
# CSV variant name -> paper name
VARIANT_LABEL = {"SymGWR": "SymGWR-T", "residual_anchored": "SymGWR-R", "residual_anchored_idw": "SymGWR-R-IDW"}
CONT_VARIANTS = ["SymGWR", "residual_anchored", "residual_anchored_idw"]
K_SHOW = 4   # k of the SymGWR-R-IDW / SymGWR-R-CS rows in Tables 1 and 3
# internal Table-1 keys for the rows that do not come from baselines_*_summary.csv
M_IDW, M_CS, M_CUBIC = f"R-IDW_k{K_SHOW}", f"R-CS_k{K_SHOW}", "cubic_T_or_arm"
VARIANT_LABEL.update({M_IDW: f"SymGWR-R-IDW (k={K_SHOW})", M_CS: f"SymGWR-R-CS (k={K_SHOW})", M_CUBIC: "Cubic along T / arm"})
AXES = ["T", "Nb", "Al", "Ti", "Mo", "Fe", "Cr", "Ni"]
AXIS_UNIT = {"T": "J/(mol K)"}
IDENT_TOL = 1e-9   # null_frac below this = identified
NOT_IDENT = 0.5    # null_frac above this = not identified
EM = "—"


# ----------------------------------------------------------------------
#  Cells: (markdown, latex) pairs
# ----------------------------------------------------------------------
class C:
    def __init__(self, md, tex=None):
        self.md, self.tex = md, (md if tex is None else tex)


_TEX_ESC = {"|": r"\textbar{}", "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
            "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def tex_esc(s):
    return "".join(_TEX_ESC.get(ch, ch) for ch in s).replace("—", "---").replace("≤", r"$\le$") \
        .replace("±", r"$\pm$").replace("−", "$-$").replace("≥", r"$\ge$") \
        .replace("–", "--").replace("×", r"$\times$").replace("²", r"$^2$")


def md_esc(s):
    return s.replace("|", r"\|")


def T(s):
    """Plain text cell (escaped for each target)."""
    return C(md_esc(s), tex_esc(s))


def bold(c):
    return C(f"**{c.md}**", rf"\textbf{{{c.tex}}}")


def cell(x):
    return x if isinstance(x, C) else T(str(x))


# ----------------------------------------------------------------------
#  Number formatting
# ----------------------------------------------------------------------
def _exp10(x):
    return int(math.floor(math.log10(abs(x))))


def num(x, sig=3):
    """Single number, `sig` significant figures; plain or scientific."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return C(EM)
    if x == 0:
        return C("0")
    ax = abs(x)
    if 1e-2 <= ax < 1e5:
        dec = max(0, sig - 1 - _exp10(ax))
        s = f"{x:.{dec}f}"
        return C(s)
    e = _exp10(ax)
    m = x / 10.0 ** e
    ms = f"{m:.{sig - 1}f}"
    if abs(float(ms)) >= 10:
        e += 1
        ms = f"{x / 10.0 ** e:.{sig - 1}f}"
    return C(f"{ms}×10{str(e).translate(_SUP)}", rf"${ms}\times10^{{{e}}}$")


ZERO_STD_REL = 1e-8   # std below this fraction of |mean| is numerical noise, shown as "± 0"
zero_std_used = False


def mean_std(m, s, sig=3):
    """mean +/- std.  Common exponent when both share a scale; separate otherwise."""
    global zero_std_used
    if m is None or not np.isfinite(m):
        return C(EM)
    if s is None or not np.isfinite(s):
        return num(m, sig)
    if m != 0 and abs(s) < ZERO_STD_REL * abs(m):
        zero_std_used = True
        a = num(m, sig)
        return C(f"{a.md} ± 0", f"{a.tex} $\\pm$ 0")
    if m == 0 and s == 0:
        return C("0")
    ref = abs(m) if m != 0 else abs(s)
    if 1e-2 <= ref < 1e5:
        dec = max(0, sig - 1 - _exp10(ref))
        if round(abs(s), dec) != 0 or s == 0:
            t = f"{m:.{dec}f} ± {s:.{dec}f}"
            return C(t, rf"${m:.{dec}f} \pm {s:.{dec}f}$")
        a, b = num(m, sig), num(s, 2)
        return C(f"{a.md} ± {b.md}", f"{a.tex} $\\pm$ {b.tex}")
    e = _exp10(ref)
    ms, ss = m / 10.0 ** e, s / 10.0 ** e
    if round(abs(ss), sig - 1) != 0 or s == 0:
        t = f"{ms:.{sig - 1}f} ± {ss:.{sig - 1}f}"
        return C(f"({t})×10{str(e).translate(_SUP)}", rf"$({ms:.{sig - 1}f} \pm {ss:.{sig - 1}f})\times10^{{{e}}}$")
    a, b = num(m, sig), num(s, 2)
    return C(f"{a.md} ± {b.md}", f"{a.tex} $\\pm$ {b.tex}")


def intc(x):
    return C(f"{int(round(x)):,}", f"{int(round(x))}") if np.isfinite(x) else C(EM)


# ----------------------------------------------------------------------
#  Table container + renderers
# ----------------------------------------------------------------------
class Table:
    def __init__(self, key, title, caption, align, header, groups=None, wide=False, size="footnotesize"):
        self.key, self.title, self.caption = key, title, caption
        self.align, self.header, self.groups = align, header, groups  # groups: [(label, span)]
        self.rows, self.notes, self.wide, self.size = [], [], wide, size

    def row(self, cells):
        self.rows.append([cell(c) for c in cells])

    def rule(self):
        self.rows.append("RULE")

    def section(self, label):
        self.rows.append(("SECTION", label))

    def note(self, s):
        self.notes.append(s)


def render_md(t, num_):
    out = [f"### Table {num_}. {t.title}", "", t.caption, ""]
    ncol = len(t.header)
    specs = _specs(t.align)
    if t.groups:  # GFM needs the first row to be the header: use the group labels there
        cells, i = [], 0
        for lab, span in t.groups:
            cells += [lab] + [""] * (span - 1)
            i += span
        cells += [""] * (ncol - i)
        out.append("| " + " | ".join(md_esc(c) for c in cells) + " |")
        out.append("|" + "|".join(":--" if a in "lp" else "--:" for a in specs) + "|")
        out.append("| " + " | ".join(f"**{md_esc(h)}**" for h in t.header) + " |")
    else:
        out.append("| " + " | ".join(md_esc(h) for h in t.header) + " |")
        out.append("|" + "|".join(":--" if a in "lp" else "--:" for a in specs) + "|")
    for r in t.rows:
        if r == "RULE":
            continue
        if isinstance(r, tuple):
            out.append("| " + " | ".join([f"**{md_esc(r[1])}**"] + [""] * (ncol - 1)) + " |")
            continue
        out.append("| " + " | ".join(c.md for c in r) + " |")
    out.append("")
    for n in t.notes:
        out.append(f"- {n}")
    out.append("")
    return "\n".join(out)


def _specs(align):
    return re.findall(r"p\{[^}]*\}|.", align)


def _tex_align(t):
    return t.align


def render_tex(t, num_):
    env = "table*" if t.wide else "table"
    ncol = len(t.header)
    L = [f"% ---- Table {num_}: {t.title}", f"\\begin{{{env}}}[t]", "\\centering", f"\\caption{{{tex_esc_caption(t.caption)}}}",
         f"\\label{{tab:{t.key}}}", f"\\{t.size}", "\\setlength{\\tabcolsep}{3pt}", f"\\begin{{tabular}}{{{_tex_align(t)}}}", "\\toprule"]
    if t.groups:
        parts, cm, col = [], [], 1
        for lab, span in t.groups:
            if span == 1 and not lab:
                parts.append("")
            else:
                parts.append(f"\\multicolumn{{{span}}}{{c}}{{{tex_esc(lab)}}}" if lab else f"\\multicolumn{{{span}}}{{c}}{{}}")
                if lab:
                    cm.append(f"\\cmidrule(lr){{{col}-{col + span - 1}}}")
            col += span
        if col <= ncol:
            parts.append("")
        L.append(" & ".join(parts) + r" \\")
        L.append("".join(cm))
    L.append(" & ".join(tex_esc(h) if not h.startswith("$") else h for h in t.header) + r" \\")
    L.append("\\midrule")
    for r in t.rows:
        if r == "RULE":
            L.append("\\midrule")
        elif isinstance(r, tuple):
            L.append(f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{{tex_esc(r[1])}}}}} \\\\")
            L.append("\\midrule")
        else:
            L.append(" & ".join(c.tex for c in r) + r" \\")
    L += ["\\bottomrule", "\\end{tabular}"]
    if t.notes:
        L.append("\\par\\smallskip")
        L.append("\\begin{minipage}{\\linewidth}\\scriptsize\\raggedright")
        for n in t.notes:
            L.append(f"{tex_esc_caption(n)}\\par")
        L.append("\\end{minipage}")
    L += [f"\\end{{{env}}}", ""]
    return "\n".join(L)


def tex_esc_caption(s):
    """Escape prose; keep `code` spans as \\texttt and **bold** off."""
    parts = re.split(r"(`[^`]*`)", s)
    out = []
    for p in parts:
        if p.startswith("`") and p.endswith("`") and len(p) > 1:
            out.append(r"\texttt{" + tex_esc(p[1:-1]) + "}")
        else:
            out.append(tex_esc(p))
    return "".join(out)


# ----------------------------------------------------------------------
#  IO helpers
# ----------------------------------------------------------------------
def rd(name):
    return pd.read_csv(os.path.join(IN_DIR, name))


def label(v):
    return VARIANT_LABEL.get(v, v)


def pnote(s):
    return " ".join(str(s).split())


# ----------------------------------------------------------------------
#  Table 1: main accuracy
# ----------------------------------------------------------------------
def _seed_stats(d):
    """Per-seed rows -> same stats dict as the baseline summaries (ddof=1)."""
    return dict(rmse=(d["rmse"].mean(), d["rmse"].std(ddof=1)), mape=(100 * d["mape_raw"].mean(), 100 * d["mape_raw"].std(ddof=1)),
                omr2=(1 - d["r2"].mean(), d["r2"].std(ddof=1)), n=int(d["seed"].nunique()), nto=0, ner=0, note=None)


def _check_combined(stats):
    """Recomputed RMSE means must agree with the strings in ridwcs_combined_table.csv (3 significant figures)."""
    comb = rd("ridwcs_combined_table.csv")
    for (key, s), st in stats.items():
        if key not in (M_IDW, M_CS, M_CUBIC):
            continue
        r = comb[(comb["split"] == s) & (comb["method"] == ("cubic_T_or_arm" if key == M_CUBIC else key.replace("R-", "SymGWR-R-")))]
        if len(r) == 0:
            continue
        ref = float(r["rmse_J_per_mol"].iloc[0].split("±")[0])
        assert abs(st["rmse"][0] - ref) <= 5e-3 * abs(ref), f"RMSE mismatch vs ridwcs_combined_table.csv: {key}/{s}: {st['rmse'][0]} vs {ref}"


def table_accuracy():
    global zero_std_used
    zero_std_used = False
    summ = {s: rd(f"baselines_{s}_summary.csv") for s in SPLITS}
    var = rd("taylor_variants_accuracy_per_seed.csv")
    # ordered union: a model may be present in some splits only while baselines are still being appended
    models = list(dict.fromkeys(m for sp in SPLITS for m in summ[sp]["model"]))
    # (rmse, mape%, 1-R2) (mean, std) per (method, split); n_seeds
    stats = {}
    for s in SPLITS:
        for _, r in summ[s].iterrows():
            stats[(r["model"], s)] = dict(
                rmse=(r["rmse_mean"], r["rmse_std"]), mape=(100 * r["mape_raw_mean"], 100 * r["mape_raw_std"]),
                omr2=(1 - r["r2_mean"], r["r2_std"]), n=int(r["n_seeds_ok"]), nto=int(r["n_timeout"]), ner=int(r["n_error"]),
                note=None if pd.isna(r["notes"]) else pnote(r["notes"]))
    # SymGWR-R from per-seed Taylor-variant file (ddof=1, same as the baseline summaries)
    r_rows = var[var["method"] == "residual_anchored"]
    for s in SPLITS:
        d = r_rows[r_rows["split"] == s]
        if len(d) == 0:
            continue
        stats[("residual_anchored", s)] = dict(
            rmse=(d["rmse"].mean(), d["rmse"].std(ddof=1)), mape=(100 * d["mape_raw"].mean(), 100 * d["mape_raw"].std(ddof=1)),
            omr2=(1 - d["r2"].mean(), d["r2"].std(ddof=1)), n=int(d["seed"].nunique()), nto=0, ner=0, note=None)
    # SymGWR-R-IDW / -CS (k=K_SHOW) and the cubic-along-T/arm interpolant, from their per-seed files
    # (full precision for 1-R2; cross-checked against ridwcs_combined_table.csv below)
    extra, cov_arm, same_rows = [], {}, {}
    try:
        acc = rd("ridwcs_accuracy_per_seed.csv")
        for key, name in ((M_IDW, f"SymGWR-R-IDW_k{K_SHOW}"), (M_CS, f"SymGWR-R-CS_k{K_SHOW}")):
            for s in SPLITS:
                d = acc[(acc["method"] == name) & (acc["split"] == s)]
                if len(d):
                    stats[(key, s)] = _seed_stats(d)
            if (key, SPLITS[0]) in stats:
                extra.append(key)
    except FileNotFoundError:
        pass
    try:
        ib = rd("interp_baselines_per_seed.csv")
        for s, meth in (("random", "b_cubic_T"), ("temp_band", "b_cubic_T"), ("composition_group", "e_cubic_arm")):
            d = ib[(ib["method"] == meth) & (ib["split"] == s) & ib["status"].isin(["ok", "partial"])]
            if len(d):
                stats[(M_CUBIC, s)] = _seed_stats(d)
                cov_arm[s] = float((d["n_eval"] / d["n_test"]).mean())
                same_rows[s] = 100 * float(d["symgwr_mape_same_rows"].mean())
        if any((M_CUBIC, s) in stats for s in SPLITS):
            extra.append(M_CUBIC)
    except FileNotFoundError:
        pass
    try:
        _check_combined(stats)
    except FileNotFoundError:
        pass
    partial = {s for s, c in cov_arm.items() if c < 1 - 1e-9}   # splits where the cubic covers only part of the test rows
    order = ["SymGWR"] + (["residual_anchored"] if ("residual_anchored", SPLITS[0]) in stats else []) + extra + \
        [m for m in models if m != "SymGWR"]

    # footnote markers for method-level notes
    letters = "abcdefghij"
    note_mark, note_list = {}, []
    if partial:
        for s in partial:
            stats[(M_CUBIC, s)]["note"] = (
                f"{SPLIT_LABEL[s]}: cubic along the composition arm is defined for only {100 * cov_arm[s]:.0f}% of the test rows "
                "(composition bracketed by reference compositions on its arm; star-centre and out-of-range rows have no value). "
                f"Its metrics are on those rows only, where SymGWR-T has MAPE {same_rows[s]:.4f}% (all rows: "
                f"{stats[('SymGWR', s)]['mape'][0]:.4f}%); it is therefore excluded from the bold best-of ranking in this split.")
    for m in order:
        for s in SPLITS:
            n_ = stats.get((m, s), {}).get("note")
            if n_ and n_ not in note_mark:
                note_mark[n_] = letters[len(note_list)]
                note_list.append(n_)

    # effective independent partitions (one value per split; must agree across models)
    eff = {}
    for s in SPLITS:
        d = summ[s]
        vals = d["effective_independent_partitions"].unique()
        assert len(vals) == 1, f"effective partitions differ across models in {s}: {vals}"
        eff[s] = (int(vals[0]), pnote(d["partition_note"].iloc[0]))
    roman = {s: "i" * (k + 1) if k < 3 else str(k + 1) for k, s in enumerate(SPLITS)}

    tab = Table("accuracy", "Main accuracy",
                "Test accuracy on the three CALPHAD splits: mean ± std over seeds (std with ddof=1). "
                "MAPE is the raw-unit MAPE in percent; R² is reported as 1−R². Bold = best mean per split and metric.",
                "lrrrl", ["Method", "RMSE (J/mol)", "MAPE (%)", "$1-R^2$", "Note"], size="scriptsize", wide=True)

    best = {}
    for s in SPLITS:
        for k in ("rmse", "mape", "omr2"):
            vals = {m: stats[(m, s)][k][0] for m in order if (m, s) in stats and not (m == M_CUBIC and s in partial)}
            best[(s, k)] = min(vals, key=vals.get)

    for s in SPLITS:
        tab.section(f"{SPLIT_LABEL[s]} split: {eff[s][0]} effective independent partition{'s' if eff[s][0] != 1 else ''} ({roman[s]})")
        for m in order:
            st = stats.get((m, s))
            cells = [T(label(m))]
            for k in ("rmse", "mape", "omr2"):
                if st is None:
                    cells.append(C(EM))
                    continue
                c = mean_std(*st[k])
                cells.append(bold(c) if best[(s, k)] == m else c)
            marks = []
            if st:
                if st["note"]:
                    marks.append(note_mark[st["note"]])
                if st["n"] < 10 or st["nto"] or st["ner"]:
                    marks.append(f"n={st['n']}")
            cells.append(T(", ".join(marks)) if marks else C(""))
            tab.row(cells)
            if m == "residual_anchored" or (m == "SymGWR" and "residual_anchored" not in order):
                tab.rule()

    n_seeds = sorted({v["n"] for v in stats.values()})
    if zero_std_used:
        tab.note(f"\"± 0\": std below {ZERO_STD_REL:g} of the mean (numerical noise; fixed test set and deterministic fit), not exactly zero.")
    tab.note("SymGWR-T = degree-5 polynomial + k=1 nearest-reference Taylor surrogate (`SymGWR` in the CSVs); "
             "SymGWR-R = residual-anchored variant (`residual_anchored`, from taylor_variants_accuracy_per_seed.csv).")
    if extra:
        tab.note(f"SymGWR-R-IDW / SymGWR-R-CS (k={K_SHOW}) = residual-anchored polynomial plus an inverse-distance / compactly supported blend of the k nearest "
                 "reference residuals (ridwcs_accuracy_per_seed.csv); the cubic row is a cubic interpolant along T (random, temp.-band) or along the "
                 "composition arm (comp.-group) through the reference rows (interp_baselines_per_seed.csv).")
    for s in SPLITS:
        tab.note(f"({roman[s]}) {SPLIT_LABEL[s]}: {eff[s][1]}")
    for n_ in note_list:
        tab.note(f"({note_mark[n_]}) {n_}")
    tab.note(f"Seeds per method: {', '.join(map(str, n_seeds))}; timeouts: {sum(v['nto'] for v in stats.values())}, "
             f"errors: {sum(v['ner'] for v in stats.values())}.")
    return tab


# ----------------------------------------------------------------------
#  Table 2: derivative fidelity
# ----------------------------------------------------------------------
def ub(x):
    """Upper bound to 2 significant figures, rounded up (so '<= ub' always holds)."""
    e = _exp10(x)
    m = math.ceil(x / 10.0 ** (e - 1) - 1e-9) / 10.0
    if m >= 10:
        m, e = m / 10.0, e + 1
    return f"{m:.1f}e{e}"


def _fmt_null(lo, hi):
    return f"{lo:.2f}" if f"{lo:.2f}" == f"{hi:.2f}" else f"{lo:.2f}–{hi:.2f}"


def table_derivative():
    fid = rd("derivative_fidelity.csv")
    ident = rd("derivative_identifiability.csv")
    n_seeds = fid["seed"].nunique()
    res = {}
    for (ax, sub, est), d in fid.groupby(["axis", "subset", "estimator"]):
        res[(ax, sub, est)] = dict(n=d["n"].mean(), err=(100 * d["normalized_error"].mean(), 100 * d["normalized_error"].std(ddof=1)),
                                   r=(d["pearson"].mean(), d["pearson"].std(ddof=1)), unit=None)
    ranks = fid["fit_rank"].unique()
    nterms = ident["n_terms"].unique()

    def ident_text(ax):
        if ax == "T":
            d = ident[ident["derivative"] == "d/dT"]
            return f"identified on all {d['evaluated_at'].nunique()} planes (null ≤ {ub(d['null_frac_max'].max())})"
        d = ident[ident["derivative"] == f"d/d{ax}"]
        if len(d) == 0:
            return "no finite difference (constant column)"
        own = d[d["evaluated_at"] == f"{ax}-plane"]
        oth = d[d["evaluated_at"] != f"{ax}-plane"]
        parts = []
        if len(own) == 0:  # no sweep plane for this axis (constant column)
            return (f"not identifiable (constant column, no FD; null {_fmt_null(d['null_frac_mean'].min(), d['null_frac_max'].max())} "
                    f"on all {d['evaluated_at'].nunique()} planes)")
        if len(own):
            nf = own["null_frac_max"].max()
            parts.append(("own plane: identified" if nf <= IDENT_TOL else "own plane: not identified")
                         + (f" (null ≤ {ub(nf)})" if nf <= IDENT_TOL else f" (null {own['null_frac_mean'].min():.2f})"))
        if len(oth):
            lo, hi = oth["null_frac_mean"].min(), oth["null_frac_mean"].max()
            parts.append(("other planes: not identified" if lo >= NOT_IDENT else "other planes: partly") + f" (null {_fmt_null(lo, hi)})")
        return "; ".join(parts)

    tab = Table("derivfid", "Derivative fidelity of the degree-5 SymGWR polynomial vs. central finite differences",
                "Analytic polynomial gradient vs. central finite differences of the raw CALPHAD table (all grid points, and held-out test points only). "
                f"Normalized error = MAE / typical |dG/dx|; r = Pearson correlation; mean ± std over {n_seeds} seeds (ddof=1). "
                "The identifiability column gives the share of the derivative functional lying in the null space of the design (null; ~0 = determined by the data).",
                "p{2.4cm}rrrrrp{4.4cm}",
                ["Axis", "n (grid)", "Norm. err. (%)", "r", "n (test)", "Norm. err., test (%)", "Identifiability (null share)"],
                wide=True)
    for ax in AXES:
        key = (ax, "all_grid", "poly")
        unit = f"dG/d{ax}" + (f" [{AXIS_UNIT[ax]}]" if ax in AXIS_UNIT else "")
        if key not in res:
            tab.row([T(unit), C(EM), C(EM), C(EM), C(EM), C(EM), T(ident_text(ax))])
            continue
        a, h = res[key], res.get((ax, "test_only", "poly"))
        tab.row([T(unit), intc(a["n"]), mean_std(*a["err"], sig=3), num(a["r"][0], 4),
                 intc(h["n"]) if h else C(EM), mean_std(*h["err"], sig=3) if h else C(EM), T(ident_text(ax))])

    # second-order identifiability summary
    d2 = ident[ident["order"] == 2].copy()
    own2 = d2[d2.apply(lambda r: bool(re.fullmatch(r"d2/d(\w+?)2|d2/d(\w+?)dT", r["derivative"])), axis=1)]
    cross2 = d2.drop(own2.index)
    rk = ", ".join(str(int(x)) for x in sorted(ranks))
    nt = ", ".join(str(int(x)) for x in sorted(nterms))
    tab.note(f"Fit design rank {rk} of {nt} polynomial terms (centred, bias-free), all {n_seeds} seeds; the minimum-norm solution sets the unconstrained part to zero.")
    tab.note("Composition-axis accuracy holds for own-axis derivatives on the sweep lines only; cross-element, Ni and interior-composition gradients are not identified by the data.")
    if len(own2) and len(cross2):
        tab.note(f"Second derivatives: pure and d/dT-mixed on the own plane identified (null ≤ {ub(own2['null_frac_max'].max())}); "
                 f"cross-element mixed terms not identified (null {_fmt_null(cross2['null_frac_mean'].min(), cross2['null_frac_max'].max())}).")
    tab.note("Held-out = FD points whose centre row is not in that seed's train+val. A random row split keeps fit neighbours in the same (T, c) plane, "
             "so this is an interpolation test, not a held-out-composition test.")
    return tab


# ----------------------------------------------------------------------
#  Table 3: continuity (short segments)
# ----------------------------------------------------------------------
def _cont_cols(df):
    """Column names (value jump J/mol, gradient jump relative) of a continuity summary file."""
    g = lambda base: next(c for c in df.columns if c.startswith(base))
    return ([g(f"value_jump_raw_{k}(") for k in ("median", "p95", "max")] +
            [g(f"grad_jump_over_gradnorm_{k}(") for k in ("median", "p95", "max")])


EVENT_LABEL = {"nearest_ref_switch": "NN switch", "kset_change": "k-set", "support_ref_change": "support"}


def table_continuity():
    summ = rd("continuity_summary.csv")
    per = rd("continuity_per_seed.csv")
    short = summ[summ["segment_type"] == "short"]
    pshort = per[per["segment_type"] == "short"]
    present = set(short["variant"])
    cols_t = _cont_cols(short)
    nseg = int(pshort["n_segments"].iloc[0])
    nseeds = int(short["n_seeds"].max())

    # SymGWR-R-IDW / SymGWR-R-CS (k=K_SHOW): short segments, ridw_*.csv / ridwcs_*.csv
    extra = {}   # variant label -> (summary, per-seed, filter, [events])
    try:
        ri, rip = rd("ridw_continuity_summary.csv"), rd("ridw_continuity_per_seed.csv")
        extra["residual_anchored_idw"] = (ri[ri["k"] == K_SHOW], rip[rip["k"] == K_SHOW], ["nearest_ref_switch", "kset_change"])
    except FileNotFoundError:
        pass
    try:
        rc, rcp = rd("ridwcs_continuity_summary.csv"), rd("ridwcs_continuity_per_seed.csv")
        extra["residual_anchored_cs"] = (rc[(rc["k"] == K_SHOW) & (rc["kind"] == "cs")], rcp[(rcp["k"] == K_SHOW) & (rcp["kind"] == "cs")],
                                         ["nearest_ref_switch", "kset_change", "support_ref_change"])
    except FileNotFoundError:
        pass
    extra_label = {"residual_anchored_idw": f"SymGWR-R-IDW (k={K_SHOW})", "residual_anchored_cs": f"SymGWR-R-CS (k={K_SHOW})"}
    variants = ["SymGWR", "residual_anchored", "residual_anchored_idw", "residual_anchored_cs"]

    tab = Table("continuity", "Continuity of the nearest-reference surrogates on short segments",
                "", "llrrrrrrr",
                ["Variant", "Event", "Events", "Median", "p95", "Max", "Median", "p95", "Max"],
                groups=[("", 3), ("Value jump (J/mol)", 3), ("Gradient jump (rel.)", 3)], wide=True, size="scriptsize")
    tab.caption = (f"Jump of the surrogate across reference-change events on {nseg:,} short segments per split and seed ({nseeds} seeds). "
                   "Short segments start at a test query and have length 0.1–1.0× the nearest-reference distance. "
                   "Value jump in J/mol; gradient jump relative to the median model gradient norm. "
                   "Median and p95 are averaged over seeds; max is the maximum over seeds.")

    label_c = lambda v: extra_label.get(v) or label(v)
    missing = []
    for s in SPLITS:
        if s != SPLITS[0]:
            tab.rule()
        tab.section(f"{SPLIT_LABEL[s]} split")
        for v in variants:
            if v in extra:
                sm, pr, events = extra[v]
                src = [(ev, sm[(sm["split"] == s) & (sm["event"] == ev)], pr[(pr["split"] == s) & (pr["event"] == ev)]) for ev in events]
                cols = _cont_cols(sm)
            elif v in present:
                r = short[short["split"] == s]
                src = [("nearest_ref_switch", r[r["variant"] == v], pshort[(pshort["split"] == s) & (pshort["variant"] == v)])]
                cols = cols_t
            else:
                src, cols = [("nearest_ref_switch", short.iloc[0:0], None)], None
                if label_c(v) not in missing:
                    missing.append(label_c(v))
            for ev, r, pr_ in src:
                if len(r) == 0:
                    tab.row([T(label_c(v)), T(EVENT_LABEL[ev]), T("not available")] + [C(EM)] * 6)
                    continue
                r = r.iloc[0]
                nsw = pr_["n_switches_evaluated" if "n_switches_evaluated" in pr_ else "n_events_evaluated"].sum()
                tab.row([T(label_c(v)), T(EVENT_LABEL[ev]), intc(nsw)] + [num(r[c]) for c in cols])
    tab.note("SymGWR-T extrapolates linearly from the nearest reference, so value and gradient both jump at cell boundaries. "
             "SymGWR-R adds a constant offset to the polynomial P, so its gradient (grad P) is continuous (jumps are zero up to rounding) and only the value jumps.")
    tab.note(f"SymGWR-R-IDW / -CS blend the residuals of the k={K_SHOW} nearest references. "
             "Event: NN switch = the nearest reference changes; k-set = the set of the k nearest references changes (k-th and (k+1)-th reference swap); "
             "support (CS only) = the (k+1)-th reference changes with the k-set unchanged, which moves the support radius. "
             "The NN-switch rows of the blended variants are the same event type as the SymGWR-T and SymGWR-R rows.")
    if missing:
        tab.note("Not available: " + ", ".join(missing) + " (input CSV missing).")
    tie = int(pshort["n_tie_plane_switches_excluded"].sum()) + sum(int(x[1]["n_tie_plane_excluded"].sum()) for x in extra.values())
    tab.note(f"Events = reference-change events located by bisection and evaluated on both sides, summed over seeds; tie-plane events excluded: {tie}. "
             "SymGWR-T/R and the blended variants use independently drawn segments, so event counts differ between them. "
             "For the blended variants at most 5,000 events per split and seed are evaluated (random subsample; 50,000 = capped in all 10 seeds).")
    return tab


# ----------------------------------------------------------------------
#  Table 4: design audit
# ----------------------------------------------------------------------
def table_design():
    path = os.path.join(IN_DIR, "design_audit.json")
    with open(path) as f:
        a = json.load(f)
    rc = a["row_counts"]
    dm = a["design_matrix"]
    comps = rc["n_distinct_compositions"]
    old = a["old_random_split_duplicates"]
    new_dups = old["new_deduped_random_split_test_dup_in_train_or_val_per_seed"]

    def design_row(key):
        runs = dm[key].get("all_seeds") or [dm[key]["seed0"]]
        ncol = runs[0]["shape"][1]
        rows = [r["shape"][0] for r in runs]
        rank = [r["rank"] for r in runs]
        cond = [r["cond_effective"] for r in runs]
        return dict(n=len(runs), rows=(min(rows), max(rows)), ncol=ncol, rank=(min(rank), max(rank)), cond=(min(cond), np.median(cond), max(cond)))

    def rng(lo, hi):
        return C(f"{lo:,}" if lo == hi else f"{lo:,}–{hi:,}", f"{lo}" if lo == hi else f"{lo}--{hi}")

    labels = {"random_dedupe": ("Random, deduped", "random"),
              "temp_band_dedupe": ("Temp.-band, deduped", None),
              "composition_group_dedupe": ("Comp.-group, deduped", None),
              "random_no_dedupe": ("Random, not deduped (old)", "old")}
    tab = Table("design", "Design audit of the degree-5 fit",
                f"Dataset: {rc['raw']:,} raw rows, {rc['after_dedupe_on_8_features']:,} after exact-duplicate removal on the 8 features "
                f"({rc['duplicate_rows_removed']} removed, deduplication {'changes' if rc['dedupe_ever_changes_target'] else 'never changes'} the target); "
                f"{comps} distinct compositions x {rc['n_distinct_temperatures']} temperatures. "
                "Design = PolynomialFeatures(degree 5) incl. bias on the train-scaled features.",
                "lrrrrp{3.2cm}p{3.2cm}",
                ["Split / dedupe", "Seeds", "Train rows", "Columns", "Rank", "Eff. cond. no. (median [min, max])", "Test rows with exact duplicate in train+val"],
                wide=True)
    for key, (name, dupkey) in labels.items():
        if key not in dm:
            continue
        d = design_row(key)
        cond = num(d["cond"][1])
        cond = C(f"{cond.md} [{num(d['cond'][0]).md}, {num(d['cond'][2]).md}]", f"{cond.tex} [{num(d['cond'][0]).tex}, {num(d['cond'][2]).tex}]")
        if dupkey == "old":
            dd = old["test_dup_in_train_or_val"]
            frac = [p["frac_test_dup_in_train_or_val"] for p in old["per_seed"]]
            dup = C(f"{dd['mean']:.1f} [{dd['min']}–{dd['max']}] ({100 * min(frac):.1f}–{100 * max(frac):.1f}%)",
                    f"{dd['mean']:.1f} [{dd['min']}--{dd['max']}] ({100 * min(frac):.1f}--{100 * max(frac):.1f}\\%)")
        elif dupkey == "random":
            dup = T(f"{int(np.mean(new_dups))} (max {int(max(new_dups))}, {len(new_dups)} seeds)")
        else:
            dup = C(EM)
        tab.row([T(name), intc(d["n"]), rng(*d["rows"]), intc(d["ncol"]), rng(*d["rank"]), cond, dup])
    tab.note(f"Rank is at most {max(design_row(k)['rank'][1] for k in labels if k in dm)} of "
             f"{design_row('random_dedupe')['ncol']:,} columns: the OFAT star design (all non-base compositions differ from the base in exactly one element) "
             "fills only the six (T, c_e) planes. Effective condition number = sv_max / smallest retained singular value.")
    tab.note("Duplicate leakage is audited for the old (non-deduplicated) random split and for the deduplicated random split; "
             "it was not computed for the temperature-band and composition-group splits (—).")
    cfeat = a.get("constant_feature_columns")
    if cfeat:
        tab.note("Constant feature column(s): " + ", ".join(cfeat) + ".")
    tab.note(f"Design statistics are over the seeds stored in design_audit.json (seeds listed per design: "
             f"{', '.join(f'{k}={design_row(k)['n']}' for k in labels if k in dm)}).")
    return tab


# ----------------------------------------------------------------------
#  Table 5: scaling
# ----------------------------------------------------------------------
def table_scaling():
    files = sorted(glob.glob(os.path.join(IN_DIR, "scaling*.csv")))
    tab = Table("scaling", "Scaling", "Scaling of fit/evaluation cost with data size.", "ll", ["Status", "Detail"], wide=False)
    if not files:
        tab.row([T("pending"), T(f"no scaling*.csv in {IN_DIR}/ yet")])
        tab.note("This table is generated automatically once results/camera_ready/scaling*.csv exists "
                 "(expected columns: n, method, measurement_type, mean, std, n_repeats).")
        return tab
    df = pd.concat([pd.read_csv(f).assign(_src=os.path.basename(f)) for f in files], ignore_index=True)
    need = {"n", "method", "measurement_type", "mean", "std"}
    if not need <= set(df.columns):
        tab.row([T("unrecognised schema"), T(f"{[os.path.basename(f) for f in files]}: columns {list(df.columns)}")])
        tab.note(f"Expected columns {sorted(need)}.")
        return tab
    mt = list(dict.fromkeys(df["measurement_type"]))
    methods = list(dict.fromkeys(df["method"]))
    ns = sorted(df["n"].unique())
    groups = [("", 1)] + [(m, len(methods)) for m in mt]
    t2 = Table("scaling", "Scaling", "Cost vs. data size n: mean ± std over repeats"
               + (f" ({int(df['n_repeats'].iloc[0])} repeats)" if "n_repeats" in df and df["n_repeats"].nunique() == 1 else "") + ".",
               "l" + "r" * (len(methods) * len(mt)), ["n"] + methods * len(mt), groups=groups, wide=len(methods) * len(mt) > 3)
    for n in ns:
        row = [intc(n)]
        for m_ in mt:
            for me in methods:
                r = df[(df["n"] == n) & (df["method"] == me) & (df["measurement_type"] == m_)]
                row.append(mean_std(r["mean"].iloc[0], r["std"].iloc[0]) if len(r) else C(EM))
        t2.row(row)
    t2.note("Source: " + ", ".join(os.path.basename(f) for f in files) + ".")
    return t2


# ----------------------------------------------------------------------
def main():
    global IN_DIR
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in-dir", default=IN_DIR)
    ap.add_argument("--out-dir", default=None, help="defaults to --in-dir")
    args = ap.parse_args()
    IN_DIR = args.in_dir
    out_dir = args.out_dir or IN_DIR
    os.makedirs(out_dir, exist_ok=True)

    builders = [table_accuracy, table_derivative, table_continuity, table_design, table_scaling]
    tables = [b() for b in builders]
    md = ["# Camera-ready paper tables", "",
          f"Generated by `experiments/export_paper_tables.py` from `{IN_DIR}/*.csv` (+ `design_audit.json`). Do not edit by hand.", ""]
    tex = ["% Generated by experiments/export_paper_tables.py -- do not edit by hand.",
           "% Requires \\usepackage{booktabs}.  Wide tables use table* (IEEEtran two-column).", ""]
    for i, t in enumerate(tables, 1):
        md.append(render_md(t, i))
        tex.append(render_tex(t, i))
    p_md, p_tex = (os.path.join(out_dir, f"paper_tables.{e}") for e in ("md", "tex"))
    with open(p_md, "w") as f:
        f.write("\n".join(md))
    with open(p_tex, "w") as f:
        f.write("\n".join(tex))
    print(f"wrote {p_md}\nwrote {p_tex}")


if __name__ == "__main__":
    sys.exit(main())
