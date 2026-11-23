#!/usr/bin/env python
"""0-D implicit thermal problem: do the SymGWR surrogates' analytic Jacobians work in Newton-Raphson?

NOT a phase-field simulation.  Per composition c (all 98 distinct compositions of
data/raw/718_all.csv after the repo's dedupe), one lumped body cools implicitly:

    H(T)    = G - T dG/dT                  (enthalpy,  dH/dT = -T d2G/dT2 = Cp)
    R(T)    = H(T) - H(T_n) + a (T - T_inf),   a = dt * kappa
    R'(T)   = dH/dT + a
    Newton  : T <- T - R / R'              from T_n, <= MAX_IT iterations, tol |R| < TOL (J/mol)

T_0 = 1800 K, T_inf = 900 K, dt = 1 s, N_STEPS = 150 steps.  kappa is calibrated PER
COMPOSITION (brentq on the reference trajectory) so that the reference reaches T_END = 1050 K
after exactly N_STEPS; the same kappa is then used by every surrogate.  The reference trajectory
therefore stays inside [1000, 1800] K and crosses the 1320-1470 K temp_band hold-out.

Surrogates (all derivatives converted to raw units, J/mol and K, through the split's scalers):
  reference     ground truth.  The stored G is rounded (1 J/mol above |G| = 1e5), so second differences
                are quantisation noise of the size of Cp itself: an interpolating cubic spline of the raw
                full table has Cp <= 0 somewhere in [1000, 1800] K for 75 of 98 compositions (seed-42
                table, checked), which makes the step ill-posed.  Truth is therefore a degree-4
                smoothing spline (UnivariateSpline, s = N * 0.3^2, i.e. the rounding-noise level);
                it has residual <~ 1 J/mol and Cp > 0 (diagnostics printed at start-up).
  Spline_full_raw  the literal "cubic spline of raw G on the full table, analytic G_T / G_TT" version
                of the reference, run against the smooth truth.  Both are computed once per
                composition (protocol "n/a", seed -1).
  SymGWR-T      G_i + gradP(r_i).(q - r_i), i = nearest reference (k=1, scaled space).
                Analytic G_T = gradP_T(r_i), G_TT = 0 (piecewise-linear surrogate).
  SymGWR-R      P(q) + G_i - P(r_i).  G_T = P_T(q), G_TT = P_TT(q).
  SymGWR-R-IDW  P(q) + sum_{i in 4-NN(q)} w_i (G_i - P(r_i)), inverse-distance weights (run_ridw.blend).
  SymGWR-R-CS   same with compact-support weights (run_ridw_cs.cs_blend).  Both reuse the repo's blend
                code (value + ANALYTIC gradient, z-space).  G_T = P_T + dB/dT is that analytic gradient;
                G_TT = P_TT + d2B/dT2 where d2B/dT2 is a central finite difference (step FD_H in scaled T)
                of the analytic gradient with the neighbour set frozen (the repo's own Hessian recipe,
                run_ridw_cs.hessian_row).  Checked against FD of the blend VALUE at start-up
                (check_rblend_derivatives); derivatives are converted to raw units like the others.
  Polynomial    P(q) (degree 5, gwr.models.fit_polynomial_regression on train+val).
  Spline_train  cubic spline (not-a-knot) along T built only from the fit-set rows of c.
  SymGWR-T_secant  SymGWR-T with a Jacobian-free secant solve of R (x0 = T_n, x1 = T_n + previous
                step, -10 K on step 1), same |R| < TOL criterion.
  SymGWR-T_brent   SymGWR-T with a Jacobian-free bracketing (Brent) solve on [900 K, T_n]; "converged" =
                sign change located (bracket < 1e-9 K), |R| NOT required; |R| at the returned point is
                recorded (abs_R_final_max).  Together these separate "analytic Jacobian" from "root
                finding at all" (on a staircase H there is generally no exact root, only a jump).
Protocols (seeds 42-51): "random" (60/20/20, fit set = train+val) and "temp_band"
(1320-1470 K held out, fit set = remainder of the rows).

Failure policy (identical for every surrogate): a step that does not reach |R| < TOL in MAX_IT
iterations, or hits a non-finite / zero R', is "failed"; the last finite iterate (clipped to
T_CLIP) is accepted and the trajectory continues, so max_dT_K measures the consequence.  If the
accepted T leaves [1000, 1800] K the run is "escaped": it stops there, the remaining steps count as
not converged and T is held for the dT error.  run_converged = all N_STEPS converged.
P(q) along T for fixed c is evaluated exactly as a 1-D polynomial in the scaled T (coefficients
collapsed once per composition; checked against gwr PolyModel in --selftest / at start-up).

Wall time = trajectory solve only (fit / spline build excluded).

Usage:
    OMP_NUM_THREADS=2 .venv/bin/python experiments/run_newton_test.py
    ... --only SymGWR-R-IDW SymGWR-R-CS   # run just these surrogates; MERGE into the existing newton_runs.csv
                                          # (rows of those surrogates replaced, others untouched), re-summarise
    ... --comps 30        # fixed random subset (seed 0) of the 98 compositions
    ... --seeds 42 43     # subset of seeds (smoke test)
Outputs: results/camera_ready/newton_runs.csv, newton_summary.csv; for the R-IDW / R-CS rows also
newton_rblend_diag.csv (per run) and newton_rblend_diag_summary.csv: for every Newton step the iterate
interval [min, max] is intersected with the surrogate's k-set breakpoints (scan of T on a 0.05 K grid +
bisection, [900, 1900] K) to see whether failures coincide with k-set changes / value jumps of H.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

import argparse
import math
import sys
import time

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PPoly, UnivariateSpline
from scipy.optimize import brentq

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import run_ridw as rw  # noqa: E402
import run_ridw_cs as rcs  # noqa: E402

from camera_ready_common import (  # noqa: E402
    OUT_DIR, SEEDS, TEMP_COL, composition_ids, fit_symgwr,
)

# ---- problem / solver constants ---------------------------------------
T0 = 1800.0
T_INF = 900.0
T_END = 1050.0
N_STEPS = 150
DT = 1.0
T_LO, T_HI = 1000.0, 1800.0  # data / physical window
T_CLIP = (900.0, 1900.0)  # accepted-iterate guard on failed steps
MAX_IT = 20
TOL = 1e-8  # |R| in J/mol
JAC_EPS = 1e-8  # R' <= JAC_EPS  -> "R' ~0 or wrong sign"
CP_EPS = 1e-6  # dH/dT <= CP_EPS -> surrogate has no / negative heat capacity
SIGMA_ROUND = 0.3  # J/mol, rounding noise of the stored G (see PPolyEval.smooth_truth)
ORDER_FLOOR = 1e-11  # residuals below this are rounding noise, excluded from order fits
PROTOCOLS = ("random", "temp_band")

SURROGATES = ("SymGWR-T", "SymGWR-R", "Polynomial", "Spline_train", "SymGWR-T_secant", "SymGWR-T_brent")
RBLEND = {"SymGWR-R-IDW": "idw", "SymGWR-R-CS": "cs"}  # name -> kind (run_ridw_cs.blend_any)
RBLEND_K = 4
FD_H = 1e-5  # central-difference step (scaled T) for d2B/dT2; neighbour set frozen
ALL_SURROGATES = SURROGATES + tuple(RBLEND)
SCAN_DT = 0.05  # K, grid for k-set breakpoint scan
JUMP_H_TOL = 1e-6  # J/mol: |H| jump counted as a value jump
JUMP_HP_TOL = 1e-2  # relative jump of R' counted as a Jacobian jump
PROTOCOL_FREE = ("reference", "Spline_full_raw")  # one run per composition, protocol "n/a"


# ----------------------------------------------------------------------
#  Surrogate evaluators:  ev(T) -> (H, dH/dT) in raw units
# ----------------------------------------------------------------------
class PPolyEval:
    """Piecewise polynomial G, G_T, G_TT (any degree) from PPoly coefficients.

    Extrapolates with the end pieces, exactly like scipy's CubicSpline / PPoly.
    """

    def __init__(self, x, c):
        self.x = np.asarray(x, dtype=np.float64)
        self.c = np.asarray(c, dtype=np.float64)  # (k+1, n_pieces), highest power first
        self.n = len(self.x)
        self.k = self.c.shape[0] - 1

    @classmethod
    def interp_cubic(cls, T, G):
        """Interpolating cubic spline (not-a-knot) through every (T, G) row."""
        cs = CubicSpline(T, G)
        return cls(cs.x, cs.c)

    @classmethod
    def smooth_truth(cls, T, G):
        """Least-squares degree-4 smoothing spline at the data's rounding-noise level.

        s = N * SIGMA_ROUND^2 (G is stored to 1 J/mol above |G| = 1e5, sigma = 1/sqrt(12) ~ 0.3).
        On this table the fit has few or no interior knots (a quartic per composition).
        """
        u = UnivariateSpline(T, G, k=4, s=len(T) * SIGMA_ROUND ** 2)
        pp = PPoly.from_spline(u._eval_args)
        return cls(pp.x, pp.c), u

    def gtt(self, T):
        i = min(max(int(np.searchsorted(self.x, T, side="right")) - 1, 0), self.n - 2)
        dx = T - self.x[i]
        c = self.c[:, i]
        g = g1 = g2 = 0.0
        for cj in c:  # Horner, tracking first and second derivatives
            g2 = g2 * dx + 2.0 * g1
            g1 = g1 * dx + g
            g = g * dx + cj
        return g, g1, g2

    def __call__(self, T):
        g, g1, g2 = self.gtt(T)
        return g - T * g1, -T * g2


class PolyAlongT:
    """P(c, x_T) = sum_k a_k x_T^k for fixed scaled composition (exact collapse of the degree-5 P)."""

    def __init__(self, pm, xc_scaled, kT):
        fac = xc_scaled[None, :] ** pm.powers  # (terms, d)
        fac[:, kT] = 1.0  # T factor handled by the polynomial in x_T
        w = fac.prod(axis=1)
        a = np.zeros(pm.deg + 1)
        np.add.at(a, pm.powers[:, kT], pm.coef * w)
        a[0] += pm.intercept
        self.a = a
        self.deg = pm.deg

    def __call__(self, x):
        a = self.a
        p = p1 = p2 = 0.0
        for k in range(self.deg, -1, -1):
            p2 = p2 * x + 2.0 * p1
            p1 = p1 * x + p
            p = p * x + a[k]
        return p, p1, p2


class ScaledSurrogate:
    """Wrap a z-space G(x_T) -> (G, G_x, G_xx) into raw (H, dH/dT)."""

    def __init__(self, zfun, ys, ym, Tm, Ts):
        self.zfun, self.ys, self.ym, self.Tm, self.Ts = zfun, ys, ym, Tm, Ts

    def __call__(self, T):
        x = (T - self.Tm) / self.Ts
        gz, g1z, g2z = self.zfun(x)
        G = self.ys * gz + self.ym
        G1 = self.ys * g1z / self.Ts
        G2 = self.ys * g2z / (self.Ts * self.Ts)
        return G - T * G1, -T * G2


class RBlendAlongT:
    """z-space G(x_T) = P + B for SymGWR-R-IDW / -CS at fixed composition: (G, G_x, G_xx).

    B and dB/dx come from the repo's run_ridw.blend / run_ridw_cs.cs_blend (analytic gradient).  d2B/dx2
    is the central difference (step FD_H) of that analytic gradient with the neighbour set frozen at the
    base query (one tree query per call).  P, P_x, P_xx are exact (PolyAlongT)."""

    def __init__(self, fit, pa, xc, kT, kind, k=RBLEND_K, h=FD_H):
        self.fit, self.pa, self.xc, self.kT, self.kind, self.k, self.h = fit, pa, xc, kT, kind, k, h
        self.nb = k if kind == "idw" else k + 1

    def blend(self, x, h=None):
        """(B, B_x, B_xx) at x_T = x."""
        h = self.h if h is None else h
        q = self.xc.copy()
        q[self.kT] = x
        _, idx = self.fit.tree.query(q, k=self.nb, workers=1)
        Q = np.tile(q, (3, 1))
        Q[1, self.kT] -= h
        Q[2, self.kT] += h
        with np.errstate(invalid="ignore", divide="ignore"):  # 0/0 at d = 0 inside cs_blend is masked there
            B, gB = rcs.blend_any(self.fit, Q, self.k, self.kind, True, np.tile(np.asarray(idx).reshape(1, -1), (3, 1)))
        return float(B[0]), float(gB[0, self.kT]), float((gB[2, self.kT] - gB[1, self.kT]) / (2.0 * h))

    def __call__(self, x):
        p, p1, p2 = self.pa(x)
        b, b1, b2 = self.blend(x)
        return p + b, p1 + b1, p2 + b2


def check_rblend_derivatives(fit, comps, cid, kT, df, n_t=25, seed=7):
    """Numeric check of the R-IDW / R-CS derivatives actually used (G_T, G_TT, z-space B part):
    compare blend()'s analytic dB/dx and FD d2B/dx2 with 4th-order central differences of the blend VALUE
    (neighbour set frozen, steps 2e-4 / 5e-5) at random T away from k-set changes.  Returns max rel errs."""
    rng = np.random.default_rng(seed)
    sp = fit.split
    fs = sp.feature_scaler
    Tm, Ts = float(fs.mean_[kT]), float(fs.scale_[kT])
    errs = {}
    for nm, kind in RBLEND.items():
        e1 = e2 = 0.0
        for c in comps[:6]:
            r = np.where(cid == c)[0][0]
            xc = fs.transform(df[sp.feature_cols].to_numpy(dtype=np.float64)[r][None, :])[0]
            sur = RBlendAlongT(fit, None, xc, kT, kind)
            T_ref = fit.split.df[TEMP_COL].to_numpy(dtype=np.float64)[fit.ref_idx[cid[fit.ref_idx] == c]]
            T_ref = T_ref[(T_ref > 1010) & (T_ref < 1790)]
            for T in np.concatenate([rng.uniform(1010, 1790, n_t), rng.choice(T_ref, 5)]):  # + exact refs (d = 0)
                x = (T - Tm) / Ts
                q = xc.copy()
                q[kT] = x
                dist, idx = fit.tree.query(q, k=RBLEND_K + 1, workers=1)
                if dist[RBLEND_K] - dist[RBLEND_K - 1] < 4e-3:  # stencil (+-4e-4) could reach a k-set change
                    continue
                nbr = np.tile(np.asarray(idx[:sur.nb]).reshape(1, -1), (5, 1))

                def val(dx):
                    Q = np.tile(q, (5, 1))
                    Q[:, kT] += dx
                    with np.errstate(invalid="ignore", divide="ignore"):
                        return rcs.blend_any(fit, Q, RBLEND_K, kind, False, nbr)[0]

                b, b1, b2 = sur.blend(x)
                c1 = c2 = np.inf
                for hh in (2e-4, 5e-5):  # two stencil widths, smaller error kept (rounding vs truncation)
                    v = val(np.array([-2 * hh, -hh, 0.0, hh, 2 * hh]))
                    d1 = (v[0] - 8 * v[1] + 8 * v[3] - v[4]) / (12 * hh)
                    d2 = (-v[0] + 16 * v[1] - 30 * v[2] + 16 * v[3] - v[4]) / (12 * hh * hh)
                    c1 = min(c1, abs(b1 - d1) / (1 + abs(d1)))
                    c2 = min(c2, abs(b2 - d2) / (1 + abs(d2)))
                e1, e2 = max(e1, c1), max(e2, c2)
        errs[nm] = (e1, e2)
    return errs


def make_surrogates(fit, c_id, comp_ids_full, kT, df):
    sp = fit.split
    fs, ts = sp.feature_scaler, sp.target_scaler
    ys, ym = float(ts.scale_[0]), float(ts.mean_[0])
    Tm, Ts = float(fs.mean_[kT]), float(fs.scale_[kT])
    rows = np.where(comp_ids_full == c_id)[0]
    x_raw = df[sp.feature_cols].to_numpy(dtype=np.float64)[rows[0]]
    xc = fs.transform(x_raw[None, :])[0]  # scaled composition, T entry overwritten per query
    pa = PolyAlongT(fit.model, xc, kT)

    def poly_z(x):
        return pa(x)

    P_ref_arr, tree, ref_X, ref_y, ref_grad, ref_P = (
        None, fit.tree, fit.ref_X, fit.ref_y, fit.ref_grad, fit.ref_P)

    def sym_t_z(x):
        q = xc.copy()
        q[kT] = x
        _, i = tree.query(q, k=1)
        g = ref_grad[i]
        return ref_y[i] + float(g @ (q - ref_X[i])), float(g[kT]), 0.0

    def sym_r_z(x):
        q = xc.copy()
        q[kT] = x
        _, i = tree.query(q, k=1)
        p, p1, p2 = pa(x)
        return p + ref_y[i] - ref_P[i], p1, p2

    out = {
        "SymGWR-T": ScaledSurrogate(sym_t_z, ys, ym, Tm, Ts),
        "SymGWR-R": ScaledSurrogate(sym_r_z, ys, ym, Tm, Ts),
        "Polynomial": ScaledSurrogate(poly_z, ys, ym, Tm, Ts),
    }
    for nm, kind in RBLEND.items():
        out[nm] = ScaledSurrogate(RBlendAlongT(fit, pa, xc, kT, kind), ys, ym, Tm, Ts)
    # spline from fit-set rows of this composition only
    fit_rows = fit.ref_idx[comp_ids_full[fit.ref_idx] == c_id]
    Tall = df[TEMP_COL].to_numpy(dtype=np.float64)
    Gall = df[sp._target].to_numpy(dtype=np.float64)
    o = np.argsort(Tall[fit_rows])
    out["Spline_train"] = PPolyEval.interp_cubic(Tall[fit_rows][o], Gall[fit_rows][o])
    out["_n_fit_rows"] = len(fit_rows)
    out["_poly_along"] = (pa, xc)
    return out


# ----------------------------------------------------------------------
#  Per-step solvers
# ----------------------------------------------------------------------
def _order(hist):
    """Observed order from the LAST valid residual triple r0 > r1 > r2 > floor (NaN if none)."""
    for k in range(len(hist) - 1, 1, -1):
        r0, r1, r2 = hist[k - 2], hist[k - 1], hist[k]
        if r2 > ORDER_FLOOR and r0 > r1 > r2:
            return math.log(r2 / r1) / math.log(r1 / r0)
    return float("nan")


def newton_step(ev, Tn, Hn, a, ref_ev):
    T = Tn
    hist, jerr = [], []
    bad_jac = cp_bad = False
    conv = False
    iters = 0
    for it in range(MAX_IT + 1):
        H, Hp = ev(T)
        R = H - Hn + a * (T - T_INF)
        Rp = Hp + a
        if not (math.isfinite(R) and math.isfinite(Rp)):
            break
        hist.append(abs(R))
        if Hp <= CP_EPS:
            cp_bad = True
        if Rp <= JAC_EPS:
            bad_jac = True
        if T_LO <= T <= T_HI:
            _, Hp_ref = ref_ev(T)
            jerr.append(abs(Rp - (Hp_ref + a)) / (Hp_ref + a))
        if abs(R) < TOL:
            conv = True
            iters = it
            break
        if it == MAX_IT or abs(Rp) < 1e-12:
            iters = it
            break
        T = T - R / Rp
        iters = it + 1
    return T, conv, iters, hist, bad_jac, cp_bad, jerr


def secant_step(ev, Tn, Hn, a, ref_ev, prev_dT):
    def F(T):
        H, Hp = ev(T)
        return H - Hn + a * (T - T_INF), Hp

    x0 = Tn
    x1 = Tn + (prev_dT if prev_dT is not None else -10.0)
    F0, _ = F(x0)
    F1, _ = F(x1)
    hist = [abs(F0), abs(F1)]
    jerr = []
    bad_jac = cp_bad = False
    conv = abs(F1) < TOL
    iters = 0
    if abs(F0) < TOL:
        return x0, True, 0, hist, False, False, jerr
    while not conv and iters < MAX_IT:
        if x1 == x0 or not (math.isfinite(F0) and math.isfinite(F1)):
            break
        slope = (F1 - F0) / (x1 - x0)
        if not math.isfinite(slope) or abs(slope) < 1e-12:
            break
        if slope <= JAC_EPS:
            bad_jac = True
        if T_LO <= x1 <= T_HI:
            _, Hp_ref = ref_ev(x1)
            jerr.append(abs(slope - (Hp_ref + a)) / (Hp_ref + a))
        x2 = x1 - F1 / slope
        F2, _ = F(x2)
        x0, F0, x1, F1 = x1, F1, x2, F2
        iters += 1
        if not math.isfinite(F1):
            break
        hist.append(abs(F1))
        conv = abs(F1) < TOL
    return x1, conv, iters, hist, bad_jac, cp_bad, jerr


def brent_step(ev, Tn, Hn, a):
    """Bracketing (Brent) solve of R on [T_CLIP[0], T_n]; derivative-free.

    "converged" here = bracket collapsed (xtol 1e-9 K) within 100 iterations, i.e. a SIGN CHANGE of R
    was located.  Unlike the other solvers it does NOT require |R| < TOL, so on a discontinuous R
    (SymGWR-T's staircase H) it can succeed at a jump; the |R| it returns is recorded.
    Iteration count = function evaluations.
    """
    def R(T):
        return ev(T)[0] - Hn + a * (T - T_INF)

    lo, hi = T_CLIP[0], Tn
    Rlo, Rhi = R(lo), R(hi)
    if not (math.isfinite(Rlo) and math.isfinite(Rhi)) or Rlo * Rhi > 0:
        return Tn, False, 0, [abs(Rhi)], False, False, []
    if Rhi == 0.0:
        return hi, True, 0, [0.0], False, False, []
    root, info = brentq(R, lo, hi, xtol=1e-9, rtol=1e-14, maxiter=100, full_output=True, disp=False)
    return root, bool(info.converged), int(info.function_calls), [abs(R(root))], False, False, []


def trajectory(ev, a, ref_ev, method="newton"):
    """Run up to N_STEPS implicit steps.  Returns (T path (N+1,), per-step stats dict).

    A trajectory that ends a step outside [T_LO, T_HI] has left the window where the data / the
    reference live: it is flagged ``escape`` (step index, else -1) and stopped, T is held at the
    accepted (clipped) value for the remaining entries and the unattempted steps count as not
    converged.  (Without this, a path that collapses to the clip value T_inf = 900 K is a trivial
    fixed point R(T_n) = 0 and would be counted as "converged".)
    """
    T = np.empty(N_STEPS + 1)
    T[0] = T0
    st = dict(conv=[], iters=[], order=[], bad_jac=[], cp_bad=[], jerr=[], rfin=[], escape=-1)
    t0 = time.perf_counter()
    prev_dT = None
    for n in range(N_STEPS):
        Tn = T[n]
        Hn = ev(Tn)[0]
        if method == "newton":
            Tk, conv, iters, hist, bj, cb, je = newton_step(ev, Tn, Hn, a, ref_ev)
        elif method == "secant":
            Tk, conv, iters, hist, bj, cb, je = secant_step(ev, Tn, Hn, a, ref_ev, prev_dT)
        else:
            Tk, conv, iters, hist, bj, cb, je = brent_step(ev, Tn, Hn, a)
        if not math.isfinite(Tk):
            Tk = Tn
        Tk = min(max(Tk, T_CLIP[0]), T_CLIP[1])
        T[n + 1] = Tk
        prev_dT = Tk - Tn
        st["conv"].append(conv)
        st["iters"].append(iters)
        st["order"].append(_order(hist) if conv else float("nan"))
        st["bad_jac"].append(bj)
        st["cp_bad"].append(cb)
        st["jerr"].extend(je)
        st["rfin"].append(hist[-1] if hist else float("nan"))
        if not (T_LO <= Tk <= T_HI):
            st["escape"] = n + 1
            T[n + 2:] = Tk
            break
    st["wall"] = time.perf_counter() - t0
    return T, st


class Recorder:
    """Callable wrapper logging every T at which ``ev`` is called (trajectory is solved unchanged)."""

    def __init__(self, ev):
        self.ev, self.calls = ev, []

    def __call__(self, T):
        self.calls.append(T)
        return self.ev(T)


def step_intervals(rec, T, st):
    """Per attempted step: [min, max] of all T the solver evaluated.  A step makes 1 (H_n) + iters + 1
    evaluations of ev (newton_step), which is asserted against the log."""
    n = np.asarray(st["iters"], int)
    assert len(rec.calls) == int(np.sum(n + 2)), (len(rec.calls), int(np.sum(n + 2)))
    c = np.asarray(rec.calls)
    pos = np.concatenate([[0], np.cumsum(n + 2)])
    lo = np.array([c[pos[i]:pos[i + 1]].min() for i in range(len(n))])
    hi = np.array([c[pos[i]:pos[i + 1]].max() for i in range(len(n))])
    assert c[pos[0]] == T[0]
    return lo, hi


def scan_breakpoints(fit, xc, kT, sur, a, Tm, Ts, k=RBLEND_K):
    """k-set / support breakpoints of the R-blend along T in [T_CLIP] (0.05 K grid, then bisection to
    ~1e-14 K).  Returns dict of arrays: T, kind (0 = k-set change, 1 = (k+1)-th reference changes with
    the k-set unchanged), dG, dH (J/mol), dHp_rel (|dR'|/R', same side-by-side evaluation)."""
    xg = (np.arange(T_CLIP[0], T_CLIP[1] + 0.5 * SCAN_DT, SCAN_DT) - Tm) / Ts
    Q = np.tile(xc, (len(xg), 1))
    Q[:, kT] = xg
    _, idx = fit.tree.query(Q, k=k + 1, workers=2)
    kset, supp = np.sort(idx[:, :k], axis=1), idx[:, k]
    chg_set = (kset[1:] != kset[:-1]).any(1)
    chg_sup = (supp[1:] != supp[:-1]) & ~chg_set
    out = dict(T=[], kind=[], dG=[], dH=[], dHp_rel=[])
    for kind, mask in ((0, chg_set), (1, chg_sup)):
        cells = np.nonzero(mask)[0]
        if not len(cells):
            continue
        key0 = kset[cells] if kind == 0 else supp[cells][:, None]
        lo, hi = xg[cells].copy(), xg[cells + 1].copy()
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            Qm = np.tile(xc, (len(cells), 1))
            Qm[:, kT] = mid
            _, im = fit.tree.query(Qm, k=k + 1, workers=1)
            key = np.sort(im[:, :k], axis=1) if kind == 0 else im[:, k:k + 1]
            same = (key == key0).all(1)
            lo = np.where(same, mid, lo)
            hi = np.where(same, hi, mid)
        for xl, xh in zip(lo, hi):
            Tl, Th = Tm + Ts * xl, Tm + Ts * xh
            Gl = sur.ys * sur.zfun(xl)[0]
            Gh = sur.ys * sur.zfun(xh)[0]
            (Hl, Hpl), (Hh, Hph) = sur(Tl), sur(Th)
            out["T"].append(0.5 * (Tl + Th))
            out["kind"].append(kind)
            out["dG"].append(abs(Gh - Gl))
            out["dH"].append(abs(Hh - Hl))
            out["dHp_rel"].append(abs(Hph - Hpl) / max(abs(Hpl + a), abs(Hph + a), 1e-300))
    o = np.argsort(out["T"])
    return {kk: np.asarray(v)[o] for kk, v in out.items()}


def _touch(Tb, lo, hi):
    """per step: number of breakpoints Tb (sorted) inside [lo, hi]"""
    return np.searchsorted(Tb, hi, side="right") - np.searchsorted(Tb, lo, side="left")


def diag_row(base, name, bp, lo, hi, conv, cp_bad, bad_jac):
    """Per-run association of non-converged / converged steps with breakpoints."""
    conv = np.asarray(conv, bool)
    cpb, bjb = np.asarray(cp_bad, bool), np.asarray(bad_jac, bool)
    ks, jv, jp = bp["kind"] == 0, bp["dH"] > JUMP_H_TOL, bp["dHp_rel"] > JUMP_HP_TOL
    flags = {
        "kset": _touch(bp["T"][ks], lo, hi) > 0,
        "support": _touch(bp["T"][bp["kind"] == 1], lo, hi) > 0,
        "Hjump": _touch(bp["T"][jv], lo, hi) > 0,
        "Hpjump": _touch(bp["T"][jp], lo, hi) > 0,
        "any": _touch(bp["T"], lo, hi) > 0,
        "cp_nonpos": cpb,  # not a breakpoint: some iterate had dH/dT <= 0 (negative surrogate Cp)
        "bad_jac": bjb,  # some iterate had R' <= 1e-8
    }
    row = dict(**base, surrogate=name, n_steps_attempted=len(conv), n_steps_conv=int(conv.sum()),
               n_steps_fail=int((~conv).sum()),
               n_break_kset=int(ks.sum()), n_break_support=int((bp["kind"] == 1).sum()),
               n_break_Hjump=int(jv.sum()), n_break_Hpjump=int(jp.sum()),
               max_dG_jump_Jmol=float(bp["dG"].max()) if len(bp["T"]) else 0.0,
               max_dH_jump_Jmol=float(bp["dH"].max()) if len(bp["T"]) else 0.0,
               max_dHp_rel_jump=float(bp["dHp_rel"].max()) if len(bp["T"]) else 0.0)
    for f, v in flags.items():
        row[f"fail_steps_touch_{f}"] = int((v & ~conv).sum())
        row[f"conv_steps_touch_{f}"] = int((v & conv).sum())
    return row


def summarize_diag(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sur, proto), g in d.groupby(["surrogate", "protocol"], sort=False):
        nf, nc = g.n_steps_fail.sum(), g.n_steps_conv.sum()
        r = dict(surrogate=sur, protocol=proto, n_runs=len(g), steps_attempted=int(g.n_steps_attempted.sum()),
                 steps_failed=int(nf), runs_with_failed_step=int((g.n_steps_fail > 0).sum()),
                 breaks_kset_mean=g.n_break_kset.mean(), breaks_support_mean=g.n_break_support.mean(),
                 breaks_Hjump_mean=g.n_break_Hjump.mean(), breaks_Hpjump_mean=g.n_break_Hpjump.mean(),
                 max_dG_jump_Jmol=g.max_dG_jump_Jmol.max(), max_dH_jump_Jmol=g.max_dH_jump_Jmol.max(),
                 max_dHp_rel_jump=g.max_dHp_rel_jump.max())
        for f in ("kset", "support", "Hjump", "Hpjump", "any", "cp_nonpos", "bad_jac"):
            r[f"pct_failed_steps_touch_{f}"] = 100.0 * g[f"fail_steps_touch_{f}"].sum() / nf if nf else float("nan")
            r[f"pct_conv_steps_touch_{f}"] = 100.0 * g[f"conv_steps_touch_{f}"].sum() / nc if nc else float("nan")
        rows.append(r)
    return pd.DataFrame(rows)


def calibrate_a(ref_ev):
    def f(a):
        T, _ = trajectory(ref_ev, a, ref_ev)
        return T[-1] - T_END

    return brentq(f, 1e-2, 1e3, xtol=1e-10, rtol=1e-12)


def run_row(base, name, st, T, ref_T, a):
    conv = np.asarray(st["conv"], bool)  # attempted steps only
    it = np.asarray(st["iters"], float)
    od = np.asarray(st["order"], float)
    je = np.asarray(st["jerr"], float)
    dT = np.abs(T - ref_T) if ref_T is not None else np.zeros_like(T)
    return dict(
        **base, surrogate=name, n_steps=N_STEPS, n_steps_attempted=len(conv), kappa=a / DT, a_dt_kappa=a,
        run_converged=bool(conv.sum() == N_STEPS), n_steps_converged=int(conv.sum()),
        n_steps_fail=int(N_STEPS - conv.sum()),  # includes unattempted steps after an escape
        escape_step=int(st["escape"]),
        iters_mean=float(it.mean()), iters_max=int(it.max()),
        order_median=float(np.nanmedian(od)) if np.isfinite(od).any() else float("nan"),
        n_order_samples=int(np.isfinite(od).sum()),
        n_steps_bad_jac=int(np.sum(st["bad_jac"])), n_steps_cp_nonpos=int(np.sum(st["cp_bad"])),
        jac_relerr_mean=float(je.mean()) if je.size else float("nan"),
        jac_relerr_max=float(je.max()) if je.size else float("nan"),
        abs_R_final_max=float(np.nanmax(st["rfin"])) if np.isfinite(st["rfin"]).any() else float("nan"),
        max_dT_K=float(dT.max()), final_dT_K=float(T[-1] - ref_T[-1]) if ref_T is not None else 0.0,
        final_T_K=float(T[-1]), wall_s=st["wall"],
    )


# ----------------------------------------------------------------------
#  Summary
# ----------------------------------------------------------------------
def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    order = ["reference", "Spline_full_raw", *ALL_SURROGATES]
    rows = []
    for (sur, proto), g in runs.groupby(["surrogate", "protocol"], sort=False):
        rows.append(dict(
            surrogate=sur, protocol=proto, n_runs=len(g),
            converged_pct=100.0 * g.run_converged.mean(),
            step_converged_pct=100.0 * g.n_steps_converged.sum() / g.n_steps.sum(),
            iters_mean=g.iters_mean.mean(), iters_max=int(g.iters_max.max()),
            order_median=g.order_median.median(), order_runs_with_estimate=int(g.order_median.notna().sum()),
            bad_jac_steps_mean=g.n_steps_bad_jac.mean(), cp_nonpos_steps_mean=g.n_steps_cp_nonpos.mean(),
            jac_relerr_mean=g.jac_relerr_mean.mean(), abs_R_final_max_median=g.abs_R_final_max.median(),
            max_dT_K_median=g.max_dT_K.median(), max_dT_K_p95=g.max_dT_K.quantile(0.95),
            max_dT_K_max=g.max_dT_K.max(),
            final_dT_K_abs_median=g.final_dT_K.abs().median(),
            escaped_pct=100.0 * (g.escape_step >= 0).mean(), wall_s_mean=g.wall_s.mean(),
        ))
    s = pd.DataFrame(rows)
    s["_o"] = s.surrogate.map({n: i for i, n in enumerate(order)})
    s["_p"] = s.protocol.map({"n/a": 0, "random": 1, "temp_band": 2})
    return s.sort_values(["_p", "_o"]).drop(columns=["_o", "_p"]).reset_index(drop=True)


def print_summary(s: pd.DataFrame):
    cols = ["surrogate", "protocol", "n_runs", "converged_pct", "step_converged_pct", "iters_mean",
            "iters_max", "order_median", "bad_jac_steps_mean", "cp_nonpos_steps_mean",
            "jac_relerr_mean", "abs_R_final_max_median", "escaped_pct", "max_dT_K_median", "max_dT_K_max", "wall_s_mean"]
    with pd.option_context("display.width", 250, "display.max_columns", 50,
                           "display.float_format", lambda v: f"{v:.4g}"):
        print(s[cols].to_string(index=False))


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comps", type=int, default=0, help="0 = all compositions, else fixed random subset (rng seed 0)")
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--only", nargs="*", choices=ALL_SURROGATES, default=None,
                    help="run only these surrogates and merge their rows into the existing newton_runs.csv "
                         "(reference rows are recomputed and checked against the file, not rewritten)")
    args = ap.parse_args()
    names = tuple(args.only) if args.only else ALL_SURROGATES
    os.makedirs(args.out, exist_ok=True)

    # data + reference trajectories (one per composition) -----------------
    fit0 = fit_symgwr("random", args.seeds[0])
    df = fit0.split.df
    feats = fit0.split.feature_cols
    kT = feats.index(TEMP_COL)
    cid = composition_ids(df, feats)
    comps = np.unique(cid)
    if args.comps and args.comps < len(comps):
        comps = np.sort(np.random.default_rng(0).choice(comps, args.comps, replace=False))
    Tall = df[TEMP_COL].to_numpy(dtype=np.float64)
    Gall = df[fit0.split._target].to_numpy(dtype=np.float64)

    refs = {}
    rows = []
    t_start = time.time()
    truth_diag = []
    for c in comps:
        r = np.where(cid == c)[0]
        o = np.argsort(Tall[r])
        assert len(r) == len(np.unique(Tall[r])), f"duplicate T on composition {c}"
        Tc, Gc = Tall[r][o], Gall[r][o]
        ref_ev, u = PPolyEval.smooth_truth(Tc, Gc)
        # self-check: PPolyEval reproduces scipy's UnivariateSpline value / 1st / 2nd derivative
        tg = np.linspace(1000, 1800, 17)
        for d, idx in ((0, 0), (1, 1), (2, 2)):
            mine = np.array([ref_ev.gtt(t)[idx] for t in tg])
            assert np.allclose(mine, u.derivative(d)(tg) if d else u(tg), rtol=1e-8, atol=1e-8)
        cpg = np.array([ref_ev(t)[1] for t in np.linspace(T_LO, T_HI, 401)])
        truth_diag.append((np.abs(u(Tc) - Gc).max(), cpg.min(), cpg.max(), len(u.get_knots()) - 2))
        a = calibrate_a(ref_ev)
        Tref, st = trajectory(ref_ev, a, ref_ev)
        refs[c] = (ref_ev, a, Tref)
        rows.append(run_row(dict(protocol="n/a", seed=-1, comp_id=int(c), n_fit_rows=len(r)),
                            "reference", st, Tref, None, a))
        # literal spec: interpolating cubic spline of the raw full table, analytic derivatives
        raw_ev = PPolyEval.interp_cubic(Tc, Gc)
        Traw, st = trajectory(raw_ev, a, ref_ev)
        rows.append(run_row(dict(protocol="n/a", seed=-1, comp_id=int(c), n_fit_rows=len(r)),
                            "Spline_full_raw", st, Traw, Tref, a))
    td = np.array(truth_diag)
    print(f"[newton] truth = degree-4 smoothing spline, s = N*{SIGMA_ROUND}^2: max |fit - G| over table "
          f"{td[:, 0].max():.2f} J/mol, Cp range over [{T_LO:.0f},{T_HI:.0f}] K "
          f"[{td[:, 1].min():.2f}, {td[:, 2].max():.2f}] J/mol/K, interior knots max {int(td[:, 3].max())}")
    print(f"[newton] {len(comps)} compositions, references done in {time.time() - t_start:.1f}s; "
          f"reference path T range [{min(v[2].min() for v in refs.values()):.1f}, "
          f"{max(v[2].max() for v in refs.values()):.1f}] K")

    # surrogates ------------------------------------------------------------
    first = True
    diag = []
    for proto in PROTOCOLS:
        for seed in args.seeds:
            fit = fit_symgwr(proto, seed)
            assert fit.split.feature_cols == feats
            for c in comps:
                ref_ev, a, Tref = refs[c]
                sur = make_surrogates(fit, c, cid, kT, df)
                if first:  # exactness check of the 1-D collapse against the repo's PolyModel
                    pa, xc = sur["_poly_along"]
                    rr = np.random.default_rng(1)
                    xs = rr.uniform(-1.5, 1.5, 5)
                    Q = np.tile(xc, (5, 1))
                    Q[:, kT] = xs
                    pv = fit.model.value(Q)
                    d1 = fit.model.grad(Q)[:, kT]
                    d2 = fit.model.hess(Q)[:, kT, kT]
                    got = np.array([pa(x) for x in xs])
                    err = max(np.max(np.abs(got[:, 0] - pv) / (1 + np.abs(pv))),
                              np.max(np.abs(got[:, 1] - d1) / (1 + np.abs(d1))),
                              np.max(np.abs(got[:, 2] - d2) / (1 + np.abs(d2))))
                    print(f"[newton] 1-D polynomial collapse vs PolyModel: max rel err {err:.2e}")
                    assert err < 1e-8
                    first = False
                    if any(n in RBLEND for n in names):
                        for nm, (e1, e2) in check_rblend_derivatives(fit, comps, cid, kT, df).items():
                            print(f"[newton] {nm} B_T / B_TT vs FD of blend value: max rel err {e1:.2e} / {e2:.2e}")
                            assert e1 < 1e-6 and e2 < 1e-4, (nm, e1, e2)
                base = dict(protocol=proto, seed=seed, comp_id=int(c), n_fit_rows=sur["_n_fit_rows"])
                for name in names:
                    if name in ("SymGWR-T_secant", "SymGWR-T_brent"):
                        T, st = trajectory(sur["SymGWR-T"], a, ref_ev, method=name.split("_")[1])
                    elif name in RBLEND:
                        rec = Recorder(sur[name])
                        T, st = trajectory(rec, a, ref_ev)
                        lo, hi = step_intervals(rec, T, st)
                        bp = scan_breakpoints(fit, sur["_poly_along"][1], kT, sur[name], a,
                                              float(fit.split.feature_scaler.mean_[kT]),
                                              float(fit.split.feature_scaler.scale_[kT]))
                        diag.append(diag_row(base, name, bp, lo, hi, st["conv"], st["cp_bad"], st["bad_jac"]))
                    else:
                        T, st = trajectory(sur[name], a, ref_ev)
                    rows.append(run_row(base, name, st, T, Tref, a))
            print(f"[newton] {proto} seed {seed} done ({time.time() - t_start:.0f}s elapsed)", flush=True)

    runs = pd.DataFrame(rows)
    path = os.path.join(args.out, "newton_runs.csv")
    if args.only:  # merge: keep everything already in the file except the surrogates being re-run
        # keep "n/a" (protocol of the reference rows) as a string; exact float round trip
        old = pd.read_csv(path, keep_default_na=False, na_values=[""], float_precision="round_trip")
        ref_old = old[old.surrogate == "reference"].set_index("comp_id").kappa
        ref_new = runs[runs.surrogate == "reference"].set_index("comp_id").kappa
        common = ref_new.index.intersection(ref_old.index)
        assert len(common) == len(ref_new), "reference compositions missing from existing file"
        assert np.allclose(ref_old.loc[common], ref_new.loc[common], rtol=1e-9), "kappa differs from existing run"
        new_rows = runs[runs.surrogate.isin(names)]
        runs = pd.concat([old[~old.surrogate.isin(names)], new_rows], ignore_index=True)
        print(f"[newton] kappa of {len(common)} compositions identical to existing file; "
              f"merged {len(new_rows)} rows ({', '.join(names)})")
    runs.to_csv(path, index=False)
    if diag:
        dd = pd.DataFrame(diag)
        dd.to_csv(os.path.join(args.out, "newton_rblend_diag.csv"), index=False)
        ds = summarize_diag(dd)
        ds.to_csv(os.path.join(args.out, "newton_rblend_diag_summary.csv"), index=False)
        with pd.option_context("display.width", 250, "display.max_columns", 50, "display.float_format", lambda v: f"{v:.4g}"):
            print(ds.T.to_string())
    summ = summarize(runs)
    summ.to_csv(os.path.join(args.out, "newton_summary.csv"), index=False)
    print_summary(summ)
    print(f"[newton] wrote {len(runs)} rows to {args.out}/newton_runs.csv and newton_summary.csv")


if __name__ == "__main__":
    main()
