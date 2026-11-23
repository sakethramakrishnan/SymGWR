"""Query-time scaling study for SymGWR (k=1 nearest-reference Taylor surrogate).

Global degree-5 polynomial P is fit once on data/raw/718_all.csv via
gwr.splits.get_split('random', 42) (references = train+val, scaled space).
Query-time methods timed (single query, pure numpy/scipy, no SymPy):

  SymGWR-T        G_i + gradP(r_i).(q - r_i)         i = nearest reference (cKDTree, k=1)
  SymGWR-R        P(q) + G_i - P(r_i)
  Spline_scipy    nearest composition line (cKDTree on line compositions), then
                  scipy CubicSpline object call along T (table-interpolation comparator)
  Spline_lean     same spline coefficients, searchsorted + Horner by hand (removes
                  scipy PPoly call overhead so the comparison is not dominated by it)
  KDTree_NN_only  diagnostic: the cKDTree.query call alone

Reference sizes: 1e3 (random subset of the real refs), real (all train+val refs),
1e5 and 1e6 (real refs tiled with jitter, sigma = 1% of each feature's std).
The tiled sizes measure INDEX/MEMORY BEHAVIOUR ONLY, not accuracy.

Tiling keeps the per-composition line structure the spline needs: copy 0 is the real
data; copies >= 1 add jitter that is shared by all rows of a composition line for the
7 composition features and independent per row for T.

Usage (orchestrator, run unpinned; workers are launched under `taskset -c CORE`):
    .venv/bin/python experiments/run_scaling_study.py --core 10
    .venv/bin/python experiments/run_scaling_study.py --summarize     # re-print from CSVs
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import gc
import json
import pickle
import platform
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results", "camera_ready")

NS = [1000, "real", 100_000, 1_000_000]
N_TIMED = 10_000
N_WARM = 2_000
REPEATS = 3
JITTER = 0.01
DEGREE = 5
SEED = 42
COMP = slice(0, 7)  # the 7 composition columns (Nb Al Ti Mo Fe Ni Cr); T is column 7
T_COL = 7
METHODS = ["SymGWR-T", "SymGWR-R", "Spline_scipy", "Spline_lean", "KDTree_NN_only"]
LOAD_LIMIT = 2.0  # discard an attempt if (external busy cores + 1 for the benchmark) > this
START_LOAD_LIMIT = 1.5
MAX_ATTEMPTS = 6


# ----------------------------------------------------------------------
#  /proc helpers
# ----------------------------------------------------------------------
def _status_kb(key):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1])
    return float("nan")


def rss_mb():
    return _status_kb("VmRSS") / 1024.0


def peak_rss_mb():
    return _status_kb("VmHWM") / 1024.0


def cpu_times():
    d = {}
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu") and line[3].isdigit():
                p = line.split()
                v = list(map(int, p[1:]))
                d[int(p[0][3:])] = (sum(v), v[3] + v[4])  # total, idle+iowait
    return d


def busy_fraction(a, b):
    return {c: 1.0 - (b[c][1] - a[c][1]) / max(1, b[c][0] - a[c][0]) for c in a}


def sibling_of(core):
    with open(f"/sys/devices/system/cpu/cpu{core}/topology/thread_siblings_list") as f:
        txt = f.read().strip()
    cpus = []
    for part in txt.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            cpus += list(range(int(lo), int(hi) + 1))
        else:
            cpus.append(int(part))
    sib = [c for c in cpus if c != core]
    return sib[0] if sib else None


def loadavg():
    with open("/proc/loadavg") as f:
        p = f.read().split()
    return float(p[0]), float(p[1]), float(p[2])


class LoadMonitor(threading.Thread):
    """Samples per-CPU busy fraction once per second while a worker runs."""

    def __init__(self, core, sib):
        super().__init__(daemon=True)
        self.core, self.sib = core, sib
        self.stop_evt = threading.Event()
        self.ext = []  # external busy cores per 1 s sample
        self.sib_busy = []
        self.la1 = []
        self.max_roll_ext = 0.0
        self.exceeded = False

    def run(self):
        roll = deque(maxlen=10)
        prev = cpu_times()
        while not self.stop_evt.wait(1.0):
            cur = cpu_times()
            b = busy_fraction(prev, cur)
            prev = cur
            ext = sum(v for c, v in b.items() if c not in (self.core, self.sib))
            self.ext.append(ext)
            self.sib_busy.append(b.get(self.sib, 0.0) if self.sib is not None else 0.0)
            self.la1.append(loadavg()[0])
            roll.append(ext)
            if len(roll) == roll.maxlen:
                r = sum(roll) / len(roll)
                self.max_roll_ext = max(self.max_roll_ext, r)
                if r + 1.0 > LOAD_LIMIT:
                    self.exceeded = True


# ----------------------------------------------------------------------
#  Prepare (orchestrator): fit P once
# ----------------------------------------------------------------------
def prepare(work):
    from gwr.models import fit_polynomial_regression
    from gwr.splits import get_split

    sp = get_split("random", SEED, verbose=False)
    X = np.vstack([sp.X_train, sp.X_val])
    y = np.concatenate([sp.y_train, sp.y_val])
    pipe, _ = fit_polynomial_regression(X, y, X_test=None, degree=DEGREE)
    with open(os.path.join(work, "pipe.pkl"), "wb") as f:
        pickle.dump(pipe, f)
    np.savez(os.path.join(work, "data.npz"), X=X, y=y, Xtest=sp.X_test, feats=np.array(sp.feature_cols))
    return dict(
        n_real_refs=len(X), n_test=len(sp.X_test), feature_cols="|".join(sp.feature_cols),
        n_rows_after_dedupe=sp.info["n_rows_after_dedupe"], n_poly_terms=int(pipe.named_steps["poly_features"].n_output_features_),
    )


# ----------------------------------------------------------------------
#  Worker
# ----------------------------------------------------------------------
def build_refs(X0, y0, n_target, rng, sigma):
    """Return (X, y, line_id, n_real_source_rows, note)."""
    N0 = len(X0)
    _, cid = np.unique(X0[:, COMP], axis=0, return_inverse=True)
    cid = np.asarray(cid).reshape(-1)
    n_comp = int(cid.max()) + 1
    if n_target == N0:
        return X0.copy(), y0.copy(), cid.copy(), "real"
    if n_target < N0:
        idx = np.sort(rng.choice(N0, size=n_target, replace=False))
        return X0[idx].copy(), y0[idx].copy(), cid[idx].copy(), "subset"
    copies = -(-n_target // N0)
    Xs, ys, ls = [X0], [y0], [cid]
    for c in range(1, copies):
        Xc = X0.copy()
        eps = rng.standard_normal((n_comp, 7)) * sigma[COMP]
        Xc[:, COMP] += eps[cid]
        Xc[:, T_COL] += rng.standard_normal(N0) * sigma[T_COL]
        Xs.append(Xc)
        ys.append(y0)
        ls.append(c * n_comp + cid)
    X = np.concatenate(Xs)[:n_target]
    y = np.concatenate(ys)[:n_target]
    line = np.concatenate(ls)[:n_target]
    return np.ascontiguousarray(X), y, line, "tiled+jitter"


def pstat(ts_ns):
    us = ts_ns / 1e3
    return dict(
        mean_us=float(us.mean()), p50_us=float(np.percentile(us, 50)),
        p95_us=float(np.percentile(us, 95)), p99_us=float(np.percentile(us, 99)),
        min_us=float(us.min()), max_us=float(us.max()), std_us=float(us.std()),
    )


def time_method(f, qs):
    for q in qs[:N_WARM]:
        f(q)
    timed = qs[N_WARM:N_WARM + N_TIMED]
    ts = np.empty(len(timed), dtype=np.int64)
    pc = time.perf_counter_ns
    gc.collect()
    gc.disable()
    try:
        for i, q in enumerate(timed):
            t0 = pc()
            f(q)
            ts[i] = pc() - t0
    finally:
        gc.enable()
    return pstat(ts)


def worker(args):
    from scipy.interpolate import CubicSpline
    from scipy.spatial import cKDTree

    from gwr.gradients import evaluate_polynomial_pipeline_gradients

    pc = time.perf_counter
    with open(os.path.join(args.work, "pipe.pkl"), "rb") as f:
        pipe = pickle.load(f)
    d = np.load(os.path.join(args.work, "data.npz"))
    X0, y0, Xtest = d["X"], d["y"], d["Xtest"]
    poly = pipe.named_steps["poly_features"]
    lin = pipe.named_steps["linear_model"]
    powers = np.asarray(poly.powers_, dtype=np.int64)
    coef = np.asarray(lin.coef_, dtype=np.float64).ravel()
    b0 = float(np.ravel(lin.intercept_)[0])
    dim = X0.shape[1]
    gc.collect()
    rss_base = rss_mb()

    n_target = len(X0) if args.n == "real" else int(args.n)
    rng = np.random.default_rng(SEED + 1000 * args.rep + (0 if args.n == "real" else int(args.n) % 997))
    sigma = JITTER * X0.std(axis=0)  # Ni is constant -> sigma 0
    t = pc()
    X, y, line_id, kind = build_refs(X0, y0, n_target, rng, sigma)
    t_build_refs = pc() - t
    n = len(X)
    del d

    # --- timed: gradient precompute for all n references (repo explicit-gradient path)
    gc.collect()
    t = pc()
    G = evaluate_polynomial_pipeline_gradients(pipe, X)
    t_grad = pc() - t

    # --- timed: P(r_i) for all refs (needed by SymGWR-R); batched to bound memory
    t = pc()
    Pref = np.empty(n)
    for s in range(0, n, 10_000):
        Pref[s:s + 10_000] = pipe.predict(X[s:s + 10_000])
    t_pref = pc() - t

    # --- timed: index builds
    r0 = rss_mb()
    t = pc()
    tree = cKDTree(X)
    t_tree = pc() - t
    index_rss_mb = rss_mb() - r0

    # --- spline table: one cubic spline per composition line, sorted along T
    t = pc()
    order = np.lexsort((X[:, T_COL], line_id))
    lid_s = line_id[order]
    xs_all = np.ascontiguousarray(X[order, T_COL])
    ys_all = y[order]
    bounds = np.flatnonzero(np.diff(lid_s)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [n]])
    keep = (ends - starts) >= 2
    starts, ends = starts[keep], ends[keep]
    Xsorted = X[order]
    splines, off, cnt, comps_l = [], [], [], []
    c_all = np.zeros((4, n))
    xs_tab = np.empty(n)
    pos = 0
    n_dup_dropped = 0
    for s, e in zip(starts, ends):
        xx, yy = xs_all[s:e], ys_all[s:e]
        good = np.concatenate([[True], np.diff(xx) > 0])  # CubicSpline needs strictly increasing x
        if not good.all():
            n_dup_dropped += int((~good).sum())
            xx, yy = xx[good], yy[good]
        m = len(xx)
        if m < 2:
            continue
        cs = CubicSpline(xx, yy)
        splines.append(cs)
        off.append(pos)
        cnt.append(m)
        comps_l.append(Xsorted[s, COMP])
        xs_tab[pos:pos + m] = xx
        c_all[:, pos:pos + m - 1] = cs.c[:, :, 0] if cs.c.ndim == 3 else cs.c
        pos += m
    comps = np.ascontiguousarray(np.array(comps_l))
    del Xsorted
    r0 = rss_mb()
    ctree = cKDTree(np.ascontiguousarray(comps))
    ctree_rss_mb = rss_mb() - r0
    t_spline = pc() - t
    xs_tab = xs_tab[:pos]
    c0, c1, c2, c3 = (np.ascontiguousarray(c_all[i, :pos]) for i in range(4))
    del c_all, order, lid_s, xs_all, ys_all

    # --- storage (explicit arrays)
    mb = 1.0 / 2**20
    ref_mb = (X.nbytes + y.nbytes) * mb
    grad_mb = G.nbytes * mb
    pref_mb = Pref.nbytes * mb
    index_indices_mb = tree.indices.nbytes * mb
    spline_mb = (xs_tab.nbytes + 4 * c0.nbytes + comps.nbytes + sum(a.nbytes for a in (np.asarray(off), np.asarray(cnt)))) * mb

    # --- single-query evaluators (no SymPy)
    ar = np.arange(DEGREE + 1)
    flat_idx = np.arange(dim)[None, :] * (DEGREE + 1) + powers
    qtree = tree.query
    cq = ctree.query
    mul_reduce = np.multiply.reduce

    def Pval(q):
        pw = q[:, None] ** ar
        return mul_reduce(pw.ravel().take(flat_idx), axis=1) @ coef + b0

    def f_T(q):
        _, i = qtree(q)
        return y[i] + G[i] @ (q - X[i])

    def f_R(q):
        _, i = qtree(q)
        return Pval(q) + (y[i] - Pref[i])

    def f_sp(q):
        _, j = cq(q[:7])
        return splines[j](q[7])

    def f_lean(q):
        _, j = cq(q[:7])
        tt = float(q[7])
        lo = off[j]
        m = cnt[j]
        k = int(np.searchsorted(xs_tab[lo:lo + m], tt, side="right")) - 1
        if k < 0:
            k = 0
        elif k > m - 2:
            k = m - 2
        i = lo + k
        dx = tt - xs_tab[i]
        return ((c0[i] * dx + c1[i]) * dx + c2[i]) * dx + c3[i]

    def f_nn(q):
        return qtree(q)

    # --- queries: held-out test rows, resampled + 1% jitter (same set for every method)
    qrng = np.random.default_rng(SEED + 7)
    Q = Xtest[qrng.integers(0, len(Xtest), N_WARM + N_TIMED)] + qrng.standard_normal((N_WARM + N_TIMED, dim)) * sigma
    Q = np.ascontiguousarray(Q)
    qs = list(Q)

    # --- correctness checks (untimed): single-query == vectorised reference
    nchk = 300
    Qc = Q[:nchk]
    _, nn = tree.query(Qc)
    ref_T = y[nn] + np.einsum("ij,ij->i", G[nn], Qc - X[nn])
    ref_R = pipe.predict(Qc) + (y[nn] - Pref[nn])
    ref_sp = np.array([splines[j](q[7]) for q, j in zip(Qc, ctree.query(Qc[:, :7])[1])])
    errs = dict(
        T=float(np.max(np.abs([f_T(q) for q in Qc] - ref_T))),
        R=float(np.max(np.abs([f_R(q) for q in Qc] - ref_R))),
        spline_lean_vs_scipy=float(np.max(np.abs([f_lean(q) for q in Qc] - ref_sp))),
        spline_scipy=float(np.max(np.abs([f_sp(q) for q in Qc] - ref_sp))),
    )
    scale = max(1.0, float(np.abs(ref_T).max()))
    if max(errs.values()) > 1e-6 * scale:
        raise RuntimeError(f"correctness check failed: {errs}")

    # --- timer overhead
    ts = np.empty(N_TIMED, dtype=np.int64)
    pcn = time.perf_counter_ns
    for i in range(N_TIMED):
        t0 = pcn()
        ts[i] = pcn() - t0
    timer_overhead_us = float(ts.mean() / 1e3)

    timings = {}
    for name, fn in [("SymGWR-T", f_T), ("SymGWR-R", f_R), ("Spline_scipy", f_sp),
                     ("Spline_lean", f_lean), ("KDTree_NN_only", f_nn)]:
        timings[name] = time_method(fn, qs)

    out = dict(
        n=n, n_label=str(args.n), rep=args.rep, ref_kind=kind,
        affinity=sorted(os.sched_getaffinity(0)),
        n_lines_spline=len(splines), n_rows_spline=int(pos), n_dup_dropped=n_dup_dropped,
        t_build_refs_s=t_build_refs, t_grad_precompute_s=t_grad, t_pref_precompute_s=t_pref,
        t_kdtree_build_s=t_tree, t_spline_build_s=t_spline,
        grad_precompute_us_per_ref=t_grad / n * 1e6,
        rss_baseline_mb=rss_base, peak_rss_mb=peak_rss_mb(), rss_end_mb=rss_mb(),
        ref_mb=ref_mb, grad_mb=grad_mb, pref_mb=pref_mb,
        index_rss_delta_mb=index_rss_mb, index_indices_mb=index_indices_mb,
        spline_mb=spline_mb, spline_line_tree_rss_delta_mb=ctree_rss_mb,
        symgwr_T_storage_mb=ref_mb + grad_mb + index_rss_mb,
        symgwr_R_storage_mb=ref_mb + pref_mb + index_rss_mb,
        timer_overhead_us=timer_overhead_us, check_max_abs_err=max(errs.values()),
        timings=timings,
    )
    with open(args.out, "w") as f:
        json.dump(out, f)


# ----------------------------------------------------------------------
#  Orchestrator
# ----------------------------------------------------------------------
def cpu_model():
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def run_attempt(core, sib, work, n, rep, attempt):
    out = os.path.join(work, f"res_n{n}_r{rep}_a{attempt}.json")
    cmd = ["taskset", "-c", str(core), sys.executable, os.path.abspath(__file__), "--worker",
           "--work", work, "--n", str(n), "--rep", str(rep), "--out", out]
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    mon = LoadMonitor(core, sib)
    la_start = loadavg()
    mon.start()
    proc = subprocess.Popen(cmd, env=env)
    killed = False
    while proc.poll() is None:
        time.sleep(1.0)
        if mon.exceeded and not killed:
            proc.terminate()
            killed = True
    mon.stop_evt.set()
    mon.join()
    rc = proc.returncode
    rec = dict(
        n_label=str(n), rep=rep, attempt=attempt, loadavg1_start=la_start[0],
        loadavg1_max_during=max(mon.la1) if mon.la1 else float("nan"),
        external_busy_cores_max_roll10s=mon.max_roll_ext,
        external_busy_cores_mean=float(np.mean(mon.ext)) if mon.ext else float("nan"),
        sibling_busy_mean=float(np.mean(mon.sib_busy)) if mon.sib_busy else float("nan"),
        sibling_busy_max=float(np.max(mon.sib_busy)) if mon.sib_busy else float("nan"),
        wall_s=len(mon.ext),
    )
    if killed or mon.exceeded:
        rec["status"] = "discarded_load"
    elif rc != 0 or not os.path.exists(out):
        rec["status"] = f"failed_rc{rc}"
    else:
        rec["status"] = "ok"
    return rec, (out if rec["status"] == "ok" else None)


def wait_quiet(core, sib, limit=1.0, tries=60):
    for _ in range(tries):
        a = cpu_times()
        time.sleep(3.0)
        b = busy_fraction(a, cpu_times())
        ext = sum(v for c, v in b.items() if c not in (core, sib))
        if ext <= limit:
            return
        time.sleep(10.0)
    print("WARNING: machine never went quiet; proceeding", flush=True)


def assemble(env_rows, attempts, results):
    import pandas as pd

    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(env_rows, columns=["key", "value"]).to_csv(os.path.join(OUT, "scaling_env.csv"), index=False)
    pd.DataFrame(attempts).to_csv(os.path.join(OUT, "scaling_attempts.csv"), index=False)
    runs, tim = [], []
    for r in results:
        flat = {k: v for k, v in r.items() if k not in ("timings", "affinity")}
        flat["affinity"] = ",".join(map(str, r["affinity"]))
        runs.append(flat)
        for m, s in r["timings"].items():
            tim.append(dict(n=r["n"], n_label=r["n_label"], rep=r["rep"], method=m, n_queries=N_TIMED, **s))
    if not runs:
        return
    pd.DataFrame(runs).sort_values(["n", "rep"]).to_csv(os.path.join(OUT, "scaling_runs.csv"), index=False)
    pd.DataFrame(tim).sort_values(["n", "method", "rep"]).to_csv(os.path.join(OUT, "scaling_query_timing.csv"), index=False)


def main(args):
    la0 = loadavg()
    print(f"load average at start: {la0}", flush=True)
    if la0[0] >= START_LOAD_LIMIT:
        sys.exit(f"1-min load {la0[0]} >= {START_LOAD_LIMIT}; not starting.")
    core = args.core
    sib = sibling_of(core)
    a = cpu_times()
    time.sleep(10.0)
    b = busy_fraction(a, cpu_times())
    print(f"core {core} busy={b[core]:.2f}  sibling {sib} busy={b.get(sib, 0):.2f}", flush=True)
    if sib is not None and b[sib] > 0.10:
        sys.exit(f"SMT sibling cpu{sib} is {b[sib]:.0%} busy; pick another core.")
    others = sorted(set(cpu_times()) - {core, sib})
    os.sched_setaffinity(0, others)  # keep orchestrator/monitor off the benchmark core + sibling

    work = args.work or tempfile.mkdtemp(prefix="scaling_")
    os.makedirs(work, exist_ok=True)
    info = prepare(work)
    import scipy
    import sklearn

    env_rows = [
        ("cpu_model", cpu_model()), ("python", platform.python_version()), ("numpy", np.__version__),
        ("scipy", scipy.__version__), ("sklearn", sklearn.__version__), ("platform", platform.platform()),
        ("pinned_core", core), ("smt_sibling", sib), ("OMP_NUM_THREADS", os.environ["OMP_NUM_THREADS"]),
        ("loadavg_start_1_5_15", "|".join(map(str, la0))), ("core_busy_at_start", f"{b[core]:.3f}"),
        ("sibling_busy_at_start", f"{b.get(sib, 0):.3f}"),
        ("git_commit", subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()),
        ("start_time", time.strftime("%Y-%m-%d %H:%M:%S")), ("n_timed_queries", N_TIMED), ("n_warmup_queries", N_WARM),
        ("repeats", REPEATS), ("jitter_frac_of_std", JITTER), ("poly_degree", DEGREE), ("split", f"random seed {SEED}"),
        ("kdtree_features", "all 8 numeric feature cols in scaled space: " + info["feature_cols"]),
        ("ni_in_tree", "yes (column present) but constant: scaled value 0 for every ref/query, so it adds 0 to every distance"),
        ("tree_effective_dims", 7), ("load_discard_rule", f"discard attempt if 10s-rolling external busy cores + 1 > {LOAD_LIMIT}"),
        ("gc_during_timing", "disabled"), ("gradient_path", "gwr.gradients.evaluate_polynomial_pipeline_gradients (explicit, no SymPy)"),
    ] + [(k, v) for k, v in info.items()]
    attempts, results = [], []
    assemble(env_rows, attempts, results)

    for rep in range(1, REPEATS + 1):
        for n in NS:
            for attempt in range(1, MAX_ATTEMPTS + 1):
                wait_quiet(core, sib)
                print(f"[rep {rep}] n={n} attempt {attempt} ...", flush=True)
                rec, path = run_attempt(core, sib, work, n, rep, attempt)
                attempts.append(rec)
                print(f"   -> {rec['status']}  ext_roll_max={rec['external_busy_cores_max_roll10s']:.2f} "
                      f"sib_busy_mean={rec['sibling_busy_mean']:.2f} loadavg_max={rec['loadavg1_max_during']:.2f}", flush=True)
                if path:
                    with open(path) as f:
                        results.append(json.load(f))
                    break
            else:
                sys.exit(f"n={n} rep={rep}: {MAX_ATTEMPTS} attempts failed; see scaling_attempts.csv")
            assemble(env_rows, attempts, results)
    summarize()


# ----------------------------------------------------------------------
#  Summary + slopes (everything read back from the CSVs)
# ----------------------------------------------------------------------
def summarize():
    import pandas as pd
    from scipy import stats

    tim = pd.read_csv(os.path.join(OUT, "scaling_query_timing.csv"))
    runs = pd.read_csv(os.path.join(OUT, "scaling_runs.csv"))
    env = pd.read_csv(os.path.join(OUT, "scaling_env.csv")).set_index("key")["value"]

    # slopes: OLS of log10(stat) on log10(n); one point per (n, rep)
    rows = []
    for m, g in tim.groupby("method"):
        for stat in ("mean_us", "p50_us", "p99_us"):
            x, yv = np.log10(g["n"].to_numpy(float)), np.log10(g[stat].to_numpy(float))
            r = stats.linregress(x, yv)
            half = stats.t.ppf(0.975, len(x) - 2) * r.stderr
            lo, hi = r.slope - half, r.slope + half
            by_n = g.groupby("n")[stat].mean()
            ratio = by_n.iloc[-1] / by_n.iloc[0]
            if lo <= 0.0 <= hi:
                verdict = "flat within CI (slope CI contains 0)"
            elif lo > 0:
                verdict = "NOT flat: grows with n"
            else:
                verdict = "NOT flat: shrinks with n"
            rows.append(dict(method=m, stat=stat, slope=r.slope, ci95_lo=lo, ci95_hi=hi, r2=r.rvalue ** 2,
                             n_points=len(x), ratio_nmax_over_nmin=ratio, verdict=verdict))
    slopes = pd.DataFrame(rows)
    slopes.to_csv(os.path.join(OUT, "scaling_slopes.csv"), index=False)

    pd.set_option("display.width", 250, "display.max_columns", 50, "display.float_format", lambda v: f"{v:,.3f}")
    print("\n=== environment ===")
    for k in ("cpu_model", "python", "numpy", "scipy", "pinned_core", "smt_sibling", "OMP_NUM_THREADS", "loadavg_start_1_5_15",
              "kdtree_features", "ni_in_tree"):
        print(f"{k:24s} {env.get(k)}")

    print("\n=== query time, microseconds (mean over repeats; [min-max of the per-repeat mean]) ===")
    agg = tim.groupby(["n", "method"]).agg(
        mean=("mean_us", "mean"), mean_min=("mean_us", "min"), mean_max=("mean_us", "max"),
        p50=("p50_us", "mean"), p95=("p95_us", "mean"), p99=("p99_us", "mean"), reps=("rep", "nunique")).reset_index()
    agg["mean [min-max]"] = agg.apply(lambda r: f"{r['mean']:.2f} [{r['mean_min']:.2f}-{r['mean_max']:.2f}]", axis=1)
    agg["spread_%"] = 100 * (agg["mean_max"] - agg["mean_min"]) / agg["mean"]
    print(agg[["n", "method", "mean [min-max]", "spread_%", "p50", "p95", "p99", "reps"]].to_string(index=False))

    print("\n=== precompute (s) and memory (MB), mean over repeats ===")
    r = runs.groupby("n").agg(
        grad_s=("t_grad_precompute_s", "mean"), grad_s_min=("t_grad_precompute_s", "min"), grad_s_max=("t_grad_precompute_s", "max"),
        Pref_s=("t_pref_precompute_s", "mean"), tree_s=("t_kdtree_build_s", "mean"), spline_s=("t_spline_build_s", "mean"),
        rss_base=("rss_baseline_mb", "mean"), peak_rss=("peak_rss_mb", "mean"), ref=("ref_mb", "mean"), grad=("grad_mb", "mean"),
        pref=("pref_mb", "mean"), index=("index_rss_delta_mb", "mean"), T_total=("symgwr_T_storage_mb", "mean"),
        R_total=("symgwr_R_storage_mb", "mean"), spline=("spline_mb", "mean"), lines=("n_lines_spline", "mean")).reset_index()
    print(r.to_string(index=False))

    print("\n=== log-log slope of query time vs n (OLS over all (n, repeat) points, 95% CI) ===")
    print(slopes[["method", "stat", "slope", "ci95_lo", "ci95_hi", "r2", "ratio_nmax_over_nmin", "verdict"]].to_string(index=False))

    att = pd.read_csv(os.path.join(OUT, "scaling_attempts.csv"))
    print("\n=== attempts ===")
    print(att["status"].value_counts().to_string())
    print(f"max sibling busy over accepted attempts: {att[att.status == 'ok']['sibling_busy_max'].max():.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--core", type=int, default=10)
    ap.add_argument("--work", default=None)
    ap.add_argument("--n", default=None)
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.worker:
        worker(a)
    elif a.summarize:
        summarize()
    else:
        main(a)
