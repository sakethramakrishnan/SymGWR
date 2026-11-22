#!/usr/bin/env python
"""Camera-ready baseline comparison: SymGWR vs. 12 baselines on one split kind.

    python experiments/run_all_baselines.py --split temp_band
    python experiments/run_all_baselines.py --split random --models SymGWR,linear,k-NN --seeds 42,43

Protocol per (model, seed): get_split(kind, seed) -> tune on validation MSE ->
refit chosen config on train+val -> evaluate once on test.  SymGWR (degree 5,
k=1) is fixed and refit on train+val.  Every (model, seed) runs in a fresh
subprocess with a wall-clock timeout; nothing is cached or reused.

Outputs (``--out-dir``, default results/camera_ready):
    baselines_{split}.csv          per-seed rows, appended after every (model, seed)
    baselines_{split}_summary.csv  written once all requested runs finish
"""

from __future__ import annotations

import os

# Thread caps must be set before numpy / torch / xgboost are imported.
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "4"
os.environ.setdefault("JULIA_NUM_THREADS", "4")

import argparse
import multiprocessing as mp
import sys
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from _baseline_common import MODEL_NOTES, MODEL_ORDER, run_one  # noqa: E402
from gwr.splits import KINDS  # noqa: E402

DEFAULT_SEEDS = list(range(42, 52))
METRICS = ["mape_raw", "mae_raw", "mape_z", "rmse", "r2", "rmse_over_std", "runtime_s"]
ROW_COLS = [
    "split", "seed", "model", "status", "chosen_hparams", "n_train_used",
    "mape_raw", "mae_raw", "mape_z", "rmse", "r2", "rmse_over_std", "runtime_s",
    "train_cap", "n_train_available", "notes", "error",
]
PARTITION_NOTE = {
    "temp_band": (1, "test set fixed (1,568 rows, same for every seed); only train/val partition varies"),
    "composition_group": (5, "5 folds x 2 shuffles: test fold = seed % 5, so 5 distinct test sets, each seen twice"),
    "random": (10, "10 random 60/20/20 resamples; test sets overlap, pairs not strictly independent"),
}


def _norm(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def parse_models(arg: str | None) -> list[str]:
    if not arg:
        return list(MODEL_ORDER)
    lookup = {_norm(m): m for m in MODEL_ORDER}
    lookup.update({"knn": "k-NN", "dt": "Decision tree", "tree": "Decision tree",
                   "gaussianprocess": "GP", "kernel": "Kernel regression",
                   "xgb": "XGBoost", "poly": "Polynomial", "linearregression": "Linear"})
    wanted = []
    for tok in arg.split(","):
        key = _norm(tok)
        if key not in lookup:
            raise SystemExit(f"unknown model {tok!r}; choose from {MODEL_ORDER}")
        wanted.append(lookup[key])
    return [m for m in MODEL_ORDER if m in wanted]  # canonical order, PySR last


# ----------------------------------------------------------------------
#  Child process
# ----------------------------------------------------------------------

def _worker(model: str, kind: str, seed: int, conn) -> None:
    try:
        if model == "PySR":
            import pysr  # noqa: F401  (sets juliacall env vars; importing juliacall directly segfaults)
        from gwr.splits import get_split

        split = get_split(kind, seed, verbose=False)
        conn.send(run_one(model, split, seed))
    except BaseException as exc:  # noqa: BLE001
        conn.send({"status": "error", "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc()})
    finally:
        conn.close()


def run_with_timeout(model: str, kind: str, seed: int, timeout_s: float) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_worker, args=(model, kind, seed, child), daemon=True)
    t0 = time.perf_counter()
    proc.start()
    child.close()
    out: dict[str, Any]
    if parent.poll(timeout_s):
        try:
            out = parent.recv()
        except EOFError:
            proc.join(5)
            out = {"status": "error", "error": f"worker died, exitcode={proc.exitcode}"}
    else:
        out = {"status": "timeout", "error": f"exceeded {timeout_s:.0f}s"}
    if proc.is_alive():
        proc.kill()
    proc.join(10)
    parent.close()
    if out["status"] != "ok":
        out["runtime_s"] = time.perf_counter() - t0
    return out


# ----------------------------------------------------------------------
#  CSV
# ----------------------------------------------------------------------

def append_row(path: str, row: dict[str, Any]) -> None:
    new = not os.path.isfile(path) or os.path.getsize(path) == 0
    df = pd.DataFrame([{c: row.get(c, np.nan) for c in ROW_COLS}])
    df.to_csv(path, mode="a", header=new, index=False)  # one append per (model, seed)


def write_summary(raw_path: str, out_path: str, kind: str) -> pd.DataFrame:
    raw = pd.read_csv(raw_path)
    n_eff, note = PARTITION_NOTE[kind]
    sym = raw[(raw["model"] == "SymGWR") & (raw["status"] == "ok")].set_index("seed")["mape_raw"]
    rows = []
    for model in [m for m in MODEL_ORDER if m in set(raw["model"])]:
        sub = raw[raw["model"] == model]
        ok = sub[sub["status"] == "ok"]
        r: dict[str, Any] = {
            "split": kind, "model": model,
            "n_seeds_ok": len(ok),
            "n_timeout": int((sub["status"] == "timeout").sum()),
            "n_error": int((sub["status"] == "error").sum()),
        }
        for m in METRICS:
            v = ok[m].to_numpy(dtype=float)
            r[f"{m}_mean"] = float(np.mean(v)) if len(v) else np.nan
            r[f"{m}_std"] = float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
        r.update(n_pairs_vs_symgwr=np.nan, symgwr_wins=np.nan, symgwr_wins_of_total="",
                 p_paired_t=np.nan, p_wilcoxon=np.nan)
        if model != "SymGWR" and len(sym):
            base = ok.set_index("seed")["mape_raw"]
            seeds = sorted(set(base.index) & set(sym.index))
            if seeds:
                b, s = base.loc[seeds].to_numpy(float), sym.loc[seeds].to_numpy(float)
                r["n_pairs_vs_symgwr"] = len(seeds)
                r["symgwr_wins"] = int(np.sum(s < b))
                r["symgwr_wins_of_total"] = f"{int(np.sum(s < b))}/{len(seeds)}"
                if len(seeds) > 1:
                    r["p_paired_t"] = float(stats.ttest_rel(b, s).pvalue)
                    try:
                        r["p_wilcoxon"] = float(stats.wilcoxon(b, s).pvalue)
                    except ValueError:  # all differences zero
                        r["p_wilcoxon"] = np.nan
        notes = sub["notes"].dropna()
        r["notes"] = notes.iloc[0] if len(notes) else ""
        r["effective_independent_partitions"] = n_eff
        r["partition_note"] = note
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    return out


# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=KINDS)
    ap.add_argument("--models", default=None, help="comma list; default all")
    ap.add_argument("--seeds", default=None, help="comma list or a-b range; default 42..51")
    ap.add_argument("--out-dir", default="results/camera_ready")
    ap.add_argument("--timeout-s", type=float, default=900.0, help="per model per seed (default 15 min)")
    ap.add_argument("--gp-timeout-s", type=float, default=1500.0, help="GP per seed (default 25 min)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--overwrite", action="store_true", help="truncate an existing per-seed CSV")
    g.add_argument("--append", action="store_true", help="append to an existing per-seed CSV")
    args = ap.parse_args()

    os.chdir(ROOT)
    seeds = DEFAULT_SEEDS
    if args.seeds:
        seeds = []
        for tok in args.seeds.split(","):
            if "-" in tok:
                a, b = tok.split("-")
                seeds += list(range(int(a), int(b) + 1))
            else:
                seeds.append(int(tok))
    models = parse_models(args.models)

    os.makedirs(args.out_dir, exist_ok=True)
    raw_path = os.path.join(args.out_dir, f"baselines_{args.split}.csv")
    sum_path = os.path.join(args.out_dir, f"baselines_{args.split}_summary.csv")
    if os.path.exists(raw_path) and os.path.getsize(raw_path) > 0:
        if args.overwrite:
            os.remove(raw_path)
        elif not args.append:
            raise SystemExit(f"{raw_path} exists; pass --overwrite or --append")

    total = len(models) * len(seeds)
    print(f"[run] split={args.split} models={models} seeds={seeds} timeout={args.timeout_s:.0f}s "
          f"out={raw_path}", flush=True)
    t_all = time.perf_counter()
    done = 0
    for model in models:
        for seed in seeds:
            t0 = time.perf_counter()
            tmo = args.gp_timeout_s if model == "GP" else args.timeout_s
            res = run_with_timeout(model, args.split, seed, tmo)
            if res["status"] == "error" and "traceback" in res:
                print(res["traceback"], flush=True)
            row = {"split": args.split, "seed": seed, "model": model,
                   "notes": MODEL_NOTES.get(model, ""), **res}
            if res["status"] != "ok":
                for m in ("mape_raw", "mae_raw", "mape_z", "rmse", "r2", "rmse_over_std"):
                    row[m] = np.nan
            append_row(raw_path, row)
            done += 1
            detail = (f"mape_raw={res['mape_raw']:.4g} rmse={res['rmse']:.1f} r2={res['r2']:.4f}"
                      if res["status"] == "ok" else res.get("error", ""))
            print(f"[{done}/{total}] split={args.split} model={model} seed={seed} "
                  f"status={res['status']} {detail} time={time.perf_counter() - t0:.1f}s "
                  f"elapsed={(time.perf_counter() - t_all) / 60:.1f}min", flush=True)

    summ = write_summary(raw_path, sum_path, args.split)
    print(f"[done] split={args.split} wall={(time.perf_counter() - t_all) / 60:.1f}min "
          f"-> {raw_path}, {sum_path}", flush=True)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summ[["model", "n_seeds_ok", "n_timeout", "mape_raw_mean", "mape_raw_std",
                    "rmse_mean", "r2_mean", "symgwr_wins_of_total", "p_paired_t", "p_wilcoxon"]],
              flush=True)


if __name__ == "__main__":
    main()
