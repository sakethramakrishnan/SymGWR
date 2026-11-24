#!/usr/bin/env python
"""Standalone pycalphad reference timing: equilibrium() AND calculate().

Two separate measurements (never averaged / conflated):

  equilibrium()
    Multi-phase Gibbs-energy minimization / phase stability. Expensive.
    Raw timings: results/calphad_reference_timing_pycalphad.csv (legacy path,
    left intact) plus a config-keyed cache CSV under results/.

  calculate()
    Direct single-phase Gibbs energy G(composition, T) evaluation -- the
    operation comparable to what the SymGWR surrogate replaces. Restricted to
    one representative matrix phase (FCC_A1). Raw timings written to a
    separate CSV; never overwrites the equilibrium CSV.
    Default path calls pycalphad.calculate(db, comps, phase, ...) each query
    WITHOUT passing phase_records (Model + PhaseRecordFactory rebuilt per call;
    only build_functions LLVM callables are LRU-cached globally).

  calculate_precompiled
    Same G(X,T) measurement, but Model + PhaseRecordFactory built ONCE before
    warm-up and passed into every calculate() via phase_records=... / model=...
    Separate config-keyed CSV; does not overwrite calculate() results.

TDB: MatCalc mc_fe_v2.059 redistributed in pycalphad examples/databases/
  License: ODbL 1.0 / DbCL 1.0

Caching
  - Parsed Database pickle under experiments/calphad_ref_data/model_cache/,
    keyed by TDB path + content SHA-256.
  - Per-query timing CSVs under results/, keyed by
    (TDB name, components, phases, n_queries, function).
  - --force-refresh bypasses timing caches (Database cache still used
    unless --force-db-refresh).

Requires Python 3.12 + pycalphad 0.11.x (.venv_pycalphad).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

from pycalphad import (
    Database,
    Model,
    __version__ as PYCALPHAD_VERSION,
    calculate,
    equilibrium,
    variables as v,
)
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad.core.utils import instantiate_models

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TDB = REPO_ROOT / "experiments" / "calphad_ref_data" / "mc_fe_v2.059.pycalphad.tdb"
TDB_URL = (
    "https://raw.githubusercontent.com/pycalphad/pycalphad/develop/"
    "examples/databases/mc_fe_v2.059.pycalphad.tdb"
)
MODEL_CACHE_DIR = REPO_ROOT / "experiments" / "calphad_ref_data" / "model_cache"
TIMING_CACHE_DIR = REPO_ROOT / "results"
LEGACY_EQUILIBRIUM_CSV = (
    REPO_ROOT / "results" / "calphad_reference_timing_pycalphad.csv"
)

COMPONENTS = ["FE", "CR", "NI", "C", "MN", "MO", "SI", "VA"]

PHASE_CANDIDATES = [
    "LIQUID",
    "FCC_A1",
    "BCC_A2",
    "HCP_A3",
    "CEMENTITE",
    "SIGMA",
    "M23C6",
    "M7C3",
]

# Austenitic Fe-Cr-Ni matrix phase: majority phase for stainless-like
# compositions in QUERY_POINTS (elevated Ni/Cr). Single-phase G(X,T)
# evaluation -- apples-to-apples with SymGWR surrogate target.
CALCULATE_PHASE = "FCC_A1"
CALCULATE_PHASE_REASON = (
    "FCC_A1 (austenite) is the substitutional matrix / majority phase for "
    "the Fe-Cr-Ni-rich compositions in this query set; single-phase G(X,T) "
    "on FCC_A1 is the operation SymGWR approximates, not multi-phase equilibrium."
)

QUERY_POINTS = [
    {"T": 900.0, "CR": 0.12, "NI": 0.08, "C": 0.005, "MN": 0.01, "MO": 0.005, "SI": 0.005},
    {"T": 1000.0, "CR": 0.18, "NI": 0.08, "C": 0.01, "MN": 0.01, "MO": 0.01, "SI": 0.005},
    {"T": 1100.0, "CR": 0.18, "NI": 0.10, "C": 0.008, "MN": 0.015, "MO": 0.008, "SI": 0.01},
    {"T": 1200.0, "CR": 0.20, "NI": 0.12, "C": 0.005, "MN": 0.02, "MO": 0.01, "SI": 0.01},
    {"T": 1050.0, "CR": 0.10, "NI": 0.20, "C": 0.002, "MN": 0.005, "MO": 0.002, "SI": 0.005},
    {"T": 950.0, "CR": 0.15, "NI": 0.05, "C": 0.015, "MN": 0.02, "MO": 0.015, "SI": 0.008},
    {"T": 1300.0, "CR": 0.18, "NI": 0.08, "C": 0.01, "MN": 0.01, "MO": 0.01, "SI": 0.005},
    {"T": 800.0, "CR": 0.12, "NI": 0.08, "C": 0.01, "MN": 0.01, "MO": 0.005, "SI": 0.005},
]

CSV_HEADER = (
    "query_index,point_index,T_K,X_CR,X_NI,X_C,X_MN,X_MO,X_SI,"
    "latency_ms,GM_J_per_mol\n"
)


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _ensure_tdb(path: Path) -> Path:
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading TDB to {path} ...")
    print(f"  URL: {TDB_URL}")
    urllib.request.urlretrieve(TDB_URL, path)
    return path


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_token(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def load_database_cached(
    tdb_path: Path, *, force_refresh: bool = False
) -> tuple[Database, dict]:
    """Load Database from pickle cache if TDB content hash matches."""
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    content_hash = _file_sha256(tdb_path)
    key = f"{_safe_token(tdb_path.name)}__{content_hash[:16]}"
    pkl_path = MODEL_CACHE_DIR / f"database__{key}.pkl"
    meta_path = MODEL_CACHE_DIR / f"database__{key}.json"

    meta = {
        "tdb_path": str(tdb_path.resolve()),
        "tdb_name": tdb_path.name,
        "content_sha256": content_hash,
        "pickle_path": str(pkl_path.resolve()),
        "meta_path": str(meta_path.resolve()),
    }

    if not force_refresh and pkl_path.is_file():
        t0 = time.perf_counter()
        with pkl_path.open("rb") as fh:
            dbf = pickle.load(fh)
        load_s = time.perf_counter() - t0
        meta.update(
            {
                "cache": "HIT",
                "load_s": load_s,
                "source": str(pkl_path),
            }
        )
        return dbf, meta

    t0 = time.perf_counter()
    dbf = Database(str(tdb_path))
    parse_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    with pkl_path.open("wb") as fh:
        pickle.dump(dbf, fh, protocol=pickle.HIGHEST_PROTOCOL)
    dump_s = time.perf_counter() - t0
    meta.update(
        {
            "cache": "MISS",
            "load_s": parse_s,
            "parse_s": parse_s,
            "pickle_write_s": dump_s,
            "source": str(tdb_path),
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return dbf, meta


def _config_id(
    *,
    tdb_name: str,
    components: list[str],
    phases: list[str],
    n_queries: int,
    function: str,
    calculate_phase: str | None = None,
) -> str:
    payload = {
        "tdb_name": tdb_name,
        "components": list(components),
        "phases": list(phases),
        "n_queries": int(n_queries),
        "function": function,
        "n_query_points": len(QUERY_POINTS),
        "query_points": QUERY_POINTS,
        "calculate_phase": calculate_phase,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
    phase_tok = calculate_phase if calculate_phase else f"{len(phases)}ph"
    return (
        f"{function}__{_safe_token(tdb_name)}__"
        f"{len(components)}comp__{_safe_token(phase_tok)}__"
        f"nq{n_queries}__{digest}"
    )


def timing_cache_paths(config_id: str) -> tuple[Path, Path]:
    csv_path = TIMING_CACHE_DIR / f"calphad_reference_timing_pycalphad__{config_id}.csv"
    meta_path = TIMING_CACHE_DIR / f"calphad_reference_timing_pycalphad__{config_id}.json"
    return csv_path, meta_path


def _conds(point: dict[str, float]) -> dict:
    return {
        v.P: 101325,
        v.T: point["T"],
        v.X("CR"): point["CR"],
        v.X("NI"): point["NI"],
        v.X("C"): point["C"],
        v.X("MN"): point["MN"],
        v.X("MO"): point["MO"],
        v.X("SI"): point["SI"],
    }


def _stats_ms(samples_ms: np.ndarray) -> dict[str, float]:
    return {
        "mean_ms": float(np.mean(samples_ms)),
        "p50_ms": float(np.percentile(samples_ms, 50)),
        "p95_ms": float(np.percentile(samples_ms, 95)),
        "p99_ms": float(np.percentile(samples_ms, 99)),
        "min_ms": float(np.min(samples_ms)),
        "max_ms": float(np.max(samples_ms)),
    }


def _print_stats(stats: dict[str, float], *, n: int, total_wall_s: float, load_s: float) -> None:
    print("--- summary ---")
    print(f"n_timed_queries:     {n}")
    print(f"timed_total_wall_s:  {total_wall_s:.6f}")
    print(f"database_load_s:     {load_s:.6f}")
    print(f"mean_ms:             {stats['mean_ms']:.6f}")
    print(f"p50_ms:              {stats['p50_ms']:.6f}")
    print(f"p95_ms:              {stats['p95_ms']:.6f}")
    print(f"p99_ms:              {stats['p99_ms']:.6f}")
    print(f"min_ms:              {stats['min_ms']:.6f}")
    print(f"max_ms:              {stats['max_ms']:.6f}")
    mono = stats["p50_ms"] <= stats["p95_ms"] <= stats["p99_ms"]
    print(f"quantiles_monotonic: {mono}")


def _write_timing_csv(
    path: Path,
    *,
    point_idx: np.ndarray,
    temps: np.ndarray,
    samples_ms: np.ndarray,
    gm_vals: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(CSV_HEADER)
        for i in range(len(samples_ms)):
            pt = QUERY_POINTS[int(point_idx[i])]
            fh.write(
                f"{i},{int(point_idx[i])},{temps[i]:.6f},"
                f"{pt['CR']:.8f},{pt['NI']:.8f},{pt['C']:.8f},"
                f"{pt['MN']:.8f},{pt['MO']:.8f},{pt['SI']:.8f},"
                f"{samples_ms[i]:.8f},{gm_vals[i]:.12g}\n"
            )


def _read_timing_csv(path: Path) -> dict[str, np.ndarray]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
        for line in fh:
            if not line.strip():
                continue
            rows.append(line.strip().split(","))
    if not rows:
        raise ValueError(f"empty timing CSV: {path}")
    col = {name: i for i, name in enumerate(header)}
    n = len(rows)
    out = {
        "point_idx": np.zeros(n, dtype=np.int64),
        "temps": np.zeros(n, dtype=np.float64),
        "samples_ms": np.zeros(n, dtype=np.float64),
        "gm_vals": np.zeros(n, dtype=np.float64),
    }
    for i, row in enumerate(rows):
        out["point_idx"][i] = int(row[col["point_index"]])
        out["temps"][i] = float(row[col["T_K"]])
        out["samples_ms"][i] = float(row[col["latency_ms"]])
        out["gm_vals"][i] = float(row[col["GM_J_per_mol"]])
    return out


def _save_timing_meta(meta_path: Path, meta: dict) -> None:
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def composition_to_fcc_bcc_site_fractions(
    point: dict[str, float], site_fractions
) -> np.ndarray:
    """Mole-fraction -> site fractions for (M):(C,VA) FCC_A1 / BCC_A2."""
    x = {k: float(point[k]) for k in ["CR", "NI", "C", "MN", "MO", "SI"]}
    x["FE"] = 1.0 - sum(x.values())
    if x["FE"] <= 0:
        raise ValueError(f"non-positive Fe fraction for point {point}")
    denom = 1.0 - x["C"]
    y_c = x["C"] / denom
    y_va = 1.0 - y_c
    y_metal = {m: x[m] / denom for m in ["CR", "FE", "MN", "MO", "NI", "SI"]}
    vals = []
    for sf in site_fractions:
        sp = sf.species.name
        if sp in y_metal:
            vals.append(y_metal[sp])
        elif sp == "C":
            vals.append(y_c)
        elif sp == "VA":
            vals.append(y_va)
        else:
            raise KeyError(f"unexpected site fraction species {sp}")
    return np.asarray(vals, dtype=np.float64)


def _try_load_timing_cache(
    *,
    config_id: str,
    n_queries: int,
    force_refresh: bool,
    legacy_csv: Path | None = None,
) -> tuple[dict[str, np.ndarray] | None, dict]:
    csv_path, meta_path = timing_cache_paths(config_id)
    info = {
        "config_id": config_id,
        "csv_path": str(csv_path.resolve()),
        "meta_path": str(meta_path.resolve()),
        "cache": "MISS",
    }
    if force_refresh:
        info["cache"] = "BYPASS (--force-refresh)"
        return None, info

    if csv_path.is_file():
        data = _read_timing_csv(csv_path)
        if len(data["samples_ms"]) == n_queries:
            info["cache"] = "HIT"
            info["source"] = str(csv_path)
            return data, info
        info["note"] = f"keyed cache row mismatch ({len(data['samples_ms'])} != {n_queries})"

    # Adopt legacy equilibrium CSV only when it matches n_queries exactly.
    if legacy_csv is not None and legacy_csv.is_file():
        data = _read_timing_csv(legacy_csv)
        if len(data["samples_ms"]) == n_queries:
            info["cache"] = "HIT (legacy CSV adopted into keyed cache)"
            info["source"] = str(legacy_csv.resolve())
            info["legacy_csv"] = str(legacy_csv.resolve())
            # Persist keyed copy without modifying the legacy file.
            _write_timing_csv(
                csv_path,
                point_idx=data["point_idx"],
                temps=data["temps"],
                samples_ms=data["samples_ms"],
                gm_vals=data["gm_vals"],
            )
            return data, info

    return None, info


def run_equilibrium_timing(
    dbf: Database,
    phases: list[str],
    *,
    n_queries: int,
    db_load_s: float,
    tdb_name: str,
    force_refresh: bool,
) -> dict:
    config_id = _config_id(
        tdb_name=tdb_name,
        components=COMPONENTS,
        phases=phases,
        n_queries=n_queries,
        function="equilibrium",
    )
    csv_path, meta_path = timing_cache_paths(config_id)

    print("=" * 72)
    print("pycalphad CALPHAD equilibrium() reference timing")
    print("=" * 72)
    print(f"function:        equilibrium()")
    print(f"config_id:       {config_id}")
    print(f"Components:      {COMPONENTS}")
    print(f"phases_used:     {phases}")
    print(f"n_query_points:  {len(QUERY_POINTS)}")
    print(f"n_timed_queries: {n_queries}")
    print(f"timing_csv:      {csv_path}")
    print(f"legacy_csv:      {LEGACY_EQUILIBRIUM_CSV} (never overwritten)")
    print()

    cached, cache_info = _try_load_timing_cache(
        config_id=config_id,
        n_queries=n_queries,
        force_refresh=force_refresh,
        legacy_csv=LEGACY_EQUILIBRIUM_CSV,
    )
    print(f"timing_cache:    {cache_info['cache']}")
    if "source" in cache_info:
        print(f"timing_source:   {cache_info['source']}")
    print()

    if cached is not None:
        samples_ms = cached["samples_ms"]
        stats = _stats_ms(samples_ms)
        # Wall clock of original run unknown when loaded from cache.
        total_wall_s = float(np.sum(samples_ms) / 1000.0)
        _print_stats(stats, n=n_queries, total_wall_s=total_wall_s, load_s=db_load_s)
        print(f"csv_written:         {csv_path} (cache; legacy untouched)")
        print(f"csv_rows:            {n_queries}")
        print(f"legacy_csv_intact:   {LEGACY_EQUILIBRIUM_CSV.is_file()}")
        meta = {
            **cache_info,
            "function": "equilibrium",
            "stats_ms": stats,
            "timed_total_wall_s_approx_from_sum": total_wall_s,
            "database_load_s": db_load_s,
            "n_queries": n_queries,
            "phases": phases,
            "components": COMPONENTS,
        }
        _save_timing_meta(meta_path, meta)
        return {"stats": stats, "cache_info": cache_info, "csv_path": csv_path, "meta": meta}

    print("--- warm-up (discarded from timing) ---")
    for i, point in enumerate(QUERY_POINTS):
        t0 = time.perf_counter()
        eq = equilibrium(dbf, COMPONENTS, phases, _conds(point))
        dt = time.perf_counter() - t0
        gm = float(np.asarray(eq.GM.values).squeeze())
        print(
            f"  warmup[{i}] T={point['T']:.0f}K "
            f"X(CR,NI,C,MN,MO,SI)="
            f"({point['CR']},{point['NI']},{point['C']},"
            f"{point['MN']},{point['MO']},{point['SI']}) "
            f"wall_s={dt:.6f} GM={gm:.6g}"
        )
    print()

    print("--- timed queries ---")
    samples_ms = np.zeros(n_queries, dtype=np.float64)
    point_idx = np.zeros(n_queries, dtype=np.int64)
    temps = np.zeros(n_queries, dtype=np.float64)
    gm_vals = np.zeros(n_queries, dtype=np.float64)
    wall0 = time.perf_counter()
    for i in range(n_queries):
        idx = i % len(QUERY_POINTS)
        point = QUERY_POINTS[idx]
        t0 = time.perf_counter_ns()
        eq = equilibrium(dbf, COMPONENTS, phases, _conds(point))
        dt_ms = (time.perf_counter_ns() - t0) / 1.0e6
        samples_ms[i] = dt_ms
        point_idx[i] = idx
        temps[i] = point["T"]
        gm_vals[i] = float(np.asarray(eq.GM.values).squeeze())
        if (i + 1) % max(1, n_queries // 10) == 0 or i == 0 or i + 1 == n_queries:
            print(
                f"  query {i + 1}/{n_queries} point={idx} T={point['T']:.0f}K "
                f"dt_ms={dt_ms:.3f} GM={gm_vals[i]:.6g}"
            )
    total_wall_s = time.perf_counter() - wall0
    print()

    stats = _stats_ms(samples_ms)
    _print_stats(stats, n=n_queries, total_wall_s=total_wall_s, load_s=db_load_s)
    _write_timing_csv(
        csv_path,
        point_idx=point_idx,
        temps=temps,
        samples_ms=samples_ms,
        gm_vals=gm_vals,
    )
    print(f"csv_written:         {csv_path}")
    print(f"csv_rows:            {n_queries}")
    print(f"legacy_csv_intact:   {LEGACY_EQUILIBRIUM_CSV.is_file()} (not overwritten)")

    meta = {
        **cache_info,
        "cache": "MISS (computed)",
        "function": "equilibrium",
        "stats_ms": stats,
        "timed_total_wall_s": total_wall_s,
        "database_load_s": db_load_s,
        "n_queries": n_queries,
        "phases": phases,
        "components": COMPONENTS,
    }
    _save_timing_meta(meta_path, meta)
    return {"stats": stats, "cache_info": cache_info, "csv_path": csv_path, "meta": meta}


def run_calculate_timing(
    dbf: Database,
    *,
    n_queries: int,
    db_load_s: float,
    tdb_name: str,
    force_refresh: bool,
) -> dict:
    phase = CALCULATE_PHASE
    config_id = _config_id(
        tdb_name=tdb_name,
        components=COMPONENTS,
        phases=[phase],
        n_queries=n_queries,
        function="calculate",
        calculate_phase=phase,
    )
    csv_path, meta_path = timing_cache_paths(config_id)
    # Also keep a stable human path for calculate (separate from equilibrium legacy).
    stable_csv = (
        REPO_ROOT / "results" / "calphad_reference_timing_pycalphad_calculate.csv"
    )

    print("=" * 72)
    print("pycalphad CALPHAD calculate() reference timing")
    print("=" * 72)
    print(f"function:        calculate()  [single-phase G; NO equilibrium]")
    print(f"config_id:       {config_id}")
    print(f"Components:      {COMPONENTS}")
    print(f"phase:           {phase}")
    print(f"phase_reason:    {CALCULATE_PHASE_REASON}")
    print(f"n_query_points:  {len(QUERY_POINTS)}")
    print(f"n_timed_queries: {n_queries}")
    print(f"timing_csv:      {csv_path}")
    print(f"stable_csv:      {stable_csv}")
    print()

    cached, cache_info = _try_load_timing_cache(
        config_id=config_id,
        n_queries=n_queries,
        force_refresh=force_refresh,
        legacy_csv=None,
    )
    print(f"timing_cache:    {cache_info['cache']}")
    if "source" in cache_info:
        print(f"timing_source:   {cache_info['source']}")
    print()

    if cached is not None:
        samples_ms = cached["samples_ms"]
        stats = _stats_ms(samples_ms)
        total_wall_s = float(np.sum(samples_ms) / 1000.0)
        _print_stats(stats, n=n_queries, total_wall_s=total_wall_s, load_s=db_load_s)
        # Refresh stable alias from keyed cache without touching equilibrium CSV.
        _write_timing_csv(
            stable_csv,
            point_idx=cached["point_idx"],
            temps=cached["temps"],
            samples_ms=cached["samples_ms"],
            gm_vals=cached["gm_vals"],
        )
        print(f"csv_written:         {csv_path}")
        print(f"stable_csv_written:  {stable_csv}")
        print(f"csv_rows:            {n_queries}")
        meta = {
            **cache_info,
            "function": "calculate",
            "phase": phase,
            "phase_reason": CALCULATE_PHASE_REASON,
            "stats_ms": stats,
            "timed_total_wall_s_approx_from_sum": total_wall_s,
            "database_load_s": db_load_s,
            "n_queries": n_queries,
            "stable_csv": str(stable_csv.resolve()),
        }
        _save_timing_meta(meta_path, meta)
        return {"stats": stats, "cache_info": cache_info, "csv_path": csv_path, "meta": meta}

    mod = Model(dbf, COMPONENTS, phase)
    points_by_idx = [
        composition_to_fcc_bcc_site_fractions(pt, mod.site_fractions).reshape(1, -1)
        for pt in QUERY_POINTS
    ]

    print("--- warm-up (discarded from timing) ---")
    for i, point in enumerate(QUERY_POINTS):
        t0 = time.perf_counter()
        res = calculate(
            dbf,
            COMPONENTS,
            phase,
            T=point["T"],
            P=101325,
            points=points_by_idx[i],
            output="GM",
        )
        dt = time.perf_counter() - t0
        gm = float(np.asarray(res.GM.values).reshape(-1)[0])
        print(
            f"  warmup[{i}] T={point['T']:.0f}K "
            f"X(CR,NI,C,MN,MO,SI)="
            f"({point['CR']},{point['NI']},{point['C']},"
            f"{point['MN']},{point['MO']},{point['SI']}) "
            f"wall_s={dt:.6f} GM={gm:.6g}"
        )
    print()

    print("--- timed queries ---")
    samples_ms = np.zeros(n_queries, dtype=np.float64)
    point_idx = np.zeros(n_queries, dtype=np.int64)
    temps = np.zeros(n_queries, dtype=np.float64)
    gm_vals = np.zeros(n_queries, dtype=np.float64)
    wall0 = time.perf_counter()
    for i in range(n_queries):
        idx = i % len(QUERY_POINTS)
        point = QUERY_POINTS[idx]
        t0 = time.perf_counter_ns()
        res = calculate(
            dbf,
            COMPONENTS,
            phase,
            T=point["T"],
            P=101325,
            points=points_by_idx[idx],
            output="GM",
        )
        dt_ms = (time.perf_counter_ns() - t0) / 1.0e6
        samples_ms[i] = dt_ms
        point_idx[i] = idx
        temps[i] = point["T"]
        gm_vals[i] = float(np.asarray(res.GM.values).reshape(-1)[0])
        if (i + 1) % max(1, n_queries // 10) == 0 or i == 0 or i + 1 == n_queries:
            print(
                f"  query {i + 1}/{n_queries} point={idx} T={point['T']:.0f}K "
                f"dt_ms={dt_ms:.3f} GM={gm_vals[i]:.6g}"
            )
    total_wall_s = time.perf_counter() - wall0
    print()

    stats = _stats_ms(samples_ms)
    _print_stats(stats, n=n_queries, total_wall_s=total_wall_s, load_s=db_load_s)
    _write_timing_csv(
        csv_path,
        point_idx=point_idx,
        temps=temps,
        samples_ms=samples_ms,
        gm_vals=gm_vals,
    )
    _write_timing_csv(
        stable_csv,
        point_idx=point_idx,
        temps=temps,
        samples_ms=samples_ms,
        gm_vals=gm_vals,
    )
    print(f"csv_written:         {csv_path}")
    print(f"stable_csv_written:  {stable_csv}")
    print(f"csv_rows:            {n_queries}")

    meta = {
        **cache_info,
        "cache": "MISS (computed)",
        "function": "calculate",
        "phase": phase,
        "phase_reason": CALCULATE_PHASE_REASON,
        "stats_ms": stats,
        "timed_total_wall_s": total_wall_s,
        "database_load_s": db_load_s,
        "n_queries": n_queries,
        "stable_csv": str(stable_csv.resolve()),
    }
    _save_timing_meta(meta_path, meta)
    return {"stats": stats, "cache_info": cache_info, "csv_path": csv_path, "meta": meta}


def run_calculate_precompiled_timing(
    dbf: Database,
    *,
    n_queries: int,
    db_load_s: float,
    tdb_name: str,
    force_refresh: bool,
) -> dict:
    """Single-phase calculate() with Model+PhaseRecordFactory built once."""
    phase = CALCULATE_PHASE
    config_id = _config_id(
        tdb_name=tdb_name,
        components=COMPONENTS,
        phases=[phase],
        n_queries=n_queries,
        function="calculate_precompiled",
        calculate_phase=phase,
    )
    csv_path, meta_path = timing_cache_paths(config_id)
    stable_csv = (
        REPO_ROOT
        / "results"
        / "calphad_reference_timing_pycalphad_calculate_precompiled.csv"
    )

    print("=" * 72)
    print("pycalphad CALPHAD calculate_precompiled reference timing")
    print("=" * 72)
    print(
        "function:        calculate() with phase_records+model reused "
        "(NOT bare calculate each call)"
    )
    print(f"config_id:       {config_id}")
    print(f"Components:      {COMPONENTS}")
    print(f"phase:           {phase}")
    print(f"phase_reason:    {CALCULATE_PHASE_REASON}")
    print(
        "precompile_note: Model + PhaseRecordFactory built once before warm-up; "
        "passed to every calculate() via phase_records=... and model=..."
    )
    print(f"n_query_points:  {len(QUERY_POINTS)}")
    print(f"n_timed_queries: {n_queries}")
    print(f"timing_csv:      {csv_path}")
    print(f"stable_csv:      {stable_csv}")
    print()

    cached, cache_info = _try_load_timing_cache(
        config_id=config_id,
        n_queries=n_queries,
        force_refresh=force_refresh,
        legacy_csv=None,
    )
    print(f"timing_cache:    {cache_info['cache']}")
    if "source" in cache_info:
        print(f"timing_source:   {cache_info['source']}")
    print()

    if cached is not None:
        samples_ms = cached["samples_ms"]
        stats = _stats_ms(samples_ms)
        total_wall_s = float(np.sum(samples_ms) / 1000.0)
        _print_stats(stats, n=n_queries, total_wall_s=total_wall_s, load_s=db_load_s)
        _write_timing_csv(
            stable_csv,
            point_idx=cached["point_idx"],
            temps=cached["temps"],
            samples_ms=cached["samples_ms"],
            gm_vals=cached["gm_vals"],
        )
        print(f"csv_written:         {csv_path}")
        print(f"stable_csv_written:  {stable_csv}")
        print(f"csv_rows:            {n_queries}")
        meta = {
            **cache_info,
            "function": "calculate_precompiled",
            "phase": phase,
            "phase_reason": CALCULATE_PHASE_REASON,
            "stats_ms": stats,
            "timed_total_wall_s_approx_from_sum": total_wall_s,
            "database_load_s": db_load_s,
            "n_queries": n_queries,
            "stable_csv": str(stable_csv.resolve()),
        }
        _save_timing_meta(meta_path, meta)
        return {"stats": stats, "cache_info": cache_info, "csv_path": csv_path, "meta": meta}

    # Build Model + PhaseRecordFactory ONCE (outside warm-up / timed loops).
    t_build0 = time.perf_counter()
    models = instantiate_models(dbf, COMPONENTS, [phase])
    # State vars the records compile against; T/P remain free inputs per call.
    statevars = {v.N: 1, v.P: 101325, v.T: 1000.0}
    phase_records = PhaseRecordFactory(dbf, COMPONENTS, statevars, models)
    _ = phase_records[phase]  # force PhaseRecord / callable construction
    build_s = time.perf_counter() - t_build0
    print(f"precompile_build_s: {build_s:.6f}  (excluded from per-query timing)")
    print()

    points_by_idx = [
        composition_to_fcc_bcc_site_fractions(
            pt, models[phase].site_fractions
        ).reshape(1, -1)
        for pt in QUERY_POINTS
    ]

    print("--- warm-up (discarded from timing) ---")
    for i, point in enumerate(QUERY_POINTS):
        t0 = time.perf_counter()
        res = calculate(
            dbf,
            COMPONENTS,
            phase,
            T=point["T"],
            P=101325,
            points=points_by_idx[i],
            output="GM",
            phase_records=phase_records,
            model=models,
        )
        dt = time.perf_counter() - t0
        gm = float(np.asarray(res.GM.values).reshape(-1)[0])
        print(
            f"  warmup[{i}] T={point['T']:.0f}K "
            f"X(CR,NI,C,MN,MO,SI)="
            f"({point['CR']},{point['NI']},{point['C']},"
            f"{point['MN']},{point['MO']},{point['SI']}) "
            f"wall_s={dt:.6f} GM={gm:.6g}"
        )
    print()

    print("--- timed queries ---")
    samples_ms = np.zeros(n_queries, dtype=np.float64)
    point_idx = np.zeros(n_queries, dtype=np.int64)
    temps = np.zeros(n_queries, dtype=np.float64)
    gm_vals = np.zeros(n_queries, dtype=np.float64)
    wall0 = time.perf_counter()
    for i in range(n_queries):
        idx = i % len(QUERY_POINTS)
        point = QUERY_POINTS[idx]
        t0 = time.perf_counter_ns()
        res = calculate(
            dbf,
            COMPONENTS,
            phase,
            T=point["T"],
            P=101325,
            points=points_by_idx[idx],
            output="GM",
            phase_records=phase_records,
            model=models,
        )
        dt_ms = (time.perf_counter_ns() - t0) / 1.0e6
        samples_ms[i] = dt_ms
        point_idx[i] = idx
        temps[i] = point["T"]
        gm_vals[i] = float(np.asarray(res.GM.values).reshape(-1)[0])
        if (i + 1) % max(1, n_queries // 10) == 0 or i == 0 or i + 1 == n_queries:
            print(
                f"  query {i + 1}/{n_queries} point={idx} T={point['T']:.0f}K "
                f"dt_ms={dt_ms:.3f} GM={gm_vals[i]:.6g}"
            )
    total_wall_s = time.perf_counter() - wall0
    print()

    stats = _stats_ms(samples_ms)
    _print_stats(stats, n=n_queries, total_wall_s=total_wall_s, load_s=db_load_s)
    _write_timing_csv(
        csv_path,
        point_idx=point_idx,
        temps=temps,
        samples_ms=samples_ms,
        gm_vals=gm_vals,
    )
    _write_timing_csv(
        stable_csv,
        point_idx=point_idx,
        temps=temps,
        samples_ms=samples_ms,
        gm_vals=gm_vals,
    )
    print(f"csv_written:         {csv_path}")
    print(f"stable_csv_written:  {stable_csv}")
    print(f"csv_rows:            {n_queries}")

    meta = {
        **cache_info,
        "cache": "MISS (computed)",
        "function": "calculate_precompiled",
        "phase": phase,
        "phase_reason": CALCULATE_PHASE_REASON,
        "stats_ms": stats,
        "timed_total_wall_s": total_wall_s,
        "precompile_build_s": build_s,
        "database_load_s": db_load_s,
        "n_queries": n_queries,
        "stable_csv": str(stable_csv.resolve()),
    }
    _save_timing_meta(meta_path, meta)
    return {"stats": stats, "cache_info": cache_info, "csv_path": csv_path, "meta": meta}


def _load_cached_calculate_stats(tdb_name: str, n_queries: int) -> dict | None:
    """Load original bare-calculate() stats from keyed cache if present."""
    config_id = _config_id(
        tdb_name=tdb_name,
        components=COMPONENTS,
        phases=[CALCULATE_PHASE],
        n_queries=n_queries,
        function="calculate",
        calculate_phase=CALCULATE_PHASE,
    )
    csv_path, _ = timing_cache_paths(config_id)
    if not csv_path.is_file():
        return None
    data = _read_timing_csv(csv_path)
    if len(data["samples_ms"]) != n_queries:
        return None
    return _stats_ms(data["samples_ms"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="pycalphad calculate() / equilibrium() reference timing with caching"
    )
    parser.add_argument("--tdb", type=Path, default=DEFAULT_TDB)
    parser.add_argument("--n-queries", type=int, default=100)
    parser.add_argument(
        "--mode",
        choices=(
            "calculate",
            "calculate_precompiled",
            "equilibrium",
            "both",
            "calculate_compare",
        ),
        default="both",
        help=(
            "which measurement(s) to run / report; "
            "calculate_compare = bare calculate cache + precompiled run"
        ),
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="recompute timing even if a matching results cache exists",
    )
    parser.add_argument(
        "--force-db-refresh",
        action="store_true",
        help="re-parse TDB even if a Database pickle cache exists",
    )
    args = parser.parse_args()

    tdb_path = _ensure_tdb(args.tdb)

    print("=" * 72)
    print("pycalphad CALPHAD reference timing (calculate + equilibrium)")
    print("=" * 72)
    print(f"TDB path:        {tdb_path}")
    print(f"TDB source URL:  {TDB_URL}")
    print(
        "TDB provenance:  MatCalc mc_fe_v2.059 (E. Povoden-Karadeniz, TU Wien); "
        "redistributed in pycalphad examples/databases/"
    )
    print("TDB license:     ODbL 1.0 / DbCL 1.0 (per TDB file header)")
    print(f"mode:            {args.mode}")
    print()
    print("--- environment ---")
    print(f"CPU:             {_cpu_model()}")
    print(f"OS:              {platform.platform()}")
    print(f"Python:          {sys.version.split()[0]} ({sys.executable})")
    print(f"pycalphad:       {PYCALPHAD_VERSION}")
    print()

    dbf, db_meta = load_database_cached(
        tdb_path, force_refresh=args.force_db_refresh
    )
    print("--- Database cache ---")
    print(f"database_cache:  {db_meta['cache']}")
    print(f"database_source: {db_meta.get('source')}")
    print(f"database_load_s: {db_meta['load_s']:.6f}")
    print(f"content_sha256:  {db_meta['content_sha256']}")
    print(f"pickle_path:     {db_meta['pickle_path']}")
    print()

    phases = [ph for ph in PHASE_CANDIDATES if ph in dbf.phases]
    missing = [ph for ph in PHASE_CANDIDATES if ph not in dbf.phases]
    print(f"equilibrium_phases: {phases}")
    if missing:
        print(f"phases_missing:     {missing}")
    print(f"calculate_phase:    {CALCULATE_PHASE}")
    print(f"calculate_reason:   {CALCULATE_PHASE_REASON}")
    print()

    eq_result = None
    calc_result = None
    calc_pre_result = None

    if args.mode in ("equilibrium", "both"):
        eq_result = run_equilibrium_timing(
            dbf,
            phases,
            n_queries=args.n_queries,
            db_load_s=float(db_meta["load_s"]),
            tdb_name=tdb_path.name,
            force_refresh=args.force_refresh,
        )
        print()

    if args.mode in ("calculate", "both", "calculate_compare"):
        calc_result = run_calculate_timing(
            dbf,
            n_queries=args.n_queries,
            db_load_s=float(db_meta["load_s"]),
            tdb_name=tdb_path.name,
            force_refresh=args.force_refresh,
        )
        print()

    if args.mode in ("calculate_precompiled", "calculate_compare"):
        calc_pre_result = run_calculate_precompiled_timing(
            dbf,
            n_queries=args.n_queries,
            db_load_s=float(db_meta["load_s"]),
            tdb_name=tdb_path.name,
            force_refresh=args.force_refresh,
        )
        print()

    if eq_result is not None and calc_result is not None:
        eq_s = eq_result["stats"]
        ca_s = calc_result["stats"]
        print("=" * 72)
        print("SIDE-BY-SIDE (separate measurements; NOT averaged)")
        print("=" * 72)
        print(
            f"{'metric':<18} {'calculate() ms':>16} {'equilibrium() ms':>18} {'eq/calc':>12}"
        )
        for key in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "min_ms", "max_ms"):
            ratio = eq_s[key] / ca_s[key] if ca_s[key] != 0 else float("inf")
            print(
                f"{key:<18} {ca_s[key]:16.6f} {eq_s[key]:18.6f} {ratio:12.2f}x"
            )
        print()
        print("NOTE: calculate() = single-phase G(X,T); equilibrium() = multiphase solve.")
        print("SymGWR surrogate compares to calculate(), not equilibrium().")
        print()

    if calc_pre_result is not None:
        ca_s = (
            calc_result["stats"]
            if calc_result is not None
            else _load_cached_calculate_stats(tdb_path.name, args.n_queries)
        )
        pre_s = calc_pre_result["stats"]
        print("=" * 72)
        print("SIDE-BY-SIDE: bare calculate() vs calculate_precompiled")
        print("=" * 72)
        if ca_s is None:
            print("(original calculate() cache not found; showing precompiled only)")
            for key in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "min_ms", "max_ms"):
                print(f"{key:<18} {pre_s[key]:16.6f}")
        else:
            print(
                f"{'metric':<18} {'bare calc ms':>16} {'precompiled ms':>16} "
                f"{'bare/pre':>12}"
            )
            for key in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "min_ms", "max_ms"):
                ratio = ca_s[key] / pre_s[key] if pre_s[key] != 0 else float("inf")
                print(
                    f"{key:<18} {ca_s[key]:16.6f} {pre_s[key]:16.6f} {ratio:12.2f}x"
                )
            print()
            print(
                "NOTE: bare calculate() rebuilds Model+PhaseRecordFactory each call "
                "(build_functions LLVM callables still LRU-cached globally). "
                "calculate_precompiled reuses phase_records+model."
            )
        print()

    print("--- cache inventory ---")
    print(f"Database pickle: {db_meta['pickle_path']} [{db_meta['cache']}]")
    if eq_result is not None:
        print(
            f"equilibrium CSV: {eq_result['csv_path']} "
            f"[{eq_result['cache_info']['cache']}]"
        )
        print(f"legacy eq CSV:   {LEGACY_EQUILIBRIUM_CSV} (untouched)")
    if calc_result is not None:
        print(
            f"calculate CSV:   {calc_result['csv_path']} "
            f"[{calc_result['cache_info']['cache']}]"
        )
        print(
            f"calculate alias: "
            f"{REPO_ROOT / 'results' / 'calphad_reference_timing_pycalphad_calculate.csv'}"
        )
    if calc_pre_result is not None:
        print(
            f"precompiled CSV: {calc_pre_result['csv_path']} "
            f"[{calc_pre_result['cache_info']['cache']}]"
        )
        print(
            f"precompiled alias: "
            f"{REPO_ROOT / 'results' / 'calphad_reference_timing_pycalphad_calculate_precompiled.csv'}"
        )


if __name__ == "__main__":
    main()
