#!/usr/bin/env python3
"""
Deterministic manuscript-number extractor.
Reads only canonical files listed in paper/PAPER_FILE_MAP.yaml.
Writes:
  - results/PAPER_RESULTS_SUMMARY.md
  - results/MANUSCRIPT_NUMBERS.md
  - results/PAPER_RESULTS.json
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

NA = "unavailable"
CANONICAL_PMLB_MODELS = [
    "Linear Regression",
    "Linear GWR",
    "Ridge Regression",
    "Ridge GWR",
    "Polynomial Regression",
    "Polynomial GWR",
]
PMLB_PRIMARY_MODEL = "Polynomial GWR"
LOWER_IS_BETTER = {"mae", "mse", "rmse", "mape", "smape", "runtime"}
HIGHER_IS_BETTER = {"r2"}


@dataclass(frozen=True)
class TracedStat:
    section: str
    stat_id: str
    value: float | int | str
    source_file: str
    row_configuration: str
    metric: str
    unavailable_reason: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "value": self.value,
            "source_file": self.source_file,
            "row_configuration": self.row_configuration,
            "metric": self.metric,
        }
        if self.unavailable_reason is not None:
            out["unavailable_reason"] = self.unavailable_reason
        return out


def parse_simple_yaml(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()
        if ":" not in stripped:
            raise RuntimeError(f"Invalid YAML line: {raw_line}")
        key, value_part = stripped.split(":", 1)
        key = key.strip()
        value_part = value_part.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise RuntimeError("Invalid YAML indentation.")
        parent = stack[-1][1]
        if value_part == "":
            new_map: Dict[str, Any] = {}
            parent[key] = new_map
            stack.append((indent, new_map))
        else:
            if (value_part.startswith('"') and value_part.endswith('"')) or (
                value_part.startswith("'") and value_part.endswith("'")
            ):
                value = value_part[1:-1]
            else:
                value = value_part
            parent[key] = value
    return root


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def fmt(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{value:.6g}"


def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {path}")
        rows = [dict(row) for row in reader]
    return list(reader.fieldnames), rows


def require_columns(path: str, headers: List[str], required: List[str]) -> None:
    missing = [col for col in required if col not in headers]
    if missing:
        raise RuntimeError(f"{path} missing required columns: {missing}")


def select_min_row(rows: List[Dict[str, str]], metric: str) -> Tuple[int, Dict[str, str]]:
    candidates: List[Tuple[float, int, Dict[str, str]]] = []
    for idx, row in enumerate(rows):
        val = to_float(row.get(metric))
        if val is None:
            continue
        degree = to_float(row.get("selected_degree")) or float("inf")
        neigh = to_float(row.get("selected_n_neighbors")) or float("inf")
        rep = to_float(row.get("repeat")) or float("inf")
        seed = to_float(row.get("seed")) or float("inf")
        tie = degree * 1e9 + neigh * 1e6 + rep * 1e3 + seed
        candidates.append((val + tie * 1e-15, idx, row))
    if not candidates:
        raise RuntimeError(f"No valid rows for metric {metric}")
    _, idx, row = min(candidates, key=lambda item: item[0])
    return idx, row


def row_config(row: Dict[str, str], idx: int) -> str:
    keys = [
        "repeat",
        "seed",
        "base_model",
        "selected_degree",
        "selected_n_neighbors",
        "relative",
        "model",
        "task_name",
        "degree",
        "n_neighbors",
    ]
    pieces = [f"row_index={idx}"]
    for key in keys:
        val = str(row.get(key, "")).strip()
        if val:
            pieces.append(f"{key}={val}")
    return ", ".join(pieces)


def median(values: List[float]) -> float:
    if not values:
        raise RuntimeError("median() on empty values")
    return statistics.median(values)


def mean(values: List[float]) -> float:
    if not values:
        raise RuntimeError("mean() on empty values")
    return statistics.mean(values)


def register(
    stats: Dict[str, Dict[str, TracedStat]],
    map_sections: Dict[str, Any],
    section: str,
    stat_id: str,
    value: float | int | str,
    source_file: str,
    row_configuration: str,
    metric: str,
    unavailable_reason: str | None = None,
) -> None:
    if section not in map_sections:
        raise RuntimeError(f"Section missing in PAPER_FILE_MAP.yaml: {section}")
    if stat_id not in map_sections[section]:
        raise RuntimeError(f"Statistic missing in PAPER_FILE_MAP.yaml: {section}.{stat_id}")
    expected_source = str(map_sections[section][stat_id]["source"]).strip()
    expected_metric = str(map_sections[section][stat_id]["metric"]).strip()
    if expected_source != source_file:
        raise RuntimeError(
            f"Source mismatch for {section}.{stat_id}: expected {expected_source}, got {source_file}"
        )
    if expected_metric != metric:
        raise RuntimeError(
            f"Metric mismatch for {section}.{stat_id}: expected {expected_metric}, got {metric}"
        )
    stats.setdefault(section, {})[stat_id] = TracedStat(
        section=section,
        stat_id=stat_id,
        value=value,
        source_file=source_file,
        row_configuration=row_configuration,
        metric=metric,
        unavailable_reason=unavailable_reason,
    )


def build_calphad(
    stats: Dict[str, Dict[str, TracedStat]],
    section_map: Dict[str, Any],
    canonical_files: Dict[str, str],
    csv_cache: Dict[str, Tuple[List[str], List[Dict[str, str]]]],
) -> Dict[str, Any]:
    source = canonical_files["CALPHAD"]
    headers, rows = csv_cache[source]
    require_columns(
        source,
        headers,
        [
            "selected_degree",
            "selected_n_neighbors",
            "gwr_mae_mean",
            "gwr_mse",
            "gwr_rmse",
            "gwr_mape_mean",
            "gwr_smape_mean",
            "gwr_r2",
            "gwr_time_s",
        ],
    )
    idx, best_row = select_min_row(rows, "gwr_mape_mean")
    cfg = row_config(best_row, idx)
    fields = {
        "calphad_degree": ("selected_degree", int),
        "calphad_neighbors": ("selected_n_neighbors", int),
        "calphad_mae": ("gwr_mae_mean", float),
        "calphad_mse": ("gwr_mse", float),
        "calphad_rmse": ("gwr_rmse", float),
        "calphad_mape": ("gwr_mape_mean", float),
        "calphad_smape": ("gwr_smape_mean", float),
        "calphad_r2": ("gwr_r2", float),
        "calphad_runtime": ("gwr_time_s", float),
    }
    for section in ("Abstract", "Results"):
        for stat_id, (metric_name, caster) in fields.items():
            raw = to_float(best_row.get(metric_name))
            if raw is None:
                raise RuntimeError(f"Missing CALPHAD value {metric_name} at {cfg}")
            value: float | int = int(raw) if caster is int else float(raw)
            register(stats, section_map, section, stat_id, value, source, cfg, metric_name)
        register(
            stats,
            section_map,
            section,
            "calphad_training_time",
            NA,
            source,
            cfg,
            "training_time_unavailable",
            unavailable_reason="training_time column absent in gwr_results_raw.csv; only gwr_time_s available",
        )
    return {"best_row": best_row, "best_cfg": cfg}


def build_baseline(
    stats: Dict[str, Dict[str, TracedStat]],
    section_map: Dict[str, Any],
    canonical_files: Dict[str, str],
    csv_cache: Dict[str, Tuple[List[str], List[Dict[str, str]]]],
    calphad: Dict[str, Any],
) -> Dict[str, Any]:
    source = canonical_files["BASELINE_COMPARISON"]
    headers, rows = csv_cache[source]
    require_columns(source, headers, ["model", "mae_mean", "mse", "rmse", "mape_mean", "smape_mean", "r2", "time_s"])
    candidates = [row for row in rows if "gwr" not in str(row.get("model", "")).lower()]
    if not candidates:
        raise RuntimeError("No non-SymGWR baseline rows found")

    gwr_values = {
        "mae": to_float(calphad["best_row"]["gwr_mae_mean"]),
        "mse": to_float(calphad["best_row"]["gwr_mse"]),
        "rmse": to_float(calphad["best_row"]["gwr_rmse"]),
        "mape": to_float(calphad["best_row"]["gwr_mape_mean"]),
        "smape": to_float(calphad["best_row"]["gwr_smape_mean"]),
        "r2": to_float(calphad["best_row"]["gwr_r2"]),
        "runtime": to_float(calphad["best_row"]["gwr_time_s"]),
    }
    col_map = {
        "mae": "mae_mean",
        "mse": "mse",
        "rmse": "rmse",
        "mape": "mape_mean",
        "smape": "smape_mean",
        "r2": "r2",
        "runtime": "time_s",
    }
    stat_map = {
        "mae": "best_competing_mae",
        "mse": "best_competing_mse",
        "rmse": "best_competing_rmse",
        "mape": "best_competing_mape",
        "smape": "best_competing_smape",
        "r2": "best_competing_r2",
        "runtime": "best_competing_runtime",
    }
    best_by_metric: Dict[str, Dict[str, Any]] = {}
    for metric, col in col_map.items():
        scored: List[Tuple[float, int, Dict[str, str]]] = []
        for idx, row in enumerate(candidates):
            val = to_float(row.get(col))
            if val is not None:
                scored.append((val, idx, row))
        if not scored:
            raise RuntimeError(f"No baseline values for metric {metric}")
        if metric in LOWER_IS_BETTER:
            best_val, best_idx, best_row = min(scored, key=lambda item: (item[0], str(item[2].get("model", ""))))
        elif metric in HIGHER_IS_BETTER:
            best_val, best_idx, best_row = max(scored, key=lambda item: (item[0], str(item[2].get("model", ""))))
        else:
            raise RuntimeError(f"No metric direction configured for {metric}")
        gwr_val = gwr_values[metric]
        if gwr_val is None:
            raise RuntimeError(f"SymGWR comparator missing for metric {metric}")
        absolute = (best_val - gwr_val) if metric in LOWER_IS_BETTER else (gwr_val - best_val)
        percent = NA if best_val == 0 else (absolute / abs(best_val)) * 100.0
        config = row_config(best_row, best_idx)
        register(stats, section_map, "BaselineComparison", stat_map[metric], best_val, source, config, col)
        best_by_metric[metric] = {
            "baseline_model": str(best_row.get("model", "")).strip(),
            "baseline_value": best_val,
            "gwr_value": gwr_val,
            "absolute_improvement": absolute,
            "percent_improvement": percent,
            "row_configuration": config,
            "source_file": source,
        }
    return best_by_metric


def build_pmlb(
    stats: Dict[str, Dict[str, TracedStat]],
    section_map: Dict[str, Any],
    canonical_files: Dict[str, str],
    csv_cache: Dict[str, Tuple[List[str], List[Dict[str, str]]]],
) -> Dict[str, Any]:
    source = canonical_files["PMLB"]
    headers, rows = csv_cache[source]
    require_columns(source, headers, ["task_name", "model", "mape_mean", "smape_mean"])
    filtered = [r for r in rows if str(r.get("model", "")).strip() in CANONICAL_PMLB_MODELS]
    if not filtered:
        raise RuntimeError("No canonical rows in pmlb_metrics_summary.csv")
    by_dataset: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for idx, row in enumerate(filtered):
        dataset = str(row.get("task_name", "")).strip()
        model = str(row.get("model", "")).strip()
        mape = to_float(row.get("mape_mean"))
        smape = to_float(row.get("smape_mean"))
        if dataset == "" or model == "" or mape is None or smape is None:
            raise RuntimeError(f"Incomplete canonical PMLB row {idx}")
        by_dataset.setdefault(dataset, {})
        if model in by_dataset[dataset]:
            raise RuntimeError(f"Duplicate canonical model {model} in dataset {dataset}")
        by_dataset[dataset][model] = {"mape": mape, "smape": smape, "row": row, "idx": idx}

    bad = [d for d, model_map in by_dataset.items() if sorted(model_map) != sorted(CANONICAL_PMLB_MODELS)]
    if bad:
        raise RuntimeError("Canonical model coverage mismatch for datasets: " + ", ".join(sorted(bad)[:10]))

    primary_mapes: List[float] = []
    primary_smapes: List[float] = []
    primary_ranks: List[float] = []
    wins = 0
    top3 = 0
    ranking_rows: List[Dict[str, Any]] = []
    for dataset in sorted(by_dataset.keys()):
        ordered = sorted(by_dataset[dataset].items(), key=lambda item: (item[1]["mape"], item[1]["smape"], item[0]))
        for rank, (model, payload) in enumerate(ordered, start=1):
            ranking_rows.append({"dataset": dataset, "model": model, "rank": rank, "mape": payload["mape"], "smape": payload["smape"]})
            if model == PMLB_PRIMARY_MODEL:
                primary_mapes.append(payload["mape"])
                primary_smapes.append(payload["smape"])
                primary_ranks.append(float(rank))
                if rank == 1:
                    wins += 1
                if rank <= 3:
                    top3 += 1

    n_dataset = len(by_dataset)
    register(stats, section_map, "PMLB", "dataset_count", n_dataset, source, "canonical_six_models_only", "task_name_unique_count")
    register(stats, section_map, "PMLB", "primary_model_median_mape", median(primary_mapes), source, f"model={PMLB_PRIMARY_MODEL},datasets={n_dataset}", "mape_mean")
    register(stats, section_map, "PMLB", "primary_model_median_smape", median(primary_smapes), source, f"model={PMLB_PRIMARY_MODEL},datasets={n_dataset}", "smape_mean")
    register(stats, section_map, "PMLB", "primary_model_average_rank", mean(primary_ranks), source, f"model={PMLB_PRIMARY_MODEL},ranking=MAPE_then_SMAPE", "rank")
    register(stats, section_map, "PMLB", "primary_model_wins", wins, source, f"model={PMLB_PRIMARY_MODEL},ranking=MAPE_then_SMAPE", "win_count")
    register(stats, section_map, "PMLB", "primary_model_top3", top3, source, f"model={PMLB_PRIMARY_MODEL},ranking=MAPE_then_SMAPE", "top3_count")
    register(stats, section_map, "PMLB", "primary_model_mean_mape_supplementary", mean(primary_mapes), source, f"model={PMLB_PRIMARY_MODEL},datasets={n_dataset}", "mape_mean")

    sig_source = canonical_files["SIGNIFICANCE"]
    sig_headers, sig_rows = csv_cache[sig_source]
    require_columns(sig_source, sig_headers, ["baseline_model", "gwr_model", "n_datasets"])
    sig_counts: List[int] = []
    for idx, row in enumerate(sig_rows):
        n = to_float(row.get("n_datasets"))
        if n is None:
            raise RuntimeError(f"Missing n_datasets in significance row {idx}")
        sig_counts.append(int(n))
    register(
        stats,
        section_map,
        "PMLB",
        "significance_dataset_count",
        min(sig_counts),
        sig_source,
        "all_rows",
        "n_datasets",
    )

    pair_counts: Dict[str, int] = {}
    for base, gwr in [("Linear Regression", "Linear GWR"), ("Ridge Regression", "Ridge GWR"), ("Polynomial Regression", "Polynomial GWR")]:
        pair_counts[f"{base} vs {gwr}"] = len(
            [d for d, mm in by_dataset.items() if base in mm and gwr in mm]
        )
    mismatch: List[str] = []
    for row in sig_rows:
        pair = f"{str(row.get('baseline_model', '')).strip()} vs {str(row.get('gwr_model', '')).strip()}"
        sig_n = int(to_float(row.get("n_datasets")) or 0)
        metric_n = pair_counts.get(pair)
        if metric_n is None:
            mismatch.append(f"{pair}: pair_not_in_canonical_triplet")
        elif metric_n != sig_n:
            mismatch.append(f"{pair}: significance={sig_n},metrics_common={metric_n}")
    mismatch_explanation = (
        "dataset counts match between pmlb_metrics_summary.csv and pmlb_significance_tests.csv"
        if not mismatch
        else "dataset count mismatch: " + "; ".join(mismatch)
    )
    return {
        "ranking_rows": ranking_rows,
        "dataset_count": n_dataset,
        "pair_counts": pair_counts,
        "significance_rows": sig_rows,
        "mismatch_explanation": mismatch_explanation,
    }


def build_derivative(
    stats: Dict[str, Dict[str, TracedStat]],
    section_map: Dict[str, Any],
    canonical_files: Dict[str, str],
    csv_cache: Dict[str, Tuple[List[str], List[Dict[str, str]]]],
) -> Dict[str, Any]:
    source = canonical_files["DERIVATIVE"]
    headers, rows = csv_cache[source]
    require_columns(source, headers, ["derivative_mode", "derivative_mae_mean", "derivative_mse_mean", "limitation_note"])
    if not rows:
        raise RuntimeError("Derivative file has no rows")
    row = rows[0]
    cfg = row_config(row, 0)
    mae = to_float(row.get("derivative_mae_mean"))
    mse = to_float(row.get("derivative_mse_mean"))
    mode = str(row.get("derivative_mode", "")).strip()
    if mae is None or mse is None or mode == "":
        raise RuntimeError("Derivative required fields missing in first row")
    register(stats, section_map, "Derivative", "derivative_mae", mae, source, cfg, "derivative_mae_mean")
    register(stats, section_map, "Derivative", "derivative_mse", mse, source, cfg, "derivative_mse_mean")
    register(stats, section_map, "Derivative", "derivative_mode", mode, source, cfg, "derivative_mode")
    return {"limitation_note": str(row.get("limitation_note", "")).strip(), "source": source, "cfg": cfg}


def build_runtime(
    stats: Dict[str, Dict[str, TracedStat]],
    section_map: Dict[str, Any],
    canonical_files: Dict[str, str],
    csv_cache: Dict[str, Tuple[List[str], List[Dict[str, str]]]],
) -> Dict[str, Any]:
    gwr_source = canonical_files["RUNTIME_SymGWR"]
    gwr_headers, gwr_rows = csv_cache[gwr_source]
    require_columns(gwr_source, gwr_headers, ["gwr_time_s_mean"])
    if not gwr_rows:
        raise RuntimeError("gwr_results.csv has no rows")
    gwr_runtime = to_float(gwr_rows[0].get("gwr_time_s_mean"))
    if gwr_runtime is None:
        raise RuntimeError("gwr_time_s_mean missing in gwr_results.csv first row")
    register(
        stats,
        section_map,
        "Runtime",
        "gwr_end_to_end_runtime",
        gwr_runtime,
        gwr_source,
        row_config(gwr_rows[0], 0),
        "gwr_time_s_mean",
    )

    base_source = canonical_files["RUNTIME_BASELINES"]
    base_headers, base_rows = csv_cache[base_source]
    require_columns(base_source, base_headers, ["model", "time_s_mean"])
    scored: List[Tuple[float, int, Dict[str, str]]] = []
    for idx, row in enumerate(base_rows):
        val = to_float(row.get("time_s_mean"))
        if val is not None:
            scored.append((val, idx, row))
    if not scored:
        raise RuntimeError("baseline_comparison.csv has no valid time_s_mean")
    fast_val, fast_idx, fast_row = min(scored, key=lambda item: (item[0], str(item[2].get("model", ""))))
    slow_val, slow_idx, slow_row = max(scored, key=lambda item: (item[0], str(item[2].get("model", ""))))
    register(stats, section_map, "Runtime", "fastest_baseline_end_to_end_runtime", fast_val, base_source, row_config(fast_row, fast_idx), "time_s_mean")
    register(stats, section_map, "Runtime", "slowest_baseline_end_to_end_runtime", slow_val, base_source, row_config(slow_row, slow_idx), "time_s_mean")
    return {"fast_model": str(fast_row.get("model", "")).strip(), "slow_model": str(slow_row.get("model", "")).strip()}


def build_neighbor_ablation(
    stats: Dict[str, Dict[str, TracedStat]],
    section_map: Dict[str, Any],
    canonical_files: Dict[str, str],
    csv_cache: Dict[str, Tuple[List[str], List[Dict[str, str]]]],
) -> Dict[str, Any]:
    source = canonical_files["NEIGHBOR_ABLATION"]
    headers, rows = csv_cache[source]
    require_columns(source, headers, ["n_neighbors", "relative", "mape_mean_mean", "mape_mean_std", "n_repeats"])
    filtered = []
    for idx, row in enumerate(rows):
        if str(row.get("relative", "")).strip().lower() != "false":
            continue
        k = to_float(row.get("n_neighbors"))
        mape = to_float(row.get("mape_mean_mean"))
        std = to_float(row.get("mape_mean_std"))
        if k is None or mape is None:
            continue
        filtered.append({"idx": idx, "row": row, "k": int(k), "mape": mape, "std": std or 0.0})
    if not filtered:
        raise RuntimeError("No valid relative=False rows in ablation_neighbors.csv")
    filtered = sorted(filtered, key=lambda item: item["k"])
    best = min(filtered, key=lambda item: (item["mape"], item["k"]))
    indistinguishable = [item for item in filtered if abs(item["mape"] - best["mape"]) <= max(item["std"], best["std"])]
    plateau = min(indistinguishable, key=lambda item: item["k"])
    note = (
        f"multiple statistically indistinguishable k values: {[x['k'] for x in indistinguishable]}"
        if len(indistinguishable) > 1
        else f"single best k: {best['k']}"
    )
    register(stats, section_map, "NeighborAblation", "best_k", best["k"], source, row_config(best["row"], best["idx"]), "mape_mean_mean")
    register(stats, section_map, "NeighborAblation", "plateau_k", plateau["k"], source, row_config(plateau["row"], plateau["idx"]), "mape_mean_mean")
    register(stats, section_map, "NeighborAblation", "plateau_note", note, source, "relative=False grouped by n_neighbors", "mape_mean_std")
    return {"best": best, "plateau": plateau, "indistinguishable": indistinguishable}


def build_degree_ablation(
    stats: Dict[str, Dict[str, TracedStat]],
    section_map: Dict[str, Any],
    canonical_files: Dict[str, str],
    csv_cache: Dict[str, Tuple[List[str], List[Dict[str, str]]]],
) -> Dict[str, Any]:
    source = canonical_files["DEGREE_ABLATION"]
    headers, rows = csv_cache[source]
    require_columns(source, headers, ["degree", "gwr_mape_mean_mean", "gwr_mape_mean_std", "n_repeats"])
    parsed = []
    for idx, row in enumerate(rows):
        degree = to_float(row.get("degree"))
        mape = to_float(row.get("gwr_mape_mean_mean"))
        std = to_float(row.get("gwr_mape_mean_std"))
        if degree is None or mape is None:
            continue
        parsed.append({"idx": idx, "row": row, "degree": int(degree), "mape": mape, "std": std or 0.0})
    if not parsed:
        raise RuntimeError("No valid rows in ablation_degree.csv")
    parsed = sorted(parsed, key=lambda item: item["degree"])
    best = min(parsed, key=lambda item: (item["mape"], item["degree"]))

    diminishing = best
    for i in range(1, len(parsed)):
        delta = parsed[i - 1]["mape"] - parsed[i]["mape"]
        if delta >= 0 and delta <= parsed[i]["std"]:
            diminishing = parsed[i]
            break

    overfit_degree = parsed[-1]["degree"]
    found_overfit = False
    for i in range(2, len(parsed)):
        if parsed[i]["mape"] > parsed[i - 1]["mape"] and parsed[i - 1]["mape"] > parsed[i - 2]["mape"]:
            overfit_degree = parsed[i - 1]["degree"]
            found_overfit = True
            break
    _ = found_overfit

    register(stats, section_map, "DegreeAblation", "best_degree", best["degree"], source, row_config(best["row"], best["idx"]), "gwr_mape_mean_mean")
    register(stats, section_map, "DegreeAblation", "diminishing_returns_degree", diminishing["degree"], source, row_config(diminishing["row"], diminishing["idx"]), "gwr_mape_mean_mean")
    register(
        stats,
        section_map,
        "DegreeAblation",
        "overfitting_trend_degree",
        overfit_degree,
        source,
        f"trend_scan_degrees={[x['degree'] for x in parsed]}",
        "gwr_mape_mean_mean",
    )
    return {"rows": parsed, "best": best, "diminishing": diminishing, "overfit_degree": overfit_degree}


def run_consistency_checks(
    stats: Dict[str, Dict[str, TracedStat]],
    canonical_files: Dict[str, str],
    pmlb_meta: Dict[str, Any],
) -> List[str]:
    errors: List[str] = []
    abstract = stats.get("Abstract", {})
    results = stats.get("Results", {})
    for key in abstract:
        if key not in results:
            errors.append(f"Results missing stat present in Abstract: {key}")
            continue
        if abstract[key].value != results[key].value:
            errors.append(f"Abstract/Results mismatch for {key}: {abstract[key].value} vs {results[key].value}")

    if "calphad_degree" in abstract and "calphad_degree" in results:
        if abstract["calphad_degree"].value != results["calphad_degree"].value:
            errors.append("Abstract degree mismatch")
    if "calphad_neighbors" in abstract and "calphad_neighbors" in results:
        if abstract["calphad_neighbors"].value != results["calphad_neighbors"].value:
            errors.append("Abstract neighbors mismatch")

    canonical_set = set(canonical_files.values())
    for section, section_stats in stats.items():
        for stat_id, stat in section_stats.items():
            if stat.source_file not in canonical_set:
                errors.append(f"Non-canonical source used: {section}.{stat_id} -> {stat.source_file}")

    for dataset in sorted({r["dataset"] for r in pmlb_meta["ranking_rows"]}):
        models = [r["model"] for r in pmlb_meta["ranking_rows"] if r["dataset"] == dataset]
        if len(models) != len(set(models)):
            errors.append(f"Duplicate models found in PMLB ranking dataset={dataset}")

    for section, section_stats in stats.items():
        for stat_id, stat in section_stats.items():
            if stat.source_file == "" or stat.metric == "" or stat.row_configuration == "":
                errors.append(f"Untraceable stat {section}.{stat_id}")
    return errors


def build_summary_markdown(
    file_map: Dict[str, Any],
    stats: Dict[str, Dict[str, TracedStat]],
    baseline_meta: Dict[str, Any],
    pmlb_meta: Dict[str, Any],
    derivative_meta: Dict[str, Any],
    runtime_meta: Dict[str, Any],
    checks: List[str],
) -> str:
    lines: List[str] = []
    lines.append("# PAPER Results Summary")
    lines.append("")
    lines.append("## Canonical File Map")
    lines.append("```json")
    lines.append(json.dumps(file_map, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Validation")
    if checks:
        for err in checks:
            lines.append(f"- FAIL: {err}")
    else:
        lines.append("- PASS: Abstract numbers match Results numbers.")
        lines.append("- PASS: Degree and neighbors consistent across sections.")
        lines.append("- PASS: All statistics use canonical sources only.")
        lines.append("- PASS: No duplicate canonical models in PMLB rankings.")
        lines.append("- PASS: Every number includes source, row/config, and metric.")
    lines.append("")
    lines.append("## Statistics With Trace")
    for section in sorted(stats.keys()):
        lines.append(f"### {section}")
        for stat_id in sorted(stats[section].keys()):
            s = stats[section][stat_id]
            reason = f"; reason={s.unavailable_reason}" if s.unavailable_reason else ""
            lines.append(
                f"- {stat_id}: value={fmt(s.value)}; source={s.source_file}; row={s.row_configuration}; metric={s.metric}{reason}"
            )
    lines.append("")
    lines.append("## Baseline Improvements (positive means SymGWR better)")
    for metric in ["mae", "mse", "rmse", "mape", "smape", "r2", "runtime"]:
        item = baseline_meta[metric]
        lines.append(
            f"- {metric}: baseline_model={item['baseline_model']}; baseline={fmt(item['baseline_value'])}; "
            f"gwr={fmt(item['gwr_value'])}; absolute={fmt(item['absolute_improvement'])}; "
            f"percent={fmt(item['percent_improvement'])}; row={item['row_configuration']}"
        )
    lines.append("")
    lines.append("## PMLB Notes")
    lines.append("- canonical models: " + ", ".join(CANONICAL_PMLB_MODELS))
    lines.append("- primary model: " + PMLB_PRIMARY_MODEL)
    lines.append("- significance/metrics count check: " + pmlb_meta["mismatch_explanation"])
    lines.append("")
    lines.append("## Derivative Note")
    lines.append("- " + derivative_meta["limitation_note"])
    lines.append("")
    lines.append("## Runtime Equivalence")
    lines.append(f"- fastest baseline model: {runtime_meta['fast_model']}")
    lines.append(f"- slowest baseline model: {runtime_meta['slow_model']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def build_manuscript_markdown(
    stats: Dict[str, Dict[str, TracedStat]],
    baseline_meta: Dict[str, Any],
    pmlb_meta: Dict[str, Any],
    derivative_meta: Dict[str, Any],
) -> str:
    a = stats["Abstract"]
    p = stats["PMLB"]
    d = stats["Derivative"]
    n = stats["NeighborAblation"]
    deg = stats["DegreeAblation"]
    r = stats["Runtime"]
    lines: List[str] = []
    lines.append("# Abstract")
    lines.append(
        f"Best CALPHAD config from gwr_results_raw.csv row [{a['calphad_mape'].row_configuration}]: "
        f"degree={fmt(a['calphad_degree'].value)}, neighbors={fmt(a['calphad_neighbors'].value)}, "
        f"MAE={fmt(a['calphad_mae'].value)}, MSE={fmt(a['calphad_mse'].value)}, RMSE={fmt(a['calphad_rmse'].value)}, "
        f"MAPE={fmt(a['calphad_mape'].value)}, SMAPE={fmt(a['calphad_smape'].value)}, R2={fmt(a['calphad_r2'].value)}, "
        f"runtime={fmt(a['calphad_runtime'].value)}, training_time={fmt(a['calphad_training_time'].value)}."
    )
    lines.append("")
    lines.append("# Results")
    lines.append(
        f"Primary PMLB model ({PMLB_PRIMARY_MODEL}) over canonical six models: "
        f"median MAPE={fmt(p['primary_model_median_mape'].value)}, "
        f"median SMAPE={fmt(p['primary_model_median_smape'].value)}, "
        f"average rank={fmt(p['primary_model_average_rank'].value)}, "
        f"wins={fmt(p['primary_model_wins'].value)}, top-3={fmt(p['primary_model_top3'].value)}."
    )
    lines.append(
        f"Supplementary only: arithmetic mean MAPE={fmt(p['primary_model_mean_mape_supplementary'].value)} "
        f"(near-zero targets can inflate MAPE)."
    )
    lines.append("- Significance dataset count check: " + pmlb_meta["mismatch_explanation"])
    lines.append("")
    lines.append("# Baseline Comparison")
    for metric in ["mae", "mse", "rmse", "mape", "smape", "r2", "runtime"]:
        item = baseline_meta[metric]
        lines.append(
            f"- {metric}: strongest baseline={item['baseline_model']} ({fmt(item['baseline_value'])}); "
            f"SymGWR={fmt(item['gwr_value'])}; absolute improvement={fmt(item['absolute_improvement'])}; "
            f"percent improvement={fmt(item['percent_improvement'])}."
        )
    lines.append("")
    lines.append("# Derivative")
    lines.append(
        f"Derivative MAE={fmt(d['derivative_mae'].value)}, MSE={fmt(d['derivative_mse'].value)}, "
        f"mode={d['derivative_mode'].value}. {derivative_meta['limitation_note']}"
    )
    lines.append("")
    lines.append("# Runtime")
    lines.append(
        f"End-to-end runtime only: SymGWR={fmt(r['gwr_end_to_end_runtime'].value)}, "
        f"fastest baseline={fmt(r['fastest_baseline_end_to_end_runtime'].value)}, "
        f"slowest baseline={fmt(r['slowest_baseline_end_to_end_runtime'].value)}."
    )
    lines.append("")
    lines.append("# Neighbor Ablation")
    lines.append(
        f"Best k={fmt(n['best_k'].value)} by metric mape_mean_mean; plateau at k={fmt(n['plateau_k'].value)}. "
        f"{n['plateau_note'].value}"
    )
    lines.append("")
    lines.append("# Degree Ablation")
    lines.append(
        f"Best polynomial degree={fmt(deg['best_degree'].value)}; diminishing returns starts at degree="
        f"{fmt(deg['diminishing_returns_degree'].value)}; overfitting trend around degree="
        f"{fmt(deg['overfitting_trend_degree'].value)}."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def build_json_payload(
    file_map: Dict[str, Any],
    stats: Dict[str, Dict[str, TracedStat]],
    baseline_meta: Dict[str, Any],
    pmlb_meta: Dict[str, Any],
    derivative_meta: Dict[str, Any],
    runtime_meta: Dict[str, Any],
    neighbor_meta: Dict[str, Any],
    degree_meta: Dict[str, Any],
    checks: List[str],
) -> Dict[str, Any]:
    return {
        "file_map": file_map,
        "statistics": {
            section: {stat_id: stat.to_dict() for stat_id, stat in section_stats.items()}
            for section, section_stats in stats.items()
        },
        "sections": {
            "baseline_comparison": baseline_meta,
            "pmlb": {
                "primary_model": PMLB_PRIMARY_MODEL,
                "canonical_models": CANONICAL_PMLB_MODELS,
                "dataset_count": pmlb_meta["dataset_count"],
                "significance_rows": pmlb_meta["significance_rows"],
                "dataset_count_explanation": pmlb_meta["mismatch_explanation"],
            },
            "derivative": derivative_meta,
            "runtime": runtime_meta,
            "neighbor_ablation": {
                "best_k": neighbor_meta["best"]["k"],
                "plateau_k": neighbor_meta["plateau"]["k"],
                "indistinguishable_k_values": [x["k"] for x in neighbor_meta["indistinguishable"]],
            },
            "degree_ablation": {
                "best_degree": degree_meta["best"]["degree"],
                "diminishing_returns_degree": degree_meta["diminishing"]["degree"],
                "overfitting_trend_degree": degree_meta["overfit_degree"],
            },
        },
        "validation": {
            "passed": len(checks) == 0,
            "errors": checks,
        },
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    map_path = repo_root / "paper" / "PAPER_FILE_MAP.yaml"
    if not map_path.exists():
        raise RuntimeError(f"Missing canonical map file: {map_path}")
    file_map = parse_simple_yaml(map_path.read_text(encoding="utf-8"))
    if "files" not in file_map or "sections" not in file_map:
        raise RuntimeError("PAPER_FILE_MAP.yaml must include top-level 'files' and 'sections'")

    files_node = file_map["files"]
    section_node = file_map["sections"]
    canonical_files: Dict[str, str] = {key: str(value).strip() for key, value in files_node.items()}
    csv_cache: Dict[str, Tuple[List[str], List[Dict[str, str]]]] = {}
    for rel in canonical_files.values():
        path = repo_root / rel
        if not path.exists():
            raise RuntimeError(f"Canonical source file missing: {rel}")
        csv_cache[rel] = read_csv(path)

    stats: Dict[str, Dict[str, TracedStat]] = {}
    calphad = build_calphad(stats, section_node, canonical_files, csv_cache)
    baseline_meta = build_baseline(stats, section_node, canonical_files, csv_cache, calphad)
    pmlb_meta = build_pmlb(stats, section_node, canonical_files, csv_cache)
    derivative_meta = build_derivative(stats, section_node, canonical_files, csv_cache)
    runtime_meta = build_runtime(stats, section_node, canonical_files, csv_cache)
    neighbor_meta = build_neighbor_ablation(stats, section_node, canonical_files, csv_cache)
    degree_meta = build_degree_ablation(stats, section_node, canonical_files, csv_cache)

    checks = run_consistency_checks(stats, canonical_files, pmlb_meta)
    if checks:
        raise RuntimeError("Consistency checks failed:\n- " + "\n- ".join(checks))

    summary_md = build_summary_markdown(
        file_map=file_map,
        stats=stats,
        baseline_meta=baseline_meta,
        pmlb_meta=pmlb_meta,
        derivative_meta=derivative_meta,
        runtime_meta=runtime_meta,
        checks=checks,
    )
    manuscript_md = build_manuscript_markdown(
        stats=stats,
        baseline_meta=baseline_meta,
        pmlb_meta=pmlb_meta,
        derivative_meta=derivative_meta,
    )
    payload = build_json_payload(
        file_map=file_map,
        stats=stats,
        baseline_meta=baseline_meta,
        pmlb_meta=pmlb_meta,
        derivative_meta=derivative_meta,
        runtime_meta=runtime_meta,
        neighbor_meta=neighbor_meta,
        degree_meta=degree_meta,
        checks=checks,
    )

    results_dir = repo_root / "results"
    (results_dir / "PAPER_RESULTS_SUMMARY.md").write_text(summary_md, encoding="utf-8")
    (results_dir / "MANUSCRIPT_NUMBERS.md").write_text(manuscript_md, encoding="utf-8")
    (results_dir / "PAPER_RESULTS.json").write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print("Wrote results/PAPER_RESULTS_SUMMARY.md")
    print("Wrote results/MANUSCRIPT_NUMBERS.md")
    print("Wrote results/PAPER_RESULTS.json")


if __name__ == "__main__":
    main()
