#!/usr/bin/env python
"""Collate final manuscript numbers from canonical result files only."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _fmt_mean_std(mean_value: float, std_value: float) -> str:
    return f"{mean_value:.6g} +- {std_value:.6g}"


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"

    calphad_summary_path = results_dir / "calphad_repeats_summary.csv"
    full_baseline_summary_path = results_dir / "full_baseline_summary.csv"
    full_baseline_improvement_path = results_dir / "full_baseline_improvement.csv"
    joint_grid_summary_path = results_dir / "joint_grid_summary.csv"
    timing_path = results_dir / "gwr_timing_detail.csv"
    pmlb_metrics_path = results_dir / "pmlb_metrics_summary.csv"
    pmlb_significance_path = results_dir / "pmlb_significance_tests.csv"
    derivative_synth_path = results_dir / "derivative_validation_synthetic.csv"
    interpolation_check_path = results_dir / "interpolation_check.csv"
    manuscript_numbers_path = results_dir / "MANUSCRIPT_NUMBERS.md"

    required = [
        calphad_summary_path,
        full_baseline_summary_path,
        full_baseline_improvement_path,
        joint_grid_summary_path,
        timing_path,
        pmlb_metrics_path,
        pmlb_significance_path,
        derivative_synth_path,
        interpolation_check_path,
    ]
    for path in required:
        _require_file(path)

    calphad_summary = pd.read_csv(calphad_summary_path)
    baseline_summary = pd.read_csv(full_baseline_summary_path)
    baseline_improvement = pd.read_csv(full_baseline_improvement_path)
    joint_grid_summary = pd.read_csv(joint_grid_summary_path)
    timing_df = pd.read_csv(timing_path)
    _ = pd.read_csv(pmlb_metrics_path)
    pmlb_significance = pd.read_csv(pmlb_significance_path)
    derivative_synth = pd.read_csv(derivative_synth_path)
    interpolation_check = pd.read_csv(interpolation_check_path)

    if len(calphad_summary) != 1:
        raise RuntimeError("Expected exactly one row in calphad_repeats_summary.csv")
    if len(joint_grid_summary) < 1:
        raise RuntimeError("Expected at least one row in joint_grid_summary.csv")
    if len(interpolation_check) != 1:
        raise RuntimeError("Expected exactly one row in interpolation_check.csv")

    cal = calphad_summary.iloc[0]
    interp = interpolation_check.iloc[0]

    best_idx = joint_grid_summary["mape_mean"].idxmin()
    best_joint = joint_grid_summary.loc[best_idx]
    best_degree = int(best_joint["degree"])
    best_k = int(best_joint["k"])
    match_4_2 = (best_degree == 4) and (best_k == 2)

    timing_single = timing_df.loc[timing_df["mode"] == "single_query"]
    if timing_single.empty:
        raise RuntimeError("gwr_timing_detail.csv missing mode=single_query row")
    timing_single_row = timing_single.iloc[0]

    gwr_row = baseline_summary.loc[baseline_summary["model"] == "Polynomial GWR"]
    if gwr_row.empty:
        raise RuntimeError("full_baseline_summary.csv missing model=Polynomial GWR")
    gwr_row = gwr_row.iloc[0]
    gwr_mape_mean = float(gwr_row["mape_mean"])
    gwr_mape_std = float(gwr_row["mape_std"])

    baseline_models = [
        "CALPHAD_NN",
        "GaussianProcess",
        "KNN",
        "KernelRegression",
        "LOESS",
        "MLP",
    ]
    baseline_lines: list[str] = []
    for model in baseline_models:
        b_row = baseline_summary.loc[baseline_summary["model"] == model]
        if b_row.empty:
            raise RuntimeError(f"full_baseline_summary.csv missing model={model}")
        b_row = b_row.iloc[0]

        imp_row = baseline_improvement.loc[
            (baseline_improvement["model"] == model) & (baseline_improvement["metric"] == "mape")
        ]
        if imp_row.empty:
            raise RuntimeError(f"full_baseline_improvement.csv missing model={model}, metric=mape")
        imp_row = imp_row.iloc[0]

        baseline_lines.append(
            "- "
            f"{model}: MAPE={_fmt_mean_std(float(b_row['mape_mean']), float(b_row['mape_std']))} "
            f"(source: results/full_baseline_summary.csv, cols=mape_mean,mape_std); "
            f"Polynomial GWR MAPE={_fmt_mean_std(gwr_mape_mean, gwr_mape_std)} "
            f"(source: results/full_baseline_summary.csv, model=Polynomial GWR); "
            f"SymGWR absolute improvement={float(imp_row['absolute_improvement']):.6g}, "
            f"percent improvement={float(imp_row['percent_improvement']):.6g}% "
            f"(source: results/full_baseline_improvement.csv, model={model}, metric=mape)."
        )

    pmlb_lines: list[str] = []
    for _, row in pmlb_significance.iterrows():
        pmlb_lines.append(
            "- "
            f"{row['baseline_model']} vs {row['gwr_model']}: "
            f"mean_rank_delta={float(row['mean_rank_delta']):.6g}, "
            f"mean_delta_mape={float(row['mean_delta_mape']):.6g}, "
            f"n_datasets={int(row['n_datasets'])}, "
            f"p_value_two_sided={float(row['p_value_two_sided']):.6g} "
            f"(source: results/pmlb_significance_tests.csv)."
        )

    dim_oor_value = interp["out_of_range_dimensions"]
    if pd.isna(dim_oor_value):
        dim_oor_text = "none"
    else:
        dim_oor = str(dim_oor_value).strip()
        dim_oor_text = dim_oor if dim_oor else "none"

    synth_n_funcs = int(derivative_synth["function"].nunique())
    synth_n_rows = int(len(derivative_synth))

    lines = [
        "# MANUSCRIPT_NUMBERS",
        "",
        "Normalization status: canonical result CSVs do not contain an explicit normalization flag; "
        "status remains not explicitly encoded in these files "
        "(sources: results/calphad_repeats_summary.csv, results/full_baseline_summary.csv, "
        "results/joint_grid_summary.csv, results/pmlb_metrics_summary.csv).",
        "",
        "Val-selected config from joint grid: "
        f"best_by_mape=(degree={best_degree}, k={best_k}) using `mape_mean` minimum "
        f"(source: results/joint_grid_summary.csv, cols=degree,k,mape_mean). "
        f"Matches degree=4/k=2: {match_4_2}.",
        "",
        "CALPHAD headline (n=10): "
        f"MAPE={_fmt_mean_std(float(cal['mape_mean']), float(cal['mape_std']))}, "
        f"MAE={_fmt_mean_std(float(cal['mae_mean']), float(cal['mae_std']))}, "
        f"RMSE={_fmt_mean_std(float(cal['rmse_mean']), float(cal['rmse_std']))}, "
        f"R2={_fmt_mean_std(float(cal['r2_mean']), float(cal['r2_std']))} "
        "(source: results/calphad_repeats_summary.csv, cols=*mean,*std,n_repeats).",
        "",
        "Baselines (n=5 each):",
        *baseline_lines,
        "",
        "Canonical inference time: "
        f"mean inference_time_s={float(cal['inference_time_s_mean']):.6g} +- {float(cal['inference_time_s_std']):.6g} "
        "(source: results/calphad_repeats_summary.csv, cols=inference_time_s_mean,inference_time_s_std).",
        "",
        "Additional timing benchmark: "
        f"single_query mean_us={float(timing_single_row['mean_us']):.6g}, "
        f"std_us={float(timing_single_row['std_us']):.6g}, p95_us={float(timing_single_row['p95_us']):.6g} "
        "(source: results/gwr_timing_detail.csv, row mode=single_query).",
        "",
        "PMLB rank deltas:",
        *pmlb_lines,
        "",
        "Interpolation regime confirmation: "
        f"fraction_test_fully_inside_training_bbox={float(interp['fraction_test_fully_inside_training_bbox']):.6g}, "
        f"fraction_with_any_out_of_range_dimension={float(interp['fraction_test_with_any_out_of_range_dimension']):.6g}, "
        f"out_of_range_dimensions={dim_oor_text} "
        "(source: results/interpolation_check.csv).",
        "",
        "Synthetic derivative validation artifact present: "
        f"{synth_n_funcs} functions, {synth_n_rows} rows "
        "(source: results/derivative_validation_synthetic.csv).",
        "",
        "CALPHAD query time: CALPHAD_QUERY_TIME_US = None [AWAITING CO-AUTHOR]",
        "",
    ]

    manuscript_numbers_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {manuscript_numbers_path}")


if __name__ == "__main__":
    main()
