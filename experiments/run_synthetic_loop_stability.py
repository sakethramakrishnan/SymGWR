#!/usr/bin/env python
"""Synthetic iterative-loop stability demo using locked GWR model."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gwr.core import GradientWeightedRegressor
from gwr.models import fit_polynomial_regression, model_to_sympy_equation
from gwr.utils import load_dataset, train_val_test_split_df


@dataclass
class RunOutcome:
    seed: int
    start_id: int
    outcome: str
    n_iters_executed: int
    in_domain_iterations: int
    out_of_domain_events: int
    converged: bool


def _fit_locked_gwr(
    df: pd.DataFrame,
    target_col: str,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    model_seed: int,
) -> tuple[GradientWeightedRegressor, np.ndarray, np.ndarray]:
    X_train, X_val, _X_test, y_train, y_val, _y_test = train_val_test_split_df(
        df,
        target_col=target_col,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        random_state=model_seed,
    )
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    base_model, _ = fit_polynomial_regression(X_train_full, y_train_full, X_test=None, degree=5)
    sympy_eq, x_symbols = model_to_sympy_equation(base_model)
    gwr = GradientWeightedRegressor(
        sympy_equation=sympy_eq,
        x_symbols=x_symbols,
        n_neighbors=1,
        relative=False,
    )
    gwr.fit(X_train_full, y_train_full)
    bbox_min = np.min(X_train_full, axis=0)
    bbox_max = np.max(X_train_full, axis=0)
    return gwr, bbox_min, bbox_max


def _in_bbox(x: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray) -> bool:
    return bool(np.all(x >= bbox_min) and np.all(x <= bbox_max))


def _write_companion(
    out_path: str,
    raw_path: str,
    fig_path: str,
    outcome_df: pd.DataFrame,
    total_rows: int,
    in_domain_frac: float,
    max_iter: int,
    n_starts: int,
    n_seeds: int,
    split_protocol: str,
    sanity_note: str,
    selected_alpha: float,
    damping_grid_summary: pd.DataFrame,
    damping_note: str,
    degenerate_dims_note: str,
) -> None:
    lines: list[str] = []
    lines.append(f"Source: {raw_path}")
    lines.append("Columns used: seed, start_id, iteration, predicted_energy, grad_norm, update_in_domain, run_outcome")
    lines.append(f"Figure: {fig_path}")
    lines.append(f"Locked model config: degree=5, k=1 (trained once at split seed=0 using protocol {split_protocol})")
    lines.append(f"Selected damping alpha for primary output files: {selected_alpha}")
    lines.append(f"Loop settings: seeds={n_seeds}, starts_per_seed={n_starts}, max_iterations={max_iter}")
    lines.append(
        "Limitation: This is a SYNTHETIC iterative query-loop test only. "
        "It is NOT a physical PF-solver-in-the-loop validation."
    )
    lines.append(
        "Domain handling: updates that leave training bounding box are recorded as out-of-domain events; "
        "the next point is projected back into the bounding box so subsequent queries remain in-domain."
    )
    lines.append(f"Total per-iteration rows recorded: {total_rows}")
    lines.append(f"In-domain iteration fraction: {in_domain_frac:.6f}")
    lines.append("Run outcomes:")
    for row in outcome_df.itertuples(index=False):
        lines.append(f"- {row.outcome}: {int(row.count)} runs ({float(row.fraction):.6f})")
    lines.append("Damping-grid outcomes (all tested, no cherry-picking):")
    for row in damping_grid_summary.itertuples(index=False):
        lines.append(
            f"- alpha={float(row.alpha):.4g}: in_domain_fraction={float(row.in_domain_fraction):.6f}, "
            f"convergence_rate={float(row.convergence_rate):.6f}, "
            f"out_of_domain_update_fraction={float(row.out_of_domain_update_fraction):.6f}"
        )
    lines.append(damping_note)
    lines.append(degenerate_dims_note)
    lines.append(sanity_note)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


_OUTCOME_SHORT_LABELS: dict[str, str] = {
    "converged": "Converged",
    "max_iter_no_convergence": "No conv.",
    "max_iter_no_convergence_with_out_of_domain_events": "OOD events",
    "numerical_failure": "Num. fail",
}


def _plot(
    raw_df: pd.DataFrame,
    outcome_counts: pd.DataFrame,
    damping_grid_summary: pd.DataFrame,
    out_png: str,
    n_trajectories: int = 12,
) -> None:
    del raw_df, n_trajectories  # trajectory panel removed; args kept for API stability

    fig, axes = plt.subplots(2, 1, figsize=(3.6, 5.8))

    short_labels = [
        _OUTCOME_SHORT_LABELS.get(outcome, outcome.replace("_", " "))
        for outcome in outcome_counts["outcome"]
    ]
    axes[0].bar(short_labels, outcome_counts["count"])
    axes[0].set_title("Run outcomes across all seeds/starts")
    axes[0].set_xlabel("Outcome")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, axis="y", alpha=0.25)

    note_lines = [
        f"{short}: {full}" for full, short in zip(outcome_counts["outcome"], short_labels, strict=True)
    ]
    axes[0].text(
        0.5,
        -0.34,
        "Outcome codes: " + "; ".join(note_lines),
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        wrap=True,
    )

    grid = damping_grid_summary.sort_values("alpha")
    axes[1].plot(grid["alpha"], grid["in_domain_fraction"], "o-", label="in-domain fraction")
    axes[1].plot(grid["alpha"], grid["convergence_rate"], "s-", label="convergence rate")
    axes[1].set_xscale("log")
    axes[1].set_title("Damping grid trend")
    axes[1].set_xlabel("alpha (log scale)")
    axes[1].set_ylabel("Fraction")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.subplots_adjust(hspace=0.48, bottom=0.16)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic iterative-loop GWR stability demo")
    parser.add_argument("--data", type=str, default="data/raw/718_all.csv")
    parser.add_argument("--target", type=str, default="Gibbs_energy")
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-starts", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--damping-grid", type=str, default="1.0,0.5,0.1,0.01")
    parser.add_argument("--selected-alpha", type=float, default=0.01)
    parser.add_argument("--grad-tol", type=float, default=1e-4)
    parser.add_argument("--degenerate-width-eps", type=float, default=1e-12)
    parser.add_argument("--output-csv", type=str, default="results/synthetic_loop_stability.csv")
    parser.add_argument("--output-figure", type=str, default="figures/synthetic_loop_stability.png")
    parser.add_argument("--output-txt", type=str, default="figures/synthetic_loop_stability.txt")
    parser.add_argument(
        "--output-report",
        type=str,
        default="results/task2_synthetic_loop_summary_report.txt",
    )
    parser.add_argument(
        "--grid-output-csv",
        type=str,
        default="results/synthetic_loop_stability_damping_grid.csv",
    )
    parser.add_argument("--no-scale", action="store_true")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Re-render figure from existing CSV outputs without re-running the loop.",
    )
    args = parser.parse_args()

    if args.plot_only:
        raw_df = pd.read_csv(args.output_csv)
        damping_grid_df = pd.read_csv(args.grid_output_csv)
        outcome_df = (
            raw_df[["seed", "start_id", "run_outcome"]]
            .drop_duplicates()
            .rename(columns={"run_outcome": "outcome"})
        )
        outcome_counts = (
            outcome_df.groupby("outcome", as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values("count", ascending=False)
            .reset_index(drop=True)
        )
        os.makedirs(os.path.dirname(args.output_figure) or ".", exist_ok=True)
        _plot(raw_df, outcome_counts, damping_grid_df, args.output_figure)
        print(f"Saved: {args.output_figure}")
        return

    _, _, df = load_dataset(args.data, args.target, scale=not args.no_scale)
    split_protocol = f"{args.train_frac:.1f}/{args.val_frac:.1f}/{args.test_frac:.1f}"
    gwr, bbox_min, bbox_max = _fit_locked_gwr(
        df=df,
        target_col=args.target,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        model_seed=args.model_seed,
    )

    feature_count = bbox_min.shape[0]
    bbox_width = bbox_max - bbox_min
    mutable_mask = bbox_width > args.degenerate_width_eps
    damping_grid = [float(x.strip()) for x in args.damping_grid.split(",") if x.strip()]
    if not damping_grid:
        raise ValueError("Damping grid must include at least one alpha.")
    if args.selected_alpha not in damping_grid:
        raise ValueError("selected-alpha must be one of damping-grid values.")

    # Precompute identical starting points for all alphas.
    start_points: dict[tuple[int, int], np.ndarray] = {}
    for seed in range(args.n_seeds):
        rng = np.random.default_rng(seed)
        for start_id in range(args.n_starts):
            start_points[(seed, start_id)] = rng.uniform(low=bbox_min, high=bbox_max)

    def run_for_alpha(alpha: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        iter_rows: list[dict[str, float | int | bool | str]] = []
        outcomes: list[RunOutcome] = []
        for seed in range(args.n_seeds):
            for start_id in range(args.n_starts):
                x = start_points[(seed, start_id)].copy()
                outcome = "max_iter_no_convergence"
                in_domain_iters = 0
                out_of_domain_events = 0
                numerical_failure = False
                converged_iter: int | None = None
                projected_on_boundary_steps = 0

                for it in range(args.max_iter):
                    pred_arr, grad_arr = gwr.predict_with_local_gradient(x.reshape(1, -1))
                    pred = float(pred_arr[0])
                    grad = grad_arr[0].astype(np.float64)
                    grad_norm = float(np.linalg.norm(grad))

                    finite_ok = bool(np.isfinite(pred) and np.all(np.isfinite(grad)))
                    if not finite_ok:
                        update_in_domain = True
                        numerical_failure = True
                        next_x = x.copy()
                    else:
                        if converged_iter is None and grad_norm < args.grad_tol:
                            converged_iter = it
                        if converged_iter is None:
                            step = -alpha * args.step_size * grad / (grad_norm + 1e-12)
                            # Keep zero-width bbox dimensions fixed; otherwise any nonzero
                            # movement is guaranteed to leave trusted domain.
                            step = step * mutable_mask.astype(np.float64)
                        else:
                            step = np.zeros_like(grad)
                        proposed_next_x = x + step
                        next_x = proposed_next_x
                        update_in_domain = _in_bbox(next_x, bbox_min, bbox_max)
                        if not update_in_domain:
                            out_of_domain_events += 1
                            next_x = np.clip(next_x, bbox_min, bbox_max)
                            if np.any(np.isclose(next_x, bbox_min) | np.isclose(next_x, bbox_max)):
                                projected_on_boundary_steps += 1

                    row: dict[str, float | int | bool | str] = {
                        "alpha": float(alpha),
                        "seed": seed,
                        "start_id": start_id,
                        "iteration": it,
                        "predicted_energy": pred,
                        "grad_norm": grad_norm,
                        "update_in_domain": bool(update_in_domain),
                        "query_in_domain": bool(_in_bbox(x, bbox_min, bbox_max)),
                        "projected_on_boundary": False,
                        "run_outcome": "",
                    }
                    for j in range(feature_count):
                        row[f"x{j}"] = float(x[j])
                        row[f"grad_x{j}"] = float(grad[j])
                    if not finite_ok:
                        row["projected_on_boundary"] = False
                    else:
                        row["projected_on_boundary"] = bool(
                            (not update_in_domain)
                            and np.any(np.isclose(next_x, bbox_min) | np.isclose(next_x, bbox_max))
                        )
                    iter_rows.append(row)

                    if finite_ok and update_in_domain:
                        in_domain_iters += 1
                    if not finite_ok:
                        break
                    x = next_x

                executed = it + 1
                if numerical_failure:
                    outcome = "numerical_failure"
                elif converged_iter is not None:
                    outcome = "converged"
                elif out_of_domain_events > 0:
                    outcome = "max_iter_no_convergence_with_out_of_domain_events"
                else:
                    outcome = "max_iter_no_convergence"
                converged = outcome == "converged"
                outcomes.append(
                    RunOutcome(
                        seed=seed,
                        start_id=start_id,
                        outcome=outcome,
                        n_iters_executed=executed,
                        in_domain_iterations=in_domain_iters,
                        out_of_domain_events=out_of_domain_events,
                        converged=converged,
                    )
                )
                for idx in range(len(iter_rows) - executed, len(iter_rows)):
                    iter_rows[idx]["run_outcome"] = outcome
                print(
                    f"alpha={alpha} seed={seed} start={start_id} "
                    f"outcome={outcome} iters={executed} boundary_proj_steps={projected_on_boundary_steps}"
                )

        return pd.DataFrame(iter_rows), pd.DataFrame([vars(o) for o in outcomes])

    grid_rows: list[dict[str, float | int | str]] = []
    selected_raw_df: pd.DataFrame | None = None
    selected_outcomes_df: pd.DataFrame | None = None
    for alpha in damping_grid:
        alpha_raw_df, alpha_outcomes_df = run_for_alpha(alpha)
        in_domain_fraction = float(alpha_raw_df["update_in_domain"].mean())
        out_domain_fraction = 1.0 - in_domain_fraction
        convergence_rate = float(alpha_outcomes_df["converged"].mean())
        boundary_projection_fraction = float(alpha_raw_df["projected_on_boundary"].mean())
        grid_rows.append(
            {
                "alpha": float(alpha),
                "n_runs": int(len(alpha_outcomes_df)),
                "n_iterations_rows": int(len(alpha_raw_df)),
                "in_domain_fraction": in_domain_fraction,
                "out_of_domain_update_fraction": out_domain_fraction,
                "convergence_rate": convergence_rate,
                "boundary_projection_fraction": boundary_projection_fraction,
            }
        )
        if np.isclose(alpha, args.selected_alpha):
            selected_raw_df = alpha_raw_df.copy()
            selected_outcomes_df = alpha_outcomes_df.copy()

    if selected_raw_df is None or selected_outcomes_df is None:
        raise RuntimeError("Selected alpha data missing after damping-grid run.")

    damping_grid_df = pd.DataFrame(grid_rows).sort_values("alpha").reset_index(drop=True)

    raw_df = selected_raw_df
    outcome_df = selected_outcomes_df
    outcome_counts = (
        outcome_df.groupby("outcome", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    outcome_counts["fraction"] = outcome_counts["count"] / max(len(outcome_df), 1)

    total_rows = int(len(raw_df))
    in_domain_frac = float(raw_df["update_in_domain"].mean()) if total_rows > 0 else float("nan")

    os.makedirs(os.path.dirname(args.grid_output_csv) or ".", exist_ok=True)
    damping_grid_df.to_csv(args.grid_output_csv, index=False)
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    raw_df.to_csv(args.output_csv, index=False)
    os.makedirs(os.path.dirname(args.output_figure) or ".", exist_ok=True)
    _plot(raw_df, outcome_counts, damping_grid_df, args.output_figure)

    sanity_note = (
        "Physical sanity check: no robust physically motivated monotonicity/convexity test was applied, "
        "because this synthetic setup does not include PF-solver physics constraints or additional domain assumptions."
    )
    damping_note = (
        "Update rule used: normalized gradient-descent style step, "
        "x_next = clip_to_bbox(x - alpha * step_size * grad / (||grad|| + 1e-12)). "
        "This is damped first-order descent, not Newton-Raphson (no Hessian / second derivative term)."
    )
    n_degenerate_dims = int(np.sum(~mutable_mask))
    degenerate_dims_note = (
        f"Degenerate bbox diagnostic: {n_degenerate_dims} feature dimension(s) had width <= {args.degenerate_width_eps:g} "
        "and were frozen during updates to avoid guaranteed out-of-domain jumps from zero-width axes."
    )
    os.makedirs(os.path.dirname(args.output_txt) or ".", exist_ok=True)
    _write_companion(
        out_path=args.output_txt,
        raw_path=args.output_csv,
        fig_path=args.output_figure,
        outcome_df=outcome_counts,
        total_rows=total_rows,
        in_domain_frac=in_domain_frac,
        max_iter=args.max_iter,
        n_starts=args.n_starts,
        n_seeds=args.n_seeds,
        split_protocol=split_protocol,
        sanity_note=sanity_note,
        selected_alpha=float(args.selected_alpha),
        damping_grid_summary=damping_grid_df,
        damping_note=damping_note,
        degenerate_dims_note=degenerate_dims_note,
    )

    summary_lines = [
        "Task 2 summary report",
        "Measured quantities:",
        "- Per-iteration queried point and gradient: results/synthetic_loop_stability.csv, columns x*, grad_x*",
        "- Predicted Gibbs energy trajectory: results/synthetic_loop_stability.csv, column predicted_energy",
        "- In-domain vs out-of-domain update flag: results/synthetic_loop_stability.csv, column update_in_domain",
        "- Convergence/divergence labels: results/synthetic_loop_stability.csv, column run_outcome",
        "- Damping-grid summary: results/synthetic_loop_stability_damping_grid.csv",
        "Aggregate outcomes:",
    ]
    for row in outcome_counts.itertuples(index=False):
        summary_lines.append(f"- {row.outcome}: count={int(row.count)}, fraction={float(row.fraction):.6f}")
    summary_lines.extend(
        [
            f"- In-domain iteration fraction: {in_domain_frac:.6f}",
            f"- Total iteration rows: {total_rows}",
            f"- Loop settings: seeds={args.n_seeds}, starts={args.n_starts}, max_iter={args.max_iter}",
            f"- Selected damping alpha for primary outputs: {args.selected_alpha}",
            f"- Locked model source convention: trained via run_calphad_repeats-style split {split_protocol}, degree=5, k=1.",
            "- Limitation: synthetic iterative query pattern only (NOT PF-solver integration/validation).",
            "- Update formula: x_next = clip_to_bbox(x - alpha * step_size * grad / (||grad|| + 1e-12)).",
            f"- Degenerate dimensions frozen in update when bbox width <= {args.degenerate_width_eps:g}; count={n_degenerate_dims}.",
        ]
    )
    summary_lines.append("Damping-grid trend:")
    for row in damping_grid_df.itertuples(index=False):
        summary_lines.append(
            f"- alpha={float(row.alpha):.4g}: in_domain_fraction={float(row.in_domain_fraction):.6f}, "
            f"convergence_rate={float(row.convergence_rate):.6f}, "
            f"out_of_domain_update_fraction={float(row.out_of_domain_update_fraction):.6f}, "
            f"boundary_projection_fraction={float(row.boundary_projection_fraction):.6f}"
        )

    os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"Saved: {args.output_csv}")
    print(f"Saved: {args.grid_output_csv}")
    print(f"Saved: {args.output_figure}")
    print(f"Saved: {args.output_txt}")
    print(f"Saved: {args.output_report}")


if __name__ == "__main__":
    main()
