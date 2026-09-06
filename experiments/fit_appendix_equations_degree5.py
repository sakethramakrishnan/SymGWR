#!/usr/bin/env python
"""Fit per-element degree-5 polynomial base equations in standardized feature space.

Each element cluster (nb, al, ti, mo, fe, cr) uses the active element mass
fraction plus temperature as a 2D polynomial regressor, matching the structure
of ``results/polynomial_training_progress.csv``.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gwr.utils import compute_all_metrics, load_dataset

ELEMENTS = {
    "nb": "Mass_fraction_Nb",
    "al": "Mass_fraction_Al",
    "ti": "Mass_fraction_Ti",
    "mo": "Mass_fraction_Mo",
    "fe": "Mass_fraction_Fe",
    "cr": "Mass_fraction_Cr",
}


def _format_equation(
    intercept: float,
    coefs: np.ndarray,
    feature_names: list[str],
    poly_names: list[str],
) -> str:
  parts = [f"{intercept:.16g}"]
  for coeff, term in zip(coefs, poly_names):
      if term == "1":
          continue
      sign = "+" if coeff >= 0 else "-"
      mag = abs(coeff)
      term_human = term
      for i, name in enumerate(feature_names):
          term_human = term_human.replace(f"x{i}", name)
      term_human = term_human.replace("^2", "**2").replace("^3", "**3")
      term_human = term_human.replace("^4", "**4").replace("^5", "**5")
      term_human = term_human.replace(" ", "*")
      parts.append(f" {sign} {mag:.16g}*{term_human}")
  return "Gibbs_energy = " + "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit appendix degree-5 element equations")
    parser.add_argument("--data", type=str, default="data/raw/718_all.csv")
    parser.add_argument("--target", type=str, default="Gibbs_energy")
    parser.add_argument("--degree", type=int, default=5)
    parser.add_argument(
        "--output",
        type=str,
        default="results/appendix_equations_degree5.csv",
    )
    parser.add_argument("--no-scale", action="store_true")
    args = parser.parse_args()

    _, _, df = load_dataset(args.data, args.target, scale=not args.no_scale)
    rows: list[dict[str, str | float | int]] = []

    for element, mass_col in ELEMENTS.items():
        subset = df[df["temp_class"] == element].copy()
        feature_cols = ["Temperature_[K]", mass_col]
        X = subset[feature_cols].to_numpy(dtype=np.float64)
        y = subset[args.target].to_numpy(dtype=np.float64)

        model = Pipeline(
            [
                ("poly_features", PolynomialFeatures(degree=args.degree, include_bias=False)),
                ("linear_model", LinearRegression()),
            ]
        )
        model.fit(X, y)
        y_pred = model.predict(X)
        metrics = compute_all_metrics(y, y_pred)

        poly = model.named_steps["poly_features"]
        linear = model.named_steps["linear_model"]
        poly_names = list(poly.get_feature_names_out(["x0", "x1"]))
        coefs = linear.coef_
        intercept = float(linear.intercept_)

        rows.append(
            {
                "element": element,
                "degree": args.degree,
                "n_points": len(subset),
                "mape": float(metrics["mape_mean"]),
                "intercept": intercept,
                "coefficients": ",".join(f"{c:.16g}" for c in coefs),
                "feature_names": "|".join(poly_names),
                "input_features": "|".join(feature_cols),
                "equation": _format_equation(intercept, coefs, feature_cols, poly_names),
            }
        )
        print(f"{element}: n={len(subset)} mape={metrics['mape_mean']:.6g}")

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
