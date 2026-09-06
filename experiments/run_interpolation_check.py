#!/usr/bin/env python
"""Fast interpolation-regime check via train bounding-box coverage."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gwr.utils import load_dataset, train_val_test_split_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Interpolation check on CALPHAD split.")
    parser.add_argument("--data", type=str, default="data/raw/718_all.csv")
    parser.add_argument("--target", type=str, default="Gibbs_energy")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/interpolation_check.csv")
    args = parser.parse_args()

    _, _, df = load_dataset(args.data, args.target, scale=False)
    feature_df = df.drop(columns=[args.target]).select_dtypes(include=[np.number])
    feature_names = list(feature_df.columns)

    X_train, _, X_test, _, _, _ = train_val_test_split_df(
        df,
        target_col=args.target,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        random_state=args.seed,
    )

    train_min = X_train.min(axis=0)
    train_max = X_train.max(axis=0)
    in_range = (X_test >= train_min[np.newaxis, :]) & (X_test <= train_max[np.newaxis, :])
    fully_inside_mask = np.all(in_range, axis=1)
    any_out_of_range_mask = ~fully_inside_mask

    out_by_dim = ~np.all(in_range, axis=0)
    out_dims = [feature_names[i] for i, flag in enumerate(out_by_dim) if flag]

    out_df = pd.DataFrame(
        [
            {
                "data_file": args.data,
                "target": args.target,
                "seed": args.seed,
                "train_frac": args.train_frac,
                "val_frac": args.val_frac,
                "test_frac": args.test_frac,
                "n_train": int(X_train.shape[0]),
                "n_test": int(X_test.shape[0]),
                "fraction_test_fully_inside_training_bbox": float(np.mean(fully_inside_mask)),
                "fraction_test_with_any_out_of_range_dimension": float(np.mean(any_out_of_range_mask)),
                "n_test_fully_inside_training_bbox": int(np.sum(fully_inside_mask)),
                "n_test_with_any_out_of_range_dimension": int(np.sum(any_out_of_range_mask)),
                "n_dimensions": int(X_train.shape[1]),
                "n_dimensions_with_out_of_range_test_points": int(np.sum(out_by_dim)),
                "out_of_range_dimensions": ";".join(out_dims) if out_dims else "",
            }
        ]
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Saved interpolation check to {args.output}")


if __name__ == "__main__":
    main()
