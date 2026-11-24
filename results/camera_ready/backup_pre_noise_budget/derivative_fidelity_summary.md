# Real-CALPHAD derivative fidelity (degree-5 SymGWR polynomial)

Script: `experiments/run_calphad_grid_fd_gradient.py`. Data: deduped `data/raw/718_all.csv` (7,938 rows = 98 compositions x 81 T).
Fit: `get_split("random", seed)` for seeds 42-51, degree-5 polynomial on train+val (6,350 rows), minimum-norm solution.
Reference: central finite differences of the raw G table. Model: analytic polynomial gradient at the same point, converted to raw units by the chain rule (`target_scale / feature_scale`).
All numbers are mean +/- std over 10 seeds (population std). Files: `derivative_fidelity.csv` (per seed/axis/subset/estimator), `derivative_fidelity_summary.csv`, `derivative_identifiability.csv` (extra, backs section 4).

## 1. dG/dT
- n = 98 lines x 79 interior points = **7,742**. Matches the expected count.
- Units J/(mol K). Typical |dG/dT| = 94.33.

| subset | n | MAE | normalized error | Pearson r |
|---|---|---|---|---|
| all grid points | 7,742 | 0.0501 +/- 0.0015 | 5.3e-4 +/- 1.6e-5 | 0.9999 |
| test points only | 1,549 (mean) | 0.0501 +/- 0.0022 | 5.3e-4 +/- 2.3e-5 | 0.9999 |

## 2. Composition axes (units J/mol per unit mass fraction)
Ni skipped: constant (0.53), no finite difference exists.
Only points with equally spaced neighbours on both sides are used. Steps are h = 0.01 (Nb, Ti, Mo, Cr, most of Fe), 0.02 (Al), and 0.005 at the Fe base value (0.195).
Al and Fe have irregular grids near the base value, so only 8 of 12 grid values qualify (648 points each).

All grid points:

| axis | n | MAE | typical abs dG/dc | normalized error | Pearson r |
|---|---|---|---|---|---|
| Nb | 1,944 | 359.1 +/- 1.9 | 27,150 | 1.32% +/- 0.007% | 0.9986 |
| Al | 648 | 1,451 +/- 37 | 93,770 | 1.55% +/- 0.04% | 0.9997 |
| Ti | 1,539 | 813.7 +/- 6.2 | 60,550 | 1.34% +/- 0.01% | 0.9996 |
| Mo | 1,539 | 290.8 +/- 0.9 | 7,630 | 3.81% +/- 0.01% | 0.9978 |
| Fe | 648 | 421.3 +/- 17.8 | 4,262 | **9.89% +/- 0.42%** | 0.9907 |
| Cr | 729 | 467.2 +/- 15.9 | 16,870 | 2.77% +/- 0.09% | 0.9868 |

## 3. Fit-data points vs held-out points
Test points = FD points whose centre row is not in that seed's train+val.

| axis | n (mean) | MAE | normalized error | Pearson r |
|---|---|---|---|---|
| T | 1,549 | 0.0501 +/- 0.0022 | 0.053% +/- 0.002% | 0.9999 |
| Nb | 383.5 | 348.0 +/- 25.6 | 1.29% +/- 0.09% | 0.9986 |
| Al | 129.0 | 1,450 +/- 74 | 1.54% +/- 0.07% | 0.9997 |
| Ti | 308.3 | 822.3 +/- 33.8 | 1.36% +/- 0.05% | 0.9996 |
| Mo | 306.9 | 289.2 +/- 22.9 | 3.86% +/- 0.19% | 0.9979 |
| Fe | 134.1 | 421.8 +/- 30.7 | 10.09% +/- 0.33% | 0.9904 |
| Cr | 149.1 | 458.8 +/- 18.2 | 2.71% +/- 0.12% | 0.9870 |

- Held-out errors match the all-grid errors within about one seed-std on every axis. The optimism concern does not show up here.
- Caveat: with a random row split, every held-out point still has fit points next to it in the same (T, c) plane. This is an interpolation test, not a held-out-composition test.
- Estimator `gwr_k1` (gradient at the nearest fit point, the convention of the old check) is also in the CSVs. It equals `poly` on fit points. On held-out points the dG/dT normalized error is 3.1e-3 vs 5.3e-4 for `poly`. Composition axes are within about 0.3 percentage points of `poly`.

## 4. What the data can and cannot identify
- Design rank on the fit rows (centred, bias-free, what the minimum-norm solve sees) = **95 of 1,286** columns in all 10 seeds. That is 96 of 1,287 with the bias column, matching `design_audit.json`.
- Singular-value gap is clean: smallest retained 3.0, largest dropped <= 2e-10. 96 = 6 (T-only monomials, shared) + 6 planes x 15 (monomials containing that element). The data fills six (T, c_e) planes that meet on the base T-line.
- `null_frac` = share of the derivative functional's norm in the null space of the design. That part is not determined by the data, and minimum-norm sets it to zero. `null_frac` is basis-dependent in size, but ~0 vs clearly nonzero is invariant.

| derivative | evaluated at | null_frac | verdict |
|---|---|---|---|
| d/dT | any plane | <= 1e-12 | identified |
| d/dc_e, d2/dc_e2, d2/dc_e dT | on plane e | <= 1e-12 | identified (in-plane only) |
| d/dNi (any order) | anywhere | 1.0 | **not identifiable**. The polynomial returns ~0 (|grad| ~ 1e-13) only because min-norm zeroes unconstrained terms. |
| d/dc_i, i != e | plane e, c_e off base | 0.71-0.999 | **not identified** (the part of the gradient from Nb x Al etc. cross monomials is free) |
| d2/dc_i dc_j, i != j | plane e | 0.91-0.998 | **not identified** |
| any derivative off the six planes | | not computed | not identified |

- At the base T-line every plane meets, so each element's own first derivative is identified there. Cross-element mixed terms still are not.
- Because the function restricted to the planes is fully determined, the minimum-norm choice cannot change any in-plane derivative. Only model bias and noise from the degree-5 form remain.
- Composition-gradient accuracy in section 2 is therefore a statement about **own-axis derivatives on the sweep lines only**. Nothing is claimed for cross-element gradients, Ni, or interior compositions.
- Poorly fitted axes (in-plane, so this is polynomial bias, not non-identifiability):
  - **Fe**: normalized error ~10%, lowest r (0.991) and the largest seed spread.
  - **Cr**: r = 0.987.
  - **Mo**: 3.8%.
- Al caveat: its FD step is h = 0.02. The stencil truncation of the polynomial itself (`stencil_trunc_mae` = 770 of the 1,451 MAE) is as large as half the error. The Al error is limited by the reference as much as by the model. Using the same stencil on the polynomial values (`fd_vs_polystencil_mae`) gives 791, or 0.84% normalized.
- For Nb, Mo, Cr, Ti and Fe, stencil truncation is small (32-158) relative to the MAE (291-814), so those errors are polynomial bias.

## 5. Seed sensitivity
- Std over seeds of the per-seed normalized error (as a fraction, all-grid): T 1.6e-5; Nb 7e-5, Ti 1.0e-4, Mo 1.2e-4, Al 3.9e-4; Cr 9.4e-4; Fe 4.2e-3.
- Pointwise cross-seed std of the analytic gradient, relative to typical |FD| (all-grid points): T 7e-5, Nb 7e-4, Al 6e-4, Ti 5e-4, Mo 2.2e-3, Cr 1.7e-3, Fe 7.3e-3. Fe varies most between seeds.
- This is expected: the in-plane function is fixed by the data, so the minimum-norm freedom does not touch it.
- Low seed spread does **not** validate the unidentified directions. Min-norm is stable across seeds there too, but it is not data-determined.
- Note: all seeds fit ~80% of the same rows. Seed spread measures sensitivity to the fit subset, not to the dataset.

## Notes
- No old check outputs exist in the repo, so there is no before/after comparison. The old script used a non-deduped table, scalers fit on all rows and seed 0.
