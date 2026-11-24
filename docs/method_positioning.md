# Method Positioning: SymGWR vs. MLS, Shepard, LOESS/LOWESS

This note positions Gradient-Weighted Regression (SymGWR) relative to nearby local-regression/interpolation families.

## Sources consulted

- Shepard (1968), inverse-distance interpolation.
- Lancaster and Salkauskas (1981) / standard MLS formulations.
- Cleveland (1979, 1981), LOWESS/LOESS local weighted regression.
- NIST/Statsmodels documentation for practical LOWESS behavior.

## Similarities

- **Locality**: all methods use nearby points to form predictions.
- **Distance weighting**: all use stronger influence for closer points.
- **Nonparametric behavior at inference**: prediction depends on local neighborhoods, not only one global parametric equation.

## Differences

### Compared to Shepard's method (IDW)

- Shepard predicts weighted averages of observed targets:
  \[
  \hat y(x)=\frac{\sum_i w_i(x) y_i}{\sum_i w_i(x)}.
  \]
- SymGWR predicts weighted averages of first-order extrapolations:
  \[
  \hat y(x)=\frac{\sum_i w_i(x)\left(y_i+\nabla f(x_i)^\top(x-x_i)\right)}{\sum_i w_i(x)}.
  \]
- So SymGWR augments IDW with directional gradient correction from an explicit base model; Shepard does not include this directional term.

### Compared to MLS

- Classical MLS solves a **new local weighted least-squares problem at each query** to obtain query-dependent coefficients.
- Current SymGWR implementation does **not** solve a per-query local regression; it reuses gradients of a pre-fitted global closed-form base model and applies neighbor-wise first-order Taylor correction.
- Therefore, SymGWR is closer to **gradient-corrected inverse-distance local averaging** than to canonical MLS optimization.

### Compared to LOESS/LOWESS

- LOESS/LOWESS fits local polynomial regressions with kernel weights (often tricube), sometimes with robust reweighting.
- SymGWR does not do robust residual reweighting and does not refit local polynomials around each query.
- SymGWR depends on analytical gradients from a symbolic/closed-form base model; LOESS/LOWESS usually does not require symbolic gradients.

## Defensible novelty claims

- Combining:
  1) closed-form symbolic/global base model,
  2) analytical gradient extraction,
  3) neighbor-wise first-order correction,
  4) normalized inverse-distance aggregation
  into one lightweight regression pipeline for this CALPHAD-surrogate use case.
- Practical contribution: an interpretable, differentiable surrogate workflow compatible with existing experiment pipelines and multiple base regressors.

## Claims that should be weakened

- Avoid claiming SymGWR is fundamentally unrelated to MLS/LOESS/IDW.
- Avoid claiming broad methodological novelty as an entirely new class of local regression.
- Prefer wording like:
  - "SymGWR is a gradient-augmented local weighting method related to IDW/MLS-style local methods."
  - "Main contribution is empirical + integration-focused: symbolic gradients plus local correction for differentiable surrogate modeling."

## Bottom line

Technically, SymGWR appears to be a hybrid local method closest to **IDW + first-order gradient correction**, with conceptual overlap to MLS/LOESS families. Novelty is most defensible as a **specific formulation and application-focused integration**, not as a wholly unprecedented local regression paradigm.
