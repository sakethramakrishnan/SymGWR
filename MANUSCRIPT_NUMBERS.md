# MANUSCRIPT_NUMBERS

## 1) Headline config
- Locked headline configuration: `(degree=5, k=1)` (binding per Section L).

## 2) Config-selection evidence (Section L, n=10 paired seeds)
- Compared configs from `results/joint_grid_n10_final.csv` (`mape`, filtered by `degree` and `k`), paired over `seed=0..9` on `split=val` (`n=10` pairs each config).
- `(degree=4, k=2)` mean val MAPE = `0.0009948425821698`, std(pop) = `0.00068578167108181` (`n=10`; source: `results/joint_grid_n10_final.csv`, col `mape`).
- `(degree=5, k=1)` mean val MAPE = `0.00075812705117366`, std(pop) = `0.000728888416446083` (`n=10`; source: `results/joint_grid_n10_final.csv`, col `mape`).
- Paired t-test on per-seed val MAPE: `t=3.450268635862887`, `p=0.007271628607374566` (`n=10`; source data: `results/joint_grid_n10_final.csv`, col `mape`).
- Wilcoxon signed-rank (paired): `p=0.00390625` (`W=1`, `n=10`; source data: `results/joint_grid_n10_final.csv`, col `mape`).

## 3) CALPHAD accuracy at headline config
- From `results/calphad_repeats_summary_locked.csv` (`n_repeats=10`, locked config `degree=5`, `k=1`, split=test, population std):
- MAPE = `0.0004352612152892793 +/- 0.00019466747891659235`.
- MAE = `7.16012741434021e-05 +/- 1.5420404882662349e-06`.
- RMSE = `9.877484822370553e-05 +/- 5.003970661158746e-06`.
- R2 = `0.9999999902625915 +/- 8.875178848411105e-10`.
- Held-out-vs-validation note (Sections 6.4/7.x): this held-out test-set MAPE differs from validation-split config-selection MAPE `7.5812705117366e-4` in Section 7.2b by design (validation drives selection; held-out test reports post-selection generalization).

## 4) Baseline comparison (Issue O locked numbers, no recompute)
- Source: `results/full_baseline_improvement.csv`, filtered to `metric=mape` (`n=5` for GWR row inherited from `full_baseline_summary.csv`).
- GWR vs `GaussianProcess` MAPE percent improvement = `59.67975374940865%` (`n=5`; source: `results/full_baseline_improvement.csv`, col `percent_improvement`).
- GWR vs `LOESS` MAPE percent improvement = `99.51286367840356%` (`n=5`; source: `results/full_baseline_improvement.csv`, col `percent_improvement`).
- GWR vs `KNN` MAPE percent improvement = `99.62332405721169%` (`n=5`; source: `results/full_baseline_improvement.csv`, col `percent_improvement`).
- GWR vs `KernelRegression` MAPE percent improvement = `99.76934676737747%` (`n=5`; source: `results/full_baseline_improvement.csv`, col `percent_improvement`).
- Documented compute-cost exception (Issue O/Q): baseline table remains `n=5` because GaussianProcess runtime cost made `n=10` baseline reruns impractical in that pass.

## 5) Derivative validation (synthetic v2)
- Source: `results/derivative_validation_synthetic_v2.csv`, grouped by `function`; mean over all dimension-seed rows (`n=10` seeds, `42..51`; 70 rows/function).
- `f1_quadratic`: mean MAE = `0.2159852743388997`, mean normalized error = `0.07441388210739075` (`n=10` seeds; source cols `mean_absolute_error`, `mean_normalized_error_gradnorm_eps1e-6`).
- `f2_mixed_trig_linear`: mean MAE = `0.1898697862620642`, mean normalized error = `0.2466551600802502` (`n=10` seeds; source cols `mean_absolute_error`, `mean_normalized_error_gradnorm_eps1e-6`).
- `f3_gaussian`: mean MAE = `0.04086134835707842`, mean normalized error = `0.1441509403191427` (`n=10` seeds; source cols `mean_absolute_error`, `mean_normalized_error_gradnorm_eps1e-6`).
- Seed range note: `42..51` is fixed analytic-diagnostic convention for this synthetic derivative check and is independent of real-data config seeds `0..9`; this is a decided convention item, not a rerun inconsistency.

## 6) Timing (single-query latency, locked v2)
- Source: `results/gwr_timing_detail_v2_locked.csv` (`locked_convention=single_canonical_run_distribution`, `n_queries=10000`, warmed single-query protocol, config `degree=5`, `n_neighbors=1`).
- Locked v2 canonical timing: mean = `65.3720727` us, p50 = `61.075` us, p95 = `94.47755` us, p99 = `114.9305100000001` us.
- Quantile monotonicity check: `quantiles_monotonic=True` (explicitly recorded in locked v2 file).
- Independent-process support (noise characterization only): `results/gwr_timing_runs_v2.csv` and across-run spread fields in locked v2 row.
- Hardware/environment disclosure for paper methodology note (from `results/gwr_timing_detail_v2_provenance.json`): CPU `AMD Ryzen 7 5700U with Radeon Graphics`; Python `3.14.4`; NumPy `2.5.1`; SciPy `1.18.0`; absolute latency values are hardware/environment dependent.
- Same-environment self-consistency check (complexity sweep at `n=8181`): `61.135827` us lies within v2 canonical range `[49.062, 184.828]` and within v2 `p50..p99` interval `[61.075, 114.9305100000001]` (source: `figures/complexity_scaling.txt`, `results/task1_complexity_summary_report.txt`).

## 7) PMLB results and significance
- Median MAPE/SMAPE summary from `results/pmlb_metrics_summary.csv` (canonical baseline/GWR models, each `n=213` datasets):
- Linear family: baseline median MAPE/SMAPE = `0.758672181874` / `0.492292632005` vs GWR median MAPE/SMAPE = `0.0965042869715` / `0.0871392082956` (source cols `mape_mean`, `smape_mean`; `n=213`).
- Ridge family: baseline median MAPE/SMAPE = `0.758659983647` / `0.4906714389` vs GWR median MAPE/SMAPE = `0.101000832059` / `0.0881274591022` (source cols `mape_mean`, `smape_mean`; `n=213`).
- Polynomial family: baseline median MAPE/SMAPE = `0.346519905539` / `0.282215561592` vs GWR median MAPE/SMAPE = `0.0979628551122` / `0.0721817218425` (source cols `mape_mean`, `smape_mean`; `n=213`).
- Significance stats from `results/pmlb_significance_tests_final.csv` (rank tests `n=213`, excl-nearzero-delta summaries `n=195`):
- `Linear Regression` vs `Linear GWR`: test_statistic=`16.1669847461`, p-value bound=`< 5e-324`, BH-FDR q=`4.94065645841e-324`, median_delta_mape (`n=213`) = `0.386757021832`, mean_delta_mape_excl_nearzero (`n=195`) = `1.516012567716656` (`n_for_rank_test=213`, `n_for_delta_excl_nearzero=195`).
- `Ridge Regression` vs `Ridge GWR`: test_statistic=`16.2498147523`, p-value bound=`< 5e-324`, BH-FDR q=`4.94065645841e-324`, median_delta_mape (`n=213`) = `0.386748792751`, mean_delta_mape_excl_nearzero (`n=195`) = `1.5141927054371715` (`n_for_rank_test=213`, `n_for_delta_excl_nearzero=195`).
- `Polynomial Regression` vs `Polynomial GWR`: test_statistic=`12.6159547708`, p-value bound=`< 5e-324`, BH-FDR q=`4.94065645841e-324`, median_delta_mape (`n=213`) = `0.106227088446`, mean_delta_mape_excl_nearzero (`n=195`) = `0.6256251767223034` (`n_for_rank_test=213`, `n_for_delta_excl_nearzero=195`).
- Median robustness check (paired delta_mape recompute with same 18-dataset near-zero exclusion list): Linear median_delta_mape `0.386757021832` (`n=213`) vs `0.323746282832` (`n=195`), Ridge `0.386748792751` (`n=213`) vs `0.323361206459` (`n=195`), Polynomial `0.106227088446` (`n=213`) vs `0.085457283181` (`n=195`). Relative shifts are about 16.29%, 16.39%, and 19.55%, respectively, so this should be treated as a denominator-sensitive discrepancy rather than a within-10% robustness confirmation.
- Coexistence rationale (Issue D): `median_delta_mape` and `mean_delta_mape_excl_nearzero` are distinct statistics with different denominators/estimands; cite both only with explicit labels (`median, n=213` vs `mean excl-near-zero, n=195`).

## 8) Known open limitations to state in manuscript
- Interpolation regime support sentence source: `results/interpolation_check_locked.csv` reports `fraction_test_fully_inside_training_bbox=1.0`, `fraction_test_with_any_out_of_range_dimension=0.0`, `out_of_range_dimensions=none` (100% inside training bounding box under locked `0.6/0.2/0.2` split).
- CALPHAD absolute single-query time is still unavailable (co-author reply pending, status per Issue R), so no CALPHAD-vs-GWR speedup ratio can be computed yet; only GWR absolute query-time numbers are currently reportable.
- Repository URL `https://github.com/sakethramakrishnan/gwr` still returns 404 (Issue R), so repo link remains blocked for manuscript citation until accessibility is fixed.

## 9) Tables 2/3 refresh (locked baseline table, n=10)
- Source: `results/baseline_table_locked.csv` (`n_repeats=10`, seeds `0..9`, split `0.6/0.2/0.2`).
- Population-std convention for manuscript consistency: Polynomial GWR std values use `results/calphad_repeats_summary_locked.csv` (ddof=0), not ddof=1 raw std from baseline table file.
- Polynomial GWR: MAPE `0.0004352612152892793 +/- 0.00019466747891659235`; MAE `7.16012741434021e-05 +/- 1.5420404882662349e-06`; RMSE `9.877484822370553e-05 +/- 5.003970661158746e-06`; R2 `0.9999999902625915 +/- 8.875178848411105e-10`.
- XGBoost: MAPE `0.022179736486445077 +/- 0.0159827035760513`.
- MLP: MAPE `0.025386994012371312 +/- 0.01232880331223052`.
- PINN: MAPE `0.05383090128325667 +/- 0.07082847733211478`.
- Decision Tree: MAPE `0.0896272467962253 +/- 0.03615896088116593`.
- PySR: MAPE `0.22833348391899597 +/- 0.07927726321561232`.

## 10) RQ4 t-test refresh (GWR vs best black-box baseline)
- Source: `results/rq4_ttest_locked.csv`.
- Best black-box baseline in refreshed table is `XGBoost` (lowest baseline MAPE).
- Welch two-sample t-test on pooled absolute percentage errors (`GWR - XGBoost`): `t=-4.658433533969057`, `p=3.211305696642254e-06`, `n=16370` per group.
- Direction note required in prose/caption: negative t-stat sign indicates GWR mean error is lower than XGBoost.
