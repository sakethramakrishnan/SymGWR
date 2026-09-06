| model | mae | mse | rmse | mape | smape | r2 | runtime_s | repeat | seed | source_row | source_file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GaussianProcess | 0.000278889 | 2.05425e-07 | 0.000453239 | 0.00106574 | 0.00105787 | 1 | 98.7524 | 1 | 42 | row_index=0, repeat=1, seed=42, model=GaussianProcess | results/baseline_comparison_raw.csv |
| MLP | 0.00362668 | 2.55985e-05 | 0.0050595 | 0.017699 | 0.0186167 | 0.999975 | 5.65308 | 1 | 42 | row_index=1, repeat=1, seed=42, model=MLP | results/baseline_comparison_raw.csv |
| CALPHAD_NN | 0.00433803 | 3.78213e-05 | 0.0061499 | 0.0198935 | 0.0175724 | 0.999963 | 5.83927 | 1 | 42 | row_index=2, repeat=1, seed=42, model=CALPHAD_NN | results/baseline_comparison_raw.csv |
| LOESS | 0.0140954 | 0.000386054 | 0.0196483 | 0.081625 | 0.0628102 | 0.999617 | 0.797735 | 1 | 42 | row_index=3, repeat=1, seed=42, model=LOESS | results/baseline_comparison_raw.csv |
| KNN | 0.017922 | 0.000550219 | 0.0234567 | 0.0917859 | 0.0637338 | 0.999455 | 0.0622022 | 1 | 42 | row_index=4, repeat=1, seed=42, model=KNN | results/baseline_comparison_raw.csv |
| KernelRegression | 0.118586 | 0.0281318 | 0.167725 | 0.255567 | 0.192563 | 0.972125 | 0.227076 | 1 | 42 | row_index=5, repeat=1, seed=42, model=KernelRegression | results/baseline_comparison_raw.csv |
