#!/usr/bin/env bash
# Resume manuscript sweeps with memory-safe parallelism.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
NEIGHBOR_PART_DIR="results/neighbor_sweep_parts"
DEGREE_PART_DIR="results/degree_matched_parts"
mkdir -p "$NEIGHBOR_PART_DIR" "$DEGREE_PART_DIR"
LOG="${LOG:-/tmp/manuscript_sweeps.log}"
START=$(date +%s)

run_seed() {
  local seed="$1"
  shift
  local neighbor_part="$NEIGHBOR_PART_DIR/seed_${seed}.csv"
  local degree_part="$DEGREE_PART_DIR/seed_${seed}.csv"
  nice -n 15 "$PY" -u -m experiments.run_manuscript_neighbor_overfit \
    --seeds "$seed" \
    --skip-plot \
    "$@" \
    --neighbor-raw "$neighbor_part" \
    --degree-raw "$degree_part"
}

run_batch() {
  local parallel="$1"
  shift
  local seeds=("$@")
  local i=0
  while [[ $i -lt ${#seeds[@]} ]]; do
    local pids=()
    for _ in $(seq 1 "$parallel"); do
      if [[ $i -ge ${#seeds[@]} ]]; then
        break
      fi
      run_seed "${seeds[$i]}" &
      pids+=($!)
      i=$((i + 1))
    done
    for pid in "${pids[@]}"; do
      wait "$pid"
    done
  done
}

echo "=== Phase 1: neighbor sweep seeds 4-9 (parallel=2) ===" | tee -a "$LOG"
run_batch 2 4 5 6 7 8 9

echo "=== Phase 2: degree sweep seeds 0-3 degrees 14-15 (parallel=1) ===" | tee -a "$LOG"
for seed in 0 1 2 3; do
  run_seed "$seed" --degree-only --degrees 14-15
done

echo "=== Phase 3: degree sweep seeds 4-9 all degrees (parallel=1) ===" | tee -a "$LOG"
for seed in 4 5 6 7 8 9; do
  run_seed "$seed" --degree-only --degrees 1-15
done

"$PY" << 'PY'
import pandas as pd
from pathlib import Path

def merge_parts(part_dir: str, out_raw: str, dedupe_cols: list[str]) -> pd.DataFrame:
    frames = []
    for p in sorted(Path(part_dir).glob("seed_*.csv")):
        if p.stat().st_size > 0:
            frames.append(pd.read_csv(p))
    if not frames:
        raise SystemExit(f"No parts in {part_dir}")
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=dedupe_cols, keep="last")
    merged.to_csv(out_raw, index=False)
    print(f"Merged {len(merged)} rows -> {out_raw}")
    return merged

merge_parts(
    "results/neighbor_sweep_parts",
    "results/neighbor_sweep_all_regressors_raw_n10.csv",
    ["base_regressor_type", "degree", "k", "seed"],
)
merge_parts(
    "results/degree_matched_parts",
    "results/degree_sweep_matched_train_val_raw_n10.csv",
    ["degree", "seed"],
)
PY

"$PY" -u -m experiments.run_manuscript_neighbor_overfit --plot-only | tee -a "$LOG"
END=$(date +%s)
echo "Manuscript sweeps wall time: $((END-START))s" | tee -a "$LOG"
