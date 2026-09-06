#!/usr/bin/env bash
# Parallel k-sweep by seed; merge checkpoints into one CSV.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
MAIN_CSV="results/k_sweep_families_n10.csv"
PART_DIR="results/k_sweep_families_parts"
mkdir -p "$PART_DIR"
START=$(date +%s)

run_seed() {
  local seed="$1"
  local part="$PART_DIR/seed_${seed}.csv"
  if [[ -f "$part" ]] && [[ $(wc -l < "$part") -ge 181 ]]; then
    echo "seed=$seed already complete"
    return 0
  fi
  nice -n 15 "$PY" -u -m experiments.run_joint_grid \
    --k-sweep \
    --seeds "$seed" \
    --k-sweep-raw-output "$part"
}

pids=()
for seed in $(seq 1 9); do
  run_seed "$seed" &
  pids+=($!)
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

"$PY" << 'PY'
import pandas as pd
from pathlib import Path

main = Path("results/k_sweep_families_n10.csv")
part_dir = Path("results/k_sweep_families_parts")
parts = [main]
for p in sorted(part_dir.glob("seed_*.csv")):
    parts.append(p)

frames = []
for p in parts:
    if p.exists() and p.stat().st_size > 0:
        frames.append(pd.read_csv(p))

if not frames:
    raise SystemExit("No k-sweep CSV parts found")

merged = pd.concat(frames, ignore_index=True)
merged = merged.drop_duplicates(
    subset=["base_model", "degree", "k", "random_state", "eval_split"],
    keep="last",
)
merged = merged.sort_values(
    ["random_state", "base_model", "degree", "k"]
).reset_index(drop=True)
merged.to_csv(main, index=False)
print(f"Merged {len(merged)} rows -> {main}")
PY

"$PY" -u -m experiments.run_joint_grid --plot-only
END=$(date +%s)
echo "Parallel k-sweep wall time: $((END-START))s"
