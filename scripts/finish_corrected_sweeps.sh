#!/usr/bin/env bash
# Complete corrected sweeps outside Cursor. Safe to interrupt/resume.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
LOG="${LOG:-/tmp/finish_corrected_sweeps.log}"
START=$(date +%s)

exec > >(tee -a "$LOG") 2>&1
echo "=== finish_corrected_sweeps start $(date) ==="

echo "--- k-sweep (skip if checkpoint complete) ---"
nice -n 15 "$PY" -m experiments.run_corrected_sweeps --k-only

echo "--- degree-sweep: one (seed,degree) per process ---"
for seed in $(seq 0 9); do
  for degree in $(seq 1 15); do
    echo "--- seed=$seed degree=$degree $(date) ---"
    nice -n 15 "$PY" -m experiments.run_corrected_sweeps \
      --degree-only \
      --seeds "$seed" \
      --degrees "$degree"
  done
done

echo "--- figures ---"
"$PY" -m experiments.run_corrected_sweeps --plot-only

END=$(date +%s)
echo "=== finish_corrected_sweeps done $(date); wall=$((END-START))s ==="
