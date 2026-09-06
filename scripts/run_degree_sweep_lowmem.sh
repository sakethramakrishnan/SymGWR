#!/usr/bin/env bash
# One (seed, degree) per process — minimal peak RAM, safe to interrupt/resume.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
LOG="${LOG:-/tmp/degree_sweep_lowmem.log}"

echo "Starting ultra-low-memory degree sweep; log=$LOG"
for seed in $(seq 0 9); do
  for degree in $(seq 1 15); do
    echo "=== seed=$seed degree=$degree ===" | tee -a "$LOG"
    nice -n 10 "$PY" -m experiments.run_corrected_sweeps \
      --degree-only \
      --seeds "$seed" \
      --degrees "$degree" \
      2>&1 | tee -a "$LOG"
  done
done

echo "=== plotting ===" | tee -a "$LOG"
"$PY" -m experiments.run_corrected_sweeps --plot-only --degree-only 2>&1 | tee -a "$LOG"
echo "Done."
