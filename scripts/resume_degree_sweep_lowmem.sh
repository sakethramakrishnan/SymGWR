#!/usr/bin/env bash
# Serial low-memory resume for degree_sweep_matched_train_val_n10.csv.
# Run outside Cursor: nohup bash scripts/resume_degree_sweep_lowmem.sh &
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
LOG="${LOG:-/tmp/degree_sweep_matched_lowmem.log}"
echo "Starting serial degree sweep at $(date)" | tee -a "$LOG"
exec python experiments/run_degree_sweep_matched.py >>"$LOG" 2>&1
