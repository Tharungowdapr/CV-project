#!/usr/bin/env bash
# Watchdog wrapper for training on a single consumer laptop GPU.
#
# The in-process safety net (Trainer's OOM-batch retry, checkpoint-per-epoch,
# graceful SIGINT/SIGTERM handling) covers everything that happens *inside*
# the python process. It cannot save you from the process itself dying: the
# Linux OOM killer ending it outright, a driver reset, or the laptop
# suspending. This script covers that outer layer: if the process exits
# non-zero, it waits, then relaunches with --resume, up to a retry limit -
# so "leave it running overnight" survives a crash instead of silently
# stopping at 2am with nothing to show for the rest of the night.
#
# Usage:
#   scripts/train_safe.sh configs/experiment/A_indist_8gb.yaml
#   MAX_RETRIES=10 scripts/train_safe.sh configs/experiment/A_indist_8gb.yaml --set train.epochs=60
set -uo pipefail   # deliberately not -e: a training crash must fall through to the retry logic below, not abort the script

CONFIG="${1:?usage: train_safe.sh <config.yaml> [-- extra reiduq train args]}"
shift || true
MAX_RETRIES="${MAX_RETRIES:-5}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-30}"
LOG_DIR="${LOG_DIR:-./outputs/train_logs}"
mkdir -p "$LOG_DIR"

cd "$(dirname "$0")/.."

echo "==> laptop-safety checks"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "    GPU:  $(nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv,noheader 2>/dev/null || echo 'nvidia-smi query failed')"
  # A lower power limit trades some speed for a cooler, quieter, longer-running
  # laptop session. Uncomment and set to your card's sensible floor if the
  # chassis runs hot; requires the script to be run with sufficient privilege.
  # nvidia-smi -pl 90
else
  echo "    nvidia-smi not found - thermal/power checks in the trainer will be skipped, not the run itself"
fi

attempt=0
resume_flag=""
while true; do
  attempt=$((attempt + 1))
  ts="$(date +%Y%m%d_%H%M%S)"
  log_file="${LOG_DIR}/attempt_${attempt}_${ts}.log"
  echo "==> attempt ${attempt}/${MAX_RETRIES}  (log: ${log_file})"

  # nice: lower CPU scheduling priority so the rest of the laptop stays
  # responsive while this runs in the background for hours.
  nice -n 10 python -m reiduq.cli train --config "$CONFIG" $resume_flag "$@" 2>&1 | tee "$log_file"
  status="${PIPESTATUS[0]}"

  if [ "$status" -eq 0 ]; then
    echo "==> training finished cleanly on attempt ${attempt}"
    exit 0
  fi

  if [ "$attempt" -ge "$MAX_RETRIES" ]; then
    echo "==> gave up after ${attempt} attempts (exit code ${status}). Check ${log_file}." >&2
    exit "$status"
  fi

  echo "==> training exited with code ${status}; retrying with --resume in ${RETRY_DELAY_SECONDS}s"
  echo "    (checkpoints are per-epoch, so at most one epoch of progress is lost)"
  resume_flag="--resume"
  sleep "$RETRY_DELAY_SECONDS"
done
