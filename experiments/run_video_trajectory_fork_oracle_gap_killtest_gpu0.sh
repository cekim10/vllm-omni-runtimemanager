#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 {cpu|baselines|controls|forks|blind|analyze} [output_dir]" >&2
  exit 2
fi
MODE="$1"
OUT="${2:-results/video_trajectory_fork_oracle_gap_killtest}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python experiments/video_trajectory_fork_oracle_gap_killtest.py \
  --mode "${MODE}" \
  --config experiments/video_trajectory_fork_oracle_gap_killtest_config.yaml \
  --output-dir "${OUT}" \
  --enable-cpu-offload
