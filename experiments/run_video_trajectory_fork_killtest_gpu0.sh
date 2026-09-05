#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
OUTPUT_DIR="${2:-results/video_trajectory_fork_killtest}"

case "${MODE}" in
  cpu|controls|primary|analyze|expansion) ;;
  *)
    echo "usage: $0 {cpu|controls|primary|analyze|expansion} [output-dir]" >&2
    exit 2
    ;;
esac

ARGS=(
  experiments/video_trajectory_fork_killtest.py
  "${MODE}"
  --config experiments/video_trajectory_fork_killtest_config.yaml
  --output-dir "${OUTPUT_DIR}"
)

if [[ "${MODE}" == "controls" || "${MODE}" == "primary" || "${MODE}" == "expansion" ]]; then
  export CUDA_VISIBLE_DEVICES=0
  ARGS+=(--enable-cpu-offload)
fi

python "${ARGS[@]}"
