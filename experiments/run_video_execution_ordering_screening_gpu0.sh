#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 {cpu|validate|screening|analyze} [output_dir]" >&2
  exit 2
fi
MODE="$1"
OUT="${2:-results/video_execution_ordering_screening}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python experiments/video_execution_ordering_screening.py \
  --mode "${MODE}" \
  --config experiments/video_execution_ordering_screening_config.yaml \
  --output-dir "${OUT}" \
  --enable-cpu-offload
