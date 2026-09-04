#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 {cpu|preflight|phase1|analyze-phase1|phase2|analyze-phase2|phase3|analyze-phase3} [output_dir]" >&2
  exit 2
fi
MODE="$1"
OUT="${2:-results/video_bf16_first_divergence_localization_phase3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python experiments/video_bf16_first_divergence_localization.py \
  --mode "${MODE}" \
  --config experiments/video_bf16_first_divergence_localization_config.yaml \
  --output-dir "${OUT}" \
  --enable-cpu-offload
