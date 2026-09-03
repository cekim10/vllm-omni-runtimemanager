#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?usage: $0 {cpu|preflight|smoke|analyze-smoke}}"
OUT="${2:-results/video_batch_determinism_killtest}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python experiments/video_batch_determinism_killtest.py \
  --mode "${MODE}" \
  --config experiments/video_batch_determinism_killtest_config.yaml \
  --output-dir "${OUT}" \
  --enable-cpu-offload
