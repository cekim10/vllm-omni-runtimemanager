#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 {cpu|preregister|reference|perturb|propagation|oracle|recompute|analyze|weakprobe} [output_dir]" >&2
  exit 2
fi
PHASE="$1"
OUT="${2:-results/regional_recompute_round0/rev4_1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python experiments/speculative_contamination_round0/run.py \
  --phase "${PHASE}" \
  --config experiments/speculative_contamination_round0/config.json \
  --output-dir "${OUT}" \
  --enable-cpu-offload
