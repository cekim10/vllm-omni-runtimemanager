#!/usr/bin/env bash
# Numerical Compatibility Round 0 (rev 0.2c). GPU 0 only. Execution configuration is chosen by run.py from --axis;
# do NOT pass --enable-cpu-offload / --enforce-eager / DIFFUSION_ATTENTION_BACKEND here (they are the axes).
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 {cpu|preregister|canonical|axis <AXIS>|freshcheck <CONFIG>|analyze} [output_dir]" >&2
  echo "  AXIS in {EAGER,OFFLOAD,ATTN_FLASHINFER,ATTN_CUDNN}; CONFIG in {CANONICAL,EAGER,OFFLOAD,ATTN_FLASHINFER,ATTN_CUDNN}" >&2
  exit 2
fi
PHASE="$1"; shift
AXIS_ARGS=()
if [ "${PHASE}" = "axis" ] || [ "${PHASE}" = "freshcheck" ]; then
  if [ $# -lt 1 ]; then echo "phase ${PHASE} requires an axis/config name" >&2; exit 2; fi
  AXIS_ARGS=(--axis "$1"); shift
fi
OUT="${1:-results/numerical_compatibility_round0/rev0_2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
unset DIFFUSION_ATTENTION_BACKEND

python experiments/numerical_compatibility_round0/run.py \
  --phase "${PHASE}" "${AXIS_ARGS[@]}" \
  --config experiments/numerical_compatibility_round0/config.json \
  --output-dir "${OUT}"
