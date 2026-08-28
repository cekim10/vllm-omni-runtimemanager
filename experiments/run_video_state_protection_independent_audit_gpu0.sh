#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-static}"
if [[ "${MODE}" != "static" && "${MODE}" != "full" ]]; then
  echo "usage: $0 [static|full]" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

python experiments/video_state_protection_independent_audit.py \
  --mode "${MODE}" \
  --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --prompt-set experiments/video_recovery_prompt_set.json \
  --prompt-id recovery_000 \
  --seed 1234 \
  --checkpoint-step 20 \
  --height 480 \
  --width 832 \
  --num-frames 33 \
  --num-inference-steps 40 \
  --guidance-scale 4.0 \
  --fps 16 \
  --flow-shift 12.0 \
  --boundary-ratio 0.875 \
  --enable-cpu-offload \
  --stage-b-dir results/video_state_protection_killtest_gpu0/run \
  --output-dir results/video_state_protection_independent_audit
