#!/usr/bin/env bash
set -euo pipefail

stage="${1:-smoke}"
root_output="${2:-results/video_error_shaped_checkpoint_killtest}"
prior_summary="${3:-}"
python_bin="${PYTHON_BIN:-python}"
output_dir="${root_output}/${stage}"

stage_args=(--stage "${stage}")
if [[ "${stage}" != "smoke" ]]; then
  if [[ -z "${prior_summary}" ]]; then
    echo "${stage} requires the prior error_shaped_checkpoint_summary.json" >&2
    exit 2
  fi
  stage_args+=(--prior-summary "${prior_summary}")
fi

mkdir -p "${output_dir}"
CUDA_VISIBLE_DEVICES=0 "${python_bin}" experiments/video_error_shaped_checkpoint_killtest.py \
  --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --prompt-set experiments/video_propagation_aware_checkpoint_prompts.json \
  --output-dir "${output_dir}" \
  --height 480 \
  --width 832 \
  --num-frames 33 \
  --num-inference-steps 40 \
  --guidance-scale 4.0 \
  --fps 16 \
  --flow-shift 12.0 \
  --boundary-ratio 0.875 \
  --sample-solver euler \
  --enable-cpu-offload \
  --resume \
  "${stage_args[@]}" \
  2>&1 | tee -a "${output_dir}/${stage}.log"
