#!/usr/bin/env bash
set -euo pipefail

stage="${1:-smoke}"
default_output="results/video_denoising_error_correction_killtest"
if [[ "${stage}" != "smoke" ]]; then
  default_output="${default_output}_${stage}"
fi
output_dir="${2:-${default_output}}"
prior_summary="${3:-}"
python_bin="${PYTHON_BIN:-python}"
sample_solver="${SAMPLE_SOLVER:-unipc}"
require_exact_resume="${REQUIRE_EXACT_RESUME:-0}"

stage_args=(--stage "${stage}")
if [[ "${require_exact_resume}" == "1" ]]; then
  stage_args+=(--require-exact-resume)
fi
if [[ "${stage}" != "smoke" ]]; then
  if [[ -z "${prior_summary}" ]]; then
    echo "${stage} requires the prior stage error_contraction_summary.json" >&2
    exit 2
  fi
  stage_args+=(--prior-summary "${prior_summary}")
fi

mkdir -p "${output_dir}"
CUDA_VISIBLE_DEVICES=0 "${python_bin}" experiments/video_denoising_error_correction_killtest.py \
  --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --prompt-set experiments/video_denoising_error_correction_prompts.json \
  --output-dir "${output_dir}" \
  --height 480 \
  --width 832 \
  --num-frames 33 \
  --num-inference-steps 40 \
  --guidance-scale 4.0 \
  --fps 16 \
  --flow-shift 12.0 \
  --boundary-ratio 0.875 \
  --sample-solver "${sample_solver}" \
  --enable-cpu-offload \
  --resume \
  "${stage_args[@]}" \
  2>&1 | tee -a "${output_dir}/${stage}.log"
