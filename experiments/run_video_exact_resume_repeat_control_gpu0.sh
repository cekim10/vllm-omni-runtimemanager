#!/usr/bin/env bash
set -euo pipefail

source_dir="${1:-results/video_denoising_error_correction_killtest}"
output_dir="${2:-${source_dir}/exact_repeat_control}"
python_bin="${PYTHON_BIN:-python}"

mkdir -p "${output_dir}"
CUDA_VISIBLE_DEVICES=0 "${python_bin}" experiments/video_exact_resume_repeat_control.py \
  --source-dir "${source_dir}" \
  --output-dir "${output_dir}" \
  --checkpoint-step 20 \
  --repeats 3 \
  --resume \
  2>&1 | tee -a "${output_dir}/exact_repeat_control.log"
