#!/usr/bin/env bash
set -euo pipefail

mode="${1:-cpu}"
output_dir="${2:-results/video_runtime_error_shape_killtest}"
python_bin="${PYTHON_BIN:-python}"
config="experiments/video_runtime_error_shape_killtest_config.yaml"

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

case "${mode}" in
  cpu)
    "${python_bin}" -m pytest -q \
      -m "core_model and cpu and diffusion" \
      tests/diffusion/test_video_runtime_error_shape_killtest.py
    "${python_bin}" experiments/video_runtime_error_shape_killtest.py \
      --mode cpu \
      --config "${config}" \
      --output-dir "${output_dir}"
    ;;
  preflight|fp16-replay|concentration-smoke|full)
    mkdir -p "${output_dir}/logs"
    "${python_bin}" experiments/video_runtime_error_shape_killtest.py \
      --mode "${mode}" \
      --config "${config}" \
      --output-dir "${output_dir}" \
      --enable-cpu-offload \
      2>&1 | tee -a "${output_dir}/logs/${mode}.log"
    ;;
  *)
    echo "usage: $0 {cpu|preflight|fp16-replay|concentration-smoke|full} [output_dir]" >&2
    exit 2
    ;;
esac
