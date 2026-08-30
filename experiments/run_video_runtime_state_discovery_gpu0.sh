#!/usr/bin/env bash
set -euo pipefail

mode="${1:-cpu}"
output_dir="${2:-results/video_runtime_state_discovery_v3_corrected}"
prompt_start="${3:-0}"
prompt_end="${4:-12}"
python_bin="${PYTHON_BIN:-python}"

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

case "${mode}" in
  cpu)
    "${python_bin}" -m pytest -q \
      -m "core_model and cpu and diffusion" \
      tests/diffusion/test_video_runtime_state_discovery.py \
      tests/diffusion/models/wan2_2/test_wan22_pipeline_diffuse.py
    "${python_bin}" experiments/video_runtime_state_discovery.py \
      --mode cpu-preflight \
      --config experiments/video_runtime_state_discovery_config.yaml \
      --output-dir "${output_dir}"
    ;;
  preflight|smoke|full|analyze)
    mkdir -p "${output_dir}/logs"
    "${python_bin}" experiments/video_runtime_state_discovery.py \
      --mode "${mode}" \
      --config experiments/video_runtime_state_discovery_config.yaml \
      --output-dir "${output_dir}" \
      --prompt-start "${prompt_start}" \
      --prompt-end "${prompt_end}" \
      --enable-cpu-offload \
      2>&1 | tee -a "${output_dir}/logs/${mode}_p${prompt_start}-${prompt_end}.log"
    ;;
  *)
    echo "usage: $0 {cpu|preflight|smoke|full|analyze} [output_dir] [prompt_start] [prompt_end]" >&2
    exit 2
    ;;
esac
