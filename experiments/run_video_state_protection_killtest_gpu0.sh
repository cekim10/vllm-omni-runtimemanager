#!/usr/bin/env bash
set -euo pipefail

stage="${1:-screening}"
output_root="${2:-results/video_state_protection_killtest_gpu0}"
prompt_start="${3:-0}"
prompt_end="${4:-12}"
seed_start="${5:-0}"
python_bin="${PYTHON_BIN:-python}"

case "${stage}" in
  screening)
    num_seeds=2
    stage_args=(--screening-stage)
    ;;
  full)
    num_seeds=5
    stage_args=()
    screening_summary="${output_root}/run/frontier_summary.json"
    if [[ ! -f "${screening_summary}" ]]; then
      echo "missing screening result: ${screening_summary}" >&2
      exit 1
    fi
    screening_judgment="$("${python_bin}" -c 'import json, sys; print(json.load(open(sys.argv[1]))["judgment"])' "${screening_summary}")"
    if [[ "${screening_judgment}" != "CONTINUE TO 5 SEEDS" ]]; then
      echo "full stage blocked by screening judgment: ${screening_judgment}" >&2
      exit 1
    fi
    ;;
  *)
    echo "usage: $0 {screening|full} [output_root] [prompt_start] [prompt_end] [seed_start] [seed_end]" >&2
    exit 2
    ;;
esac

seed_end="${6:-${num_seeds}}"
mkdir -p "${output_root}/logs"

log_path="${output_root}/logs/${stage}_p${prompt_start}-${prompt_end}_s${seed_start}-${seed_end}.log"

CUDA_VISIBLE_DEVICES=0 "${python_bin}" experiments/video_state_protection_killtest.py \
  --prompt-set experiments/video_recovery_prompt_set.json \
  --output-dir "${output_root}/run" \
  --num-prompts 12 \
  --num-seeds "${num_seeds}" \
  --prompt-start "${prompt_start}" \
  --prompt-end "${prompt_end}" \
  --seed-start "${seed_start}" \
  --seed-end "${seed_end}" \
  --height 480 \
  --width 832 \
  --num-frames 33 \
  --num-inference-steps 40 \
  --checkpoint-steps 10 20 30 \
  --guidance-scale 4.0 \
  --fps 16 \
  --flow-shift 12.0 \
  --boundary-ratio 0.875 \
  --enable-cpu-offload \
  --resume \
  "${stage_args[@]}" \
  2>&1 | tee -a "${log_path}"

echo "${stage} results: ${output_root}/run"
