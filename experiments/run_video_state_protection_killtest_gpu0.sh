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
    raw_frontier="${output_root}/run/frontier_raw.csv"
    if [[ ! -f "${screening_summary}" || ! -f "${raw_frontier}" ]]; then
      echo "missing screening result: ${screening_summary}" >&2
      exit 1
    fi
    current_rows="$("${python_bin}" -c 'import csv, sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1]))))' "${raw_frontier}")"
    if [[ "${current_rows}" -le 432 ]]; then
      screening_judgment="$("${python_bin}" -c 'import json, sys; print(json.load(open(sys.argv[1]))["judgment"])' "${screening_summary}")"
      if [[ "${screening_judgment}" != "CONTINUE TO 5 SEEDS" ]]; then
        echo "full stage blocked by screening judgment: ${screening_judgment}" >&2
        exit 1
      fi
    fi
    ;;
  noise)
    "${python_bin}" -c 'import sys; from pathlib import Path; from experiments import video_state_protection_analysis as a; root=Path(sys.argv[1]); rows=a._read_csv(root / "frontier_raw.csv"); v=a.validate_frontier(rows, 12, 5); a.validate_preregistered_config(root / "preregistered_config.json", v, 5); print("validated complete n=5 frontier: 1080 rows")' \
      "${output_root}/run"
    CUDA_VISIBLE_DEVICES=0 "${python_bin}" experiments/video_recovery_noise_floor.py \
      --prompt-set experiments/video_recovery_prompt_set.json \
      --frontier-dir "${output_root}/run" \
      --output-dir "${output_root}/run/noise_floor" \
      --height 480 \
      --width 832 \
      --num-frames 33 \
      --num-inference-steps 40 \
      --checkpoint-steps 10 20 30 \
      --guidance-scale 4.0 \
      --fps 16 \
      --flow-shift 12.0 \
      --boundary-ratio 0.875 \
      --enable-cpu-offload
    exit 0
    ;;
  analyze)
    "${python_bin}" experiments/video_state_protection_analysis.py \
      --input-dir "${output_root}/run" \
      --output-dir "${output_root}/run" \
      --expected-prompts 12 \
      --expected-seeds 5 \
      --analysis-config experiments/video_state_protection_stage_b_analysis_config.yaml \
      --noise-floor-csv "${output_root}/run/noise_floor/noise_floor_results.csv"
    exit 0
    ;;
  *)
    echo "usage: $0 {screening|full|noise|analyze} [output_root] [prompt_start] [prompt_end] [seed_start] [seed_end]" >&2
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

expected_rows=432
if [[ "${stage}" == "full" ]]; then
  expected_rows=1080
fi
if [[ "${prompt_start}" != "0" || "${prompt_end}" != "12" || "${seed_start}" != "0" || "${seed_end}" != "${num_seeds}" ]]; then
  expected_rows=0
fi
"${python_bin}" -c 'import csv, sys; p, expected = sys.argv[1], int(sys.argv[2]); rows = list(csv.DictReader(open(p))); keys = [(r["prompt_id"], r["seed_index"], r["checkpoint_step"], r["variant"]) for r in rows]; assert len(keys) == len(set(keys)), "duplicate frontier rows"; assert not expected or len(rows) == expected, (len(rows), expected); print(f"validated frontier rows: {len(rows)}" + (" (partial range)" if not expected else ""))' \
  "${output_root}/run/frontier_raw.csv" "${expected_rows}"
