#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 {cpu|profile-on|profile-off|analyze} [output_dir]" >&2
  exit 2
fi
MODE="$1"
OUT="${2:-results/video_resource_lifetime_profile}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CFG=experiments/video_resource_lifetime_profile_config.yaml

case "${MODE}" in
  cpu|analyze)
    python experiments/video_resource_lifetime_profile.py --mode "${MODE}" --config "${CFG}" --output-dir "${OUT}" ;;
  profile-on)
    python experiments/video_resource_lifetime_profile.py --mode profile --offload on --config "${CFG}" --output-dir "${OUT}" ;;
  profile-off)
    python experiments/video_resource_lifetime_profile.py --mode profile --offload off --config "${CFG}" --output-dir "${OUT}" ;;
  *)
    echo "unknown mode ${MODE}" >&2; exit 2 ;;
esac
