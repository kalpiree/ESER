#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 4 ]]; then
  echo "Usage: bash scripts/ihpc_run_recommendation_rq1_detached.sh <config-path> <model-path> <prompt-jsonl> <score-jsonl> [score-extra-args...]" >&2
  echo "Optional env: IHPC_CUDA_VISIBLE_DEVICES=1 to pin the scoring step to a specific remote GPU." >&2
  exit 1
fi

config_path="$1"
model_path="$2"
prompt_jsonl="$3"
score_jsonl="$4"
shift 4

config_name="$(basename "${config_path}" .json)"
job_label="rec-rq1-${config_name}"
remote_prefix=""
if [[ -n "${IHPC_CUDA_VISIBLE_DEVICES:-}" ]]; then
  remote_prefix="CUDA_VISIBLE_DEVICES=${IHPC_CUDA_VISIBLE_DEVICES} "
fi

remote_command="${remote_prefix}bash scripts/ihpc_recommendation_rq1_remote.sh ${config_path} ${model_path} ${prompt_jsonl} ${score_jsonl}"
if [[ "$#" -gt 0 ]]; then
  remote_command="${remote_command} $*"
fi

bash scripts/ihpc_exec_detached.sh --label "${job_label}" "${remote_command}"
