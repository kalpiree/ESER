#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 6 ]]; then
  echo "Usage: bash scripts/ihpc_run_recommendation_rq1_qwen_parallel_detached.sh <config-path> <model-path> <prompt-jsonl> <merged-score-jsonl> <num-shards> <shard-index:gpu>... [-- score-extra-args...]" >&2
  exit 1
fi

config_path="$1"
model_path="$2"
prompt_jsonl="$3"
merged_score_jsonl="$4"
num_shards="$5"
shift 5

config_name="$(basename "${config_path}" .json)"
job_label="rec-rq1-par-${config_name}"
remote_command="bash scripts/ihpc_recommendation_rq1_qwen_parallel_remote.sh ${config_path} ${model_path} ${prompt_jsonl} ${merged_score_jsonl} ${num_shards}"

while [[ "$#" -gt 0 ]]; do
  remote_command="${remote_command} $1"
  shift
done

bash scripts/ihpc_exec_detached.sh --label "${job_label}" "${remote_command}"
