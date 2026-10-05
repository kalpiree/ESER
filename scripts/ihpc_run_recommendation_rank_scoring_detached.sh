#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 ]]; then
  echo "Usage: bash scripts/ihpc_run_recommendation_rank_scoring_detached.sh <model-path> <input-jsonl> <output-jsonl> [extra-args...]" >&2
  exit 1
fi

model_path="$1"
input_jsonl="$2"
output_jsonl="$3"
shift 3

job_label="rec-rank-$(basename "${output_jsonl}" .jsonl)"
remote_command="./.venv/bin/python scripts/score_recommendation_rank_prompts.py --model-path ${model_path} --input-jsonl ${input_jsonl} --output-jsonl ${output_jsonl}"
if [[ "$#" -gt 0 ]]; then
  remote_command="${remote_command} $*"
fi

bash scripts/ihpc_exec_detached.sh --label "${job_label}" "${remote_command}"
