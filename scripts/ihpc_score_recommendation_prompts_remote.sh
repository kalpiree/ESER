#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 ]]; then
  echo "Usage: bash scripts/ihpc_score_recommendation_prompts_remote.sh <model-path> <input-jsonl> <output-jsonl> [extra-args...]" >&2
  exit 1
fi

model_path="$1"
input_jsonl="$2"
output_jsonl="$3"
shift 3

./.venv/bin/python scripts/score_recommendation_prompts.py \
  --model-path "${model_path}" \
  --input-jsonl "${input_jsonl}" \
  --output-jsonl "${output_jsonl}" \
  "$@"
