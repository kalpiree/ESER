#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 4 ]]; then
  echo "Usage: bash scripts/ihpc_recommendation_rq1_remote.sh <config-path> <model-path> <prompt-jsonl> <score-jsonl> [score-extra-args...]" >&2
  exit 1
fi

config_path="$1"
model_path="$2"
prompt_jsonl="$3"
score_jsonl="$4"
shift 4

REC_PROMPT_LIMIT="${REC_PROMPT_LIMIT:-250000}"
REC_PROMPT_HISTORY_LIMIT="${REC_PROMPT_HISTORY_LIMIT:-50}"
export REC_MAX_EVAL_USERS="${REC_MAX_EVAL_USERS:-2450}"
export REC_NEGATIVE_SAMPLE_SIZE_OVERRIDE="${REC_NEGATIVE_SAMPLE_SIZE_OVERRIDE:-50}"

if [[ -s "${prompt_jsonl}" ]]; then
  echo "[rec:qwen] reusing existing prompt batch ${prompt_jsonl}"
else
  ./.venv/bin/python -u scripts/export_recommendation_prompts.py \
    --config "${config_path}" \
    --output "${prompt_jsonl}" \
    --max-prompts "${REC_PROMPT_LIMIT}" \
    --max-history-items "${REC_PROMPT_HISTORY_LIMIT}"
fi

./.venv/bin/python -u scripts/score_recommendation_prompts.py \
  --model-path "${model_path}" \
  --input-jsonl "${prompt_jsonl}" \
  --output-jsonl "${score_jsonl}" \
  --use-chat-template \
  --limit "${REC_PROMPT_LIMIT}" \
  "$@"

config_dir="$(dirname "${config_path}")"
runtime_config="$(mktemp "${config_dir}/.runtime-rec-rq1-XXXXXX.json")"
./.venv/bin/python - "${config_path}" "${score_jsonl}" "${runtime_config}" <<'PY'
import json
import sys
from pathlib import Path

config_path, score_path, runtime_path = sys.argv[1:4]
with open(config_path, encoding="utf-8") as handle:
    config = json.load(handle)
config["scorer"]["scores_path"] = str(Path(score_path).resolve())
with open(runtime_path, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2)
PY

./.venv/bin/python -u scripts/run_recommendation_experiment.py \
  --config "${runtime_config}"
