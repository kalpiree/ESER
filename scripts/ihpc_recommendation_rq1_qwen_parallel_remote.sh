#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 5 ]]; then
  echo "Usage: bash scripts/ihpc_recommendation_rq1_qwen_parallel_remote.sh <config-path> <model-path> <prompt-jsonl> <merged-score-jsonl> <num-shards> <shard-index:gpu>... [-- score-extra-args...]" >&2
  exit 1
fi

config_path="$1"
model_path="$2"
prompt_jsonl="$3"
merged_score_jsonl="$4"
num_shards="$5"
shift 5

REC_PROMPT_LIMIT="${REC_PROMPT_LIMIT:-250000}"
REC_PROMPT_HISTORY_LIMIT="${REC_PROMPT_HISTORY_LIMIT:-50}"
export REC_MAX_EVAL_USERS="${REC_MAX_EVAL_USERS:-2450}"
export REC_NEGATIVE_SAMPLE_SIZE_OVERRIDE="${REC_NEGATIVE_SAMPLE_SIZE_OVERRIDE:-50}"

shard_specs=()
score_extra_args=()
parsing_specs=1
for arg in "$@"; do
  if [[ "${arg}" == "--" && "${parsing_specs}" -eq 1 ]]; then
    parsing_specs=0
    continue
  fi
  if [[ "${parsing_specs}" -eq 1 ]]; then
    shard_specs+=("${arg}")
  else
    score_extra_args+=("${arg}")
  fi
done

if [[ "${#shard_specs[@]}" -eq 0 ]]; then
  echo "At least one shard spec is required." >&2
  exit 1
fi

if [[ -s "${prompt_jsonl}" ]]; then
  echo "[rec:qwen] reusing existing prompt batch ${prompt_jsonl}"
else
  ./.venv/bin/python -u scripts/export_recommendation_prompts.py \
    --config "${config_path}" \
    --output "${prompt_jsonl}" \
    --max-prompts "${REC_PROMPT_LIMIT}" \
    --max-history-items "${REC_PROMPT_HISTORY_LIMIT}"
fi

score_base="${merged_score_jsonl%.jsonl}"
pids=()
shard_files=()

for spec in "${shard_specs[@]}"; do
  shard_index="${spec%%:*}"
  gpu_id="${spec##*:}"
  shard_output="${score_base}.shard${shard_index}.jsonl"
  shard_files+=("${shard_output}")
  echo "[rec:qwen] launching shard=${shard_index}/${num_shards} gpu=${gpu_id} output=${shard_output}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" ./.venv/bin/python -u scripts/score_recommendation_prompts.py \
    --model-path "${model_path}" \
    --input-jsonl "${prompt_jsonl}" \
    --output-jsonl "${shard_output}" \
    --resume \
    --num-shards "${num_shards}" \
    --shard-index "${shard_index}" \
    --limit "${REC_PROMPT_LIMIT}" \
    "${score_extra_args[@]}" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

./.venv/bin/python -u scripts/merge_score_shards.py \
  --output-jsonl "${merged_score_jsonl}" \
  --input-jsonl "${shard_files[@]}"

config_dir="$(dirname "${config_path}")"
runtime_config="$(mktemp "${config_dir}/.runtime-rec-rq1-XXXXXX.json")"
./.venv/bin/python - "${config_path}" "${merged_score_jsonl}" "${runtime_config}" <<'PY'
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
