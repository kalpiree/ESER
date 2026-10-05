#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: bash scripts/ihpc_run_recommendation_model_download_detached.sh <model-id>" >&2
  exit 1
fi

model_id="$1"
job_label="rec-model-${model_id}"
remote_command="./.venv/bin/python scripts/download_recommendation_assets.py download-model ${model_id}"

bash scripts/ihpc_exec_detached.sh --label "${job_label}" "${remote_command}"
