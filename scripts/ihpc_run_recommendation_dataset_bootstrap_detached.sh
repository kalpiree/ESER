#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: bash scripts/ihpc_run_recommendation_dataset_bootstrap_detached.sh <dataset-id> [extra-args...]" >&2
  exit 1
fi

dataset_id="$1"
shift || true

job_label="rec-data-${dataset_id}"
if [[ "$#" -gt 0 ]]; then
  extra_slug="$(printf '%s-' "$@" | tr ' /=' '___' | tr -cd '[:alnum:]_-')"
  extra_slug="${extra_slug%-}"
  if [[ -n "${extra_slug}" ]]; then
    job_label="${job_label}-${extra_slug:0:48}"
  fi
fi
remote_command="./.venv/bin/python scripts/download_recommendation_assets.py download-dataset ${dataset_id} && ./.venv/bin/python scripts/prepare_recommendation_dataset.py --dataset ${dataset_id}"
if [[ "$#" -gt 0 ]]; then
  remote_command="${remote_command} $*"
fi

bash scripts/ihpc_exec_detached.sh --label "${job_label}" "${remote_command}"
