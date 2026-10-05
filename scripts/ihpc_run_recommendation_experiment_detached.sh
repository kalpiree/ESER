#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: bash scripts/ihpc_run_recommendation_experiment_detached.sh <config-path> [extra-args...]" >&2
  exit 1
fi

config_path="$1"
shift || true

config_name="$(basename "${config_path}")"
job_label="rec-exp-${config_name%.json}"
remote_command="./.venv/bin/python -u scripts/run_recommendation_experiment.py --config ${config_path}"
if [[ "$#" -gt 0 ]]; then
  remote_command="${remote_command} $*"
fi

bash scripts/ihpc_exec_detached.sh --label "${job_label}" "${remote_command}"
