#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: bash scripts/ihpc_recommendation_experiment_remote.sh <config-path> [extra-args...]" >&2
  exit 1
fi

config_path="$1"
shift || true

./.venv/bin/python -u scripts/run_recommendation_experiment.py --config "${config_path}" "$@"
