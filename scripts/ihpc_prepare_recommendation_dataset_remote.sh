#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: bash scripts/ihpc_prepare_recommendation_dataset_remote.sh <dataset-id> [extra-args...]" >&2
  exit 1
fi

dataset_id="$1"
shift || true

./.venv/bin/python scripts/download_recommendation_assets.py download-dataset "${dataset_id}"
./.venv/bin/python scripts/prepare_recommendation_dataset.py --dataset "${dataset_id}" "$@"
