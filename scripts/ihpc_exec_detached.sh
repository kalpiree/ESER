#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ihpc_common.sh"

JOB_LABEL=""

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --label)
      JOB_LABEL="${2:-}"
      shift 2
      ;;
    -h|--help)
      echo "Usage: bash scripts/ihpc_exec_detached.sh [--label job-label] [command...]"
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

if [[ "$#" -gt 0 ]]; then
  REMOTE_COMMAND="$*"
else
  : "${IHPC_REMOTE_COMMAND:?Set IHPC_REMOTE_COMMAND or pass a command}"
  REMOTE_COMMAND="${IHPC_REMOTE_COMMAND}"
fi

timestamp="$(timestamp_now)"
if [[ -z "${JOB_LABEL}" ]]; then
  JOB_LABEL="ihpc-detached-${timestamp}"
fi

LOG_BASENAME="${IHPC_LOG_NAME:-${JOB_LABEL}-${timestamp}.log}"
JOB_ID="${LOG_BASENAME%.log}"
REMOTE_LOG_DIR="${IHPC_REMOTE_DIR%/}/logs/ihpc"
REMOTE_RECEIPT_DIR="${REMOTE_LOG_DIR}/receipts"
REMOTE_LOG_PATH="${REMOTE_LOG_DIR}/${LOG_BASENAME}"
REMOTE_RECEIPT_PATH="${REMOTE_RECEIPT_DIR}/${JOB_ID}.receipt"
LOCAL_RECEIPT_DIR="${LOCAL_LOG_DIR}/receipts"
LOCAL_RECEIPT_PATH="${LOCAL_RECEIPT_DIR}/${JOB_ID}.receipt"

mkdir -p "${LOCAL_RECEIPT_DIR}"

FULL_COMMAND="$(build_remote_command "${REMOTE_COMMAND}")"
REMOTE_START_COMMAND="$(cat <<EOF
mkdir -p $(printf '%q' "${REMOTE_LOG_DIR}") $(printf '%q' "${REMOTE_RECEIPT_DIR}")
nohup bash -lc $(printf '%q' "${FULL_COMMAND}") > $(printf '%q' "${REMOTE_LOG_PATH}") 2>&1 < /dev/null &
pid=\$!
{
  printf 'job_id=%s\n' $(printf '%q' "${JOB_ID}")
  printf 'job_label=%s\n' $(printf '%q' "${JOB_LABEL}")
  printf 'pid=%s\n' "\${pid}"
  printf 'node=%s\n' $(printf '%q' "${IHPC_EXEC_HOST:-access-host}")
  printf 'remote_log=%s\n' $(printf '%q' "${REMOTE_LOG_PATH}")
  printf 'remote_receipt=%s\n' $(printf '%q' "${REMOTE_RECEIPT_PATH}")
  printf 'remote_dir=%s\n' $(printf '%q' "${IHPC_REMOTE_DIR}")
  printf 'submitted_at=%s\n' $(printf '%q' "${timestamp}")
} > $(printf '%q' "${REMOTE_RECEIPT_PATH}")
cat $(printf '%q' "${REMOTE_RECEIPT_PATH}")
EOF
)"

REMOTE_INVOCATION="$(build_remote_invocation "${REMOTE_START_COMMAND}")"
receipt_contents="$(ssh "${IHPC_HOST}" "${REMOTE_INVOCATION}")"
filtered_receipt="$(printf "%s\n" "${receipt_contents}" | grep -E '^(job_id|job_label|pid|node|remote_log|remote_receipt|remote_dir|submitted_at)=' || true)"
if [[ -z "${filtered_receipt}" ]]; then
  echo "Detached job submission returned no parseable receipt lines." >&2
  echo "${receipt_contents}" >&2
  exit 1
fi
printf "%s\n" "${filtered_receipt}" > "${LOCAL_RECEIPT_PATH}"

echo "Submitted detached IHPC job."
echo "  receipt: ${LOCAL_RECEIPT_PATH}"
echo "  remote log: ${REMOTE_LOG_PATH}"
echo "  check: bash scripts/ihpc_job_status.sh ${JOB_ID}"
