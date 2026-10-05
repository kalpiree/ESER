#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ihpc_common.sh"

RECEIPT_DIR="${LOCAL_LOG_DIR}/receipts"

if [[ "$#" -ne 1 ]]; then
  echo "Usage: bash scripts/ihpc_job_status.sh <job-id-or-receipt-path>" >&2
  exit 1
fi

target="$1"
if [[ -f "${target}" ]]; then
  receipt_path="${target}"
else
  receipt_path="$(ls -1t "${RECEIPT_DIR}"/*"${target}"*.receipt 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "${receipt_path}" || ! -f "${receipt_path}" ]]; then
  echo "Could not find receipt for '${target}' in ${RECEIPT_DIR}" >&2
  exit 1
fi

receipt_value() {
  local key="$1"
  awk -F= -v k="${key}" '$1 == k { value=$0 } END { sub(/^[^=]*=/, "", value); print value }' "${receipt_path}"
}

job_id="$(receipt_value job_id)"
job_label="$(receipt_value job_label)"
pid="$(receipt_value pid)"
node="$(receipt_value node)"
remote_log="$(receipt_value remote_log)"

remote_status_command="$(cat <<EOF
if kill -0 $(printf '%q' "${pid}") 2>/dev/null; then
  echo "status=running"
else
  echo "status=finished"
fi
if [[ -f $(printf '%q' "${remote_log}") ]]; then
  echo "remote_log_size=\$(wc -c < $(printf '%q' "${remote_log}"))"
  echo "remote_log_mtime=\$(date -r $(printf '%q' "${remote_log}") +%Y-%m-%dT%H:%M:%S)"
fi
EOF
)"

remote_shell_command="bash -lc $(printf '%q' "${remote_status_command}")"
if [[ -n "${node}" && "${node}" != "access-host" ]]; then
  status_output="$(ssh "${IHPC_HOST}" "ssh $(printf '%q' "${node}") $(printf '%q' "${remote_shell_command}")")"
else
  status_output="$(ssh "${IHPC_HOST}" "${remote_shell_command}")"
fi

printf "receipt=%s\n" "${receipt_path}"
printf "job_id=%s\n" "${job_id}"
printf "job_label=%s\n" "${job_label}"
printf "node=%s\n" "${node}"
printf "pid=%s\n" "${pid}"
printf "remote_log=%s\n" "${remote_log}"
printf "%s\n" "${status_output}"
