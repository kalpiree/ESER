#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${IHPC_CONFIG:-${ROOT_DIR}/configs/ihpc.env}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing IHPC config at ${CONFIG_PATH}" >&2
  echo "Create it from ${ROOT_DIR}/configs/ihpc.env.example" >&2
  exit 1
fi

OVERRIDABLE_VARS=(
  IHPC_HOST
  IHPC_REMOTE_DIR
  IHPC_EXEC_HOST
  IHPC_REMOTE_PRELUDE
  IHPC_REMOTE_COMMAND
  IHPC_LOCAL_LOG_DIR
  IHPC_LOG_NAME
)

for var_name in "${OVERRIDABLE_VARS[@]}"; do
  if [[ -n "${!var_name+x}" ]]; then
    declare "OVERRIDE_${var_name}=${!var_name}"
  fi
done

set -a
source "${CONFIG_PATH}"
set +a

for var_name in "${OVERRIDABLE_VARS[@]}"; do
  override_name="OVERRIDE_${var_name}"
  if [[ -n "${!override_name+x}" ]]; then
    export "${var_name}=${!override_name}"
  fi
done

: "${IHPC_HOST:?Set IHPC_HOST in ${CONFIG_PATH}}"
: "${IHPC_REMOTE_DIR:?Set IHPC_REMOTE_DIR in ${CONFIG_PATH}}"

LOCAL_LOG_DIR="${ROOT_DIR}/${IHPC_LOCAL_LOG_DIR:-logs/ihpc}"
mkdir -p "${LOCAL_LOG_DIR}"

timestamp_now() {
  date +"%Y%m%d-%H%M%S"
}

build_remote_command() {
  local command_string="$1"
  local prelude="${IHPC_REMOTE_PRELUDE:-}"
  local full_command=""

  if [[ -n "${prelude}" ]]; then
    full_command="${prelude} && "
  fi
  full_command="${full_command}cd \"${IHPC_REMOTE_DIR}\" && ${command_string}"
  printf "%s" "${full_command}"
}

build_remote_shell() {
  local command_string="$1"
  local full_command
  full_command="$(build_remote_command "${command_string}")"
  printf "bash -lc %q" "${full_command}"
}

build_remote_invocation() {
  local command_string="$1"
  local remote_shell
  remote_shell="$(build_remote_shell "${command_string}")"

  if [[ -n "${IHPC_EXEC_HOST:-}" ]]; then
    printf "ssh %q %s" "${IHPC_EXEC_HOST}" "${remote_shell}"
  else
    printf "%s" "${remote_shell}"
  fi
}
