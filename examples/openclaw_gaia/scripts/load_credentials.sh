#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "This script must be sourced: source ${BASH_SOURCE[0]}" >&2
  exit 1
fi

credential_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
credential_repo_dir="$(cd -- "${credential_script_dir}/../../.." && pwd)"
credential_secret_file="${HOME}/.config/sigmetrics-2027/openclaw-gaia.env"
credential_alcf_cli="${credential_repo_dir}/.venv/bin/alcf-ai"

if [[ ! -r "${credential_secret_file}" ]]; then
  echo "Missing credential file: ${credential_secret_file}" >&2
  return 1
fi
if [[ ! -x "${credential_alcf_cli}" ]]; then
  echo "Missing alcf-ai CLI: ${credential_alcf_cli}" >&2
  return 1
fi

set -a
# shellcheck disable=SC1090
source "${credential_secret_file}"
set +a

if [[ -z "${BRAVE_API_KEY:-}" ]]; then
  echo "BRAVE_API_KEY is missing from ${credential_secret_file}" >&2
  return 1
fi

export OPENBLAS_NUM_THREADS="${SIGMETRICS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${SIGMETRICS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${SIGMETRICS_NUM_THREADS:-1}"
export ALCF_AI_TOKEN="$("${credential_alcf_cli}" auth get-access-token)"

unset credential_script_dir credential_repo_dir credential_secret_file credential_alcf_cli
