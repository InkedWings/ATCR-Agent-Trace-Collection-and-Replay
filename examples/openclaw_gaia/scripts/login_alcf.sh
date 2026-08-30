#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/../../.." && pwd)"
alcf_cli="${repo_dir}/.venv/bin/alcf-ai"

export OPENBLAS_NUM_THREADS="${SIGMETRICS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${SIGMETRICS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${SIGMETRICS_NUM_THREADS:-1}"

if [[ ! -x "${alcf_cli}" ]]; then
  echo "alcf-ai is not installed; run bootstrap_python.sh first." >&2
  exit 1
fi

"${alcf_cli}" auth login

# Validate retrieval without printing or storing the access token.
alcf_access_token="$("${alcf_cli}" auth get-access-token)"
if [[ -z "${alcf_access_token}" ]]; then
  echo "ALCF authentication completed but no access token was returned." >&2
  exit 1
fi
unset alcf_access_token

echo "ALCF inference authentication is ready."
"${alcf_cli}" auth get-token-expiration --units hours
