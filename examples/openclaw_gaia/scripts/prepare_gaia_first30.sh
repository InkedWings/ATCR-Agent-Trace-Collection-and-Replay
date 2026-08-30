#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/../../.." && pwd)"

"${script_dir}/download_gaia_validation.sh" >/dev/null

export OPENBLAS_NUM_THREADS="${SIGMETRICS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${SIGMETRICS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${SIGMETRICS_NUM_THREADS:-1}"

cd "${repo_dir}"
exec "${repo_dir}/.venv/bin/python" \
  "${repo_dir}/examples/openclaw_gaia/select_validation_prefix.py" \
  --count 30 \
  --output "${repo_dir}/data/gaia/pilots/validation_first30.jsonl" \
  "$@"
