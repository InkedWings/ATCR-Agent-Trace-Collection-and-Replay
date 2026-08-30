#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
component_dir="$(cd -- "${script_dir}/.." && pwd)"
repo_dir="$(cd -- "${component_dir}/../.." && pwd)"
uv_bin="${UV_BIN:-${HOME}/.local/bin/uv}"
venv_dir="${repo_dir}/.venv"

# Keep scientific Python imports courteous on shared Polaris login nodes.
export OPENBLAS_NUM_THREADS="${SIGMETRICS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${SIGMETRICS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${SIGMETRICS_NUM_THREADS:-1}"

if [[ ! -x "${uv_bin}" ]]; then
  echo "uv is missing at ${uv_bin}" >&2
  exit 1
fi

if [[ ! -x "${venv_dir}/bin/python" ]]; then
  "${uv_bin}" venv --python 3.12 "${venv_dir}"
fi

"${uv_bin}" pip install \
  --python "${venv_dir}/bin/python" \
  --requirement "${component_dir}/requirements-bootstrap.txt"
"${uv_bin}" pip install --python "${venv_dir}/bin/python" --editable "${repo_dir}"

"${venv_dir}/bin/python" --version
"${venv_dir}/bin/python" -c 'from importlib.metadata import version; print("alcf-ai " + version("alcf-ai"))'
