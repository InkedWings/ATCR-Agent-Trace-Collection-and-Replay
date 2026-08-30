#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/../../.." && pwd)"

(
  # Keep credentials scoped to this subprocess and its HTTP clients.
  # shellcheck disable=SC1091
  source "${script_dir}/load_credentials.sh"
  exec "${repo_dir}/.venv/bin/python" \
    "${repo_dir}/examples/openclaw_gaia/preflight.py" "$@"
)
