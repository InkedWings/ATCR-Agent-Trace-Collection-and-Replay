#!/usr/bin/env bash
set -euo pipefail
umask 077
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/environment.sh"
cd "${script_dir}/../.."
exec .venv/bin/python -m agenttrace.experiments.scale "$@"
