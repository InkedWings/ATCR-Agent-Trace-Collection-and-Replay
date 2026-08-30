#!/usr/bin/env bash
set -euo pipefail
umask 077

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/../../.." && pwd)"
openclaw_bin="${repo_dir}/.tools/openclaw/bin/openclaw"
run_id="${GAIA_RUN_ID:-direct-first30-$(date -u +%Y%m%dT%H%M%SZ)}"

cd "${repo_dir}"

export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
export HTTP_PROXY="${HTTP_PROXY:-${http_proxy}}"
export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy}}"
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="${no_proxy}"

export OPENBLAS_NUM_THREADS="${SIGMETRICS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${SIGMETRICS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${SIGMETRICS_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export PATH="${repo_dir}/.venv/bin:${repo_dir}/.tools/openclaw/bin:${PATH}"

export OPENCLAW_BIN="${openclaw_bin}"
export OPENCLAW_STATE_DIR="${OPENCLAW_STATE_DIR:-${repo_dir}/.openclaw-state/gaia}"
export OPENCLAW_CONFIG_PATH="${repo_dir}/examples/openclaw_gaia/config/openclaw.capture.json5"
export OPENCLAW_WORKSPACE_DIR="${OPENCLAW_STATE_DIR}/config-validation-workspace"
export LLM_CAPTURE_BASE_URL="http://127.0.0.1:1"
export TRACE_ID="config-validation"

mkdir -p "${OPENCLAW_WORKSPACE_DIR}"

# shellcheck disable=SC1091
source "${script_dir}/load_credentials.sh"

echo "host=$(hostname) run_id=${run_id}"
"${script_dir}/auth_status.sh"
"${script_dir}/preflight_services.sh" --skip-inference
"${openclaw_bin}" config validate --json

exec "${repo_dir}/.venv/bin/python" \
  "${repo_dir}/examples/openclaw_gaia/collect_traces.py" \
  --manifest "${repo_dir}/data/gaia/pilots/validation_first30.jsonl" \
  --gaia-dir "${repo_dir}/data/gaia" \
  --run-root "${repo_dir}/runs/openclaw-gaia" \
  --run-id "${run_id}" \
  --start-index "${GAIA_START_INDEX:-0}" \
  --max-tasks "${GAIA_MAX_TASKS:-30}" \
  --task-timeout "${GAIA_TASK_TIMEOUT:-240}" \
  --run-timeout "${GAIA_RUN_TIMEOUT:-3150}" \
  --llm-upstream \
  "https://inference-api.alcf.anl.gov/resource_server/minerva/api/v1"
