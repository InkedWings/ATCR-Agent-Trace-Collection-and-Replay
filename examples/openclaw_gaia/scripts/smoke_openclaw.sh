#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/../../.." && pwd)"
openclaw_bin="${repo_dir}/.tools/openclaw/bin/openclaw"

export OPENCLAW_STATE_DIR="${repo_dir}/.openclaw-state/gaia"
export OPENCLAW_CONFIG_PATH="${repo_dir}/examples/openclaw_gaia/config/openclaw.json5"
export OPENCLAW_WORKSPACE_DIR="${repo_dir}/runs/openclaw-smoke/workspace"

mkdir -p "${OPENCLAW_STATE_DIR}" "${OPENCLAW_WORKSPACE_DIR}"
chmod 700 "${OPENCLAW_STATE_DIR}"

if [[ ! -x "${openclaw_bin}" ]]; then
  echo "OpenClaw is not installed; run ${script_dir}/install_openclaw.sh" >&2
  exit 1
fi

(
  # Keep both credentials and the short-lived ALCF token inside this process.
  # shellcheck disable=SC1091
  source "${script_dir}/load_credentials.sh"

  "${openclaw_bin}" config validate --json
  "${openclaw_bin}" agent \
    --local \
    --json \
    --session-id "sigmetrics-prep-smoke" \
    --model "alcf-minerva/nemotron-3-ultra" \
    --timeout 600 \
    --message "Reply with exactly READY."
)
