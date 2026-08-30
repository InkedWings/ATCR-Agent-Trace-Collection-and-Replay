#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/../../.." && pwd)"
openclaw_version="2026.7.1-2"
brave_plugin_version="2026.7.1"
node_version="24.19.0"
installer_url="https://raw.githubusercontent.com/openclaw/openclaw/v${openclaw_version}/scripts/install-cli.sh"
openclaw_bin="${repo_dir}/.tools/openclaw/bin/openclaw"
openclaw_state_dir="${repo_dir}/.openclaw-state/gaia"

curl -fsSL --proto '=https' --tlsv1.2 "${installer_url}" | \
  bash -s -- \
    --prefix "${repo_dir}/.tools/openclaw" \
    --version "${openclaw_version}" \
    --node-version "${node_version}" \
    --no-onboard \
    --json

"${openclaw_bin}" --version
"${repo_dir}/.tools/openclaw/tools/node/bin/node" --version

mkdir -p "${openclaw_state_dir}"
chmod 700 "${openclaw_state_dir}"

# Keep the official web-search plugin and its generated install metadata in the
# project-local, git-ignored OpenClaw state directory.
if ! OPENCLAW_STATE_DIR="${openclaw_state_dir}" \
  "${openclaw_bin}" plugins info brave --json >/dev/null 2>&1; then
  OPENCLAW_STATE_DIR="${openclaw_state_dir}" \
    "${openclaw_bin}" plugins install --pin \
      "@openclaw/brave-plugin@${brave_plugin_version}"
fi
