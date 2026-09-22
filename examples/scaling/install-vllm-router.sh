#!/usr/bin/env bash
# One-time login-node installation; compute jobs never download router packages.
set -euo pipefail
router_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${router_repo}"
router_env="${router_repo}/.tools/vllm-router-venv"
if [[ ! -x "${router_env}/bin/python" ]]; then
  .venv/bin/python -m venv "${router_env}"
fi
"${router_env}/bin/python" -m pip install --no-cache-dir --only-binary=:all: \
  -r examples/scaling/vllm-router-requirements.txt
.venv/bin/python -m agenttrace.experiments.multinode check \
  --config examples/scaling/routing.json --group routing
