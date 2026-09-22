#!/usr/bin/env bash
# Render only. Submit the resulting PBS scripts manually.
set -euo pipefail
router_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${router_repo}"
router_bundle="${1:-runs/routing-jobs-$(date -u +%Y%m%dT%H%M%SZ)}"
.venv/bin/python -m agenttrace.experiments.multinode render \
  --config examples/scaling/routing.json --group routing --output "${router_bundle}"
.venv/bin/python -m agenttrace.experiments.multinode render \
  --config examples/scaling/routing.json --group routing --smoke --output "${router_bundle}/smoke"
