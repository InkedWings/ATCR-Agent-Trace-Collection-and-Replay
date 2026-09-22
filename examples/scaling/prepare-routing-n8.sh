#!/usr/bin/env bash
# Prepare the four 1+8 routing experiments; never submit jobs.
set -euo pipefail
routing_n8_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${routing_n8_repo}"
routing_n8_bundle="${1:-runs/routing-n8-prod-$(date -u +%Y%m%dT%H%M%SZ)}"
.venv/bin/python - "${routing_n8_bundle}" <<'PY'
import json
import sys
from pathlib import Path

from agenttrace.experiments.multi.config import read_config, render, write_json
from agenttrace.experiments.multi.packed import render as render_packs

output = Path(sys.argv[1]).resolve()
sources = output.with_name(output.name + "-sources")
if output.exists() or sources.exists():
    raise SystemExit("Choose a new output directory; existing bundles are never overwritten.")
config = read_config(Path("examples/scaling/routing.json"))
config["router"]["inference_nodes"] = [8]
config["router"]["policies"] = ["round_robin", "cache_aware"]
render(config, sources, group="routing")
spec = {"repo": config["repo"], "account": config["pbs"]["account"], "packs": []}
for workload, short in (("openclaw", "oc"), ("minisweagent", "ms")):
    for policy, suffix in (("round_robin", "rr"), ("cache_aware", "ca")):
        spec["packs"].append({
            "id": f"at-r-{short}-{suffix}-n8",
            "configs": [str(sources / "configs" / f"{workload}-routing-n8-{policy}.json")],
        })
result = render_packs(spec, output)
write_json(output / "pack-spec.json", spec)
# This batch adds points and does not replace any queued jobs.
commands = output / "submit-commands.txt"
commands.write_text("# Submit these four new jobs manually.\n" + "\n".join(
    line for line in commands.read_text().splitlines() if not line.startswith("#")) + "\n")
print(json.dumps(result, indent=2))
print(commands.read_text(), end="")
PY
