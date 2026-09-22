"""Freeze four controlled followups into three 10-node prod jobs; do not submit."""

import argparse
import copy
import json
import shlex
import subprocess
import tempfile
from pathlib import Path

from agenttrace.experiments.multi.config import write_json
from agenttrace.experiments.multi.packed import partition, render
from agenttrace.experiments.multi.runtime import prepare_run


def prepare(output: Path, baselines: Path) -> dict:
    output, baselines = output.resolve(), baselines.resolve()
    sources = output.with_name(output.name + "-sources")
    if output.exists() or sources.exists():
        raise ValueError("choose a new output directory; existing bundles are never overwritten")

    def baseline(name):
        path = baselines / "configs" / f"{name}.json"
        config = json.loads(path.read_text())
        config.pop("allocation_pack", None)
        config["baseline_config"] = str(path)
        return config

    routing = []
    for workload in ("openclaw", "minisweagent"):
        config = baseline(f"{workload}-routing-n4-round_robin")
        config["point"].update(id=f"{workload}-routing-n4-power_of_two", routing="power_of_two")
        config["router"]["policies"] = ["power_of_two"]
        routing.append(config)

    frontend = []
    for count, builds in ((2, 2), (1, 8)):
        config = baseline("minisweagent-fixed_frontend-n8")
        config["point"].update(id=f"minisweagent-frontend-f{count}-n8-build{builds}",
                               layout="frontend_ratio", frontend_nodes=count, physical_nodes=count + 8)
        config["replay_profile"]["tool_executor"]["config"]["max_parallel_sandbox_builds"] = builds
        frontend.append(config)

    configs = routing + frontend
    repo = Path(configs[0]["repo"]).resolve()
    checks = []
    # Uses only local validation: no SSH, GPU launch, image staging, or job submission.
    with tempfile.TemporaryDirectory(prefix="agenttrace-followups-", dir="/tmp") as temporary:
        for config in configs:
            point = config["point"]
            if Path(config["repo"]).resolve() != repo:
                raise ValueError("baseline repositories differ")
            if len(config["trace_paths"]) != config["workloads"][point["workload"]]["expected_trace_count"]:
                raise ValueError("frozen pool count differs from baseline")
            nodes = [f"check-node-{i}" for i in range(point["physical_nodes"])]
            mapping = prepare_run(copy.deepcopy(config), Path(temporary) / point["id"], nodes, nodes[0])
            builds = config["replay_profile"]["tool_executor"]["config"].get("max_parallel_sandbox_builds")
            checks.append({"id": point["id"], "trace_count": len(config["trace_paths"]),
                           "frontend_nodes": point["frontend_nodes"], "inference_nodes": len(mapping["inference"]),
                           "workers_per_frontend": [sum(r["frontend_host"] == h for r in mapping["replicas"])
                                                    for h in mapping["frontends"]],
                           "total_task_cc": len(mapping["replicas"]) * point["task_cc_per_replica"],
                           "build_slots_per_frontend": builds,
                           "total_build_slots": builds * point["frontend_nodes"] if builds else None})

    (sources / "configs").mkdir(parents=True)
    for config in configs:
        write_json(sources / "configs" / f"{config['point']['id']}.json", config)
    spec = {"repo": str(repo), "account": configs[0]["pbs"]["account"], "packs": []}
    for name, members in (("at-p2-n4", routing), ("at-ms-f2-n8-b2", frontend[:1]),
                          ("at-ms-f1-n8-b8", frontend[1:])):
        spec["packs"].append({"id": name, "configs": [
            str(sources / "configs" / f"{config['point']['id']}.json") for config in members]})
    result = render(spec, output)
    write_json(output / "pack-spec.json", spec)
    commands = []
    for item in spec["packs"]:
        name = item["id"]
        pack = json.loads((output / "packs" / f"{name}.json").read_text())
        partition(pack, [f"node-{i}.alcf" for i in range(10) for _ in range(64)], "node-3.alcf")
        script = output / "jobs" / f"{name}.pbs"
        subprocess.run(["bash", "-n", str(script)], check=True)
        commands.append("qsub " + shlex.quote(str(script)))
    (output / "submit-commands.txt").write_text("# Three new jobs; submit manually.\n" + "\n".join(commands) + "\n")
    write_json(output / "validation.json", {"experiments": checks, "pbs_syntax_ok": True,
               "pack_partition_ok": True, "submitted_jobs": 0, "gpu_experiments_started": 0,
               "scope": "frozen inputs, schema, referenced paths, SIF presence, router installation and node mapping"})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, default=Path("runs/prod-packs-20260921"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.baselines), indent=2))
    print((args.output / "submit-commands.txt").read_text(), end="")
