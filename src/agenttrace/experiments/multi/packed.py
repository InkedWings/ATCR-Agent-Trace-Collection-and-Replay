"""Run existing experiments on disjoint subsets of one PBS allocation."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import re
import shlex
import socket
import sys
from pathlib import Path

from agenttrace.cli import _run_interruptible
from .config import node_layout, positive_int, walltime_seconds, write_json


def partition(pack: dict, allocated: list[str], batch_host: str) -> dict:
    nodes = list(dict.fromkeys(host.split(".")[0] for host in allocated))
    host = batch_host.split(".")[0]
    if len(nodes) != pack["allocated_nodes"] or host not in nodes:
        raise ValueError("pack does not match the PBS allocation")
    nodes.remove(host)
    nodes.insert(0, host)
    offset, members = 0, []
    for member in pack["members"]:
        point = json.loads(Path(member["config"]).read_text())["point"]
        count = positive_int(point["physical_nodes"], "physical_nodes")
        subset = nodes[offset:offset + count]
        if len(subset) != count:
            raise ValueError("experiments exceed the allocated nodes")
        mapping = node_layout(point, subset, subset[0])
        members.append({**member, "nodes": subset, "coordinator": subset[0], "node_mapping": mapping})
        offset += count
    return {"allocated_nodes": nodes, "members": members, "unused_nodes": nodes[offset:]}


def render(spec: dict, output: Path) -> dict:
    """Copy frozen inputs; never alter an existing bundle or submit a job."""
    repo = Path(spec["repo"]).resolve()
    output = output.resolve()
    packs, inputs, seen = [], {}, set()
    for item in spec["packs"]:
        name = item["id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or any(p["id"] == name for p in packs):
            raise ValueError("pack IDs must be unique and shell-safe")
        pack = {"id": name, "repo": str(repo), "allocated_nodes": 10,
                "queue": "prod", "walltime": "03:00:00", "members": []}
        for source_name in item["configs"]:
            source = (repo / source_name).resolve(strict=True)
            config = json.loads(source.read_text())
            point = config["point"]
            if point["id"] in seen or Path(config["repo"]).resolve() != repo:
                raise ValueError("duplicate experiment or inconsistent repository")
            seen.add(point["id"])
            required = point["warmup_seconds"] + point["duration_seconds"] + config["drain_budget_seconds"] + 930
            if required > walltime_seconds(pack["walltime"]):
                raise ValueError("experiment does not fit the pack walltime")
            config = copy.deepcopy(config)
            config["pbs"] = {"account": spec["account"], "queue": "prod", "max_nodes": 10,
                             "walltime": pack["walltime"]}
            config["point"]["pbs"] = {"queue": "prod", "walltime": pack["walltime"],
                                      "drain_budget_seconds": config["drain_budget_seconds"]}
            config["allocation_pack"] = {"id": name, "allocated_nodes": 10,
                                         "experiment_nodes": point["physical_nodes"], "source_config": str(source)}
            target = output / "configs" / f"{point['id']}.json"
            inputs[target] = config
            pack["members"].append({"id": point["id"], "config": str(target),
                                    "physical_nodes": point["physical_nodes"]})
        if not pack["members"] or sum(m["physical_nodes"] for m in pack["members"]) > 10:
            raise ValueError("each pack must fit in ten nodes")
        pack["replaces_jobs"] = item.get("replaces_jobs", [])
        packs.append(pack)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", spec["account"]):
        raise ValueError("invalid PBS account")
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("configs", "jobs", "packs"):
        (output / folder).mkdir()
    for path, config in inputs.items():
        write_json(path, config)
    commands = []
    for pack in packs:
        manifest = output / "packs" / f"{pack['id']}.json"
        write_json(manifest, pack)
        script = output / "jobs" / f"{pack['id']}.pbs"
        script.write_text(f'''#!/bin/bash -l
#PBS -N {pack['id']}
#PBS -A {spec['account']}
#PBS -q prod
#PBS -l select=10:system=polaris
#PBS -l place=scatter
#PBS -l walltime=03:00:00
#PBS -l filesystems=home:eagle
#PBS -j oe
#PBS -r n
set -euo pipefail
umask 077
: "${{PBS_JOBID:?Run inside PBS}}"
: "${{PBS_NODEFILE:?PBS_NODEFILE is required}}"
cd {shlex.quote(str(repo))}
exec > {shlex.quote(str(output))}/pack-${{PBS_JOBID}}-{pack['id']}.log 2>&1
exec .venv/bin/python -u -m agenttrace.experiments.multi.packed run --manifest {shlex.quote(str(manifest))}
''')
        commands.append("qsub " + shlex.quote(str(script)))
    (output / "submit-commands.txt").write_text("# Submit manually after cancelling the replaced queued jobs.\n" + "\n".join(commands) + "\n")
    write_json(output / "matrix.json", {"packs": packs, "submitted_jobs": 0})
    return {"output": str(output), "packs": len(packs), "experiments": len(inputs), "submitted_jobs": 0}


async def until_disconnect(operation, disconnected):
    task = asyncio.create_task(operation)
    try:
        done, _ = await asyncio.wait([task, disconnected], return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            return await task
        raise RuntimeError("pack coordinator disconnected; cancelling this experiment")
    finally:
        task.cancel()
        disconnected.cancel()
        await asyncio.gather(task, disconnected, return_exceptions=True)


async def child(config: Path):
    from .runtime import run
    reader = asyncio.StreamReader()
    transport, _ = await asyncio.get_running_loop().connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    try:
        return await until_disconnect(run(config), asyncio.create_task(reader.read()))
    finally:
        transport.close()


async def run_member(member: dict, pack: dict, root: Path, job_id: str) -> dict:
    nodefile = root / f"{member['id']}.nodes"
    nodefile.write_text("\n".join(member["nodes"]) + "\n")
    command = "cd " + shlex.quote(pack["repo"]) + " && exec " + shlex.join([
        "env", f"PBS_JOBID={job_id}", f"PBS_NODEFILE={nodefile}", ".venv/bin/python", "-u", "-m",
        "agenttrace.experiments.multi.packed", "child", "--config", member["config"]])
    process = None
    try:
        with (root / f"{member['id']}.log").open("x") as log:
            # Each coordinator runs on its own frontend, including the second half of a 5+5 pack.
            process = await asyncio.create_subprocess_exec("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                member["coordinator"], command, stdin=asyncio.subprocess.PIPE,
                stdout=log, stderr=asyncio.subprocess.STDOUT)
            code = await process.wait()
        result = {"id": member["id"], "exit_code": code, "status": "completed" if code == 0 else "failed"}
    except Exception as error:
        result = {"id": member["id"], "exit_code": 1, "status": "failed", "error": repr(error)}
    finally:
        if process is not None:
            process.stdin.close()
            if process.returncode is None:
                try:
                    await asyncio.wait_for(process.wait(), 120)
                except TimeoutError:
                    process.kill()
                    await process.wait()
    write_json(root / f"{member['id']}.status.json", result)
    return result


async def run_pack(manifest: Path) -> dict:
    pack = json.loads(manifest.read_text())
    job_id, nodefile = os.environ.get("PBS_JOBID", ""), os.environ.get("PBS_NODEFILE")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", job_id) or not nodefile:
        raise RuntimeError("packed run requires a PBS allocation")
    layout = partition(pack, Path(nodefile).read_text().split(), socket.gethostname())
    root = manifest.parent.parent / "allocations" / f"{pack['id']}-{job_id}"
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "layout.json", {"job_id": job_id, **layout})
    print(f"Pack layout and coordinator logs: {root}", flush=True)
    # A member failure is recorded without cancelling the other independent experiment.
    tasks = [asyncio.create_task(run_member(m, pack, root, job_id)) for m in layout["members"]]
    joined = asyncio.gather(*tasks)
    try:
        # Deliver cancellation once, in finally, so a second cancellation cannot cut cleanup short.
        results = await asyncio.shield(joined)
        result = {"status": "completed" if all(r["exit_code"] == 0 for r in results) else "failed", "members": results}
    except BaseException:
        write_json(root / "status.json", {"status": "interrupted"})
        raise
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(joined, *tasks, return_exceptions=True)
    write_json(root / "status.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("render")
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--manifest", type=Path, required=True)
    p = sub.add_parser("child")
    p.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "render":
        result = render(json.loads(args.spec.read_text()), args.output)
    elif args.action == "run":
        result = _run_interruptible(run_pack(args.manifest.resolve()))
    else:
        result = _run_interruptible(child(args.config.resolve()))
    print(json.dumps(result, indent=2))
    if result.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
