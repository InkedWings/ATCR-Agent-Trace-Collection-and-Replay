"""Experiment matrix, input checks, node layout and literal PBS rendering."""

from __future__ import annotations

import copy
import json
import math
import os
import re
import shlex
from pathlib import Path

WORKLOADS = ("openclaw", "minisweagent")
GROUPS = ("all", "weak-scaling", "fixed-frontend", "mechanism", "routing")
REPO = Path(__file__).resolve().parents[4]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def walltime_seconds(value: str) -> int:
    if not re.fullmatch(r"\d{2,3}:\d{2}:\d{2}", value):
        raise ValueError("PBS walltime must be HH:MM:SS")
    hours, minutes, seconds = map(int, value.split(":"))
    if minutes >= 60 or seconds >= 60 or hours*3600+minutes*60+seconds < 1:
        raise ValueError("invalid PBS walltime")
    return hours*3600+minutes*60+seconds


def pbs_for_point(config: dict, point: dict) -> dict | None:
    """Resolve explicit node-count routing before rendering; never submit to execution queues."""
    pbs = config["pbs"]
    selected = {k: v for k, v in pbs.items() if k != "overflow_routes"}
    selected["drain_budget_seconds"] = config.get("drain_budget_seconds", 7200)
    count = point["physical_nodes"]
    if pbs.get("max_nodes") is None or count <= pbs["max_nodes"]:
        return selected
    for route in pbs.get("overflow_routes", []):
        if route["min_nodes"] <= count <= route["max_nodes"]:
            return {**selected, **route}
    return None


def read_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("multi-node config schema_version must be 1")
    config["repo"] = str(Path(config.get("repo", REPO)).expanduser().resolve())
    repo = Path(config["repo"])
    for workload in WORKLOADS:
        item = config["workloads"][workload]
        for key in ("pool", "profile"):
            item[key] = str((repo / item[key]).resolve())
        positive_int(item["expected_trace_count"], "expected_trace_count")
        positive_int(item["duration_seconds"], "duration_seconds")
        if (not isinstance(item["warmup_seconds"], (float, int)) or not math.isfinite(item["warmup_seconds"])
                or item["warmup_seconds"] < 0):
            raise ValueError("warmup_seconds must be nonnegative")
    counts = config["inference_nodes"]
    if counts != sorted(set(counts)) or not counts or counts[0] != 1:
        raise ValueError("inference_nodes must be unique, increasing and start with 1")
    for n in counts:
        positive_int(n, "inference_nodes")
    serve = config["serve"]
    if str(serve["VLLM_TENSOR_PARALLEL_SIZE"]) != "4" or str(serve["VLLM_MAX_MODEL_LEN"]) != "262144":
        raise ValueError("this matrix fixes TP4 and context=262144")
    if str(serve.get("VLLM_ENABLE_THINKING")) != "1" or str(serve.get("VLLM_PREFIX_CACHING")) != "1":
        raise ValueError("the matrix requires thinking and prefix caching enabled")
    pbs = config["pbs"]
    for key in ("account", "queue"):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", pbs[key]):
            raise ValueError(f"invalid PBS {key}")
    if pbs.get("max_nodes") is not None:
        positive_int(pbs["max_nodes"], "pbs.max_nodes")
    walltime_seconds(pbs["walltime"])
    previous_max = pbs.get("max_nodes")
    for route in pbs.get("overflow_routes", []):
        if previous_max is None:
            raise ValueError("overflow routes require pbs.max_nodes")
        positive_int(route["min_nodes"], "overflow min_nodes")
        positive_int(route["max_nodes"], "overflow max_nodes")
        if route["min_nodes"] <= previous_max or route["max_nodes"] < route["min_nodes"]:
            raise ValueError("PBS node routes must be increasing and nonoverlapping")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", route["queue"]):
            raise ValueError("invalid overflow PBS queue")
        walltime_seconds(route["walltime"])
        if "drain_budget_seconds" in route:
            positive_int(route["drain_budget_seconds"], "overflow drain_budget_seconds")
        previous_max = route["max_nodes"]
    if type(config.get("seed", 42)) is not int:
        raise ValueError("seed must be an integer")
    for port in (serve["VLLM_PORT"], config.get("proxy_port", 18010)):
        if not 1 <= int(port) <= 65535:
            raise ValueError("ports must be in 1..65535")
    positive_int(config.get("preparation_timeout_seconds", 7200), "preparation_timeout_seconds")
    positive_int(config.get("drain_budget_seconds", 7200), "drain_budget_seconds")
    if "router" in config:
        from .official_router import POLICIES, VERSION
        router = config["router"]
        router["python"] = str((repo / router["python"]).absolute())
        if router["version"] != VERSION:
            raise ValueError(f"official routing requires version {VERSION}")
        if not router["policies"] or len(set(router["policies"])) != len(router["policies"]) or any(
                p not in POLICIES for p in router["policies"]):
            raise ValueError("invalid official router policies")
        if not router["inference_nodes"] or len(set(router["inference_nodes"])) != len(router["inference_nodes"]):
            raise ValueError("router inference_nodes must be nonempty and unique")
        for n in router["inference_nodes"]:
            positive_int(n, "router inference_nodes")
        for key in ("balance_abs_threshold", "eviction_interval_secs", "max_tree_size",
                    "request_timeout_secs", "max_concurrent_requests", "threads"):
            positive_int(router[key], "router." + key)
        if not 0 < router["cache_threshold"] <= 1 or not 1 < router["balance_rel_threshold"] < float("inf"):
            raise ValueError("invalid cache/load thresholds")
        if not 1 <= router["prometheus_port"] <= 65535 or router["prometheus_port"] == config.get("proxy_port", 18010):
            raise ValueError("router metrics port must be valid and distinct from proxy port")
    return config


def matrix(config: dict, *, smoke: bool = False, group: str = "all") -> list[dict]:
    if group not in GROUPS or (smoke and group not in ("all", "routing")):
        raise ValueError("select a supported group; smoke uses group=all or routing")
    points = []
    if group == "routing":
        if "router" not in config:
            raise ValueError("routing group requires official router configuration")
        specs = [(n, "fixed_frontend", policy, "normal", True)
                 for n in config["router"]["inference_nodes"] for policy in config["router"]["policies"]]
    elif smoke:
        # Single-inference-node replay has already been validated; exercise cross-node routing here.
        specs = [(2, "balanced", "round_robin", "none", True)]
    else:
        specs = [(n, layout, "sticky", "normal", False)
                 for n in config["inference_nodes"]
                 for layout in (["balanced"] if n == 1 else ["balanced", "fixed_frontend"])]
        specs += [(4, "balanced", routing, reuse, True)
                  for routing in ("sticky", "round_robin") for reuse in ("normal", "none")]
    for n, layout, routing, reuse, proxy in specs:
        for workload in WORKLOADS:
            item = config["workloads"][workload]
            official = group == "routing"
            mechanism = proxy and not smoke and not official
            kind = ("routing-smoke" if smoke else "routing") if official else "smoke" if smoke else "mechanism" if mechanism else layout
            name = f"{workload}-{kind}-n{n}"
            if mechanism:
                name += f"-{routing}-{reuse}"
            if official:
                name += f"-{routing}"
            points.append({"id": name, "workload": workload, "inference_replicas": n,
                "layout": layout, "frontend_nodes": n if layout == "balanced" else 1,
                "physical_nodes": 2*n if layout == "balanced" else n+1,
                "routing": routing, "prefix_reuse": reuse, "proxy": proxy,
                "smoke": smoke, "task_cc_per_replica": 2 if smoke else item.get("task_cc_per_replica"),
                "warmup_seconds": 30 if smoke else item["warmup_seconds"],
                "duration_seconds": 60 if smoke else item["duration_seconds"],
                "seed": config.get("seed", 42)})
            if official:
                points[-1]["router_impl"] = "vllm-router"
    if group != "all":
        points = [p for p in points if (
            (group == "weak-scaling" and p["layout"] == "balanced" and not p["proxy"])
            or (group == "fixed-frontend" and p["layout"] == "fixed_frontend")
            or (group == "mechanism" and p["proxy"])
            or (group == "routing" and p.get("router_impl") == "vllm-router"))]
    for point in points:
        selection = pbs_for_point(config, point)
        if selection is not None:
            point["pbs"] = {k: selection[k] for k in ("queue", "walltime", "drain_budget_seconds")}
    return points


def pool_paths(path: Path) -> list[Path]:
    return [(path.parent / line.strip()).resolve(strict=True)
            for line in path.read_text().splitlines() if line.strip()]


def check(config: dict, *, smoke: bool = False, group: str = "all") -> dict:
    pending = []
    if group == "routing":
        from .official_router import verify_installation
        verify_installation(config["router"])
    budget = config["serve"].get("VLLM_MAX_NUM_BATCHED_TOKENS")
    if budget is None:
        pending.append("serve.VLLM_MAX_NUM_BATCHED_TOKENS")
    elif not str(budget).isdigit() or int(budget) < int(config["serve"]["VLLM_MAX_NUM_SEQS"]):
        raise ValueError("token budget must be an integer >= max_num_seqs")
    pools = {}
    for name in WORKLOADS:
        item = config["workloads"][name]
        cc = item.get("task_cc_per_replica")
        if cc is None:
            pending.append(f"workloads.{name}.task_cc_per_replica")
        else:
            positive_int(cc, f"{name}.task_cc_per_replica")
        paths = pool_paths(Path(item["pool"]))
        if len(paths) != item["expected_trace_count"] or len(set(paths)) != len(paths):
            raise ValueError(f"{name}: pool count/uniqueness differs from frozen configuration")
        if any(not path.is_file() for path in paths):
            raise ValueError(f"{name}: trace path is not a file")
        from agenttrace.schema import load_trace
        images = set()
        for path in paths:
            trace = load_trace(path)
            if name == "minisweagent" and group == "routing":
                images.add(trace["context"]["container_image"])
            seed = trace["context"].get("workspace_seed")
            if seed and not (path.parent / seed).is_dir():
                raise ValueError(f"missing workspace seed: {path.parent / seed}")
            for artifact in trace["artifacts"]:
                if not (path.parent / artifact["path"]).is_file():
                    raise ValueError(f"missing trace artifact: {path.parent / artifact['path']}")
        profile = json.loads(Path(item["profile"]).read_text())
        if images:
            from agenttrace.miniswe_images import cached_image
            cache = Path(profile["tool_executor"]["config"].get("image_cache_dir")
                         or os.environ.get("AGENTTRACE_MINISWE_IMAGE_CACHE")
                         or Path(config["repo"]) / "runs/cache/minisweagent-images")
            for source in images:
                cached_image(cache, source)
        llm = profile["llm_executor"]["config"]
        if (llm.get("model_override") != config["serve"]["VLLM_SERVED_MODEL_NAME"] or not llm.get("ignore_eos")
                or llm.get("max_tokens_field", "max_tokens") != "max_tokens"):
            raise ValueError(f"{name}: profile must use the configured model and exact output targets")
        if name == "openclaw" and profile["tool_executor"]["config"].get("web_search_mode") != "recorded_delay":
            raise ValueError("OpenClaw must use web_search recorded_delay")
        pools[name] = len(paths)
    repo = Path(config["repo"])
    for relative in (".venv/bin/python", "examples/minisweagent_swebench/scripts/vllm_qwen3_32b.sh",
                     ".tools/openclaw/bin/openclaw"):
        if not (repo / relative).exists():
            raise ValueError(f"required runtime path is missing: {repo / relative}")
    points = matrix(config, smoke=smoke, group=group)
    if not points:
        raise ValueError("the selected experiment group contains no points")
    limit = config["pbs"].get("max_nodes")
    deferred = [{"id": p["id"], "physical_nodes": p["physical_nodes"],
                 "reason": "no configured PBS queue route covers this node count"}
                for p in points if "pbs" not in p]
    queue_counts = {}
    for point in points:
        if "pbs" not in point:
            continue
        selection = point["pbs"]
        required = point["warmup_seconds"] + point["duration_seconds"] + selection["drain_budget_seconds"] + 930
        if walltime_seconds(selection["walltime"]) < required:
            raise ValueError(f"{point['id']}: PBS walltime cannot fit startup, warmup, measurement and drain budget")
        queue_counts[selection["queue"]] = queue_counts.get(selection["queue"], 0) + 1
    return {"ready_to_render": (smoke or not pending) and len(deferred) < len(points), "pending_baseline_fields": pending,
        "queue": config["pbs"]["queue"], "queue_node_limit": limit,
        "queue_counts": queue_counts,
        "queue_deferred_points": deferred, "renderable_points": len(points)-len(deferred),
        "pool_counts": pools, "points": len(points), "max_nodes": max(p["physical_nodes"] for p in points),
        "warmup_and_measurement_node_hours": sum(p["physical_nodes"] *
            (p["warmup_seconds"] + p["duration_seconds"]) / 3600 for p in points),
        "scope": "count/schema/referenced-path/profile checks; no inference, submission or dataset mutation"}


def node_layout(point: dict, allocated: list[str], batch_host: str) -> dict:
    nodes = list(dict.fromkeys(n.split(".")[0] for n in allocated))
    batch_host = batch_host.split(".")[0]
    if len(nodes) != point["physical_nodes"] or batch_host not in nodes:
        raise ValueError("PBS allocation does not match the requested topology")
    nodes.remove(batch_host)
    nodes.insert(0, batch_host)
    f = point["frontend_nodes"]
    frontends, inference = nodes[:f], nodes[f:]
    return {"frontends": frontends, "inference": inference,
        "replicas": [{"id": f"r{i:02d}", "index": i, "inference_host": host,
                      "frontend_host": frontends[i % f]}
                     for i, host in enumerate(inference)]}


def render(config: dict, output: Path, *, smoke: bool = False, group: str = "all") -> dict:
    status = check(config, smoke=smoke, group=group)
    if not status["ready_to_render"]:
        if status["pending_baseline_fields"] and not smoke:
            raise ValueError("fill the frozen baseline fields first: " + ", ".join(status["pending_baseline_fields"]))
        raise ValueError("no experiment points fit the configured queue node limit")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "jobs").mkdir()
    (output / "configs").mkdir()
    planned = matrix(config, smoke=smoke, group=group)
    deferred_ids = {p["id"] for p in status["queue_deferred_points"]}
    points = [p for p in planned if p["id"] not in deferred_ids]
    commands = []
    index = []
    for i, point in enumerate(points, 1):
        job = copy.deepcopy(config)
        job["point"] = point
        job["pbs"] = pbs_for_point(config, point)
        job["drain_budget_seconds"] = job["pbs"].pop("drain_budget_seconds")
        if smoke and job["serve"].get("VLLM_MAX_NUM_BATCHED_TOKENS") is None:
            job["serve"]["VLLM_MAX_NUM_BATCHED_TOKENS"] = "8192"
        job["serve"] = {k: str(v) for k, v in job["serve"].items()}
        # Freeze profile and pool contents into the bundle. Trace artifacts stay in place.
        source = config["workloads"][point["workload"]]
        job["trace_paths"] = [str(p) for p in pool_paths(Path(source["pool"]))]
        job["replay_profile"] = json.loads(Path(source["profile"]).read_text())
        config_path = output / "configs" / f"{point['id']}.json"
        write_json(config_path, job)
        script = output / "jobs" / f"{point['id']}.pbs"
        repo = shlex.quote(config["repo"])
        job_name = f"at-mn-{i:02d}"
        if point.get("router_impl") == "vllm-router":
            short_workload = "oc" if point["workload"] == "openclaw" else "ms"
            short_policy = {"round_robin": "rr", "cache_aware": "ca", "power_of_two": "p2", "consistent_hash": "ch"}[point["routing"]]
            job_name = f"at-{'s' if smoke else 'r'}-{short_workload}-{short_policy}-n{point['inference_replicas']}"
        text = f'''#!/bin/bash -l
#PBS -N {job_name}
#PBS -A {config["pbs"]["account"]}
#PBS -q {job["pbs"]["queue"]}
#PBS -l select={point["physical_nodes"]}:system=polaris
#PBS -l place=scatter
#PBS -l walltime={job["pbs"]["walltime"]}
#PBS -l filesystems=home:eagle
#PBS -j oe
#PBS -r n
set -euo pipefail
umask 077
: "${{PBS_JOBID:?Run this script inside PBS}}"
: "${{PBS_NODEFILE:?PBS_NODEFILE is required}}"
cd {repo}
# PBS output can be delayed; keep a live, unique coordinator log in the bundle.
exec > {shlex.quote(str(output))}/coordinator-${{PBS_JOBID}}-{point['id']}.log 2>&1
exec .venv/bin/python -u -m agenttrace.experiments.multinode run --config {shlex.quote(str(config_path))}
'''
        script.write_text(text)
        commands.append("qsub " + shlex.quote(str(script)))
        index.append({**point, "config": str(config_path), "pbs_script": str(script)})
    (output / "submit-commands.txt").write_text("# Submit manually, in this order; avoid overlapping measurement jobs.\n" +
                                                 f"# Queue-deferred points: {len(deferred_ids)}; see matrix.json.\n" +
                                                 "\n".join(commands) + "\n")
    if not smoke:
        groups = {
            "weak-scaling": [p for p in index if p["layout"] == "balanced" and not p["proxy"]],
            "fixed-frontend": [p for p in index if p["layout"] == "fixed_frontend" and not p["proxy"]],
            "mechanism": [p for p in index if p["proxy"] and p.get("router_impl") != "vllm-router"],
            "routing": [p for p in index if p.get("router_impl") == "vllm-router"],
        }
        for name, selected in groups.items():
            (output / f"submit-{name}.txt").write_text(
                f"# {name}: {len(selected)} jobs. Submit manually, one at a time.\n" +
                "".join("qsub " + shlex.quote(p["pbs_script"]) + "\n" for p in selected))
    write_json(output / "matrix.json", {"check": status, "points": index, "planned_points": planned,
                                       "deferred_points": status["queue_deferred_points"]})
    return {"output": str(output), "pbs_scripts": len(points), "submitted_jobs": 0, **status}
