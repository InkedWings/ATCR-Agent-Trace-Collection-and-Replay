"""PBS coordinator. Owns services, freezes a common window, then drains and validates."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import socket
import time
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from agenttrace.experiments.scale import allocation_remaining, inference_service
from agenttrace.schema import load_trace
from .config import node_layout, positive_int, write_json
from .frontend import wait_files


async def guarded(awaitable, processes):
    """Watch owned services even when a frontend is blocked inside native tools."""
    task = asyncio.ensure_future(awaitable)
    try:
        while not task.done():
            if any(p.returncode is not None for p in processes):
                raise RuntimeError("an owned backend/frontend exited; inspect its service.log")
            await asyncio.wait([task], timeout=.25)
        if any(p.returncode is not None for p in processes):
            raise RuntimeError("an owned backend/frontend exited before completion")
        return await task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@asynccontextmanager
async def frontend_service(repo: Path, root: Path, config_path: Path, host: str, timeout: float):
    front = root / "frontends" / host
    command = shlex.join(["bash", str(repo / "examples/scaling/multinode-node.sh"), str(root), str(config_path)])
    with (front / "service.log").open("x") as log:
        process = await asyncio.create_subprocess_exec("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            host, command, stdin=asyncio.subprocess.PIPE, stdout=log, stderr=asyncio.subprocess.STDOUT)
        try:
            await guarded(wait_files([front / "ready.json"], timeout=timeout), [process])
            yield process
        finally:
            process.stdin.close()
            # Worker cancels native children on EOF, then flushes its sampler.
            code = await asyncio.wait_for(process.wait(), 60)
            if code:
                raise RuntimeError(f"frontend worker/cleanup failed ({code}): {front / 'service.log'}")


async def enter_services(stack, contexts):
    # TaskGroup cancels and joins unfinished startups before AsyncExitStack tears down ready peers.
    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(stack.enter_async_context(context)) for context in contexts]
    return [task.result() for task in tasks]


async def check_clocks(hosts: list[str], tolerance: float = .5) -> dict:
    # Start one responder per host before timing requests. SSH authentication,
    # shell setup and Python startup are not clock/network measurements.
    responder = (
        "import sys, time\n"
        "print('agenttrace-clock-ready', flush=True)\n"
        "for line in sys.stdin:\n"
        "    print(time.time(), flush=True)\n"
    )

    async def probe(host):
        samples = []
        process = await asyncio.create_subprocess_exec("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            host, shlex.join(["python3", "-u", "-c", responder]),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            ready = await asyncio.wait_for(process.stdout.readline(), 15)
            if ready.strip() != b"agenttrace-clock-ready":
                raise RuntimeError(f"clock responder did not become ready on {host}")
            for _ in range(3):
                left = time.time()
                process.stdin.write(b"probe\n")
                await process.stdin.drain()
                stdout = await asyncio.wait_for(process.stdout.readline(), 15)
                right = time.time()
                remote = float(stdout.strip())
                samples.append({"rtt_seconds": right-left, "offset_seconds": remote-(left+right)/2})
        except (TimeoutError, ValueError, ConnectionError) as error:
            raise RuntimeError(f"clock probe failed on {host}: {error}") from error
        finally:
            process.stdin.close()
            try:
                await asyncio.wait_for(process.communicate(), 5)
            except TimeoutError:
                process.kill()
                await process.communicate()
        if process.returncode:
            raise RuntimeError(f"clock responder exited with status {process.returncode} on {host}")
        best = min(samples, key=lambda s: s["rtt_seconds"])
        best["error_bound_seconds"] = abs(best["offset_seconds"]) + best["rtt_seconds"]/2
        return host, best
    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(probe(host)) for host in hosts]
    results = dict(task.result() for task in tasks)
    return {"method": "ssh_ready_request_response", "tolerance_seconds": tolerance, "hosts": results,
            "valid": all(r["error_bound_seconds"] <= tolerance for r in results.values())}


async def clock_diagnostics(hosts: list[str]) -> dict:
    """Best-effort timing diagnostics; never gate a throughput experiment on a probe."""
    try:
        async with asyncio.timeout(10):
            result = await check_clocks(hosts)
    except Exception as error:
        result = {"valid": False, "hosts": {}, "error": repr(error)}
    result["policy"] = "diagnostic_only"
    if not result["valid"]:
        print("Warning: clock probe unavailable or imprecise; continuing (see clocks*.json)", flush=True)
    return result


def prepare_run(config: dict, root: Path, nodes: list[str], host: str):
    """Validate a rendered bundle before any SSH or model launch."""
    point = config["point"]
    positive_int(point["task_cc_per_replica"], "task_cc_per_replica")
    positive_int(int(config["serve"]["VLLM_MAX_NUM_BATCHED_TOKENS"]), "token budget")
    if point.get("router_impl") == "vllm-router":
        from .official_router import verify_installation
        verify_installation(config["router"])
    mapping = node_layout(point, nodes, host)
    config["node_mapping"] = mapping
    paths = [Path(p).resolve(strict=True) for p in config["trace_paths"]]
    if len(paths) != len(set(paths)) or not paths:
        raise ValueError("rendered pool must be nonempty and unique")
    eligible = []
    image_cache = None
    if point["workload"] == "minisweagent":
        from agenttrace.miniswe_images import cached_image
        tool_config = config["replay_profile"]["tool_executor"].setdefault("config", {})
        image_cache = Path(tool_config.get("image_cache_dir") or os.environ.get("AGENTTRACE_MINISWE_IMAGE_CACHE")
                           or Path(config["repo"]) / "runs/cache/minisweagent-images").resolve()
        tool_config["image_cache_dir"] = str(image_cache)
    for path in paths:
        trace = load_trace(path)  # Validate one record at a time; do not hold full prompts for the pool.
        if image_cache is not None:
            cached_image(image_cache, trace["context"]["container_image"])
        seed = trace["context"].get("workspace_seed")
        if seed and not (path.parent / seed).is_dir():
            raise ValueError(f"missing workspace seed: {path.parent / seed}")
        if any(not (path.parent / a["path"]).is_file() for a in trace["artifacts"]):
            raise ValueError(f"missing referenced artifact: {path}")
        has_native = any(n["type"] == "tool" and n["request"]["name"] != "web_search" for n in trace["nodes"])
        if has_native and any(n["type"] == "llm" for n in trace["nodes"]):
            eligible.append((path, len(trace["nodes"])))
    if point["smoke"]:
        # Use complete short DAGs with both LLM and native-tool nodes whenever available.
        if not eligible:
            raise ValueError("smoke requires a complete trace containing an LLM and a native tool other than web_search")
        config["source_trace_paths"] = config["trace_paths"]
        config["trace_paths"] = [str(p) for p, _ in sorted(eligible, key=lambda pt: pt[1])[:2]]
    config["sample_vllm_metrics"] = True
    from agenttrace.replay.benchmark import MULTINODE_REPLAY_POLICY
    config["replay_runtime_policy"] = dict(MULTINODE_REPLAY_POLICY)
    config.setdefault("preparation_timeout_seconds", 7200)
    config.setdefault("drain_budget_seconds", 7200)
    root.mkdir(parents=True, exist_ok=False)
    for directory in ("control", "workers", "backends", "frontends"):
        (root / directory).mkdir()
    for replica in mapping["replicas"]:
        (root / "backends" / replica["id"]).mkdir()
    for front in mapping["frontends"]:
        (root / "frontends" / front).mkdir()
    write_json(root / "config.json", config)
    write_json(root / "node_mapping.json", mapping)
    return mapping


async def run(config_path: Path):
    job_id = os.environ.get("PBS_JOBID", "")
    nodefile = os.environ.get("PBS_NODEFILE")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", job_id) or not nodefile:
        raise RuntimeError("run requires PBS_JOBID and PBS_NODEFILE; use render to prepare jobs")
    config = json.loads(config_path.read_text())
    repo = Path(config["repo"])
    point = config["point"]
    root = repo / "runs/scaling/multinode" / f"{point['id']}-{job_id}"
    mapping = prepare_run(config, root, Path(nodefile).read_text().split(), socket.gethostname().split(".")[0])
    print(f"Run directory: {root}", flush=True)
    config_path = root / "config.json"
    window_seconds = point["warmup_seconds"] + point["duration_seconds"]
    try:
        remaining = await asyncio.to_thread(allocation_remaining, job_id)
        required = window_seconds + config["drain_budget_seconds"] + 930
        if remaining < required:
            raise RuntimeError(f"allocation has {remaining:.0f}s, requires at least {required}s before preparation")
        write_json(root / "allocation.json", {"job_id": job_id, "remaining_at_start_seconds": remaining})
        clocks = await clock_diagnostics(mapping["frontends"] + mapping["inference"])
        write_json(root / "clocks-startup.json", clocks)
        async with AsyncExitStack() as stack:
            print(f"Starting {len(mapping['inference'])} fresh inference replicas", flush=True)
            backends = await enter_services(stack, [inference_service(repo, root / "backends" / r["id"],
                config_path, r["inference_host"]) for r in mapping["replicas"]])
            fronts = await guarded(enter_services(stack, [frontend_service(repo, root, config_path, host,
                config["preparation_timeout_seconds"] + 120) for host in mapping["frontends"]]), backends)
            clocks = await guarded(clock_diagnostics(mapping["frontends"] + mapping["inference"]), [*backends, *fronts])
            write_json(root / "clocks.json", clocks)
            remaining = await asyncio.to_thread(allocation_remaining, job_id)
            if remaining < window_seconds + config["drain_budget_seconds"] + 60:
                raise RuntimeError("insufficient allocation after cache preparation; no measurement admitted")
            t0 = time.time() + 10
            start = {"start_unix": t0, "measurement_start_unix": t0 + point["warmup_seconds"],
                     "measurement_end_unix": t0 + window_seconds}
            write_json(root / "control/start.pending", start)
            (root / "control/start.pending").replace(root / "control/start.json")
            print(f"All nodes ready; common window: {start}", flush=True)
            await guarded(wait_files([root / "frontends" / host / "complete.json" for host in mapping["frontends"]],
                timeout=window_seconds + config["drain_budget_seconds"] + 10), [*backends, *fronts])
            # Give asynchronous Prometheus counters a final idle scrape before stopping samplers.
            await guarded(asyncio.sleep(3), [*backends, *fronts])
            after = await guarded(clock_diagnostics(mapping["frontends"] + mapping["inference"]), [*backends, *fronts])
            clocks["after_drain"] = after
            clocks["valid"] = clocks["valid"] and after["valid"]
            write_json(root / "clocks.json", clocks)
        from .report import summarize_run
        result = summarize_run(root)
        if not result["valid"]:
            raise RuntimeError(f"global count/window validation failed (collection_complete="
                f"{result['collection_complete']}, failed_tasks={result['failed_tasks']}); inspect summary.json")
        write_json(root / "status.json", {"status": "completed", "ended_unix": time.time(),
                                         "valid": True, "steady": result["steady"]})
        print(f"Validated: {root / 'summary.json'} (steady={result['steady']})", flush=True)
        return {"root": str(root), "valid": True, "steady": result["steady"]}
    except BaseException as error:
        write_json(root / "status.json", {"status": "failed", "valid": False,
            "error": str(error), "error_type": type(error).__name__, "ended_unix": time.time()})
        raise
