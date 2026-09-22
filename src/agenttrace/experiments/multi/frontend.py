"""One owned frontend service per physical node, possibly hosting many replicas."""

from __future__ import annotations

import asyncio
import copy
import getpass
import json
import socket
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path

from agenttrace.experiments.scale import precache
from agenttrace.metrics import monitor
from agenttrace.replay.benchmark import benchmark
from .config import write_json
from .proxy import Router


async def wait_files(paths: list[Path], tasks=(), timeout: float = 7200):
    async with asyncio.timeout(timeout):
        while not all(p.is_file() for p in paths):
            for task in tasks:
                if task.done():
                    await task
                    raise RuntimeError("worker stopped before readiness")
            await asyncio.sleep(.1)


async def work(root: Path, config: dict, host: str):
    front = root / "frontends" / host
    point = config["point"]
    mapping = config["node_mapping"]
    replicas = [r for r in mapping["replicas"] if r["frontend_host"] == host]
    if not replicas:
        raise ValueError("frontend has no assigned replicas")
    profile = copy.deepcopy(config["replay_profile"])
    profile["llm_executor"]["config"].update(base_url_env="AGENTTRACE_LLM_BASE_URL", trust_env=False)
    profile["llm_executor"]["config"].pop("base_url", None)
    if point.get("router_impl") == "vllm-router":
        profile["llm_executor"]["config"]["routing_envelope"] = True
    if point["workload"] == "minisweagent":
        from agenttrace.miniswe_images import stage_images

        tool_config = profile["tool_executor"]["config"]
        shared = Path(tool_config["image_cache_dir"])
        local = Path("/local/scratch") / getpass.getuser() / "agenttrace-multinode/minisweagent-images"
        if tool_config.get("max_parallel_sandbox_builds", 0):
            tool_config["sandbox_build_lock_dir"] = str(local.parent / "sandbox-build-slots")
        # Smoke also stages the complete source pool, then replays only its short traces.
        sources = [json.loads(Path(p).read_text())["context"]["container_image"]
                   for p in config.get("source_trace_paths", config["trace_paths"])]
        async with asyncio.timeout(config["preparation_timeout_seconds"]):
            staged = await asyncio.to_thread(stage_images, shared, local, sources)
        write_json(front / "image-cache.json", staged)
        tool_config["image_cache_dir"] = str(local)
    profile_path = front / "profile.json"
    write_json(profile_path, profile)
    paths = [Path(p) for p in config["trace_paths"]]
    pool = front / "traces.txt"
    pool.write_text("".join(str(p) + "\n" for p in paths))
    print(f"Precache {len(paths)} complete traces on {host}", flush=True)
    async with asyncio.timeout(config["preparation_timeout_seconds"]):
        await precache(pool, front / "precache", profile_path)
    stop, ready = asyncio.Event(), asyncio.Event()
    sampler = asyncio.create_task(monitor(front / "metrics.jsonl", label="frontend", interval=1,
                                           stop=stop, ready=ready))
    workers = []
    try:
        readiness = asyncio.create_task(ready.wait())
        try:
            await asyncio.wait([readiness, sampler], return_when=asyncio.FIRST_COMPLETED)
            if sampler.done():
                await sampler
                raise RuntimeError("frontend sampler stopped before readiness")
        finally:
            readiness.cancel()
            await asyncio.gather(readiness, return_exceptions=True)
        async with AsyncExitStack() as stack:
            monitors = [sampler]
            endpoints = {r["id"]: f"http://{r['inference_host']}:{config['serve']['VLLM_PORT']}/v1"
                         for r in mapping["replicas"]}
            proxy_url = None
            if point["proxy"]:
                if point.get("router_impl") == "vllm-router":
                    from .official_router import OfficialRouter
                    proxy = OfficialRouter(endpoints, point["routing"], config["router"], front / "router")
                else:
                    proxy = Router(endpoints, point["routing"], point["prefix_reuse"], root.name,
                                   front / "routing.jsonl", offset=mapping["frontends"].index(host))
                proxy_url = await stack.enter_async_context(proxy.serve(config.get("proxy_port", 18010)))
                if point.get("router_impl") == "vllm-router":
                    monitors.append(proxy.failure)
            for replica in replicas:
                rid = replica["id"]
                workers.append(asyncio.create_task(benchmark(paths, profile=profile_path,
                    output=root / "workers" / rid, concurrency=point["task_cc_per_replica"],
                    warmup_seconds=point["warmup_seconds"], duration_seconds=point["duration_seconds"],
                    seed=point["seed"] + replica["index"], sample_hardware=False,
                    continue_on_http_error=True,
                    control_dir=root / "control", task_namespace=f"{root.name}/{rid}",
                    child_env={"AGENTTRACE_HOME_REPLICA": rid,
                               "AGENTTRACE_LLM_BASE_URL": proxy_url or endpoints[rid]})))
            try:
                await wait_files([root / "workers" / r["id"] / "ready.json" for r in replicas],
                                 [*monitors, *workers], config["preparation_timeout_seconds"])
                write_json(front / "ready.json", {"ready_unix": time.time(), "replicas": replicas})
                pending = set(workers)
                while pending:
                    done, _ = await asyncio.wait([*pending, *monitors], return_when=asyncio.FIRST_COMPLETED)
                    for service in monitors:
                        if service in done:
                            await service
                            raise RuntimeError("frontend sampler/router stopped during replay")
                    for task in done:
                        await task
                        pending.remove(task)
                write_json(front / "complete.json", {"completed_unix": time.time()})
                # Keep the sampler alive until every frontend has drained and coordinator closes stdin.
                await asyncio.shield(sampler)
                raise RuntimeError("frontend sampler stopped before coordinator shutdown")
            finally:
                for task in workers:
                    task.cancel()
                await asyncio.gather(*workers, return_exceptions=True)
    finally:
        stop.set()
        await sampler


async def serve_frontend(root: Path, config: dict):
    host = socket.gethostname().split(".")[0]
    reader = asyncio.StreamReader()
    transport, _ = await asyncio.get_running_loop().connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    disconnected = asyncio.create_task(reader.read())
    task = asyncio.create_task(work(root, config, host))
    try:
        done, _ = await asyncio.wait([task, disconnected], return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            await task
        elif not (root / "frontends" / host / "complete.json").is_file():
            print("Coordinator disconnected; cancelling owned replay children", flush=True)
    finally:
        task.cancel()
        disconnected.cancel()
        results = await asyncio.gather(task, disconnected, return_exceptions=True)
        transport.close()
        if isinstance(results[0], BaseException) and not isinstance(results[0], asyncio.CancelledError):
            raise results[0]
