"""Own an unmodified, separately installed official vLLM Router process."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx

VERSION = "0.1.15"
POLICIES = ("round_robin", "cache_aware", "power_of_two")


def verify_installation(settings: dict) -> dict:
    probe = subprocess.run([settings["python"], "-c",
        "import json, importlib.metadata, vllm_router_rs; "
        "print(json.dumps({'version': importlib.metadata.version('vllm-router'), 'rust_extension': True}))"],
        check=True, capture_output=True, text=True, timeout=30)
    result = json.loads(probe.stdout)
    if result["version"] != VERSION or settings["version"] != VERSION:
        raise ValueError(f"routing experiment requires vllm-router {VERSION}")
    return result


def command(settings: dict, policy: str, endpoints: dict[str, str], port: int) -> list[str]:
    if policy not in POLICIES or not endpoints:
        raise ValueError("invalid official router policy/endpoints")
    args = [settings["python"], "-m", "vllm_router.launch_router", "--host", "127.0.0.1",
            "--port", str(port), "--policy", policy, "--worker-urls",
            *(url.removesuffix("/v1") for url in endpoints.values()),
            "--intra-node-data-parallel-size", "1", "--disable-retries", "--disable-circuit-breaker",
            "--worker-startup-timeout-secs", "120", "--worker-startup-check-interval", "1",
            "--prometheus-host", "127.0.0.1", "--log-level", "info"]
    for key in ("prometheus_port", "cache_threshold", "balance_abs_threshold", "balance_rel_threshold",
                "eviction_interval_secs", "max_tree_size", "request_timeout_secs", "max_concurrent_requests"):
        args.extend(["--" + key.replace("_", "-"), str(settings[key])])
    return args


class OfficialRouter:
    def __init__(self, endpoints: dict, policy: str, settings: dict, output: Path):
        self.endpoints, self.policy, self.settings, self.output = endpoints, policy, settings, output

    async def _watch(self):
        code = await self.process.wait()
        raise RuntimeError(f"official router exited ({code}); see {self.output / 'service.log'}")

    async def _sample(self, client):
        url = f"http://127.0.0.1:{self.settings['prometheus_port']}/metrics"
        with (self.output / "metrics.jsonl").open("x", buffering=1) as handle:
            while True:
                row = {"timestamp_unix": time.time()}
                try:
                    response = await client.get(url)
                    response.raise_for_status()
                    row["prometheus"] = response.text
                except httpx.HTTPError as error:
                    row["error"] = type(error).__name__
                handle.write(json.dumps(row) + "\n")
                await asyncio.sleep(1)

    @asynccontextmanager
    async def serve(self, port: int):
        # Do not mistake another process's health/metrics endpoints for our own.
        for candidate in (port, self.settings["prometheus_port"]):
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("127.0.0.1", candidate))
        self.output.mkdir(exist_ok=False)
        args = command(self.settings, self.policy, self.endpoints, port)
        env = {k: v for k, v in os.environ.items()
               if k.lower() not in ("http_proxy", "https_proxy", "all_proxy", "no_proxy")
               and k not in ("RUST_LOG", "RUST_LOG_STYLE")}
        env.update(NO_PROXY="*", no_proxy="*", TOKIO_WORKER_THREADS=str(self.settings.get("threads", 4)))
        (self.output / "config.json").write_text(json.dumps({
            "component": "vllm-router", "version": VERSION, "policy": self.policy,
            "command": args, "settings": self.settings, "endpoints": self.endpoints,
            "transport": "lossless_chat_via_completions", "retries": False}, indent=2) + "\n")
        self.failure = None
        sampler = None
        with (self.output / "service.log").open("x") as log:
            self.process = await asyncio.create_subprocess_exec(*args, env=env, stdout=log,
                                                               stderr=asyncio.subprocess.STDOUT)
            try:
                async with httpx.AsyncClient(trust_env=False, timeout=3) as client:
                    deadline = time.monotonic() + 130
                    url = f"http://127.0.0.1:{port}"
                    while True:
                        if self.process.returncode is not None:
                            raise RuntimeError(f"official router startup failed; see {self.output / 'service.log'}")
                        try:
                            response = await client.get(url + "/health")
                            if response.is_success:
                                break
                        except httpx.HTTPError:
                            pass
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f"official router startup timed out; see {self.output / 'service.log'}")
                        await asyncio.sleep(.2)
                    self.failure = asyncio.create_task(self._watch())
                    sampler = asyncio.create_task(self._sample(client))
                    yield url + "/v1"
            finally:
                for task in (sampler, self.failure):
                    if task is not None:
                        task.cancel()
                await asyncio.gather(*(t for t in (sampler, self.failure) if t is not None), return_exceptions=True)
                if self.process.returncode is None:
                    self.process.terminate()
                    try:
                        await asyncio.wait_for(self.process.wait(), 10)
                    except TimeoutError:
                        self.process.kill()
                        await self.process.wait()
