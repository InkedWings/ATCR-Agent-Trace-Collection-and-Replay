"""Owned inference-node service: stdin EOF shuts down only its process group."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path

import httpx

from agenttrace.cli import _run_interruptible
from agenttrace.metrics import monitor


def _group_members(pgid: int) -> list[int]:
    members = []
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if int(fields[2]) == pgid and fields[0] != "Z":
            members.append(int(path.parent.name))
    return members


async def stop_process_group(process, grace_seconds: float = 15) -> None:
    """Wait for the owned group, not just its shell/launcher, to exit."""
    for sig, timeout in ((signal.SIGTERM, grace_seconds), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + timeout
        while _group_members(process.pid):
            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(.1)
        else:
            break
    await asyncio.wait_for(process.wait(), 5)
    if _group_members(process.pid):
        raise RuntimeError("owned inference processes did not exit; refusing next backend")


async def serve(point: Path, repo: Path, config: dict) -> None:
    port = int(config["serve"]["VLLM_PORT"])
    # Refuse an occupied port. Never search/kill an unowned server.
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", port))
    env = dict(os.environ, **config["serve"], VLLM_BACKEND_EVENTS=str(point / "backend.jsonl"),
        VLLM_LOG_DIR=str(point / "serve-logs"))
    stop = asyncio.Event()
    sampler = None
    process = await asyncio.create_subprocess_exec("bash",
        str(repo / "examples/minisweagent_swebench/scripts/vllm_qwen3_32b.sh"), "serve",
        env=env, start_new_session=True)
    (point / "service.json").write_text(json.dumps({"hostname": socket.gethostname(),
        "service_pid": os.getpid(), "server_process_group": process.pid,
        "config": config["serve"], "started_unix": time.time()}, indent=2) + "\n")
    reader = asyncio.StreamReader()
    transport, _ = await asyncio.get_running_loop().connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    disconnected = asyncio.create_task(reader.read())
    exited = asyncio.create_task(process.wait())
    try:
        async with httpx.AsyncClient(trust_env=False, timeout=2) as client:
            deadline = time.monotonic() + 900
            while True:
                if disconnected.done():
                    return
                if exited.done():
                    raise RuntimeError(f"vLLM exited during startup: {process.returncode}")
                try:
                    response = await client.get(f"http://127.0.0.1:{port}/health")
                    if response.is_success:
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("vLLM startup exceeded 900 seconds")
                await asyncio.sleep(1)
        ready = asyncio.Event()
        sampler = asyncio.create_task(monitor(point / "inference-metrics.jsonl",
            interval=1, label="inference", stop=stop, ready=ready))
        readiness = asyncio.create_task(ready.wait())
        await asyncio.wait([readiness, sampler], return_when=asyncio.FIRST_COMPLETED)
        if sampler.done():
            readiness.cancel()
            await sampler
        (point / "ready.json").write_text(json.dumps({"ready_unix": time.time()}))
        print("Inference ready; fresh prefix cache, hardware sampler active", flush=True)
        await asyncio.wait([disconnected, exited, sampler], return_when=asyncio.FIRST_COMPLETED)
        if exited.done():
            raise RuntimeError(f"vLLM exited unexpectedly: {process.returncode}")
        if sampler.done():
            await sampler
            raise RuntimeError("inference sampler stopped unexpectedly")
    finally:
        stop.set()
        transport.close()
        disconnected.cancel()
        try:
            if sampler:
                await sampler
        finally:
            # The launcher execs Apptainer with a private PID namespace, so
            # container workers cannot survive its init process. Also wait
            # for surviving host-side group members (including log tee).
            await stop_process_group(process)
            await exited
            await asyncio.gather(disconnected, return_exceptions=True)
        print("Owned inference service stopped", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--point", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    _run_interruptible(serve(args.point.resolve(), Path.cwd(), json.loads(args.config.read_text())))


if __name__ == "__main__":
    main()
