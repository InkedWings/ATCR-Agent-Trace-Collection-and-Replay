"""Foreground single-backend scale coordinator, run on the replay node."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import shlex
import socket
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path

from agenttrace.cli import _run_interruptible
from agenttrace.loader import create_from_spec, load_profile
from agenttrace.metrics import summarize_metrics
from agenttrace.replay.benchmark import benchmark
from agenttrace.replay.engine import _restore_workspace
from agenttrace.schema import load_trace


def allocation_remaining(job: str) -> float:
    result = subprocess.run(["qstat", "-f", "-F", "json", job], check=True, capture_output=True, text=True)
    data = next(iter(json.loads(result.stdout)["Jobs"].values()))
    def seconds(value):
        h, m, s = map(int, value.split(":"))
        return h * 3600 + m * 60 + s
    return seconds(data["Resource_List"]["walltime"]) - seconds(data["resources_used"]["walltime"])


@asynccontextmanager
async def inference_service(repo: Path, point: Path, config_path: Path, host: str):
    command = f"cd {shlex.quote(str(repo))} && exec .venv/bin/python -m agenttrace.experiments.service " + shlex.join([
        "--point", str(point), "--config", str(config_path)])
    with (point / "service.log").open("x") as log:
        process = await asyncio.create_subprocess_exec("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            host, command, stdin=asyncio.subprocess.PIPE, stdout=log, stderr=asyncio.subprocess.STDOUT)
        try:
            start = time.monotonic()
            while not (point / "ready.json").exists():
                if process.returncode is not None:
                    raise RuntimeError(f"inference service failed; see {point / 'service.log'}")
                if time.monotonic() - start > 930:
                    raise TimeoutError("inference service did not become ready")
                await asyncio.sleep(1)
            yield process
        finally:
            process.stdin.close()  # remote worker drains sampler and stops its own vLLM group
            returncode = await asyncio.wait_for(process.wait(), 30)
            if returncode:
                raise RuntimeError(f"inference worker/cleanup failed ({returncode}); see {point / 'service.log'}")


async def precache(pool: Path, output: Path, profile: Path) -> None:
    """Warm image/plugin caches without LLM or tool execution."""
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for i, line in enumerate(pool.read_text().splitlines(), 1):
        path = Path(line)
        trace = load_trace(path)
        root = output / f"{i:03d}"
        root.mkdir()
        _restore_workspace(trace, path, root / "workspace", root / "tmp")
        executor = create_from_spec(load_profile(profile)["tool_executor"])
        print(f"Precache {i}: {trace['trace_id']}", flush=True)
        try:
            await executor.setup(trace, root / "workspace")
        finally:
            await executor.close()
    (output / "complete.json").write_text(json.dumps({"pool": str(pool.resolve()), "completed_unix": time.time()}))


async def guarded_benchmark(service, *args, **kwargs):
    """A crashed inference worker must also stop admission during a tool call."""
    replay = asyncio.create_task(benchmark(*args, **kwargs))
    exited = asyncio.create_task(service.wait())
    try:
        await asyncio.wait([replay, exited], return_when=asyncio.FIRST_COMPLETED)
        if exited.done():
            raise RuntimeError("inference worker exited during benchmark")
        return await replay
    finally:
        replay.cancel()
        exited.cancel()
        await asyncio.gather(replay, exited, return_exceptions=True)


def export_summary(root: Path) -> None:
    rows = []
    for path in sorted(root.glob("*/benchmark/summary.json")):
        r = json.loads(path.read_text())
        m = r.get("measurement", {})
        backend = m.get("backend_throughput", {})
        rows.append({"point": path.parent.parent.name, "status": r["status"], "valid": m.get("valid", False),
            "task_cc": r["concurrency"], "task_per_s": m.get("task_throughput_per_second"),
            "task_latency_n": m.get("task_lifecycle_seconds", {}).get("count"),
            "task_latency_p95_s": m.get("task_lifecycle_seconds", {}).get("p95"),
            "llm_latency_p95_s": m.get("llm_latency_seconds", {}).get("p95"),
            "tool_latency_p95_s": m.get("tool_latency_seconds", {}).get("p95"),
            "client_ttft_p95_s": m.get("llm_ttft_seconds", {}).get("p95"),
            "backend_active_output_token_per_s": backend.get("active_output_tokens_per_second"),
            "backend_decode_token_per_s": backend.get("decode_tokens_per_second"),
            "prefix_hit_ratio": m.get("metrics", {}).get("vllm", {}).get("prefix_cache_hit_ratio"),
            "admitted_traces": len(m.get("admitted_trace_coverage", [])),
            "native_tool_errors": m.get("native_tool_errors")})
    if rows:
        with (root / "summary.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        lines = ["# Single-backend exploratory scaling", "", "One repetition per point; this is not an SLO capacity claim.", "",
            "| Point | Valid | Tasks/s | Backend active tokens/s | Task p95 (s) |", "|---|---|---|---|---|"]
        for row in rows:
            def fmt(value):
                if isinstance(value, bool):
                    return str(value)
                return f"{value:.3f}" if isinstance(value, (int, float)) else str(value)
            lines.append("| " + " | ".join(fmt(row[key]) for key in ("point", "valid", "task_per_s",
                "backend_active_output_token_per_s", "task_latency_p95_s")) + " |")
        (root / "summary.md").write_text("\n".join(lines) + "\n")


async def run(args):
    repo = Path.cwd()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    if socket.gethostname().split(".")[0] != config["replay_node"]:
        raise RuntimeError(f"run this coordinator on replay node {config['replay_node']}")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if args.job_id:
        remaining = allocation_remaining(args.job_id)
        required = (len(config["workloads"]) * len(config["concurrency"]) *
            (config["warmup_seconds"] + config["duration_seconds"] + 600))
        if not args.smoke and remaining < required:
            raise RuntimeError(f"only {remaining:.0f}s remain; windows + startup/drain budget need {required}s")
        print(f"Allocation remaining: {remaining / 3600:.2f} h (drain duration is workload-dependent)", flush=True)
        (root / "allocation.json").write_text(json.dumps({"job_id": args.job_id,
            "remaining_seconds_at_start": remaining, "checked_unix": time.time()}, indent=2) + "\n")
    else:
        raise ValueError("--job-id is required for allocation-time checks")
    base_url = f"http://{config['inference_node']}:{config['serve']['VLLM_PORT']}/v1"
    os.environ["AGENTTRACE_LLM_BASE_URL"] = base_url
    try:
        for workload in config["workloads"]:
            pool = args.pools.resolve() / workload / "traces.txt"
            profile = repo / config["profiles"][workload]
            paths = [Path(line) for line in pool.read_text().splitlines() if line]
            if not paths:
                raise ValueError(f"empty pool: {pool}")
            (root / f"{workload}-traces.txt").write_text(pool.read_text())
            for cc in ([2] if args.smoke else config["concurrency"]):
                if allocation_remaining(args.job_id) < config["warmup_seconds"] + config["duration_seconds"] + 600:
                    raise RuntimeError("insufficient remaining allocation for another point and drain budget")
                point = root / f"{workload}-cc{cc}"
                point.mkdir()
                print(f"Starting {point.name}: new dedicated vLLM, prefix cache initially empty", flush=True)
                async with inference_service(repo, point, config_path, config["inference_node"]) as service:
                    options = dict(profile=profile, sample_interval=1, metrics_url=base_url.removesuffix("/v1") + "/metrics",
                                   backend_events=point / "backend.jsonl")
                    if args.smoke:
                        # Shortest complete traces, still real full DAGs and tools.
                        selected = sorted(paths, key=lambda p: len(load_trace(p)["nodes"]))[:2]
                        first = await guarded_benchmark(service, selected[:1], output=point / "single", concurrency=1, **options)
                        if first["backend_throughput"]["status"] != "measured":
                            raise RuntimeError("single-trace backend/client verification failed")
                        r = await guarded_benchmark(service, selected, output=point / "benchmark", concurrency=2,
                            warmup_seconds=30, duration_seconds=30, seed=config["seed"], **options)
                    else:
                        r = await guarded_benchmark(service, paths, output=point / "benchmark", concurrency=cc,
                            warmup_seconds=config["warmup_seconds"], duration_seconds=config["duration_seconds"],
                            seed=config["seed"], **options)
                    if service.returncode is not None:
                        raise RuntimeError("inference service exited during replay")
                m = r["measurement"]
                m["inference_metrics"] = summarize_metrics(point / "inference-metrics.jsonl",
                    started_unix=m["started_unix"], ended_unix=m["ended_unix"])
                counters = r["metrics"].get("vllm", {}).get("counter_deltas", {})
                generation = sum(value for key, value in counters.items()
                    if key.split("{", 1)[0] == "vllm:generation_tokens_total")
                r["validation"] = {"backend_client_counts_match": r["backend_throughput"]["status"] == "measured",
                    "vllm_generation_counter_delta": generation,
                    "client_output_tokens": r["actual_output_tokens"],
                    "counter_counts_match": generation == r["actual_output_tokens"]}
                r["validation"]["inference_gpu_samples_ok"] = (
                    m["inference_metrics"]["samples"] > 0 and
                    m["inference_metrics"]["gpu_status_counts"] == {"ok": m["inference_metrics"]["samples"]})
                r["validation"]["window_has_backend_tokens"] = m["backend_throughput"].get("output_tokens", 0) > 0
                m["valid"] = m["valid"] and all((r["validation"]["backend_client_counts_match"],
                    r["validation"]["counter_counts_match"], m["backend_throughput"]["status"] == "measured",
                    r["validation"]["inference_gpu_samples_ok"], r["validation"]["window_has_backend_tokens"],
                    m["metrics"]["error_samples"] == 0, m["inference_metrics"]["error_samples"] == 0))
                (point / "benchmark/summary.json").write_text(json.dumps(r, indent=2) + "\n")
                export_summary(root)
                if not m["valid"]:
                    raise RuntimeError(f"point validation failed: {point}")
                print(f"Validated {point.name}; backend tokens={generation:g}", flush=True)
    finally:
        export_summary(root)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    cache = sub.add_parser("precache")
    cache.add_argument("--pool", type=Path, required=True)
    cache.add_argument("--output", type=Path, required=True)
    cache.add_argument("--profile", type=Path, required=True)
    sweep = sub.add_parser("run")
    sweep.add_argument("--config", type=Path, default=Path("examples/scaling/single-backend.json"))
    sweep.add_argument("--pools", type=Path, required=True)
    sweep.add_argument("--output", type=Path, required=True)
    sweep.add_argument("--job-id", default=os.environ.get("PBS_JOBID"))
    sweep.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    _run_interruptible(precache(args.pool, args.output, args.profile) if args.action == "precache" else run(args))


if __name__ == "__main__":
    main()
