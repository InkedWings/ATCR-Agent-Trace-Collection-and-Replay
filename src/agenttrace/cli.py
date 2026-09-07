"""AgentTrace command-line interface."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
from pathlib import Path
from typing import Any

from agenttrace.capture.openai import CaptureProxy
from agenttrace.loader import create_from_spec, load_profile
from agenttrace.replay.engine import replay_trace
from agenttrace.schema import load_trace


def _write_or_print(value: dict[str, Any], output: Path | None) -> None:
    text = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


def _validate(args: argparse.Namespace) -> int:
    trace = load_trace(args.trace)
    counts = {
        kind: sum(node["type"] == kind for node in trace["nodes"])
        for kind in ("llm", "tool")
    }
    print(json.dumps({"ok": True, "trace_id": trace["trace_id"], **counts}))
    return 0


def _replay(args: argparse.Namespace) -> int:
    llm_executor = None
    tool_executor = None
    if args.profile:
        profile = load_profile(args.profile)
        if "llm_executor" in profile:
            llm_executor = create_from_spec(profile["llm_executor"])
        if "tool_executor" in profile:
            tool_executor = create_from_spec(profile["tool_executor"])
    report = asyncio.run(
        replay_trace(
            args.trace,
            llm_executor=llm_executor,
            tool_executor=tool_executor,
            run_dir=args.run_dir,
            dry_run=args.dry_run,
            event_path=args.events,
        )
    )
    _write_or_print(report, args.output)
    return 0


def _run_interruptible(coroutine):
    async def run():
        task = asyncio.current_task()
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGTERM, task.cancel)
        try:
            return await coroutine
        finally:
            loop.remove_signal_handler(signal.SIGTERM)
    try:
        return asyncio.run(run())
    except (KeyboardInterrupt, asyncio.CancelledError):
        raise SystemExit(130)


def _benchmark(args: argparse.Namespace) -> int:
    from agenttrace.replay.benchmark import benchmark

    traces = list(args.traces)
    if args.trace_list:
        # One path per line; paths are relative to the list file.
        traces.extend(args.trace_list.resolve().parent / line.strip()
            for line in args.trace_list.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#"))
    report = _run_interruptible(benchmark(traces, profile=args.profile, output=args.output_dir,
        concurrency=args.concurrency, repeat=args.repeat, dry_run=args.dry_run,
        sample_interval=args.sample_interval, metrics_url=args.vllm_metrics_url,
        backend_events=args.backend_events, warmup_seconds=args.warmup_seconds,
        duration_seconds=args.duration_seconds, seed=args.seed))
    print(json.dumps({key: value for key, value in report.items() if key != "tasks"}, indent=2))
    return 0


def _monitor(args: argparse.Namespace) -> int:
    from agenttrace.metrics import monitor

    _run_interruptible(monitor(args.output, interval=args.interval,
        metrics_url=args.vllm_metrics_url, duration=args.duration, label=args.label))
    return 0


def _backend_throughput(args: argparse.Namespace) -> int:
    from agenttrace.vllm_backend import summarize_backend

    result = summarize_backend(args.events, started_unix=args.started_unix,
        ended_unix=args.ended_unix)
    _write_or_print(result, args.output)
    return 0


def _capture_openai(args: argparse.Namespace) -> int:
    proxy = CaptureProxy(args.upstream, args.output, args.host, args.port)
    proxy.start()
    print(json.dumps({"base_url": proxy.base_url, "output": str(args.output)}), flush=True)
    try:
        proxy._thread.join()
    except KeyboardInterrupt:
        proxy.close()
    return 0


def _build_openclaw(args: argparse.Namespace) -> int:
    from agenttrace.adapters.openclaw import write_openclaw_trace

    trace = write_openclaw_trace(
        trace_id=args.trace_id,
        trajectory_path=args.trajectory,
        capture_path=args.capture,
        output_path=args.output,
        workload=args.workload,
        framework_version=args.framework_version,
        captured_workspace_root=args.captured_workspace_root,
        workspace_seed=args.workspace_seed,
        artifacts=[],
    )
    print(json.dumps({"ok": True, "nodes": len(trace["nodes"])}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agenttrace")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate a schema-v2 trace")
    validate.add_argument("trace", type=Path)
    validate.set_defaults(function=_validate)

    replay = subparsers.add_parser("replay", help="replay one trace DAG")
    replay.add_argument("trace", type=Path)
    replay.add_argument("--profile", type=Path)
    replay.add_argument("--run-dir", type=Path)
    replay.add_argument("--output", type=Path)
    replay.add_argument("--dry-run", action="store_true")
    replay.add_argument("--events", type=Path, help="append live performance events to a new JSONL file")
    replay.set_defaults(function=_replay)

    bench = subparsers.add_parser("benchmark", help="bounded finite-list or steady-load replay")
    bench.add_argument("traces", nargs="*", type=Path)
    bench.add_argument("--trace-list", type=Path)
    bench.add_argument("--profile", type=Path)
    bench.add_argument("--output-dir", type=Path, required=True)
    bench.add_argument("--concurrency", type=int, default=1)
    mode = bench.add_mutually_exclusive_group()
    mode.add_argument("--repeat", type=int)
    mode.add_argument("--duration-seconds", type=float)
    bench.add_argument("--warmup-seconds", type=float, default=0)
    bench.add_argument("--seed", type=int, default=42)
    bench.add_argument("--sample-interval", type=float, default=1)
    bench.add_argument("--vllm-metrics-url")
    bench.add_argument("--backend-events", type=Path, help="shared vLLM request-timing JSONL file")
    bench.add_argument("--dry-run", action="store_true")
    bench.set_defaults(function=_benchmark)

    metrics = subparsers.add_parser("monitor", help="sample this node's hardware and optional vLLM metrics")
    metrics.add_argument("--output", type=Path, required=True)
    metrics.add_argument("--interval", type=float, default=1)
    metrics.add_argument("--duration", type=float)
    metrics.add_argument("--vllm-metrics-url")
    metrics.add_argument("--label", default="node")
    metrics.set_defaults(function=_monitor)

    backend = subparsers.add_parser("backend-throughput", help="summarize backend busy-time token throughput")
    backend.add_argument("events", type=Path)
    backend.add_argument("--started-unix", type=float)
    backend.add_argument("--ended-unix", type=float)
    backend.add_argument("--output", type=Path)
    backend.set_defaults(function=_backend_throughput)

    capture = subparsers.add_parser(
        "capture-openai", help="run an OpenAI-compatible capture proxy"
    )
    capture.add_argument("--upstream", required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--host", default="127.0.0.1")
    capture.add_argument("--port", type=int, default=18080)
    capture.set_defaults(function=_capture_openai)

    build = subparsers.add_parser(
        "build-openclaw-trace", help="convert OpenClaw trajectory plus LLM capture"
    )
    build.add_argument("--trace-id", required=True)
    build.add_argument("--trajectory", type=Path, required=True)
    build.add_argument("--capture", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--workload", default="unknown")
    build.add_argument("--framework-version", default="unknown")
    build.add_argument("--captured-workspace-root", required=True)
    build.add_argument("--workspace-seed")
    build.set_defaults(function=_build_openclaw)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
