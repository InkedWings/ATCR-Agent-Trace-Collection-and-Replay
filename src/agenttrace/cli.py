"""AgentTrace command-line interface."""

from __future__ import annotations

import argparse
import asyncio
import json
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
        )
    )
    _write_or_print(report, args.output)
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
    replay.set_defaults(function=_replay)

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
