"""Framework-independent single-trace DAG replay."""

from __future__ import annotations

import asyncio
import copy
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from agenttrace.interfaces import LLMExecutor, ToolExecutor
from agenttrace.schema import load_trace

from .bindings import Bindings


def _restore_workspace(
    trace: dict[str, Any], trace_path: Path, workspace: Path, tmp: Path
) -> Bindings:
    context = trace["context"]
    seed_value = context.get("workspace_seed")
    if seed_value:
        seed = (trace_path.parent / seed_value).resolve()
        if not seed.is_dir():
            raise FileNotFoundError(f"workspace seed is missing: {seed}")
        shutil.copytree(seed, workspace)
    else:
        workspace.mkdir()
    tmp.mkdir()

    bindings = Bindings(
        {
            context["captured_workspace_root"]: str(workspace),
            context.get("captured_tmp_root", "/tmp"): str(tmp),
        }
    )
    for artifact in trace["artifacts"]:
        source = (trace_path.parent / artifact["path"]).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"trace artifact is missing: {source}")
        workspace_path = artifact.get("workspace_path")
        if workspace_path:
            destination = workspace / workspace_path
        else:
            destination = tmp / f"{artifact['id']}-{source.name}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        bindings.add(artifact["captured_path"], str(destination))
    return bindings


async def replay_trace(
    trace_path: str | Path,
    *,
    llm_executor: LLMExecutor | None = None,
    tool_executor: ToolExecutor | None = None,
    run_dir: str | Path | None = None,
    dry_run: bool = False,
    event_path: str | Path | None = None,
) -> dict[str, Any]:
    """Replay one trace and return performance-only measurements."""

    path = Path(trace_path).resolve()
    trace = load_trace(path)
    if run_dir is None:
        root = Path(tempfile.mkdtemp(prefix=f"agenttrace-{trace['trace_id']}-"))
    else:
        root = Path(run_dir).resolve()
        root.mkdir(parents=True, exist_ok=False)
    workspace = root / "workspace"
    tmp = root / "tmp"

    setup_started = time.perf_counter()
    started_unix = time.time()
    events = Path(event_path).open("x", encoding="utf-8", buffering=1) if event_path else None

    def emit(event: str, **fields: Any) -> None:
        if events:
            events.write(json.dumps({"event": event, "trace_id": trace["trace_id"],
                "timestamp_unix": time.time(), "task_elapsed_seconds": time.perf_counter() - setup_started,
                **fields}) + "\n")

    emit("setup_started")
    llm_started = False
    tool_started = False
    try:
        bindings = _restore_workspace(trace, path, workspace, tmp)
        if not dry_run and any(node["type"] == "llm" for node in trace["nodes"]):
            if llm_executor is None:
                raise RuntimeError("trace contains LLM nodes but no LLM executor was configured")
            llm_started = True
            await llm_executor.setup(trace, workspace)
        if not dry_run and any(node["type"] == "tool" for node in trace["nodes"]):
            if tool_executor is None:
                raise RuntimeError("trace contains tool nodes but no tool executor was configured")
            tool_started = True
            await tool_executor.setup(trace, workspace)
        setup_seconds = time.perf_counter() - setup_started
        emit("replay_started", setup_seconds=setup_seconds)

        by_id = {node["id"]: node for node in trace["nodes"]}
        tasks: dict[str, asyncio.Task[None]] = {}
        results: dict[str, dict[str, Any]] = {}
        node_starts: dict[str, float] = {}
        start_gate = asyncio.Event()
        replay_started = time.perf_counter()

        async def execute(node_id: str) -> None:
            node = by_id[node_id]
            await start_gate.wait()
            await asyncio.gather(*(tasks[parent] for parent in node["depends_on"]))
            started = time.perf_counter()
            node_starts[node_id] = started
            result: dict[str, Any] = {
                "node_id": node_id,
                "type": node["type"],
                "started_seconds": round(started - replay_started, 6),
                "started_unix": time.time(),
                "status": "completed",
            }
            emit("node_started", node_id=node_id, type=node["type"])
            if dry_run:
                if node["type"] == "llm":
                    result["target_output_tokens"] = node["output_tokens"]
                    result["actual_output_tokens"] = node["output_tokens"]
                else:
                    result["native_error"] = node["recorded_result"]["isError"]
            elif node["type"] == "llm":
                assert llm_executor is not None
                execution = await llm_executor.execute(node)
                result["target_output_tokens"] = node["output_tokens"]
                result["actual_output_tokens"] = execution.actual_output_tokens
                result["ttft_seconds"] = execution.ttft_seconds
                result["output_stream_seconds"] = execution.output_stream_seconds
                result["output_chunks"] = execution.output_chunks
                result["tpot_estimate_seconds"] = (
                    execution.output_stream_seconds / (execution.actual_output_tokens - 1)
                    if execution.output_stream_seconds is not None and execution.actual_output_tokens > 1
                    and execution.output_chunks > 1
                    else None
                )
            else:
                assert tool_executor is not None
                replay_node = copy.deepcopy(node)
                replay_node["request"]["arguments"] = bindings.rewrite(
                    replay_node["request"]["arguments"]
                )
                execution = await tool_executor.execute(replay_node)
                result["native_error"] = execution.is_error
                bindings.learn_from_results(node["recorded_result"], execution.result)
            result["elapsed_seconds"] = round(time.perf_counter() - started, 6)
            results[node_id] = result
            emit("node_completed", **result)

        async def dispatch(node_id: str) -> None:
            try:
                await execute(node_id)
            except BaseException as error:
                emit("node_cancelled" if isinstance(error, asyncio.CancelledError) else "node_failed",
                    node_id=node_id, error_type=type(error).__name__,
                    execution_started=node_id in node_starts,
                    elapsed_seconds=time.perf_counter() - node_starts[node_id] if node_id in node_starts else None)
                raise

        for node in trace["nodes"]:
            tasks[node["id"]] = asyncio.create_task(dispatch(node["id"]))
        start_gate.set()
        try:
            await asyncio.gather(*tasks.values())
        except BaseException as error:
            emit("replay_failed", error_type=type(error).__name__)
            for task in tasks.values():
                task.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)
            raise
        makespan = time.perf_counter() - replay_started
        emit("replay_completed", replay_makespan_seconds=makespan)
        return {
            "schema_version": 1,
            "trace_id": trace["trace_id"],
            "trace_path": str(path),
            "run_dir": str(root),
            "dry_run": dry_run,
            "started_unix": started_unix,
            "replay_started_unix": started_unix + (replay_started - setup_started),
            "setup_seconds": round(setup_seconds, 6),
            "replay_makespan_seconds": round(makespan, 6),
            "nodes": [results[node["id"]] for node in trace["nodes"]],
        }
    finally:
        try:
            if tool_started and tool_executor is not None:
                await tool_executor.close()
        finally:
            try:
                if llm_started and llm_executor is not None:
                    await llm_executor.close()
            finally:
                emit("closed")
                if events:
                    events.close()
