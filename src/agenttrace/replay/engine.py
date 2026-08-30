"""Framework-independent single-trace DAG replay."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from agenttrace.interfaces import LLMExecutor, ToolExecutor
from agenttrace.schema import load_trace

from .bindings import Bindings


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        if _sha256(source) != artifact["sha256"]:
            raise ValueError(f"trace artifact checksum mismatch: {source}")
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
    bindings = _restore_workspace(trace, path, workspace, tmp)
    llm_started = False
    tool_started = False
    try:
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

        by_id = {node["id"]: node for node in trace["nodes"]}
        tasks: dict[str, asyncio.Task[None]] = {}
        results: dict[str, dict[str, Any]] = {}
        start_gate = asyncio.Event()
        replay_started = time.perf_counter()

        async def execute(node_id: str) -> None:
            node = by_id[node_id]
            await start_gate.wait()
            await asyncio.gather(*(tasks[parent] for parent in node["depends_on"]))
            started = time.perf_counter()
            result: dict[str, Any] = {
                "node_id": node_id,
                "type": node["type"],
                "started_seconds": round(started - replay_started, 6),
                "status": "completed",
            }
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

        for node in trace["nodes"]:
            tasks[node["id"]] = asyncio.create_task(execute(node["id"]))
        start_gate.set()
        try:
            await asyncio.gather(*tasks.values())
        except BaseException:
            for task in tasks.values():
                task.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)
            raise
        makespan = time.perf_counter() - replay_started
        return {
            "schema_version": 1,
            "trace_id": trace["trace_id"],
            "trace_path": str(path),
            "run_dir": str(root),
            "dry_run": dry_run,
            "setup_seconds": round(setup_seconds, 6),
            "replay_makespan_seconds": round(makespan, 6),
            "nodes": [results[node["id"]] for node in trace["nodes"]],
        }
    finally:
        if tool_started and tool_executor is not None:
            await tool_executor.close()
        if llm_started and llm_executor is not None:
            await llm_executor.close()
