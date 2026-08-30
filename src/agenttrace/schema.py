"""Validation for the framework-independent AgentTrace schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def validate_trace(trace: dict[str, Any]) -> None:
    """Validate one schema-v2 trace, including its dependency DAG."""

    version = trace.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported trace schema_version {version!r}; expected {SCHEMA_VERSION}"
        )
    _string(trace.get("trace_id"), "trace_id")

    source = _object(trace.get("source"), "source")
    for field in ("framework", "workload", "version"):
        _string(source.get(field), f"source.{field}")

    context = _object(trace.get("context"), "context")
    _string(context.get("captured_workspace_root"), "context.captured_workspace_root")
    workspace_seed = context.get("workspace_seed")
    if workspace_seed is not None:
        _string(workspace_seed, "context.workspace_seed")
    captured_tmp_root = context.get("captured_tmp_root", "/tmp")
    _string(captured_tmp_root, "context.captured_tmp_root")

    artifacts = trace.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("artifacts must be a list")
    artifact_ids: set[str] = set()
    for index, artifact_value in enumerate(artifacts):
        artifact = _object(artifact_value, f"artifacts[{index}]")
        artifact_id = _string(artifact.get("id"), f"artifacts[{index}].id")
        if artifact_id in artifact_ids:
            raise ValueError(f"duplicate artifact id: {artifact_id}")
        artifact_ids.add(artifact_id)
        _string(artifact.get("kind"), f"artifacts[{index}].kind")
        _string(artifact.get("captured_path"), f"artifacts[{index}].captured_path")
        _string(artifact.get("path"), f"artifacts[{index}].path")
        _string(artifact.get("sha256"), f"artifacts[{index}].sha256")
        workspace_path = artifact.get("workspace_path")
        if workspace_path is not None:
            _string(workspace_path, f"artifacts[{index}].workspace_path")

    nodes = trace.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("trace must contain at least one node")

    by_id: dict[str, dict[str, Any]] = {}
    for index, node_value in enumerate(nodes):
        node = _object(node_value, f"nodes[{index}]")
        node_id = _string(node.get("id"), f"nodes[{index}].id")
        if node_id in by_id:
            raise ValueError(f"duplicate node id: {node_id}")
        node_type = node.get("type")
        if node_type not in {"llm", "tool"}:
            raise ValueError(f"invalid node type for {node_id}: {node_type!r}")
        dependencies = node.get("depends_on")
        if not isinstance(dependencies, list) or not all(
            isinstance(parent, str) and parent for parent in dependencies
        ):
            raise ValueError(f"depends_on for {node_id} must be a list of node IDs")
        if len(dependencies) != len(set(dependencies)):
            raise ValueError(f"duplicate dependency for {node_id}")

        request = _object(node.get("request"), f"request for {node_id}")
        _string(request.get("protocol"), f"request.protocol for {node_id}")
        if node_type == "llm":
            _string(request.get("endpoint"), f"request.endpoint for {node_id}")
            _object(request.get("payload"), f"request.payload for {node_id}")
            output_tokens = node.get("output_tokens")
            if not isinstance(output_tokens, int) or isinstance(output_tokens, bool):
                raise ValueError(f"output_tokens for {node_id} must be an integer")
            if output_tokens < 0:
                raise ValueError(f"output_tokens for {node_id} must be non-negative")
        else:
            _string(request.get("name"), f"request.name for {node_id}")
            _object(request.get("arguments"), f"request.arguments for {node_id}")
            result = _object(node.get("recorded_result"), f"recorded_result for {node_id}")
            _string(result.get("toolCallId"), f"recorded_result.toolCallId for {node_id}")
            _string(result.get("toolName"), f"recorded_result.toolName for {node_id}")
            if result["toolName"] != request["name"]:
                raise ValueError(f"tool name mismatch for {node_id}")
            if not isinstance(result.get("content"), list):
                raise ValueError(f"recorded_result.content for {node_id} must be a list")
            _object(result.get("details"), f"recorded_result.details for {node_id}")
            if not isinstance(result.get("isError"), bool):
                raise ValueError(f"recorded_result.isError for {node_id} must be boolean")
        by_id[node_id] = node

    for node_id, node in by_id.items():
        for parent in node["depends_on"]:
            if parent not in by_id:
                raise ValueError(f"unknown dependency {parent!r} for {node_id}")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise ValueError("trace dependency graph contains a cycle")
        if node_id in visited:
            return
        visiting.add(node_id)
        for parent in by_id[node_id]["depends_on"]:
            visit(parent)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in by_id:
        visit(node_id)


def load_trace(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("trace root must be an object")
    validate_trace(value)
    return value


def write_trace(path: str | Path, trace: dict[str, Any]) -> None:
    validate_trace(trace)
    Path(path).write_text(
        json.dumps(trace, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
