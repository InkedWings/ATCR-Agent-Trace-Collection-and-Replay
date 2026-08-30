from __future__ import annotations

import json

import pytest

from agenttrace.adapters.openclaw import build_openclaw_trace


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_tool_call_result_pairing_and_sibling_dag(tmp_path):
    trajectory = tmp_path / "trajectory.jsonl"
    capture = tmp_path / "calls.jsonl"
    write_jsonl(
        trajectory,
        [
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "responseId": "response-1",
                    "content": [
                        {"type": "toolCall", "id": "call-a", "name": "read", "arguments": {"path": "a"}},
                        {"type": "toolCall", "id": "call-b", "name": "read", "arguments": {"path": "b"}},
                    ],
                },
            },
            {
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "call-b",
                    "toolName": "read",
                    "content": [],
                    "isError": True,
                },
            },
            {
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "call-a",
                    "toolName": "read",
                    "content": [],
                    "details": {"bytes": 1},
                    "isError": False,
                },
            },
            {
                "type": "message",
                "message": {"role": "assistant", "responseId": "response-2", "content": []},
            },
        ],
    )
    write_jsonl(
        capture,
        [
            {"sequence": 1, "trace_id": "t", "call_id": "l1", "response_id": "response-1", "protocol": "openai-chat-completions", "endpoint": "/chat/completions", "request": {}, "output_tokens": 3},
            {"sequence": 2, "trace_id": "t", "call_id": "l2", "response_id": "response-2", "protocol": "openai-chat-completions", "endpoint": "/chat/completions", "request": {}, "output_tokens": 2},
        ],
    )
    trace = build_openclaw_trace(
        trace_id="t",
        trajectory_path=trajectory,
        capture_path=capture,
        workload="GAIA",
        framework_version="test",
        captured_workspace_root="/old/workspace",
        workspace_seed=None,
        artifacts=[],
    )
    by_id = {node["id"]: node for node in trace["nodes"]}
    assert by_id["tool-001"]["depends_on"] == ["llm-001"]
    assert by_id["tool-002"]["depends_on"] == ["llm-001"]
    assert by_id["llm-002"]["depends_on"] == ["tool-001", "tool-002"]
    assert by_id["tool-002"]["recorded_result"]["toolCallId"] == "call-b"
    assert by_id["tool-002"]["recorded_result"]["details"] == {}


def test_missing_tool_result_is_rejected(tmp_path):
    trajectory = tmp_path / "trajectory.jsonl"
    capture = tmp_path / "calls.jsonl"
    write_jsonl(
        trajectory,
        [{"type": "message", "message": {"role": "assistant", "responseId": "r", "content": [{"type": "toolCall", "id": "missing", "name": "read", "arguments": {}}]}}],
    )
    write_jsonl(
        capture,
        [{"sequence": 1, "trace_id": "t", "call_id": "l", "response_id": "r", "protocol": "openai-chat-completions", "endpoint": "/chat/completions", "request": {}, "output_tokens": 1}],
    )
    with pytest.raises(ValueError, match="missing OpenClaw tool result"):
        build_openclaw_trace(
            trace_id="t",
            trajectory_path=trajectory,
            capture_path=capture,
            workload="GAIA",
            framework_version="test",
            captured_workspace_root="/old",
            workspace_seed=None,
            artifacts=[],
        )
