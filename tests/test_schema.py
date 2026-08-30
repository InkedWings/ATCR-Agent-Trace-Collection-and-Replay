from __future__ import annotations

import pytest

from agenttrace.schema import validate_trace


def tool_node(node_id: str, depends_on: list[str], call_id: str = "call-1") -> dict:
    return {
        "id": node_id,
        "type": "tool",
        "depends_on": depends_on,
        "request": {
            "protocol": "openclaw-tools-invoke",
            "name": "read",
            "arguments": {"path": "a.txt"},
        },
        "recorded_result": {
            "toolCallId": call_id,
            "toolName": "read",
            "content": [],
            "details": {},
            "isError": False,
        },
    }


def test_valid_v2(minimal_trace):
    validate_trace(minimal_trace)


def test_v1_has_explicit_version_error(minimal_trace):
    minimal_trace["schema_version"] = 1
    with pytest.raises(ValueError, match="unsupported trace schema_version 1"):
        validate_trace(minimal_trace)


def test_unknown_dependency(minimal_trace):
    minimal_trace["nodes"].append(tool_node("tool-001", ["missing"]))
    with pytest.raises(ValueError, match="unknown dependency"):
        validate_trace(minimal_trace)


def test_cycle(minimal_trace):
    minimal_trace["nodes"] = [
        tool_node("tool-001", ["tool-002"], "call-1"),
        tool_node("tool-002", ["tool-001"], "call-2"),
    ]
    with pytest.raises(ValueError, match="cycle"):
        validate_trace(minimal_trace)


def test_tool_result_is_required(minimal_trace):
    node = tool_node("tool-001", ["llm-001"])
    del node["recorded_result"]["details"]
    minimal_trace["nodes"].append(node)
    with pytest.raises(ValueError, match="recorded_result.details"):
        validate_trace(minimal_trace)
