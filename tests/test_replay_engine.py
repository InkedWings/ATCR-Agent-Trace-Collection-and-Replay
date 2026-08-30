from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import pytest

from agenttrace.interfaces import LLMExecutionResult, ToolExecutionResult
from agenttrace.replay.engine import replay_trace


def tool(node_id: str, parent: str, call_id: str, *, is_error: bool = False) -> dict:
    return {
        "id": node_id,
        "type": "tool",
        "depends_on": [parent],
        "request": {
            "protocol": "openclaw-tools-invoke",
            "name": "read",
            "arguments": {"path": "/captured/workspace/input.txt"},
        },
        "recorded_result": {
            "toolCallId": call_id,
            "toolName": "read",
            "content": [],
            "details": {},
            "isError": is_error,
        },
    }


class FakeLLM:
    def __init__(self, order: list[str]):
        self.order = order
        self.closed = False

    async def setup(self, trace, workspace):
        self.workspace = workspace

    async def execute(self, node):
        self.order.append(node["id"])
        return LLMExecutionResult(node["output_tokens"])

    async def close(self):
        self.closed = True


class ParallelTools:
    def __init__(self, order: list[str]):
        self.order = order
        self.active = 0
        self.both_started = asyncio.Event()
        self.closed = False

    async def setup(self, trace, workspace):
        self.workspace = workspace

    async def execute(self, node):
        assert str(self.workspace) in node["request"]["arguments"]["path"]
        self.order.append(f"start:{node['id']}")
        self.active += 1
        if self.active == 2:
            self.both_started.set()
        await asyncio.wait_for(self.both_started.wait(), 1)
        self.order.append(f"end:{node['id']}")
        return ToolExecutionResult(
            {"content": [], "details": {}, "isError": node["id"] == "tool-002"}
        )

    async def close(self):
        self.closed = True


def test_parallel_siblings_join_and_native_error_continue(tmp_path, minimal_trace):
    trace = copy.deepcopy(minimal_trace)
    trace["nodes"].extend(
        [
            tool("tool-001", "llm-001", "call-1"),
            tool("tool-002", "llm-001", "call-2", is_error=True),
            {
                "id": "llm-002",
                "type": "llm",
                "depends_on": ["tool-001", "tool-002"],
                "request": {
                    "protocol": "openai-chat-completions",
                    "endpoint": "/v1/chat/completions",
                    "payload": {"messages": [{"role": "user", "content": "recorded"}]},
                },
                "output_tokens": 2,
            },
        ]
    )
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    order: list[str] = []
    llm = FakeLLM(order)
    tools = ParallelTools(order)
    report = asyncio.run(
        replay_trace(
            trace_path,
            llm_executor=llm,
            tool_executor=tools,
            run_dir=tmp_path / "replay",
        )
    )
    assert order[0] == "llm-001"
    assert order.index("start:tool-001") < order.index("llm-002")
    assert order.index("end:tool-001") < order.index("llm-002")
    assert order.index("end:tool-002") < order.index("llm-002")
    by_id = {row["node_id"]: row for row in report["nodes"]}
    assert by_id["tool-002"]["native_error"] is True
    assert llm.closed and tools.closed


class FailingTool:
    def __init__(self):
        self.calls = 0

    async def setup(self, trace, workspace):
        pass

    async def execute(self, node):
        self.calls += 1
        raise http_error()

    async def close(self):
        pass


def http_error():
    return RuntimeError("transport failed")


def test_transport_failure_is_fail_fast_without_retry(tmp_path, minimal_trace):
    minimal_trace["nodes"] = [tool("tool-001", "llm-001", "call-1")]
    minimal_trace["nodes"][0]["depends_on"] = []
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(minimal_trace), encoding="utf-8")
    executor = FailingTool()
    with pytest.raises(RuntimeError, match="transport failed"):
        asyncio.run(
            replay_trace(path, tool_executor=executor, run_dir=tmp_path / "replay")
        )
    assert executor.calls == 1
