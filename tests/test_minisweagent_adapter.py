from __future__ import annotations

import json

from agenttrace.adapters.minisweagent import (
    MiniSWEAgentToolExecutor,
    build_minisweagent_trace,
)


def test_minisweagent_call_pairing_and_sequential_tools(tmp_path):
    trajectory = tmp_path / "trajectory.json"
    capture = tmp_path / "calls.jsonl"
    trajectory.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "fix it"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [],
                        "extra": {
                            "response": {"id": "response-1"},
                            "actions": [
                                {"command": "pwd", "tool_call_id": "call-1"},
                                {"command": "ls", "tool_call_id": "call-2"},
                            ],
                        },
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call-1",
                        "content": "one",
                        "extra": {"returncode": 0},
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call-2",
                        "content": "two",
                        "extra": {"returncode": 0},
                    },
                    {
                        "role": "assistant",
                        "content": "",
                        "extra": {"response": {"id": "response-2"}, "actions": []},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    capture.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in [
                {
                    "sequence": 1,
                    "trace_id": "t",
                    "call_id": "l1",
                    "response_id": "response-1",
                    "protocol": "openai-chat-completions",
                    "endpoint": "/chat/completions",
                    "request": {},
                    "output_tokens": 3,
                },
                {
                    "sequence": 2,
                    "trace_id": "t",
                    "call_id": "failed-retry",
                    "response_id": None,
                    "protocol": "openai-chat-completions",
                    "endpoint": "/chat/completions",
                    "request": {},
                    "status": 502,
                    "output_tokens": None,
                },
                {
                    "sequence": 3,
                    "trace_id": "t",
                    "call_id": "l2",
                    "response_id": "response-2",
                    "protocol": "openai-chat-completions",
                    "endpoint": "/chat/completions",
                    "request": {},
                    "output_tokens": 2,
                },
            ]
        ),
        encoding="utf-8",
    )

    trace = build_minisweagent_trace(
        trace_id="t",
        trajectory_path=trajectory,
        capture_path=capture,
        workload="SWE-bench_Lite/dev",
        framework_version="2.4.5",
        captured_workspace_root="/old/workspace",
        source_repository="owner/repo",
        base_commit="abc",
        instance_id="owner__repo-1",
    )
    by_id = {node["id"]: node for node in trace["nodes"]}
    assert by_id["tool-001"]["depends_on"] == ["llm-001"]
    assert by_id["tool-002"]["depends_on"] == ["tool-001"]
    assert by_id["llm-002"]["depends_on"] == ["tool-002"]
    assert by_id["tool-001"]["recorded_result"]["toolCallId"] == "call-1"
    assert by_id["tool-001"]["request"]["protocol"] == "minisweagent-singularity"
    assert trace["context"]["source_repository"] == "owner/repo"
    assert trace["context"]["container_cwd"] == "/testbed"
    assert trace["context"]["container_image"].endswith(
        "sweb.eval.x86_64.owner_1776_repo-1:latest"
    )


def test_minisweagent_rewrites_replay_paths_to_container_paths():
    executor = MiniSWEAgentToolExecutor({})
    executor.replay_workspace = "/replay/workspace"
    executor.replay_tmp = "/replay/tmp"
    executor.container_cwd = "/testbed"

    arguments = {"command": "cat /replay/tmp/input > /replay/workspace/output"}
    rewritten = executor._rewrite_arguments(arguments)
    assert rewritten == {"command": "cat /tmp/input > /testbed/output"}
