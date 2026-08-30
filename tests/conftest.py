from __future__ import annotations

import copy

import pytest


@pytest.fixture
def minimal_trace() -> dict:
    return {
        "schema_version": 2,
        "trace_id": "trace-1",
        "source": {"framework": "test", "workload": "unit", "version": "1"},
        "context": {
            "captured_workspace_root": "/captured/workspace",
            "captured_tmp_root": "/tmp",
            "workspace_seed": None,
        },
        "artifacts": [],
        "nodes": [
            {
                "id": "llm-001",
                "type": "llm",
                "depends_on": [],
                "request": {
                    "protocol": "openai-chat-completions",
                    "endpoint": "/v1/chat/completions",
                    "payload": {"model": "test", "messages": []},
                },
                "output_tokens": 4,
            }
        ],
    }


@pytest.fixture
def clone_trace(minimal_trace):
    return lambda: copy.deepcopy(minimal_trace)
