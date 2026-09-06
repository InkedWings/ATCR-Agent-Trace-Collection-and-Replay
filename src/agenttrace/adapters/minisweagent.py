"""mini-SWE-agent trajectory conversion and native shell replay."""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from typing import Any

from agenttrace.interfaces import ToolExecutionResult
from agenttrace.schema import validate_trace, write_trace


def swebench_container_image(instance_id: str) -> str:
    """Return the official per-instance SWE-bench OCI image URI."""

    docker_id = instance_id.replace("__", "_1776_")
    return (
        "docker://docker.io/swebench/"
        f"sweb.eval.x86_64.{docker_id}:latest"
    ).lower()


def _load_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("mini-SWE-agent trajectory must be an object")
    return value


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _response_id(message: dict[str, Any]) -> str | None:
    response = (message.get("extra") or {}).get("response")
    if isinstance(response, dict) and isinstance(response.get("id"), str):
        return response["id"]
    return None


def _result_content(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content", "")
    if isinstance(content, list):
        return content
    return [{"type": "text", "text": str(content)}]


def _recorded_result(
    call_id: str, message: dict[str, Any] | None
) -> dict[str, Any]:
    if message is None:
        return {
            "toolCallId": call_id,
            "toolName": "bash",
            "content": [],
            "details": {"missing_result": True},
            "isError": True,
        }
    extra = message.get("extra") or {}
    returncode = extra.get("returncode")
    exception = extra.get("exception_info")
    exit_status = extra.get("exit_status")
    is_error = bool(exception) or (
        isinstance(returncode, int) and returncode != 0
    )
    if exit_status == "Submitted":
        is_error = False
    details = {
        key: value
        for key, value in {
            "returncode": returncode,
            "exception_info": exception,
            "exit_status": exit_status,
        }.items()
        if value not in (None, "")
    }
    return {
        "toolCallId": call_id,
        "toolName": "bash",
        "content": _result_content(message),
        "details": details,
        "isError": is_error,
    }


def build_minisweagent_trace(
    *,
    trace_id: str,
    trajectory_path: str | Path,
    capture_path: str | Path,
    workload: str,
    framework_version: str,
    captured_workspace_root: str,
    source_repository: str,
    base_commit: str,
    instance_id: str,
) -> dict[str, Any]:
    """Convert one native mini-SWE-agent trajectory into schema v2."""

    trajectory = _load_object(trajectory_path)
    messages = trajectory.get("messages")
    if not isinstance(messages, list):
        raise ValueError("mini-SWE-agent trajectory has no messages list")

    response_events = [
        (index, message)
        for index, message in enumerate(messages)
        if isinstance(message, dict)
        and isinstance((message.get("extra") or {}).get("response"), dict)
    ]
    calls = sorted(
        (
            call
            for call in _read_jsonl(capture_path)
            if call.get("trace_id") == trace_id
            and isinstance(call.get("output_tokens"), int)
            and (
                call.get("status") is None
                or 200 <= call["status"] < 300
            )
        ),
        key=lambda call: call["sequence"],
    )
    if len(calls) != len(response_events):
        raise ValueError(
            f"LLM call mismatch for {trace_id}: {len(calls)} captured, "
            f"{len(response_events)} trajectory responses"
        )

    by_response_id = {
        call["response_id"]: call for call in calls if call.get("response_id")
    }
    unused_calls = list(calls)
    ordered_calls: list[dict[str, Any]] = []
    for _, message in response_events:
        call = by_response_id.get(_response_id(message))
        if call not in unused_calls:
            call = unused_calls[0]
        unused_calls.remove(call)
        ordered_calls.append(call)

    tool_results = {
        message["tool_call_id"]: message
        for message in messages
        if isinstance(message, dict)
        and message.get("role") == "tool"
        and isinstance(message.get("tool_call_id"), str)
    }
    exit_messages = [
        (index, message)
        for index, message in enumerate(messages)
        if isinstance(message, dict) and message.get("role") == "exit"
    ]

    nodes: list[dict[str, Any]] = []
    previous_llm: str | None = None
    pending_tools: list[str] = []
    tool_index = 0
    for llm_index, ((message_index, message), call) in enumerate(
        zip(response_events, ordered_calls, strict=True), 1
    ):
        output_tokens = call.get("output_tokens")
        if not isinstance(output_tokens, int):
            raise ValueError(f"missing output_tokens for captured call {call['call_id']}")
        llm_id = f"llm-{llm_index:03d}"
        dependencies = pending_tools or ([previous_llm] if previous_llm else [])
        nodes.append(
            {
                "id": llm_id,
                "type": "llm",
                "depends_on": dependencies,
                "request": {
                    "protocol": call["protocol"],
                    "endpoint": call["endpoint"],
                    "payload": call["request"],
                },
                "output_tokens": output_tokens,
            }
        )

        actions = (message.get("extra") or {}).get("actions") or []
        if not isinstance(actions, list):
            raise ValueError(f"actions for {llm_id} must be a list")
        pending_tools = []
        prior_node = llm_id
        for action in actions:
            if not isinstance(action, dict):
                raise ValueError(f"action for {llm_id} must be an object")
            call_id = action.get("tool_call_id")
            command = action.get("command")
            if not isinstance(call_id, str) or not isinstance(command, str):
                raise ValueError(f"invalid bash action for {llm_id}")
            result_message = tool_results.get(call_id)
            if result_message is None:
                result_message = next(
                    (
                        candidate
                        for index, candidate in exit_messages
                        if index > message_index
                    ),
                    None,
                )
            tool_index += 1
            tool_id = f"tool-{tool_index:03d}"
            nodes.append(
                {
                    "id": tool_id,
                    "type": "tool",
                    "depends_on": [prior_node],
                    "request": {
                        "protocol": "minisweagent-singularity",
                        "name": "bash",
                        "arguments": {"command": command},
                    },
                    "recorded_result": _recorded_result(call_id, result_message),
                }
            )
            prior_node = tool_id
            pending_tools = [tool_id]
        previous_llm = llm_id

    trace = {
        "schema_version": 2,
        "trace_id": trace_id,
        "source": {
            "framework": "mini-swe-agent",
            "workload": workload,
            "version": framework_version,
        },
        "context": {
            "captured_workspace_root": captured_workspace_root,
            "captured_tmp_root": "/tmp",
            "workspace_seed": None,
            "source_repository": source_repository,
            "base_commit": base_commit,
            "instance_id": instance_id,
            "container_image": swebench_container_image(instance_id),
            "container_cwd": "/testbed",
        },
        "artifacts": [],
        "nodes": nodes,
    }
    validate_trace(trace)
    return trace


def write_minisweagent_trace(output_path: str | Path, **kwargs: Any) -> dict[str, Any]:
    trace = build_minisweagent_trace(**kwargs)
    write_trace(output_path, trace)
    return trace


def _replace_string(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_replace_string(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: _replace_string(item, old, new) for key, item in value.items()}
    return value


class MiniSWEAgentToolExecutor:
    """Execute bash nodes in one persistent mini-SWE Singularity sandbox."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.timeout = int(config.get("timeout", 60))
        self.executable = config.get("executable")
        self.environment: Any = None
        self.replay_workspace = ""
        self.replay_tmp = ""
        self.container_cwd = "/testbed"

    async def setup(self, trace: dict[str, Any], workspace: Path) -> None:
        from minisweagent.environments.singularity import SingularityEnvironment

        context = trace["context"]
        self.replay_workspace = str(workspace)
        self.replay_tmp = str(workspace.parent / "tmp")
        self.container_cwd = context.get("container_cwd", "/testbed")
        options: dict[str, Any] = {
            "image": context["container_image"],
            "cwd": self.container_cwd,
            "timeout": self.timeout,
            "exec_args": [
                "--contain",
                "--cleanenv",
                "--fakeroot",
                "--bind",
                f"{self.replay_tmp}:/tmp",
            ],
            "forward_env": [
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "http_proxy",
                "https_proxy",
                "NO_PROXY",
                "no_proxy",
            ],
            "env": {
                "PAGER": "cat",
                "MANPAGER": "cat",
                "PIP_PROGRESS_BAR": "off",
                "BASH_ENV": "/root/.bashrc",
            },
        }
        if self.executable:
            options["executable"] = self.executable
        self.environment = await asyncio.to_thread(SingularityEnvironment, **options)

    def _rewrite_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        rewritten = _replace_string(
            copy.deepcopy(arguments), self.replay_workspace, self.container_cwd
        )
        return _replace_string(rewritten, self.replay_tmp, "/tmp")

    async def execute(self, node: dict[str, Any]) -> ToolExecutionResult:
        if self.environment is None:
            raise RuntimeError("mini-SWE-agent tool executor is not set up")
        request = node["request"]
        if (
            request["protocol"] != "minisweagent-singularity"
            or request["name"] != "bash"
        ):
            raise ValueError("unsupported mini-SWE-agent tool request")
        arguments = self._rewrite_arguments(request["arguments"])
        try:
            result = await asyncio.to_thread(
                self.environment.execute, arguments
            )
            content = [{"type": "text", "text": str(result.get("output", ""))}]
            details = {
                "returncode": result.get("returncode"),
                "exception_info": result.get("exception_info", ""),
            }
            is_error = bool(details["exception_info"]) or details["returncode"] != 0
        except Exception as error:
            if type(error).__name__ != "Submitted":
                raise
            messages = getattr(error, "messages", [])
            message = messages[0] if messages else {}
            content = _result_content(message)
            details = {"exit_status": "Submitted"}
            is_error = False
        return ToolExecutionResult(
            {
                "content": content,
                "details": details,
                "isError": is_error,
            }
        )

    async def close(self) -> None:
        if self.environment is not None:
            await asyncio.to_thread(self.environment.cleanup)
        self.environment = None


def create_tool_executor(config: dict[str, Any]) -> MiniSWEAgentToolExecutor:
    return MiniSWEAgentToolExecutor(config)
