"""OpenClaw trajectory collection and native ``/tools/invoke`` replay."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import shutil
import socket
from pathlib import Path
from typing import Any

import httpx

from agenttrace.interfaces import CapturedArtifacts, ToolExecutionResult
from agenttrace.schema import validate_trace, write_trace

SUPPORTED_TOOLS = {"web_search", "web_fetch", "exec", "read", "write", "edit"}
CODING_TOOL_BRIDGE = {
    "read": "agenttrace_read",
    "write": "agenttrace_write",
    "edit": "agenttrace_edit",
    "exec": "agenttrace_exec",
}


def _write_tool_bridge(plugin_dir: Path) -> None:
    """Expose OpenClaw's native coding tools to its Gateway HTTP surface.

    OpenClaw 2026.7's ``/tools/invoke`` resolver includes web tools but omits
    the four coding tools used by normal agent turns. This replay-only plugin
    registers aliases backed directly by the public OpenClaw agent-sessions
    SDK, keeping invocation on the official Gateway endpoint without copying
    any tool implementation into AgentTrace.
    """

    plugin_dir.mkdir()
    (plugin_dir / "openclaw.plugin.json").write_text(
        json.dumps(
            {
                "id": "agenttrace-replay",
                "name": "AgentTrace replay bridge",
                "description": "Replay-only aliases for native OpenClaw coding tools",
                "contracts": {"tools": sorted(CODING_TOOL_BRIDGE.values())},
                "configSchema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (plugin_dir / "package.json").write_text(
        json.dumps(
            {
                "name": "agenttrace-openclaw-replay",
                "version": "0.1.0",
                "private": True,
                "type": "module",
                "openclaw": {"extensions": ["./index.js"]},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (plugin_dir / "index.js").write_text(
        """import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { createCodingTools } from "openclaw/plugin-sdk/agent-sessions";

const workspace = process.env.AGENTTRACE_REPLAY_WORKSPACE;
if (!workspace) throw new Error("AGENTTRACE_REPLAY_WORKSPACE is required");

const nativeTools = new Map(createCodingTools(workspace).map((tool) => [tool.name, tool]));
const aliases = {
  agenttrace_read: "read",
  agenttrace_write: "write",
  agenttrace_edit: "edit",
  agenttrace_exec: "bash",
};

export default definePluginEntry({
  id: "agenttrace-replay",
  name: "AgentTrace replay bridge",
  description: "Replay-only aliases for native OpenClaw coding tools",
  register(api) {
    for (const [alias, nativeName] of Object.entries(aliases)) {
      const nativeTool = nativeTools.get(nativeName);
      if (!nativeTool) throw new Error(`OpenClaw native tool unavailable: ${nativeName}`);
      api.registerTool({
        ...nativeTool,
        name: alias,
        label: alias,
        async execute(id, params) {
          try {
            return await nativeTool.execute(id, params);
          } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            return {
              content: [{ type: "text", text: message }],
              details: { error: message },
              isError: true,
            };
          }
        },
      });
    }
  },
});
""",
        encoding="utf-8",
    )


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        # JSON strings may legally contain Unicode line separators. JSONL is
        # delimited by the ASCII newline byte, so do not use str.splitlines().
        for line in Path(path).read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_spill_artifacts(
    trajectory_path: Path, artifact_dir: Path, trace_parent: Path
) -> list[dict[str, Any]]:
    """Copy tool spill files while the collection node's ``/tmp`` still exists."""

    artifacts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in read_jsonl(trajectory_path):
        message = event.get("message") or {}
        if message.get("role") != "toolResult":
            continue
        details = message.get("details") or {}
        recorded_path = details.get("fullOutputPath")
        if not isinstance(recorded_path, str) or recorded_path in seen:
            continue
        seen.add(recorded_path)
        source = Path(recorded_path)
        if not source.is_file():
            continue
        artifact_dir.mkdir(parents=True, exist_ok=True)
        destination = artifact_dir / f"spill-{len(artifacts) + 1:03d}-{source.name}"
        shutil.copy2(source, destination)
        artifacts.append(
            {
                "id": f"tool-output-{len(artifacts) + 1:03d}",
                "kind": "tool_output",
                "captured_path": recorded_path,
                "path": destination.relative_to(trace_parent).as_posix(),
                "sha256": sha256(destination),
            }
        )
    return artifacts


def _ordered_calls(
    trace_id: str, events: list[dict[str, Any]], calls: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assistants = [
        event
        for event in events
        if event.get("type") == "message"
        and (event.get("message") or {}).get("role") == "assistant"
    ]
    matching = sorted(
        (call for call in calls if call.get("trace_id") == trace_id),
        key=lambda call: call["sequence"],
    )
    if len(matching) != len(assistants):
        raise ValueError(
            f"LLM call mismatch for {trace_id}: {len(matching)} captured, "
            f"{len(assistants)} assistant events"
        )

    by_response_id = {
        call["response_id"]: call for call in matching if call.get("response_id")
    }
    unused = list(matching)
    ordered: list[dict[str, Any]] = []
    for assistant in assistants:
        response_id = (assistant.get("message") or {}).get("responseId")
        call = by_response_id.get(response_id)
        if call not in unused:
            call = unused[0]
        unused.remove(call)
        ordered.append(call)
    return assistants, ordered


def build_openclaw_trace(
    *,
    trace_id: str,
    trajectory_path: str | Path,
    capture_path: str | Path,
    workload: str,
    framework_version: str,
    captured_workspace_root: str,
    workspace_seed: str | None,
    artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    events = read_jsonl(trajectory_path)
    assistants, calls = _ordered_calls(trace_id, events, read_jsonl(capture_path))
    tool_results: dict[str, dict[str, Any]] = {}
    for event in events:
        message = event.get("message") or {}
        if message.get("role") != "toolResult":
            continue
        tool_call_id = message.get("toolCallId")
        if not isinstance(tool_call_id, str) or tool_call_id in tool_results:
            raise ValueError(f"invalid or duplicate OpenClaw tool result ID: {tool_call_id}")
        tool_results[tool_call_id] = {
            "toolCallId": tool_call_id,
            "toolName": str(message.get("toolName") or "unknown"),
            "content": message.get("content") or [],
            "details": message.get("details") or {},
            "isError": bool(message.get("isError", False)),
        }

    nodes: list[dict[str, Any]] = []
    previous_llm: str | None = None
    pending_tools: list[str] = []
    used_results: set[str] = set()
    tool_index = 0
    for llm_index, (assistant, call) in enumerate(
        zip(assistants, calls, strict=True), 1
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
        pending_tools = []
        for part in (assistant.get("message") or {}).get("content") or []:
            if not isinstance(part, dict) or part.get("type") != "toolCall":
                continue
            tool_call_id = part.get("id")
            if not isinstance(tool_call_id, str) or tool_call_id not in tool_results:
                raise ValueError(f"missing OpenClaw tool result for {tool_call_id}")
            recorded_result = tool_results[tool_call_id]
            tool_name = str(part.get("name") or "unknown")
            if recorded_result["toolName"] != tool_name:
                raise ValueError(f"OpenClaw tool name mismatch for {tool_call_id}")
            tool_index += 1
            tool_id = f"tool-{tool_index:03d}"
            nodes.append(
                {
                    "id": tool_id,
                    "type": "tool",
                    "depends_on": [llm_id],
                    "request": {
                        "protocol": "openclaw-tools-invoke",
                        "name": tool_name,
                        "arguments": part.get("arguments") or {},
                    },
                    "recorded_result": recorded_result,
                }
            )
            used_results.add(tool_call_id)
            pending_tools.append(tool_id)
        previous_llm = llm_id

    if used_results != set(tool_results):
        unused = sorted(set(tool_results) - used_results)
        raise ValueError(f"unpaired OpenClaw tool results: {unused}")
    trace = {
        "schema_version": 2,
        "trace_id": trace_id,
        "source": {
            "framework": "openclaw",
            "workload": workload,
            "version": framework_version or "unknown",
        },
        "context": {
            "captured_workspace_root": captured_workspace_root,
            "captured_tmp_root": "/tmp",
            "workspace_seed": workspace_seed,
        },
        "artifacts": artifacts,
        "nodes": nodes,
    }
    validate_trace(trace)
    return trace


def write_openclaw_trace(output_path: str | Path, **kwargs: Any) -> dict[str, Any]:
    trace = build_openclaw_trace(**kwargs)
    write_trace(output_path, trace)
    return trace


class OpenClawCollectionAdapter:
    """Launch one local OpenClaw agent case and retain its native artifacts."""

    def __init__(
        self,
        *,
        openclaw_bin: str,
        model: str,
        state_dir: Path,
        capture_proxy: Any,
        timeout_seconds: int = 240,
        workload: str = "unknown",
        framework_version: str = "unknown",
    ) -> None:
        binary_path = Path(openclaw_bin).expanduser()
        self.openclaw_bin = (
            str(binary_path.resolve()) if binary_path.parent != Path(".") else openclaw_bin
        )
        self.model = model
        self.state_dir = state_dir
        self.capture_proxy = capture_proxy
        self.timeout_seconds = timeout_seconds
        self.workload = workload
        self.framework_version = framework_version

    async def collect_case(
        self, case: dict[str, Any], context: dict[str, Any]
    ) -> CapturedArtifacts:
        trace_id = str(case["trace_id"])
        output_dir = Path(context["output_dir"])
        workspace = Path(context["workspace"])
        prompt_path = Path(context["prompt_path"])
        session_id = str(context.get("session_id", f"agenttrace-{trace_id}"))
        workspace_seed = output_dir / "workspace_seed"
        shutil.copytree(workspace, workspace_seed)
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in context.get("env", {}).items()})
        env["OPENCLAW_STATE_DIR"] = str(self.state_dir)
        env["OPENCLAW_WORKSPACE_DIR"] = str(workspace)
        env["TRACE_ID"] = trace_id
        env["LLM_CAPTURE_BASE_URL"] = self.capture_proxy.base_url
        process = await asyncio.create_subprocess_exec(
            self.openclaw_bin,
            "agent",
            "--local",
            "--json",
            "--session-id",
            session_id,
            "--model",
            self.model,
            "--timeout",
            str(self.timeout_seconds),
            "--message-file",
            str(prompt_path),
            cwd=workspace,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(), timeout=self.timeout_seconds + 30
        )
        stdout = stdout_bytes.decode(errors="replace")
        stderr = stderr_bytes.decode(errors="replace")
        (output_dir / "openclaw.stdout.json").write_text(stdout, encoding="utf-8")
        (output_dir / "openclaw.stderr.log").write_text(stderr, encoding="utf-8")
        payload = json.loads(stdout)
        session_file = ((payload.get("meta") or {}).get("agentMeta") or {}).get(
            "sessionFile"
        )
        source = Path(session_file) if session_file else (
            self.state_dir / "agents/main/sessions" / f"{session_id}.jsonl"
        )
        if not source.resolve().is_relative_to(self.state_dir.resolve()):
            raise ValueError("OpenClaw session file escaped its state directory")
        trajectory = output_dir / "trajectory.jsonl"
        shutil.copy2(source, trajectory)
        self.capture_proxy.wait_idle(trace_id)
        artifacts = capture_spill_artifacts(
            trajectory, output_dir / "artifacts", output_dir
        )
        return CapturedArtifacts(
            trace_id=trace_id,
            trajectory_path=trajectory,
            capture_path=self.capture_proxy.output,
            output_dir=output_dir,
            workspace_root=workspace,
            workspace_seed=workspace_seed,
            artifacts=artifacts,
            framework_version=self.framework_version,
        )

    def build_trace(self, artifacts: CapturedArtifacts) -> dict[str, Any]:
        return build_openclaw_trace(
            trace_id=artifacts.trace_id,
            trajectory_path=artifacts.trajectory_path,
            capture_path=artifacts.capture_path,
            workload=self.workload,
            framework_version=artifacts.framework_version,
            captured_workspace_root=str(artifacts.workspace_root),
            workspace_seed=artifacts.workspace_seed.relative_to(
                artifacts.output_dir
            ).as_posix()
            if artifacts.workspace_seed
            else None,
            artifacts=artifacts.artifacts,
        )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class OpenClawToolExecutor:
    """Execute recorded tools through one long-lived native OpenClaw Gateway."""

    def __init__(
        self,
        *,
        openclaw_bin: str = "openclaw",
        gateway_url: str | None = None,
        token_env: str | None = None,
        plugin_state_dir: str | None = None,
        startup_timeout: float = 30.0,
    ) -> None:
        self.openclaw_bin = openclaw_bin
        self.gateway_url = gateway_url
        self.token_env = token_env
        self.plugin_state_dir = (
            Path(plugin_state_dir).resolve() if plugin_state_dir else None
        )
        self.startup_timeout = startup_timeout
        self.token = ""
        self.session_key = ""
        self.client: httpx.AsyncClient | None = None
        self.process: asyncio.subprocess.Process | None = None
        self._stdout: Any = None
        self._stderr: Any = None
        self._use_coding_bridge = False

    async def setup(self, trace: dict[str, Any], workspace: Path) -> None:
        self.session_key = f"agenttrace-{trace['trace_id']}"
        self.token = (
            os.environ.get(self.token_env, "") if self.token_env else ""
        ) or secrets.token_urlsafe(32)
        if self.gateway_url is None:
            run_root = workspace.parent
            state_dir = run_root / "openclaw-state"
            state_dir.mkdir()
            config_path = state_dir / "openclaw.json"
            bridge_dir = run_root / "openclaw-replay-plugin"
            _write_tool_bridge(bridge_dir)
            self._use_coding_bridge = True
            needs_brave = any(
                node["type"] == "tool" and node["request"]["name"] == "web_search"
                for node in trace["nodes"]
            )
            plugin_paths: list[str] = []
            if needs_brave and self.plugin_state_dir:
                plugin_paths = [
                    str(path)
                    for path in sorted(
                        self.plugin_state_dir.glob(
                            "npm/projects/openclaw-brave-plugin-*/node_modules/@openclaw/brave-plugin"
                        )
                    )
                    if path.is_dir()
                ]
            if needs_brave and not plugin_paths:
                raise RuntimeError(
                    "web_search replay requires the installed Brave plugin; "
                    "configure plugin_state_dir"
                )
            plugin_ids = ["agenttrace-replay"]
            plugin_load_paths = [str(bridge_dir), *plugin_paths]
            plugin_entries: dict[str, Any] = {
                "agenttrace-replay": {"enabled": True}
            }
            if plugin_paths:
                plugin_ids.append("brave")
                plugin_entries["brave"] = {"enabled": True}
            plugin_config: dict[str, Any] = {
                "allow": plugin_ids,
                "load": {"paths": plugin_load_paths},
                "entries": plugin_entries,
            }
            tools_config: dict[str, Any] = {
                "profile": "full",
                "allow": sorted(
                    {"web_search", "web_fetch", *CODING_TOOL_BRIDGE.values()}
                ),
                "exec": {"host": "gateway", "mode": "full"},
                "web": {"fetch": {"useTrustedEnvProxy": True}},
            }
            if needs_brave:
                tools_config["web"]["search"] = {
                    "enabled": True,
                    "provider": "brave",
                    "apiKey": "${BRAVE_API_KEY}",
                    "maxResults": 10,
                    "timeoutSeconds": 30,
                }
            config = {
                "agents": {
                    "defaults": {
                        "workspace": str(workspace),
                        "model": {"primary": "agenttrace/replay-tools"},
                    }
                },
                "models": {
                    "mode": "merge",
                    "providers": {
                        "agenttrace": {
                            "baseUrl": "http://127.0.0.1:1/v1",
                            "apiKey": "not-used-by-tools-invoke",
                            "api": "openai-completions",
                            "models": [
                                {
                                    "id": "replay-tools",
                                    "name": "AgentTrace tool replay",
                                    "input": ["text"],
                                    "contextWindow": 1024,
                                    "maxTokens": 1,
                                    "cost": {
                                        "input": 0,
                                        "output": 0,
                                        "cacheRead": 0,
                                        "cacheWrite": 0,
                                    },
                                    "compat": {"supportsTools": True},
                                }
                            ],
                        }
                    },
                },
                "gateway": {
                    "mode": "local",
                    "bind": "loopback",
                    "auth": {"mode": "token"},
                    "tools": {"allow": ["exec"]},
                },
                "tools": tools_config,
            }
            config["plugins"] = plugin_config
            config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            (state_dir / "exec-approvals.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "defaults": {
                            "security": "full",
                            "ask": "off",
                            "askFallback": "full",
                        },
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            port = _free_port()
            self.gateway_url = f"http://127.0.0.1:{port}"
            env = os.environ.copy()
            env.update(
                {
                    "OPENCLAW_STATE_DIR": str(state_dir),
                    "OPENCLAW_CONFIG_PATH": str(config_path),
                    "OPENCLAW_WORKSPACE_DIR": str(workspace),
                    "OPENCLAW_GATEWAY_TOKEN": self.token,
                    "AGENTTRACE_REPLAY_WORKSPACE": str(workspace),
                    "TMPDIR": str(run_root / "tmp"),
                }
            )
            self._stdout = (run_root / "gateway.stdout.log").open("wb")
            self._stderr = (run_root / "gateway.stderr.log").open("wb")
            self.process = await asyncio.create_subprocess_exec(
                self.openclaw_bin,
                "gateway",
                "run",
                "--port",
                str(port),
                "--bind",
                "loopback",
                "--auth",
                "token",
                "--token",
                self.token,
                "--compact",
                cwd=workspace,
                env=env,
                stdout=self._stdout,
                stderr=self._stderr,
            )

        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {self.token}"}, timeout=None, trust_env=False
        )
        if self.process is not None:
            deadline = asyncio.get_running_loop().time() + self.startup_timeout
            while True:
                if self.process.returncode is not None:
                    raise RuntimeError(
                        f"OpenClaw Gateway exited during setup with {self.process.returncode}"
                    )
                try:
                    response = await self.client.get(f"{self.gateway_url}/health")
                    if response.is_success:
                        break
                except httpx.HTTPError:
                    pass
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError("OpenClaw Gateway did not become ready")
                await asyncio.sleep(0.1)

    async def execute(self, node: dict[str, Any]) -> ToolExecutionResult:
        if self.client is None or self.gateway_url is None:
            raise RuntimeError("OpenClaw tool executor is not set up")
        request = node["request"]
        if request["protocol"] != "openclaw-tools-invoke":
            raise ValueError(f"unsupported OpenClaw tool protocol: {request['protocol']}")
        if request["name"] not in SUPPORTED_TOOLS:
            raise ValueError(f"unsupported OpenClaw tool: {request['name']}")
        tool_name = request["name"]
        if self._use_coding_bridge:
            tool_name = CODING_TOOL_BRIDGE.get(tool_name, tool_name)
        response = await self.client.post(
            f"{self.gateway_url}/tools/invoke",
            json={
                "tool": tool_name,
                "args": request["arguments"],
                "sessionKey": self.session_key,
                "idempotencyKey": node["recorded_result"]["toolCallId"],
            },
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok"):
            raise RuntimeError(f"OpenClaw tool invocation failed: {payload.get('error')}")
        result = payload.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("OpenClaw Gateway returned a non-object tool result")
        result.setdefault("isError", False)
        return ToolExecutionResult(result=result)

    async def close(self) -> None:
        if self.client is not None:
            await self.client.aclose()
            self.client = None
        if self.process is not None and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()
        self.process = None
        for handle in (self._stdout, self._stderr):
            if handle is not None:
                handle.close()
        self._stdout = None
        self._stderr = None


def create_tool_executor(config: dict[str, Any]) -> OpenClawToolExecutor:
    openclaw_bin = str(config.get("openclaw_bin") or "openclaw")
    openclaw_bin_env = config.get("openclaw_bin_env")
    if openclaw_bin_env:
        openclaw_bin = os.environ.get(str(openclaw_bin_env), openclaw_bin)
    gateway_url = config.get("gateway_url")
    gateway_url_env = config.get("gateway_url_env")
    if gateway_url_env:
        gateway_url = os.environ.get(str(gateway_url_env), gateway_url)
    plugin_state_dir = config.get("plugin_state_dir")
    plugin_state_dir_env = config.get("plugin_state_dir_env")
    if plugin_state_dir_env:
        plugin_state_dir = os.environ.get(str(plugin_state_dir_env), plugin_state_dir)
    return OpenClawToolExecutor(
        openclaw_bin=openclaw_bin,
        gateway_url=str(gateway_url) if gateway_url else None,
        token_env=config.get("token_env"),
        plugin_state_dir=str(plugin_state_dir) if plugin_state_dir else None,
        startup_timeout=float(config.get("startup_timeout", 30)),
    )
