from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from agenttrace.adapters.openclaw import OpenClawToolExecutor, _stage_brave_plugins, _write_tool_bridge, _free_port, _reserve_gateway_port
from agenttrace.replay.benchmark import benchmark
from agenttrace.replay.engine import replay_trace


def recorded_result(call_id, tool_name, details=None, is_error=False):
    return {
        "toolCallId": call_id,
        "toolName": tool_name,
        "content": [],
        "details": details or {},
        "isError": is_error,
    }


def tool_node(node_id, depends, name, arguments, result):
    return {
        "id": node_id,
        "type": "tool",
        "depends_on": depends,
        "request": {"protocol": "openclaw-tools-invoke", "name": name, "arguments": arguments},
        "recorded_result": result,
    }


def base_trace(nodes):
    return {
        "schema_version": 2,
        "trace_id": "gateway-test",
        "source": {"framework": "openclaw", "workload": "test", "version": "test"},
        "context": {"captured_workspace_root": "/old/workspace", "captured_tmp_root": "/tmp", "workspace_seed": None},
        "artifacts": [],
        "nodes": nodes,
    }


def run_gateway(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_gateway_port_reservation_spans_processes_until_close(tmp_path):
    candidate = _free_port()
    script = """
import sys
from pathlib import Path
from agenttrace.adapters import openclaw
original = openclaw._free_port
first = [int(sys.argv[2])]
openclaw._free_port = lambda: first.pop() if first else original()
port, lease = openclaw._reserve_gateway_port(Path(sys.argv[1]))
print(port, flush=True)
sys.stdin.read()
lease.close()
"""
    workers = []
    try:
        for _ in range(2):
            workers.append(subprocess.Popen([sys.executable, "-c", script, str(tmp_path), str(candidate)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True))
        ports = [int(worker.stdout.readline()) for worker in workers]
        assert candidate in ports and len(set(ports)) == 2
    finally:
        for worker in workers:
            worker.stdin.close()
            worker.wait(timeout=10)
            worker.stdout.close()


@pytest.mark.parametrize("authenticated", [True, False])
def test_readiness_checks_bearer_without_executing_a_tool(tmp_path, monkeypatch, authenticated):
    requests = []
    class Process:
        returncode = None
        def terminate(self):
            self.returncode = 0
        async def wait(self):
            return self.returncode
    async def spawn(*args, **kwargs):
        return Process()
    def handler(request):
        requests.append(request)
        if request.url.path == "/health":
            return httpx.Response(200, json={"ok": True})
        assert request.url.path == "/tools/invoke" and request.method == "POST"
        assert json.loads(request.content) == {}
        assert request.headers["Authorization"].startswith("Bearer ")
        if not authenticated:
            return httpx.Response(401, json={"error": "unauthorized"})
        return httpx.Response(400, json={"ok": False, "error": {
            "type": "invalid_request", "message": "tools.invoke requires name"}})
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client(**kwargs, transport=httpx.MockTransport(handler)))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    async def exercise():
        executor = OpenClawToolExecutor()
        try:
            if authenticated:
                await executor.setup(base_trace([]), workspace)
            else:
                with pytest.raises(httpx.HTTPStatusError, match="401"):
                    await executor.setup(base_trace([]), workspace)
        finally:
            await executor.close()
        assert executor._port_lease is None
    asyncio.run(exercise())
    assert len(requests) == 2


def test_brave_payload_is_seeded_in_each_private_state(tmp_path):
    source = tmp_path / "source"
    project = source / "npm/projects/openclaw-brave-plugin-fixture"
    package = project / "node_modules/@openclaw/brave-plugin"
    (package / "node_modules").mkdir(parents=True)
    metadata = {"dependencies": {"@openclaw/brave-plugin": "2026.7.1"}}
    (project / "package.json").write_text(json.dumps(metadata))
    (package / "openclaw.plugin.json").write_text('{"id":"brave"}')
    (package / "index.js").write_text("original plugin")
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    (package / "node_modules/openclaw").symlink_to(sdk, target_is_directory=True)
    (source / "credentials.json").write_text("do not copy")

    first = Path(_stage_brave_plugins(source, tmp_path / "run1")[0])
    second = Path(_stage_brave_plugins(source, tmp_path / "run2")[0])
    assert first != second and first != package
    assert json.loads((first.parents[2] / "package.json").read_text()) == metadata
    assert (first / "node_modules/openclaw").is_symlink()
    assert (first / "node_modules/openclaw").resolve() == sdk
    (first / "index.js").write_text("private change")
    assert (second / "index.js").read_text() == (package / "index.js").read_text() == "original plugin"
    assert not (tmp_path / "run1/credentials.json").exists()
    assert _stage_brave_plugins(tmp_path / "missing", tmp_path / "run3") == []


def test_no_search_trace_disables_env_selected_search_plugin(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "test-only")

    async def intercept_spawn(*args, **kwargs):
        raise RuntimeError("intercepted before Gateway spawn")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", intercept_spawn)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace = base_trace([tool_node("tool-001", [], "read", {}, recorded_result("call-1", "read"))])

    async def setup():
        executor = OpenClawToolExecutor()
        try:
            with pytest.raises(RuntimeError, match="intercepted before Gateway spawn"):
                await executor.setup(trace, workspace)
        finally:
            await executor.close()

    asyncio.run(setup())
    config = json.loads((tmp_path / "openclaw-state/openclaw.json").read_text())
    assert config["tools"]["web"]["search"] == {"enabled": False}
    assert "brave" not in config["plugins"]["allow"]
    assert not (tmp_path / "openclaw-state/npm").exists()


def test_same_session_dynamic_spill_binding_and_native_error(tmp_path):
    requests: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            request = json.loads(body)
            requests.append(request)
            if request["tool"] == "web_fetch":
                result = {"content": [], "details": {"fullOutputPath": "/fresh/spill.log"}, "isError": False}
            else:
                result = {"content": [], "details": {}, "isError": True}
            encoded = json.dumps({"ok": True, "result": result}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server, thread = run_gateway(Handler)
    try:
        trace = base_trace(
            [
                tool_node("tool-001", [], "web_fetch", {"url": "https://example.test"}, recorded_result("call-1", "web_fetch", {"fullOutputPath": "/tmp/old.log"})),
                tool_node("tool-002", ["tool-001"], "read", {"path": "/tmp/old.log"}, recorded_result("call-2", "read", is_error=True)),
            ]
        )
        path = tmp_path / "trace.json"
        path.write_text(json.dumps(trace), encoding="utf-8")
        executor = OpenClawToolExecutor(gateway_url=f"http://127.0.0.1:{server.server_port}")
        report = asyncio.run(replay_trace(path, tool_executor=executor, run_dir=tmp_path / "replay"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert requests[0]["sessionKey"] == requests[1]["sessionKey"]
    assert requests[1]["args"]["path"] == "/fresh/spill.log"
    assert report["nodes"][1]["native_error"] is True


def test_write_edit_exec_side_effects_share_workspace(tmp_path):
    requests: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            args = request["args"]
            if request["tool"] == "write":
                Path(args["path"]).write_text(args["content"], encoding="utf-8")
            elif request["tool"] == "edit":
                path = Path(args["path"])
                content = path.read_text(encoding="utf-8")
                for edit in args["edits"]:
                    content = content.replace(edit["oldText"], edit["newText"])
                path.write_text(content, encoding="utf-8")
            elif request["tool"] == "exec":
                assert Path(args["command"].split()[-1]).read_text() == "after"
            encoded = json.dumps({"ok": True, "result": {"content": [], "details": {}, "isError": False}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server, thread = run_gateway(Handler)
    try:
        old_path = "/old/workspace/state.txt"
        trace = base_trace(
            [
                tool_node("tool-001", [], "write", {"path": old_path, "content": "before"}, recorded_result("call-1", "write")),
                tool_node(
                    "tool-002",
                    ["tool-001"],
                    "edit",
                    {
                        "path": old_path,
                        "edits": [{"oldText": "before", "newText": "after"}],
                    },
                    recorded_result("call-2", "edit"),
                ),
                tool_node("tool-003", ["tool-002"], "exec", {"command": f"check {old_path}"}, recorded_result("call-3", "exec")),
            ]
        )
        path = tmp_path / "trace.json"
        path.write_text(json.dumps(trace), encoding="utf-8")
        executor = OpenClawToolExecutor(gateway_url=f"http://127.0.0.1:{server.server_port}")
        asyncio.run(replay_trace(path, tool_executor=executor, run_dir=tmp_path / "replay"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert (tmp_path / "replay/workspace/state.txt").read_text() == "after"
    assert len({request["sessionKey"] for request in requests}) == 1


@pytest.mark.parametrize("tool_error_status", [400, 403, 500])
def test_gateway_tool_error_continues_dag_and_next_task(tmp_path, monkeypatch, tool_error_status):
    monkeypatch.setenv("TMPDIR", str(tmp_path / "local"))
    requests = []
    recorded_messages = [{"role": "user", "content": "recorded input"}]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, request))
            if self.path == "/tools/invoke":
                if request["tool"] == "web_fetch":
                    status = tool_error_status
                    result = {"ok": False, "error": {"type": "tool_error", "message": "PRIVATE fetch failed"}}
                else:
                    status = 200
                    result = {"ok": True, "result": {"isError": True, "content": [], "details": {}}}
                body = json.dumps(result).encode()
                content_type = "application/json"
            else:
                status = 200
                body = ('data: {"choices":[{"delta":{"content":"PRIVATE output"}}]}\n\n'
                        'data: {"usage":{"completion_tokens":2}}\n\ndata: [DONE]\n\n').encode()
                content_type = "text/event-stream"
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server, thread = run_gateway(Handler)
    url = f"http://127.0.0.1:{server.server_port}"
    trace = base_trace([
        tool_node("tool-001", [], "web_fetch", {"url": "http://example.test"}, recorded_result("call-1", "web_fetch")),
        tool_node("tool-002", ["tool-001"], "read", {"path": "/tmp/missing.txt"}, recorded_result("call-2", "read")),
        {"id": "llm-001", "type": "llm", "depends_on": ["tool-002"], "output_tokens": 2,
         "request": {"protocol": "openai-chat-completions", "endpoint": "/v1/chat/completions",
                     "payload": {"model": "fake", "messages": recorded_messages}}},
    ])
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(trace))
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({
        "tool_executor": {"factory": "agenttrace.adapters.openclaw:create_tool_executor", "config": {"gateway_url": url}},
        "llm_executor": {"factory": "agenttrace.replay.llm:create_openai_executor",
                         "config": {"base_url": url, "trust_env": False, "ignore_eos": True}},
    }))
    output = tmp_path / "bench"
    try:
        report = asyncio.run(benchmark([path], profile=profile, output=output, repeat=2, concurrency=1))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    # Both failed tools finish before the recorded LLM call; the next task runs.
    # Each call executes exactly once, even though recorded tools had succeeded.
    assert [route for route, _ in requests] == ["/tools/invoke", "/tools/invoke", "/v1/chat/completions"] * 2
    assert [body["tool"] for route, body in requests if route == "/tools/invoke"] == ["web_fetch", "read"] * 2
    for route, body in requests:
        if route == "/v1/chat/completions":
            assert body["messages"] == recorded_messages
            assert body["max_tokens"] == 2
    assert report["completed"] == 2
    assert report["native_tool_errors"] == 4
    assert report["actual_output_tokens"] == report["target_output_tokens"] == 4
    assert report["tool_latency_seconds"]["count"] == 4
    for task in report["tasks"]:
        nodes = json.loads((output / task["report"]).read_text())["nodes"]
        assert all(node["status"] == "completed" for node in nodes)
        assert all(node["native_error"] is True and node["elapsed_seconds"] > 0 for node in nodes[:2])
    events = [json.loads(line) for path in output.glob("tasks/*/events.jsonl") for line in path.read_text().splitlines()]
    assert sum(event["event"] == "node_completed" for event in events) == 6
    assert not any(event["event"] in ("node_failed", "replay_failed") for event in events)
    assert "PRIVATE" not in "".join(path.read_text() for path in output.rglob("*.json*"))


@pytest.mark.parametrize(("status", "error_type"), [
    (401, "unauthorized"), (403, "tool_call_blocked"), (404, "not_found"),
    (500, "internal_error"), (500, None),
])
def test_gateway_infrastructure_errors_are_not_tool_results(status, error_type):
    requests = []

    def respond(request):
        requests.append(request)
        if error_type is None:
            return httpx.Response(status, text="not a tool response")
        return httpx.Response(status, json={"ok": False, "error": {"type": error_type}})

    async def invoke():
        executor = OpenClawToolExecutor(gateway_url="http://gateway.test")
        executor.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await executor.execute(tool_node("tool-001", [], "web_fetch", {}, recorded_result("call-1", "web_fetch")))
        finally:
            await executor.close()

    asyncio.run(invoke())
    assert len(requests) == 1


def test_replay_bridge_uses_openclaw_native_sdk(tmp_path):
    plugin = tmp_path / "plugin"
    _write_tool_bridge(plugin)
    manifest = json.loads((plugin / "openclaw.plugin.json").read_text())
    source = (plugin / "index.js").read_text()

    assert manifest["contracts"]["tools"] == [
        "agenttrace_edit",
        "agenttrace_exec",
        "agenttrace_read",
        "agenttrace_write",
    ]
    assert 'from "openclaw/plugin-sdk/agent-sessions"' in source
    assert 'agenttrace_exec: "bash"' in source
    assert "isError: true" in source
