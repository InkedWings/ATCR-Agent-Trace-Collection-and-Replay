from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agenttrace.adapters.openclaw import OpenClawToolExecutor, _write_tool_bridge
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
