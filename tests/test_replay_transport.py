"""Real HTTP/1.1 sockets: idle reuse, ambiguous POST failure, and task isolation."""

import asyncio
import json
import socket
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from agenttrace.adapters.openclaw import OpenClawToolExecutor
from agenttrace.replay.benchmark import benchmark
from agenttrace.replay.benchmark import task_failure
from agenttrace.replay.engine import replay_trace
from agenttrace.replay.llm import OpenAICompatibleExecutor


@pytest.fixture
def endpoint():
    state = {"requests": [], "failure": None}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                state["requests"].append({"peer": self.client_address, "connection": self.headers.get("Connection")})
                fail = state["failure"] if len(state["requests"]) == 1 else None
            if fail == "reset":
                # The POST has already been received; repeating it could execute twice.
                self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                self.close_connection = True
                return
            if isinstance(fail, int):
                self.send_response(fail)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if self.path == "/tools/invoke":
                data = json.dumps({"ok": True, "result": {"content": [], "isError": False}}).encode()
                content_type = "application/json"
            else:
                chunks = [{"choices": [{"delta": {"content": "ignored"}}]},
                          {"choices": [], "usage": {"completion_tokens": payload["max_tokens"]}}]
                data = ("".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n").encode()
                content_type = "text/event-stream"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if fail == "partial":
                self.wfile.write(data[:len(data)//2])
                self.wfile.flush()
                self.close_connection = True
            else:
                self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("kind", ["llm", "tool"])
@pytest.mark.parametrize("failure", [None, "reset", "partial", 500, 503])
def test_each_call_uses_new_connection_and_never_retries_ambiguous_post(endpoint, tmp_path, minimal_trace, kind, failure):
    url, state = endpoint
    state["failure"] = failure
    if kind == "llm":
        executor = OpenAICompatibleExecutor(url, trust_env=False)
        node = minimal_trace["nodes"][0]
    else:
        executor = OpenClawToolExecutor(gateway_url=url)
        node = {"id": "tool-001", "type": "tool", "depends_on": [],
                "request": {"protocol": "openclaw-tools-invoke", "name": "web_fetch",
                            "arguments": {"url": "https://example.test"}},
                "recorded_result": {"toolCallId": "t1"}}

    async def run():
        await executor.setup(minimal_trace, tmp_path)
        try:
            if failure:
                with pytest.raises(httpx.HTTPError):
                    await executor.execute(node)
            else:
                await executor.execute(node)
            assert len(state["requests"]) == 1
            await executor.execute({**node, "id": "next-node"})
        finally:
            await executor.close()

    asyncio.run(run())
    assert len(state["requests"]) == 2
    assert len({r["peer"] for r in state["requests"]}) == 2
    assert all(r["connection"] == "close" for r in state["requests"])


@pytest.mark.parametrize("timed", [False, True])
@pytest.mark.parametrize("failure", ["reset", 500, 503])
def test_transport_failure_does_not_cancel_peers_or_shorten_collection(endpoint, tmp_path, minimal_trace, timed, failure):
    url, state = endpoint
    state["failure"] = failure
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps(minimal_trace))
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"llm_executor": {"factory": "agenttrace.replay.llm:create_openai_executor",
                                                  "config": {"base_url": url, "trust_env": False}}}))
    root = tmp_path / "benchmark"
    options = {"warmup_seconds": .1, "duration_seconds": 2} if timed else {"repeat": 4}
    report = asyncio.run(benchmark([trace], profile=profile, output=root, concurrency=2,
        sample_hardware=False, continue_on_http_error=True, **options))
    assert report["status"] == "completed_with_errors"
    assert report["failed_tasks"] == 1 and report["completed"] >= 1
    assert len(state["requests"]) == len(report["tasks"])  # no duplicated inference
    assert all(t["status"] in ("failed", "completed") for t in report["tasks"])
    assert report["task_failure_fraction"] == 1 / len(report["tasks"])
    failed = next(t for t in report["tasks"] if t["status"] == "failed")
    assert failed["failure"]["kind"] == ("http_transport" if failure == "reset" else "http_server")
    if isinstance(failure, int):
        assert failed["failure"]["http_status"] == failure
    events = [json.loads(line) for line in (root / "tasks" / failed["task_id"] / "events.jsonl").read_text().splitlines()]
    assert any(e["event"] == "task_failed" for e in events)
    assert not any(e["event"] == "node_completed" for e in events)
    assert report["actual_output_tokens"] == report["completed"] * minimal_trace["nodes"][0]["output_tokens"]
    if timed:
        assert report["measurement"]["window_complete"]
        assert not report["measurement"]["valid"]
        assert report["ended_unix"] >= report["measurement"]["ended_unix"]
    else:
        assert report["completed"] == 3


def test_non_transport_failure_still_fails_fast(tmp_path, minimal_trace):
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps(minimal_trace))
    root = tmp_path / "benchmark"
    with pytest.raises(RuntimeError, match="failed"):
        asyncio.run(benchmark([trace], profile=None, output=root, repeat=3,
                             sample_hardware=False, continue_on_http_error=True))
    report = json.loads((root / "summary.json").read_text())
    assert report["status"] == "failed"
    assert [t["status"] for t in report["tasks"]] == ["failed", "pending", "pending"]
    assert report["tasks"][0]["failure"]["kind"] == "execution"


@pytest.mark.parametrize("status", [400, 401])
def test_bad_request_or_credentials_still_fail_fast(endpoint, tmp_path, minimal_trace, status):
    url, state = endpoint
    state["failure"] = status
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps(minimal_trace))
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"llm_executor": {"factory": "agenttrace.replay.llm:create_openai_executor",
                                                  "config": {"base_url": url, "trust_env": False}}}))
    root = tmp_path / "benchmark"
    with pytest.raises(RuntimeError, match="failed"):
        asyncio.run(benchmark([trace], profile=profile, output=root, repeat=3,
                             sample_hardware=False, continue_on_http_error=True))
    report = json.loads((root / "summary.json").read_text())
    assert len(state["requests"]) == 1
    assert [t["status"] for t in report["tasks"]] == ["failed", "pending", "pending"]
    assert report["tasks"][0]["failure"]["http_status"] == status


def test_cleanup_failure_after_read_error_remains_fatal(tmp_path, minimal_trace):
    class Executor:
        async def setup(self, trace, workspace):
            pass
        async def execute(self, node):
            raise httpx.ReadError("injected")
        async def close(self):
            raise RuntimeError("cleanup failed")
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps(minimal_trace))
    events = tmp_path / "events.jsonl"
    with pytest.raises(RuntimeError, match="cleanup failed"):
        asyncio.run(replay_trace(trace, llm_executor=Executor(), run_dir=tmp_path / "run", event_path=events))
    assert task_failure(events) == {"kind": "cleanup", "error_type": "RuntimeError"}
