from __future__ import annotations

import asyncio
import copy
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from agenttrace.replay.benchmark import benchmark


def add_tool(trace, operation="write"):
    trace["nodes"].append({"id": "tool-001", "type": "tool", "depends_on": ["llm-001"],
        "request": {"protocol": "fixture", "name": "fixture", "arguments": {"operation": operation}},
        "recorded_result": {"toolCallId": "c", "toolName": "fixture", "content": [], "details": {}, "isError": True}})


def fixture_profile(tmp_path, monkeypatch, url=None):
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).parent))
    # A parent environment's TMPDIR must not collide across tests.
    monkeypatch.setenv("TMPDIR", str(tmp_path / "local"))
    spec = {"tool_executor": {"factory": "benchmark_fixture:create_tools", "config": {}}}
    if url:
        spec["llm_executor"] = {"factory": "agenttrace.replay.llm:create_openai_executor",
            "config": {"base_url": url, "trust_env": False, "ignore_eos": True}}
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(spec))
    return path


@pytest.mark.parametrize(("concurrency", "llm_branches"), [(1, 1), (2, 1), (2, 2)])
def test_bounded_http_replay_metrics_and_workspace_isolation(
    tmp_path, monkeypatch, minimal_trace, concurrency, llm_branches
):
    active = peak = requests = 0
    lock = threading.Lock()
    pair = threading.Event()
    backend_path = tmp_path / "backend.jsonl"
    backend_path.write_text("")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            body = f"vllm:generation_tokens_total {requests * 4}\n".encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            nonlocal active, peak, requests
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            scheduled = time.monotonic()
            scheduled_unix = time.time()
            with lock:
                active += 1
                requests += 1
                peak = max(peak, active)
                if active == concurrency * llm_branches:
                    pair.set()
            if concurrency == 2:
                pair.wait(5)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for delta in ({"role": "assistant"}, {"reasoning_content": "PRIVATE"}, {"content": "PRIVATE"}):
                self.wfile.write(("data: " + json.dumps({"choices": [{"delta": delta}]}) + "\n\n").encode())
                self.wfile.flush()
                time.sleep(.05)
            ended = time.monotonic()
            with lock, backend_path.open("a") as handle:
                handle.write(json.dumps({"hostname": "fake-backend", "finish_reason": "length",
                    "scheduled_monotonic": scheduled, "first_token_monotonic": scheduled + .05,
                    "last_token_monotonic": ended, "scheduled_unix": scheduled_unix,
                    "finished_unix": scheduled_unix + ended - scheduled,
                    "output_tokens": payload["max_tokens"]}) + "\n")
            self.wfile.write(("data: " + json.dumps({"usage": {"completion_tokens": payload["max_tokens"]}}) + "\n\n").encode())
            with lock:
                active -= 1

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    profile = fixture_profile(tmp_path, monkeypatch, url)
    if llm_branches == 2:
        branch = copy.deepcopy(minimal_trace["nodes"][0])
        branch["id"] = "llm-002"
        minimal_trace["nodes"].append(branch)
    add_tool(minimal_trace)
    minimal_trace["nodes"][-1]["depends_on"] = [node["id"] for node in minimal_trace["nodes"][:-1]]
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(minimal_trace))
    try:
        report = asyncio.run(benchmark([path], profile=profile, output=tmp_path / "bench",
            concurrency=concurrency, repeat=4, sample_interval=.05, metrics_url=url + "/metrics",
            backend_events=backend_path))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert peak == concurrency * llm_branches and requests == 4 * llm_branches
    assert report["completed"] == 4
    assert report["peak_in_flight_tasks"] == concurrency
    assert report["actual_output_tokens"] == report["target_output_tokens"] == 16 * llm_branches
    backend = report["backend_throughput"]
    assert backend["status"] == "measured" and backend["requests"] == 4 * llm_branches
    assert backend["output_tokens"] == report["actual_output_tokens"]
    assert backend["active_output_tokens_per_second"] > report["output_tokens_per_second"]
    assert report["native_tool_errors"] == 4
    assert report["llm_ttft_seconds"]["count"] == 4 * llm_branches
    assert report["metrics"]["samples"] >= 2
    assert len(list((tmp_path / "bench/tasks").glob("*/run/workspace/marker"))) == 4
    for task in report["tasks"]:
        task_report = json.loads((tmp_path / "bench" / task["report"]).read_text())
        tool = task_report["nodes"][-1]
        for llm in task_report["nodes"][:-1]:
            assert llm["output_chunks"] == 2  # role-only event excluded
            assert llm["ttft_seconds"] > .02
            assert llm["output_stream_seconds"] > .02
            assert tool["started_seconds"] >= llm["started_seconds"] + llm["elapsed_seconds"] - .001
    assert "PRIVATE" not in "".join(p.read_text() for p in (tmp_path / "bench").rglob("*.json*"))


def test_failure_stops_queue_and_saves_summary(tmp_path, minimal_trace):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(minimal_trace))
    with pytest.raises(RuntimeError, match="failed"):
        asyncio.run(benchmark([path], profile=None, output=tmp_path / "failed", repeat=3))
    summary = json.loads((tmp_path / "failed/summary.json").read_text())
    assert summary["status"] == "failed"
    assert [row["status"] for row in summary["tasks"]] == ["failed", "pending", "pending"]


def test_sigint_stops_descendants_and_leaves_parent_alive(tmp_path, monkeypatch, minimal_trace):
    profile = fixture_profile(tmp_path, monkeypatch)
    trace = copy.deepcopy(minimal_trace)
    add_tool(trace, "block")
    trace["nodes"] = trace["nodes"][1:]
    trace["nodes"][0]["depends_on"] = []
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(trace))
    root = tmp_path / "cancel"
    process = subprocess.Popen([sys.executable, "-m", "agenttrace.cli", "benchmark", str(path),
        "--profile", str(profile), "--output-dir", str(root)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        marker = root / "tasks/00001/run/workspace/child.pid"
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(.05)
        assert marker.exists()
        pid = int(marker.read_text())
        process.send_signal(signal.SIGINT)
        assert process.wait(timeout=10) == 130
        proc = Path(f"/proc/{pid}/stat")
        assert not proc.exists() or proc.read_text().split()[2] == "Z"
        assert json.loads((root / "summary.json").read_text())["status"] == "cancelled"
        os.kill(os.getpid(), 0)  # parent test process was not signalled
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=10)
