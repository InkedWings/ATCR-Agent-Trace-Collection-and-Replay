from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

from agenttrace.capture.openai import CaptureProxy


def test_incomplete_upstream_stream_preserves_received_usage(tmp_path, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    body = b'data: {"id":"response-1","usage":{"completion_tokens":7}}\n\n'

    class BrokenStream(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            return

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self.wfile.write(f"{len(body):x}\r\n".encode() + body + b"\r\n")
            self.wfile.flush()
            self.close_connection = True

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), BrokenStream)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    capture_path = tmp_path / "calls.jsonl"
    proxy = CaptureProxy(
        f"http://127.0.0.1:{upstream.server_port}", capture_path
    )
    proxy.start()
    try:
        httpx.post(
            proxy.base_url + "/chat/completions",
            headers={"X-Trace-Id": "trace-1"},
            json={"stream": True},
        )
        proxy.wait_idle("trace-1")
    finally:
        proxy.close()
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join()

    record = json.loads(capture_path.read_text())
    assert record["response_id"] == "response-1"
    assert record["output_tokens"] == 7
    assert record["transport_error"].startswith("RemoteProtocolError:")


def test_disconnect_before_headers_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")

    class Disconnect(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_POST(self):
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Disconnect)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    capture_path = tmp_path / "calls.jsonl"
    proxy = CaptureProxy(
        f"http://127.0.0.1:{upstream.server_port}", capture_path
    )
    proxy.start()
    try:
        response = httpx.post(
            proxy.base_url + "/chat/completions",
            headers={"X-Trace-Id": "trace-2"},
            json={"stream": True},
        )
        assert response.status_code == 502
        proxy.wait_idle("trace-2")
    finally:
        proxy.close()
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join()

    record = json.loads(capture_path.read_text())
    assert record["status"] is None
    assert record["output_tokens"] is None
    assert record["transport_error"].startswith("RemoteProtocolError:")
