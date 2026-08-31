"""Transparent OpenAI-compatible chat-completion capture proxy."""

from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx


def _response_metadata(body: bytes, content_type: str) -> tuple[str | None, int | None]:
    values: list[dict[str, Any]] = []
    text = body.decode("utf-8", errors="replace")
    if "text/event-stream" in content_type:
        for line in text.splitlines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                value = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                values.append(value)
    else:
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict):
            values.append(value)

    response_id: str | None = None
    output_tokens: int | None = None
    for value in values:
        if isinstance(value.get("id"), str):
            response_id = value["id"]
        usage = value.get("usage") or {}
        if isinstance(usage.get("completion_tokens"), int):
            output_tokens = usage["completion_tokens"]
    return response_id, output_tokens


class CaptureProxy:
    """Forward requests while recording only requests and response metadata."""

    def __init__(
        self,
        upstream: str,
        output: Path,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self.upstream = upstream.rstrip("/") + "/"
        self.output = output
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self._active_condition = threading.Condition()
        self._active_requests = 0
        self._active_trace_requests: dict[str, int] = {}
        self._sequence = 0
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                return

            def do_GET(self) -> None:
                if self.path == "/health":
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"ok\n")
                    return
                self.send_error(404)

            def do_POST(self) -> None:
                trace_id = self.headers.get("X-Trace-Id") or ""
                with proxy._active_condition:
                    proxy._active_requests += 1
                    proxy._active_trace_requests[trace_id] = (
                        proxy._active_trace_requests.get(trace_id, 0) + 1
                    )
                try:
                    body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                    payload = json.loads(body)
                    if not isinstance(payload, dict):
                        raise ValueError("OpenAI request payload must be an object")
                    if payload.get("stream"):
                        payload.setdefault("stream_options", {})["include_usage"] = True
                    forwarded_body = json.dumps(payload, separators=(",", ":")).encode()
                    headers = {
                        key: value
                        for key, value in self.headers.items()
                        if key.lower()
                        not in {"host", "content-length", "connection", "x-trace-id"}
                    }
                    target = urljoin(proxy.upstream, self.path.lstrip("/"))
                    response_body = bytearray()
                    content_type = ""
                    response_status: int | None = None
                    response_started = False
                    transport_error: str | None = None
                    try:
                        with httpx.Client(timeout=None, trust_env=True) as client:
                            with client.stream(
                                "POST", target, headers=headers, content=forwarded_body
                            ) as response:
                                response_status = response.status_code
                                response_started = True
                                self.send_response(response.status_code)
                                content_type = response.headers.get("content-type", "")
                                if content_type:
                                    self.send_header("Content-Type", content_type)
                                self.end_headers()
                                connected = True
                                for chunk in response.iter_bytes():
                                    response_body.extend(chunk)
                                    if connected:
                                        try:
                                            self.wfile.write(chunk)
                                            self.wfile.flush()
                                        except (BrokenPipeError, ConnectionResetError):
                                            connected = False
                    except (httpx.TransportError, httpx.StreamError) as error:
                        # Preserve metadata already received before an upstream
                        # response or chunked stream terminates abruptly. If
                        # usage never arrived, trace construction still rejects
                        # the incomplete LLM call.
                        transport_error = f"{type(error).__name__}: {error}"
                        if not response_started:
                            self.send_error(502, "upstream transport error")

                    response_id, output_tokens = _response_metadata(
                        bytes(response_body), content_type
                    )
                    with proxy._write_lock:
                        proxy._sequence += 1
                        record = {
                            "sequence": proxy._sequence,
                            "call_id": str(uuid.uuid4()),
                            "trace_id": trace_id or None,
                            "protocol": "openai-chat-completions",
                            "endpoint": self.path,
                            "request": payload,
                            "status": response_status,
                            "response_id": response_id,
                            "output_tokens": output_tokens,
                            "transport_error": transport_error,
                        }
                        with proxy.output.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                finally:
                    with proxy._active_condition:
                        proxy._active_requests -= 1
                        remaining = proxy._active_trace_requests[trace_id] - 1
                        if remaining:
                            proxy._active_trace_requests[trace_id] = remaining
                        else:
                            del proxy._active_trace_requests[trace_id]
                        proxy._active_condition.notify_all()

        self._server = ThreadingHTTPServer((host, port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()

    def wait_idle(self, trace_id: str | None = None, timeout: float = 30.0) -> None:
        """Wait for the upstream response to be fully recorded.

        OpenClaw can exit immediately after consuming the final stream chunk,
        while the proxy still needs a few seconds to parse usage and append the
        capture record.  Five seconds was too short on Polaris under load and
        caused complete calls to be reported as missing.
        """
        deadline = time.monotonic() + timeout
        with self._active_condition:
            while (
                self._active_trace_requests.get(trace_id, 0)
                if trace_id is not None
                else self._active_requests
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    scope = f" for trace {trace_id}" if trace_id is not None else ""
                    raise TimeoutError(f"LLM capture proxy did not become idle{scope}")
                self._active_condition.wait(remaining)
