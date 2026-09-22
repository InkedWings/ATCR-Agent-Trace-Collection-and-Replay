"""Local streaming router for the four mechanism arms; never retries a call."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web


class Router:
    def __init__(self, backends: dict[str, str], routing: str, reuse: str, run_id: str,
                 ledger: Path, offset: int = 0):
        if not backends or routing not in ("sticky", "round_robin") or reuse not in ("normal", "none"):
            raise ValueError("invalid proxy configuration")
        self.backends, self.routing, self.reuse = backends, routing, reuse
        self.ids = list(backends)
        self.sequence = offset
        self.run_id, self.ledger_path = run_id, ledger

    async def request(self, request: web.Request) -> web.StreamResponse:
        task, call, home = (request.headers.get(f"X-Agenttrace-{key}") for key in ("Task", "Call", "Home"))
        if not task or not call or home not in self.backends:
            raise web.HTTPBadRequest(text="task, call and home replica headers are required")
        payload = await request.json()
        destination = home if self.routing == "sticky" else self.ids[self.sequence % len(self.ids)]
        self.sequence += 1
        # A different cache namespace per HTTP call prevents reuse, while preserving APC/KV sizing.
        if self.reuse == "none":
            payload["cache_salt"] = f"{self.run_id}/{task}/{call}/{self.sequence}"
        record = {"task_instance_id": task, "node_id": call, "home": home, "destination": destination,
                  "started_unix": time.time(), "routing": self.routing, "prefix_reuse": self.reuse,
                  "target_output_tokens": payload.get("max_tokens"), "status": "failed"}
        response = None
        try:
            async with self.client.post(self.backends[destination].rstrip("/") + "/chat/completions",
                                        json=payload, allow_redirects=False) as upstream:
                record["http_status"] = upstream.status
                response = web.StreamResponse(status=upstream.status, headers={
                    "Content-Type": upstream.headers.get("Content-Type", "text/event-stream"),
                    "X-Agenttrace-Replica": destination})
                await response.prepare(request)
                async for chunk in upstream.content.iter_any():
                    await response.write(chunk)
                await response.write_eof()
                record["status"] = "completed" if 200 <= upstream.status < 300 else "http_error"
                return response
        except (Exception, asyncio.CancelledError) as error:
            record["error_type"] = type(error).__name__
            if response is not None and response.prepared and request.transport is not None:
                request.transport.close()  # A truncated stream must not appear successfully completed.
            raise
        finally:
            record["ended_unix"] = time.time()
            self.ledger.write(json.dumps(record) + "\n")

    @asynccontextmanager
    async def serve(self, port: int = 18010):
        self.ledger = self.ledger_path.open("x", buffering=1)
        runner = None
        try:
            async with ClientSession(trust_env=False, connector=TCPConnector(limit=0),
                    timeout=ClientTimeout(total=None, sock_connect=10, sock_read=None)) as self.client:
                app = web.Application(client_max_size=64 * 1024 * 1024)
                app.router.add_post("/v1/chat/completions", self.request)
                runner = web.AppRunner(app, access_log=None, handler_cancellation=True, shutdown_timeout=5)
                await runner.setup()
                site = web.TCPSite(runner, "127.0.0.1", port)
                try:
                    await site.start()
                    actual_port = site._server.sockets[0].getsockname()[1]
                    yield f"http://127.0.0.1:{actual_port}/v1"
                finally:
                    await runner.cleanup()
        finally:
            self.ledger.close()
