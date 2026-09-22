"""Lossless chat transport through vLLM Router 0.1.15's completion policy path.

The official chat path extracts only session_id and rewrites message fields.
CompletionRequest accepts a text prefix and preserves unknown top-level fields.
Both experiment arms use this same envelope; routing decisions remain upstream.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

PAYLOAD_FIELD = "agenttrace_chat_payload"


def wrap_chat(payload: dict) -> dict:
    # Keep prompt-affecting configuration before the append-only conversation.
    # This is a character-prefix estimate, not the model's tokenized chat template.
    context = {key: payload[key] for key in (
        "model", "tools", "tool_choice", "parallel_tool_calls", "functions", "function_call",
        "chat_template", "chat_template_kwargs", "add_generation_prompt", "continue_final_message",
        "documents") if key in payload}
    prefix = json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    prefix += "\n" + json.dumps(payload["messages"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {"model": payload["model"], "prompt": prefix, "max_tokens": payload["max_tokens"],
            "stream": payload["stream"], "stream_options": payload.get("stream_options", {}),
            PAYLOAD_FIELD: payload}


class RoutingEnvelopeMiddleware:
    """Restore original chat payload and record actual destination at the backend.

    Loaded by vLLM's --middleware option only for official-router experiments.
    It neither chooses a destination nor retries, and does not buffer SSE output.
    """

    def __init__(self, app):
        self.app = app
        self.replica = os.environ["AGENTTRACE_ROUTING_REPLICA"]
        self.ledger = Path(os.environ["AGENTTRACE_ROUTING_EVENTS"])

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/v1/completions":
            return await self.app(scope, receive, send)
        chunks = []
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        envelope = json.loads(body)
        payload = envelope.get(PAYLOAD_FIELD)
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            scope = {**scope, "path": "/v1/chat/completions", "raw_path": b"/v1/chat/completions",
                     "headers": [(k, v) for k, v in scope["headers"] if k.lower() != b"content-length"]
                                + [(b"content-length", str(len(body)).encode())]}
        delivered = False

        async def restored_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        headers = {k.lower(): v.decode("latin-1") for k, v in scope["headers"]}
        task, call = headers.get(b"x-agenttrace-task"), headers.get(b"x-agenttrace-call")
        record = {"task_instance_id": task, "node_id": call, "home": headers.get(b"x-agenttrace-home"),
                  "destination": self.replica, "started_unix": time.time(), "status": "failed"}
        if payload is not None:
            record["target_output_tokens"] = payload.get("max_tokens")

        async def observed_send(message):
            if message["type"] == "http.response.start":
                record["http_status"] = message["status"]
                message = {**message, "headers": list(message.get("headers", []))
                           + [(b"x-agenttrace-replica", self.replica.encode())]}
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                record["status"] = "completed" if 200 <= record.get("http_status", 0) < 300 else "http_error"

        try:
            await self.app(scope, restored_receive, observed_send)
        except (Exception, asyncio.CancelledError) as error:
            record["status"] = "failed"
            record["error_type"] = type(error).__name__
            raise
        finally:
            if task and call:
                record["ended_unix"] = time.time()
                with self.ledger.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
