from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agenttrace.replay.engine import replay_trace
from agenttrace.replay.llm import OpenAICompatibleExecutor


def test_fake_openai_stream_enforces_target_and_discards_text(tmp_path, minimal_trace):
    received: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            payload = json.loads(body)
            received.append(payload)
            target = payload["max_tokens"]
            chunks = [
                {"id": "new-response", "choices": [{"delta": {"content": "discard me"}}]},
                {"id": "new-response", "choices": [], "usage": {"completion_tokens": target}},
            ]
            data = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
            data += "data: [DONE]\n\n"
            encoded = data.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        trace_path = tmp_path / "trace.json"
        trace_path.write_text(json.dumps(minimal_trace), encoding="utf-8")
        executor = OpenAICompatibleExecutor(
            f"http://127.0.0.1:{server.server_port}",
            ignore_eos=True,
            model_override="replay-model",
            trust_env=False,
        )
        report = asyncio.run(
            replay_trace(
                trace_path, llm_executor=executor, run_dir=tmp_path / "replay"
            )
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert received[0]["max_tokens"] == 4
    assert received[0]["model"] == "replay-model"
    assert received[0]["ignore_eos"] is True
    assert received[0]["stream_options"]["include_usage"] is True
    assert report["nodes"][0]["actual_output_tokens"] == 4
    assert "discard me" not in json.dumps(report)
