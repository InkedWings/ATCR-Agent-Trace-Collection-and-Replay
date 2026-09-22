import asyncio
import json
import socket
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

import pytest
from aiohttp import ClientSession, web

from agenttrace.experiments.multi.config import read_config, matrix
from agenttrace.experiments.multi.official_router import OfficialRouter, verify_installation
from agenttrace.routing_envelope import PAYLOAD_FIELD, RoutingEnvelopeMiddleware, wrap_chat


def settings():
    config = read_config(Path("examples/scaling/routing.json"))
    return config["router"]


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_routing_matrix_keeps_existing_groups_separate():
    config = read_config(Path("examples/scaling/routing.json"))
    points = matrix(config, group="routing")
    assert len(points) == 4 and len({p["id"] for p in points}) == 4
    assert all(p["physical_nodes"] == 5 and p["frontend_nodes"] == 1 for p in points)
    assert all(p["prefix_reuse"] == "normal" and p["router_impl"] == "vllm-router" for p in points)
    assert {p["routing"] for p in points} == {"round_robin", "cache_aware"}
    assert {p["task_cc_per_replica"] * p["inference_replicas"] for p in points} == {64, 128}
    assert all(p["pbs"]["queue"] == "preemptable" and p["pbs"]["walltime"] == "03:00:00" for p in points)
    assert len(matrix(config)) == 26
    smoke = matrix(config, group="routing", smoke=True)
    assert len(smoke) == 4 and all(p["smoke"] and p["duration_seconds"] == 60 for p in smoke)


def test_chat_prefix_excludes_output_target_and_preserves_history():
    first = {"model": "qwen", "messages": [{"role": "user", "content": "long input"}], "max_tokens": 4, "stream": True}
    second = {**first, "max_tokens": 800}
    assert wrap_chat(first)["prompt"] == wrap_chat(second)["prompt"]
    assert wrap_chat(first)[PAYLOAD_FIELD] is first
    later = {**first, "messages": first["messages"] + [{"role": "tool", "content": "result", "tool_call_id": "t"}]}
    assert wrap_chat(later)["prompt"].startswith(wrap_chat(first)["prompt"][:-1])


@asynccontextmanager
async def worker(replica, ledger, received, release, status, connections=None):
    # Exercise the actual ASGI middleware while aiohttp supplies a small HTTP backend.
    async def asgi_app(scope, receive, send):
        assert scope["path"] == "/v1/chat/completions"
        body = await receive()
        payload = json.loads(body["body"])
        received.append((replica, payload))
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"text/event-stream")]})
        if status != 200:
            await send({"type": "http.response.body", "body": b"unavailable"})
            return
        await send({"type": "http.response.body", "body": b'data: {"choices":[{"delta":{"reasoning_content":"output"}}]}\n\n',
                    "more_body": True})
        await release.wait()
        tail = 'data: ' + json.dumps({"usage": {"completion_tokens": payload["max_tokens"]}}) + '\n\ndata: [DONE]\n\n'
        await send({"type": "http.response.body", "body": tail.encode()})

    middleware = object.__new__(RoutingEnvelopeMiddleware)
    middleware.app, middleware.replica, middleware.ledger = asgi_app, replica, ledger

    async def handler(request):
        if connections is not None:
            connections.append((request.transport.get_extra_info("peername"), request.headers.get("Connection")))
        body = await request.read()
        response = None

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message):
            nonlocal response
            if message["type"] == "http.response.start":
                response = web.StreamResponse(status=message["status"],
                    headers={k.decode(): v.decode() for k, v in message["headers"]})
                await response.prepare(request)
            else:
                await response.write(message.get("body", b""))
                if not message.get("more_body", False):
                    await response.write_eof()
        await middleware({"type": "http", "path": "/v1/completions", "headers": request.raw_headers}, receive, send)
        return response

    async def health(request):
        return web.json_response({"status": "ok", "data": [{"id": "qwen"}], "model_path": "qwen"})

    app = web.Application()
    app.router.add_post("/v1/completions", handler)
    app.router.add_get("/{tail:.*}", health)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        yield f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/v1"
    finally:
        await runner.cleanup()


@pytest.mark.parametrize("policy", ["round_robin", "cache_aware", "power_of_two"])
@pytest.mark.parametrize("status", [200, 503])
def test_real_official_router_stream_payload_affinity_and_no_retry(tmp_path, monkeypatch, policy, status):
    cfg = settings()
    if not Path(cfg["python"]).is_file():
        pytest.skip("run examples/scaling/install-vllm-router.sh for official component integration tests")
    assert verify_installation(cfg)["rust_extension"]
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    cfg["prometheus_port"] = free_port()
    received = []
    payload = {"model": "qwen", "messages": [
        {"role": "system", "content": [{"type": "text", "text": "preserve array content"}]},
        {"role": "user", "content": "repeat this long prefix " * 500},
        {"role": "assistant", "content": None, "reasoning_content": "recorded reasoning",
         "tool_calls": [{"id": "tool1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "tool1", "content": "tool result"}],
        "max_tokens": 4, "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": True},
        "tools": [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}]}

    async def run():
        release = asyncio.Event()
        async with AsyncExitStack() as stack:
            endpoints = {f"r{i:02d}": await stack.enter_async_context(worker(
                f"r{i:02d}", tmp_path / f"r{i:02d}.jsonl", received, release, status)) for i in range(2)}
            router = OfficialRouter(endpoints, policy, cfg, tmp_path / "router")
            url = await stack.enter_async_context(router.serve(free_port()))
            async with ClientSession(trust_env=False) as client:
                destinations = []
                for i in range(4 if status == 200 else 1):
                    request_payload = {**payload, "messages": payload["messages"] + [{"role": "user", "content": str(i)}]}
                    async with client.post(url+"/completions", json=wrap_chat(request_payload), headers={
                        "X-Agenttrace-Task": "task1", "X-Agenttrace-Call": str(i), "X-Agenttrace-Home": "r01"}) as response:
                        assert response.status == status, await response.text() if response.status != status else ""
                        destinations.append(response.headers["X-Agenttrace-Replica"])
                        if status == 200:
                            line = await asyncio.wait_for(response.content.readline(), 5)
                            assert b"reasoning_content" in line
                            release.set()
                            assert b"[DONE]" in await response.read()
                        else:
                            assert await response.text() == "unavailable"
                    assert received[-1][1] == request_payload
                return destinations
    destinations = asyncio.run(run())
    if status == 200 and policy in ("round_robin", "cache_aware"):
        assert len(set(destinations)) == (2 if policy == "round_robin" else 1)
        if policy == "round_robin":
            assert destinations[0] == destinations[2] != destinations[1] == destinations[3]
    assert len(received) == (4 if status == 200 else 1)
    rows = [json.loads(line) for path in tmp_path.glob("r*.jsonl") for line in path.read_text().splitlines()]
    assert len(rows) == len(received)
    assert {row["status"] for row in rows} == {"completed" if status == 200 else "http_error"}
    assert all("recorded reasoning" not in path.read_text() for path in tmp_path.glob("r*.jsonl"))


@pytest.mark.parametrize("policy", ["cache_aware", "power_of_two"])
def test_real_load_aware_policies_balance_outstanding_streams(tmp_path, policy):
    cfg = settings()
    if not Path(cfg["python"]).is_file():
        pytest.skip("official router is not installed")
    cfg.update(prometheus_port=free_port(), balance_abs_threshold=1)
    received = []
    payload = {"model": "qwen", "messages": [{"role": "user", "content": "shared prefix " * 500}],
               "max_tokens": 4, "stream": True}

    async def run():
        release = asyncio.Event()
        async with AsyncExitStack() as stack:
            endpoints = {f"r{i:02d}": await stack.enter_async_context(worker(
                f"r{i:02d}", tmp_path / f"r{i:02d}.jsonl", received, release, 200)) for i in range(2)}
            router = OfficialRouter(endpoints, policy, cfg, tmp_path / "router")
            url = await stack.enter_async_context(router.serve(free_port()))
            async with ClientSession(trust_env=False) as client:
                responses = []
                try:
                    for i in range(3):
                        response = await client.post(url+"/completions", json=wrap_chat(payload), headers={
                            "X-Agenttrace-Task": "task1", "X-Agenttrace-Call": str(i), "X-Agenttrace-Home": "r01"})
                        responses.append(response)
                        assert response.status == 200
                        await asyncio.wait_for(response.content.readline(), 5)
                    destinations = [r.headers["X-Agenttrace-Replica"] for r in responses]
                    if policy == "cache_aware":
                        assert destinations[0] == destinations[1] != destinations[2]
                    else:
                        # The first response is still streaming: P2 must prefer the idle backend.
                        assert destinations[0] != destinations[1]
                finally:
                    release.set()
                    for response in responses:
                        await response.read()
                        response.close()
    asyncio.run(run())


def test_replay_executor_through_official_router(tmp_path, monkeypatch):
    from agenttrace.replay.llm import create_openai_executor
    cfg = settings()
    if not Path(cfg["python"]).is_file():
        pytest.skip("official router is not installed")
    cfg["prometheus_port"] = free_port()
    monkeypatch.setenv("AGENTTRACE_TASK_INSTANCE_ID", "job/r01/task")
    monkeypatch.setenv("AGENTTRACE_HOME_REPLICA", "r01")
    received = []
    connections = []

    async def run():
        release = asyncio.Event()
        release.set()
        async with worker("r00", tmp_path / "routing.jsonl", received, release, 200, connections) as backend:
            router = OfficialRouter({"r00": backend}, "cache_aware", cfg, tmp_path / "router")
            async with router.serve(free_port()) as url:
                executor = create_openai_executor({"base_url": url, "model_override": "qwen", "ignore_eos": True,
                                                   "routing_envelope": True, "trust_env": False})
                await executor.setup({}, tmp_path)
                try:
                    node = {"id": "llm1", "output_tokens": 4, "request": {
                        "endpoint": "/chat/completions", "payload": {"model": "original",
                        "messages": [{"role": "user", "content": "hello"}]}}}
                    first = await executor.execute(node)
                    second = await executor.execute({**node, "id": "llm2"})
                    return first, second
                finally:
                    await executor.close()
    results = asyncio.run(run())
    assert all(result.backend_replica_id == "r00" and result.actual_output_tokens == 4 for result in results)
    assert received[0][1]["ignore_eos"] is True
    assert len(connections) == len(received) == 2
    assert all(header == "close" for _, header in connections)
    assert len({peer for peer, _ in connections}) == 2
