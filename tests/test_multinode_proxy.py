import asyncio
import json
from contextlib import AsyncExitStack, asynccontextmanager

import pytest
from aiohttp import ClientSession, web

from agenttrace.experiments.multi.proxy import Router


@asynccontextmanager
async def upstream(handler):
    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app, access_log=None, handler_cancellation=True)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        yield f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/v1"
    finally:
        await runner.cleanup()


@pytest.mark.parametrize("routing", ["sticky", "round_robin"])
@pytest.mark.parametrize("reuse", ["normal", "none"])
def test_stream_routing_and_payload_preservation(tmp_path, routing, reuse):
    received = []
    continue_stream = asyncio.Event()
    async def handler(request):
        payload = await request.json()
        received.append(payload)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b'data: {"choices":[{"delta":{"reasoning_content":"private"}}]}\n\n')
        if len(received) == 1:
            await continue_stream.wait()
        await response.write(f'data: {json.dumps({"usage": {"completion_tokens": payload["max_tokens"]}})}\n\n'.encode())
        await response.write(b"data: [DONE]\n\n")
        return response
    original = {"model": "qwen", "max_tokens": 4, "ignore_eos": True, "stream": True,
        "messages": [{"role": "assistant", "reasoning_content": "keep recorded field", "content": "abc"}],
        "tools": [{"type": "function", "function": {"name": "web_search"}}]}
    async def run():
        async with AsyncExitStack() as stack:
            endpoints = {f"r{i:02d}": await stack.enter_async_context(upstream(handler)) for i in range(2)}
            proxy = Router(endpoints, routing, reuse, "job", tmp_path / "routing.jsonl")
            url = await stack.enter_async_context(proxy.serve(0))
            async with ClientSession() as client:
                destinations = []
                for i in range(4):
                    async with client.post(url+"/chat/completions", json=original, headers={
                        "X-Agenttrace-Task": "job/r00/1", "X-Agenttrace-Call": str(i), "X-Agenttrace-Home": "r01"}) as response:
                        assert response.status == 200
                        destinations.append(response.headers["X-Agenttrace-Replica"])
                        line = await asyncio.wait_for(response.content.readline(), 2)
                        assert b"reasoning_content" in line  # first output forwarded before upstream completes
                        continue_stream.set()
                        body = await response.read()
                        assert b'"completion_tokens": 4' in body
                return destinations
    destinations = asyncio.run(run())
    assert destinations == (["r01"]*4 if routing == "sticky" else ["r00", "r01"]*2)
    assert len(received) == 4
    salts = [p.pop("cache_salt", None) for p in received]
    assert received == [original]*4
    assert len(set(salts)) == (4 if reuse == "none" else 1)
    assert all(salts) if reuse == "none" else salts == [None]*4
    ledger = [json.loads(line) for line in (tmp_path / "routing.jsonl").read_text().splitlines()]
    assert all(r["status"] == "completed" for r in ledger)
    assert "private" not in (tmp_path / "routing.jsonl").read_text()


def test_http_failure_is_propagated_once(tmp_path):
    calls = []
    async def handler(request):
        calls.append(1)
        return web.Response(status=503, text="unavailable")
    async def run():
        async with upstream(handler) as base:
            router = Router({"r00": base}, "sticky", "normal", "j", tmp_path / "ledger")
            async with router.serve(0) as url, ClientSession() as client:
                async with client.post(url+"/chat/completions", json={"max_tokens": 4}, headers={
                    "X-Agenttrace-Task": "j/r/1", "X-Agenttrace-Call": "llm1", "X-Agenttrace-Home": "r00"}) as response:
                    assert response.status == 503
                    assert await response.text() == "unavailable"
    asyncio.run(run())
    assert calls == [1]
    assert json.loads((tmp_path / "ledger").read_text())["status"] == "http_error"


def test_llm_executor_records_actual_destination(tmp_path, monkeypatch, minimal_trace):
    from agenttrace.replay.llm import OpenAICompatibleExecutor
    from agenttrace.replay.engine import replay_trace
    async def handler(request):
        payload = await request.json()
        return web.Response(content_type="text/event-stream", text=
            'data: {"choices":[{"delta":{"content":"x"}}]}\n\n' +
            'data: '+json.dumps({"usage": {"completion_tokens": payload["max_tokens"]}})+'\n\n')
    monkeypatch.setenv("AGENTTRACE_TASK_INSTANCE_ID", "j/r01/1")
    monkeypatch.setenv("AGENTTRACE_HOME_REPLICA", "r01")
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps(minimal_trace))
    async def run():
        async with upstream(handler) as base:
            router = Router({"r00": base, "r01": base}, "round_robin", "none", "j", tmp_path / "ledger")
            async with router.serve(0) as url:
                executor = OpenAICompatibleExecutor(url.removesuffix("/v1"), ignore_eos=True, trust_env=False)
                # minimal_trace uses /v1/chat/completions, while frozen pool profiles use /chat/completions.
                return await replay_trace(trace, llm_executor=executor, run_dir=tmp_path / "replay")
    result = asyncio.run(run())
    assert result["nodes"][0]["backend_replica_id"] == "r00"
    assert result["nodes"][0]["actual_output_tokens"] == 4


def test_client_disconnect_cancels_upstream(tmp_path):
    closed = asyncio.Event()
    async def handler(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b'data: {"choices":[]}\n\n')
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()
    async def run():
        async with upstream(handler) as base:
            router = Router({"r00": base}, "sticky", "normal", "j", tmp_path / "ledger")
            async with router.serve(0) as url, ClientSession() as client:
                response = await client.post(url+"/chat/completions", json={"max_tokens": 4}, headers={
                    "X-Agenttrace-Task": "j/r/1", "X-Agenttrace-Call": "llm1", "X-Agenttrace-Home": "r00"})
                await response.content.readline()
                response.close()
                await asyncio.wait_for(closed.wait(), 3)
    asyncio.run(run())
    record = json.loads((tmp_path / "ledger").read_text())
    assert record["status"] == "failed"
