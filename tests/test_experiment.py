import asyncio
import copy
import json

import httpx

from agenttrace.experiments.prepare import prepare


def test_preflight_keeps_exact_template_and_excludes_whole_trace(tmp_path, monkeypatch, minimal_trace):
    paths = []
    for i in range(2):
        trace = copy.deepcopy(minimal_trace)
        trace["nodes"][0]["request"]["endpoint"] = "/chat/completions"
        payload = trace["nodes"][0]["request"]["payload"]
        payload.update(messages=[{"role": "user", "content": str(i)}], tools=[],
                       chat_template_kwargs={"enable_thinking": False})
        path = tmp_path / f"{i}.json"
        path.write_text(json.dumps(trace))
        paths.append(path)
    def handler(request):
        assert request.url.path == "/tokenize"
        payload = json.loads(request.content)
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert payload["tools"] == [] and payload["model"] == "qwen/qwen3-32b"
        return httpx.Response(200, json={"count": 32765 if payload["messages"][0]["content"] == "1" else 32764})
    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client(**kwargs, transport=httpx.MockTransport(handler)))
    result = asyncio.run(prepare(paths, tmp_path / "pool", "http://test/v1"))
    assert result["candidate_count"] == 2 and result["eligible_count"] == 1
    assert result["traces"][1]["exclusion_reason"] == "context_overflow"
    assert (tmp_path / "pool/traces.txt").read_text() == str(paths[0]) + "\n"
    assert "sha256" not in json.dumps(result)
