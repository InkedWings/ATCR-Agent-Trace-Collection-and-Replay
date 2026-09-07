"""OpenAI-compatible fixed-output-length LLM replay."""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx

from agenttrace.interfaces import LLMExecutionResult


class OpenAICompatibleExecutor:
    def __init__(
        self,
        base_url: str,
        api_key_env: str | None = None,
        ignore_eos: bool = False,
        max_tokens_field: str = "max_tokens",
        model_override: str | None = None,
        trust_env: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.ignore_eos = ignore_eos
        self.max_tokens_field = max_tokens_field
        self.model_override = model_override
        self.trust_env = trust_env
        self.client: httpx.AsyncClient | None = None

    async def setup(self, trace: dict[str, Any], workspace: Path) -> None:
        headers: dict[str, str] = {}
        if self.api_key_env:
            token = os.environ.get(self.api_key_env)
            if not token:
                raise RuntimeError(f"missing LLM credential environment: {self.api_key_env}")
            headers["Authorization"] = f"Bearer {token}"
        self.client = httpx.AsyncClient(headers=headers, timeout=None, trust_env=self.trust_env)

    async def execute(self, node: dict[str, Any]) -> LLMExecutionResult:
        if self.client is None:
            raise RuntimeError("LLM executor is not set up")
        target = node["output_tokens"]
        payload = copy.deepcopy(node["request"]["payload"])
        if self.model_override:
            payload["model"] = self.model_override
        payload[self.max_tokens_field] = target
        if self.ignore_eos:
            payload["ignore_eos"] = True
        payload["stream"] = True
        payload.setdefault("stream_options", {})["include_usage"] = True
        endpoint = node["request"]["endpoint"]
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        actual: int | None = None
        started = time.perf_counter()
        first_output: float | None = None
        last_output: float | None = None
        output_chunks = 0
        async with self.client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                value = json.loads(data)
                # Role/usage/finish-only events are not generated output. A
                # stream chunk can contain several tokens (including tool args).
                if any(_has_output(choice.get("delta") or {}) for choice in value.get("choices", [])):
                    last_output = time.perf_counter()
                    if first_output is None:
                        first_output = last_output
                    output_chunks += 1
                usage = value.get("usage") or {}
                if isinstance(usage.get("completion_tokens"), int):
                    actual = usage["completion_tokens"]
        if actual is None:
            raise RuntimeError(f"LLM endpoint returned no completion token usage for {node['id']}")
        if actual != target:
            raise RuntimeError(
                f"LLM output token mismatch for {node['id']}: target {target}, actual {actual}"
            )
        return LLMExecutionResult(
            actual_output_tokens=actual,
            ttft_seconds=first_output - started if first_output is not None else None,
            output_stream_seconds=last_output - first_output if first_output is not None else None,
            output_chunks=output_chunks,
        )

    async def close(self) -> None:
        if self.client is not None:
            await self.client.aclose()
            self.client = None


def _has_output(delta: dict[str, Any]) -> bool:
    if any(delta.get(key) for key in ("content", "reasoning", "reasoning_content")):
        return True
    calls = delta.get("tool_calls") or []
    return any((call.get("function") or {}).get("arguments") for call in calls)


def create_openai_executor(config: dict[str, Any]) -> OpenAICompatibleExecutor:
    base_url = str(config.get("base_url") or "")
    base_url_env = config.get("base_url_env")
    if base_url_env:
        base_url = os.environ.get(str(base_url_env), base_url)
    if not base_url:
        raise ValueError("OpenAI executor requires base_url or base_url_env")
    return OpenAICompatibleExecutor(
        base_url=base_url,
        api_key_env=config.get("api_key_env"),
        ignore_eos=bool(config.get("ignore_eos", False)),
        max_tokens_field=str(config.get("max_tokens_field", "max_tokens")),
        model_override=config.get("model_override"),
        trust_env=bool(config.get("trust_env", True)),
    )
