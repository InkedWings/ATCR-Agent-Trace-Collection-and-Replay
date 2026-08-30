from __future__ import annotations

from agenttrace.loader import create_from_spec, load_symbol
from agenttrace.replay.llm import OpenAICompatibleExecutor


def test_module_factory_loading():
    factory = load_symbol("agenttrace.replay.llm:create_openai_executor")
    assert callable(factory)
    executor = create_from_spec(
        {
            "factory": "agenttrace.replay.llm:create_openai_executor",
            "config": {"base_url": "http://localhost:8000", "ignore_eos": True},
        }
    )
    assert isinstance(executor, OpenAICompatibleExecutor)
    assert executor.ignore_eos is True
