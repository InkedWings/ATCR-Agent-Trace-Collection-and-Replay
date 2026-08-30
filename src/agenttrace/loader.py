"""Profile and ``module:factory`` loading."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any


def load_profile(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("profile root must be an object")
    return value


def load_symbol(reference: str) -> Any:
    if ":" not in reference:
        raise ValueError("adapter reference must use module:factory syntax")
    module_name, symbol_name = reference.split(":", 1)
    if not module_name or not symbol_name:
        raise ValueError("adapter reference must use module:factory syntax")
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def create_from_spec(spec: dict[str, Any]) -> Any:
    if not isinstance(spec, dict):
        raise ValueError("executor profile must be an object")
    factory = load_symbol(str(spec.get("factory", "")))
    config = spec.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("executor config must be an object")
    return factory(config)
