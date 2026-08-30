"""Runtime path and resource bindings for fixed-path tool replay."""

from __future__ import annotations

import re
from typing import Any


class Bindings:
    def __init__(self, initial: dict[str, str] | None = None) -> None:
        self._values = dict(initial or {})

    @property
    def values(self) -> dict[str, str]:
        return dict(self._values)

    def add(self, recorded: str, replayed: str) -> None:
        if recorded and replayed and recorded != replayed:
            self._values[recorded] = replayed

    def rewrite(self, value: Any) -> Any:
        if isinstance(value, str):
            if not self._values:
                return value
            recorded_values = sorted(self._values, key=len, reverse=True)
            pattern = re.compile("|".join(re.escape(item) for item in recorded_values))
            return pattern.sub(lambda match: self._values[match.group(0)], value)
        if isinstance(value, list):
            return [self.rewrite(item) for item in value]
        if isinstance(value, dict):
            return {key: self.rewrite(item) for key, item in value.items()}
        return value

    def learn_from_results(self, recorded: Any, replayed: Any, key: str = "") -> None:
        if isinstance(recorded, dict) and isinstance(replayed, dict):
            for child_key in recorded.keys() & replayed.keys():
                self.learn_from_results(recorded[child_key], replayed[child_key], child_key)
            return
        if isinstance(recorded, list) and isinstance(replayed, list):
            for left, right in zip(recorded, replayed):
                self.learn_from_results(left, right, key)
            return
        dynamic_keys = {"fullOutputPath", "sessionId", "resourceId", "processId"}
        if key in dynamic_keys and isinstance(recorded, str) and isinstance(replayed, str):
            self.add(recorded, replayed)
