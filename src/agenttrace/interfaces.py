"""Small framework extension interfaces used by collection and replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(slots=True)
class CapturedArtifacts:
    trace_id: str
    trajectory_path: Path
    capture_path: Path
    output_dir: Path
    workspace_root: Path
    workspace_seed: Path | None = None
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    framework_version: str = "unknown"


@dataclass(slots=True)
class ToolExecutionResult:
    result: dict[str, Any]

    @property
    def is_error(self) -> bool:
        return bool(self.result.get("isError", False))


@dataclass(slots=True)
class LLMExecutionResult:
    actual_output_tokens: int


class CollectionAdapter(Protocol):
    async def collect_case(
        self, case: dict[str, Any], context: dict[str, Any]
    ) -> CapturedArtifacts: ...

    def build_trace(self, artifacts: CapturedArtifacts) -> dict[str, Any]: ...


class ToolExecutor(Protocol):
    async def setup(self, trace: dict[str, Any], workspace: Path) -> None: ...

    async def execute(self, node: dict[str, Any]) -> ToolExecutionResult: ...

    async def close(self) -> None: ...


class LLMExecutor(Protocol):
    async def setup(self, trace: dict[str, Any], workspace: Path) -> None: ...

    async def execute(self, node: dict[str, Any]) -> LLMExecutionResult: ...

    async def close(self) -> None: ...
