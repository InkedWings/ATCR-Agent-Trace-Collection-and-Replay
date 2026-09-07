"""Subprocess fixtures for benchmark lifecycle tests (never installed)."""

import asyncio
import os
import subprocess
from pathlib import Path

from agenttrace.interfaces import ToolExecutionResult


class Tools:
    async def setup(self, trace, workspace):
        self.workspace = workspace

    async def execute(self, node):
        operation = node["request"]["arguments"]["operation"]
        if operation == "block":
            # A descendant process lets the test check process-group cleanup.
            child = subprocess.Popen(["sleep", "60"])
            (self.workspace / "child.pid").write_text(str(child.pid))
            await asyncio.sleep(60)
        else:
            marker = self.workspace / "marker"
            assert not marker.exists()
            marker.write_text(str(os.getpid()))
            await asyncio.sleep(.05)
        return ToolExecutionResult({"isError": True})

    async def close(self):
        pass


def create_tools(config):
    return Tools()
