import asyncio
import shlex
import sys

import pytest

from agenttrace.experiments.multi.runtime import check_clocks


@pytest.mark.parametrize("skew,valid", [(0, True), (2, False), (-2, False)])
def test_clock_sampling_excludes_startup_but_detects_skew(monkeypatch, skew, valid):
    spawn = asyncio.create_subprocess_exec
    processes = []

    async def local_responder(*args, **kwargs):
        assert args[0] == "ssh"
        responder = shlex.split(args[-1])[-1]
        # Emulate slow SSH/shell/Python startup, then an independently offset clock.
        setup = (
            "import time\n"
            "time.sleep(0.7)\n"
            "original_time = time.time\n"
            f"time.time = lambda: original_time() + {skew}\n"
        )
        process = await spawn(sys.executable, "-u", "-c", setup + responder, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr("agenttrace.experiments.multi.runtime.asyncio.create_subprocess_exec", local_responder)
    result = asyncio.run(check_clocks(["node-a", "node-b"]))
    assert result["valid"] is valid
    assert result["tolerance_seconds"] == .5
    for sample in result["hosts"].values():
        assert sample["rtt_seconds"] < .5
        assert sample["offset_seconds"] == pytest.approx(skew, abs=.1)
    assert len(processes) == 2  # All samples reuse one ready connection per node.
    assert all(process.returncode == 0 for process in processes)


@pytest.mark.parametrize("ready", [False, True])
def test_clock_responder_failure_is_not_validated(monkeypatch, ready):
    spawn = asyncio.create_subprocess_exec
    processes = []

    async def broken_responder(*args, **kwargs):
        code = "import sys\n"
        if ready:
            code += "print('agenttrace-clock-ready', flush=True)\nsys.stdin.readline()\n"
        code += "sys.exit(7)\n"
        process = await spawn(sys.executable, "-u", "-c", code, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr("agenttrace.experiments.multi.runtime.asyncio.create_subprocess_exec", broken_responder)
    with pytest.raises(ExceptionGroup) as error:
        asyncio.run(check_clocks(["broken-node"]))
    assert "broken-node" in str(error.value.exceptions[0])
    assert all(process.returncode == 7 for process in processes)
