import asyncio
import json
import os
import shlex
import signal
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agenttrace.experiments.service import serve, stop_process_group, _group_members
from agenttrace.experiments.scale import inference_service


def test_coordinator_does_not_accept_failed_remote_cleanup(tmp_path, monkeypatch):
    (tmp_path / "ready.json").write_text("{}")
    closed = []
    class Process:
        returncode = None
        stdin = SimpleNamespace(close=lambda: closed.append(True))
        async def wait(self):
            self.returncode = 1
            return 1
    async def spawn(*args, **kwargs):
        return Process()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    async def exercise():
        with pytest.raises(RuntimeError, match="cleanup failed"):
            async with inference_service(tmp_path, tmp_path, tmp_path / "config.json", "fake-host"):
                pass
    asyncio.run(exercise())
    assert closed == [True]


def test_cleanup_waits_for_worker_even_when_launcher_exits(tmp_path):
    async def exercise():
        log_path = tmp_path / "child-pid"
        child_code = ("import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                      "print(os.getpid(),flush=True); time.sleep(60)")
        with log_path.open("w") as log:
            process = await asyncio.create_subprocess_exec("bash", "-c",
                shlex.join([sys.executable, "-c", child_code]) + " & wait",
                stdout=log, start_new_session=True)
            try:
                async with asyncio.timeout(5):
                    while not log_path.read_text().strip():
                        await asyncio.sleep(.02)
                child_pid = int(log_path.read_text())
                await stop_process_group(process, grace_seconds=.2)
                assert not _group_members(process.pid)
                path = Path(f"/proc/{child_pid}/stat")
                assert not path.exists() or path.read_text().rsplit(")", 1)[1].split()[0] == "Z"
                os.kill(os.getpid(), 0)  # The test runner/user shell was not signalled.
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()
    asyncio.run(exercise())


def test_remote_stdin_eof_stops_only_owned_server(tmp_path, monkeypatch):
    config = {"serve": {"VLLM_PORT": "0"}}
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        config["serve"]["VLLM_PORT"] = str(sock.getsockname()[1])
    read_fd, write_fd = os.pipe()
    stream = os.fdopen(read_fd)
    monkeypatch.setattr(sys, "stdin", stream)
    native_spawn = asyncio.create_subprocess_exec
    async def fake_spawn(*args, **kwargs):
        return await native_spawn(sys.executable, "-c",
            "import os; from http.server import HTTPServer,BaseHTTPRequestHandler; "
            "exec('class Handler(BaseHTTPRequestHandler):\\n def do_GET(self):\\n  self.send_response(200); self.end_headers()\\n def log_message(self,*a): pass'); "
            "HTTPServer(('127.0.0.1', int(os.environ['VLLM_PORT'])), Handler).serve_forever()",
            **kwargs)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)
    async def fake_monitor(path, *, stop, ready, **kwargs):
        path.write_text("")
        ready.set()
        await stop.wait()
    monkeypatch.setattr("agenttrace.experiments.service.monitor", fake_monitor)
    async def exercise():
        worker = asyncio.create_task(serve(tmp_path, tmp_path, config))
        try:
            async with asyncio.timeout(10):
                while not (tmp_path / "ready.json").exists():
                    if worker.done():
                        await worker
                    await asyncio.sleep(.02)
            assert not worker.done()
            os.close(write_fd)
            await asyncio.wait_for(worker, 10)
            pid = json.loads((tmp_path / "service.json").read_text())["server_process_group"]
            assert not __import__("pathlib").Path(f"/proc/{pid}").exists()
            os.kill(os.getpid(), 0)
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
    try:
        asyncio.run(exercise())
    finally:
        stream.close()
