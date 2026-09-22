import asyncio
import json
import multiprocessing
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agenttrace.adapters.minisweagent import MiniSWEAgentToolExecutor
from agenttrace.miniswe_images import cached_image, prepare, sandbox_build_slot, stage_images


def test_sequential_cache_resumes_and_uses_only_local_image(tmp_path, monkeypatch):
    source = "docker://docker.io/swebench/test:latest"
    trace = {"context": {"container_image": source, "instance_id": "django__django-1"}}
    (tmp_path / "trace.json").write_text(json.dumps(trace))
    pool = tmp_path / "pool.txt"
    pool.write_text("trace.json\ntrace.json\n")
    cache = tmp_path / "cache"
    commands = []

    def execute(args, **kwargs):
        commands.append(args)
        if args[1] == "pull":
            Path(args[3]).write_bytes(b"test image")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("agenttrace.miniswe_images.subprocess.run", execute)
    prepare(pool, cache, "apptainer")
    assert [c[1] for c in commands] == ["pull", "exec"]
    prepare(pool, cache, "apptainer")
    assert len(commands) == 2
    assert json.loads((cache / "status.json").read_text())["completed"] == 1
    image = cached_image(cache, source)
    options = []
    monkeypatch.setitem(sys.modules, "minisweagent.environments.singularity",
                        SimpleNamespace(SingularityEnvironment=lambda **kw: options.append(kw)))
    executor = MiniSWEAgentToolExecutor({"image_cache_dir": str(cache)})
    asyncio.run(executor.setup(trace, tmp_path / "workspace"))
    assert options[0]["image"] == str(image)
    image.unlink()
    with pytest.raises(FileNotFoundError, match="missing or empty"):
        asyncio.run(executor.setup(trace, tmp_path / "workspace"))
    assert len(options) == 1  # A cache miss never falls back to a registry build.


@pytest.mark.parametrize("error", ["TOOMANYREQUESTS: pull rate limit", "error writing layer: unexpected EOF"])
def test_download_failure_retries_same_image_before_next(tmp_path, monkeypatch, error):
    pool = tmp_path / "pool.txt"
    for i in (1, 2):
        (tmp_path / f"{i}.json").write_text(json.dumps({"context": {
            "instance_id": f"test-{i}", "container_image": f"docker://test/{i}:latest"}}))
    pool.write_text("1.json\n2.json\n")
    sources, sleeps = [], []

    def execute(args, **kwargs):
        if args[1] == "pull":
            sources.append(args[-1])
            if len(sources) == 1:
                kwargs["stdout"].write(error)
                return SimpleNamespace(returncode=255)
            Path(args[3]).write_bytes(b"image")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("agenttrace.miniswe_images.subprocess.run", execute)
    monkeypatch.setattr("agenttrace.miniswe_images.time.sleep", sleeps.append)
    prepare(pool, tmp_path / "cache", "apptainer", retry_wait=300)
    assert sources == ["docker://test/1:latest", "docker://test/1:latest", "docker://test/2:latest"]
    assert sleeps == [300]


@pytest.mark.parametrize("error, attempts", [
    ("unexpected EOF", 5), ("Disk quota exceeded", 1),
])
def test_download_failure_stops_on_persistent_or_storage_error(tmp_path, monkeypatch, error, attempts):
    (tmp_path / "trace.json").write_text(json.dumps({"context": {
        "instance_id": "test-1", "container_image": "docker://test/1:latest"}}))
    pool = tmp_path / "pool.txt"
    pool.write_text("trace.json\n")
    commands, sleeps = [], []

    def execute(args, **kwargs):
        commands.append(args)
        kwargs["stdout"].write(error)
        return SimpleNamespace(returncode=255)

    monkeypatch.setattr("agenttrace.miniswe_images.subprocess.run", execute)
    monkeypatch.setattr("agenttrace.miniswe_images.time.sleep", sleeps.append)
    cache = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="image pull failed"):
        prepare(pool, cache, "apptainer")
    assert len(commands) == attempts
    assert sleeps == [300] * (attempts - 1)
    assert json.loads((cache / "status.json").read_text())["state"] == "failed"
    assert not (cache / "index.json").exists()


def test_unknown_image_and_unprepared_cache_do_not_resolve(tmp_path):
    with pytest.raises(FileNotFoundError, match="not prepared"):
        cached_image(tmp_path, "docker://test:latest")
    (tmp_path / "index.json").write_text(json.dumps({"images": {}}))
    with pytest.raises(FileNotFoundError, match="not cached"):
        cached_image(tmp_path, "docker://test:latest")


def test_local_staging_resumes_after_interrupted_copy(tmp_path, monkeypatch):
    shared, local = tmp_path / "shared", tmp_path / "local"
    shared.mkdir()
    sources = ["docker://test/1:latest", "docker://test/2:latest"]
    index = {"images": {source: {"file": f"test-{i}.sif"} for i, source in enumerate(sources, 1)}}
    (shared / "index.json").write_text(json.dumps(index))
    for i in (1, 2):
        (shared / f"test-{i}.sif").write_bytes(b"complete image" * i)
    copies = []

    def copy(source, target):
        copies.append(source.name)
        target.write_bytes(source.read_bytes())
        if len(copies) == 2:
            target.write_bytes(b"incomplete")
            raise OSError("interrupted copy")

    monkeypatch.setattr("agenttrace.miniswe_images.shutil.copyfile", copy)
    with pytest.raises(OSError, match="interrupted copy"):
        stage_images(shared, local, sources)
    assert cached_image(local, sources[0]).read_bytes() == b"complete image"
    with pytest.raises(FileNotFoundError, match="not cached"):
        cached_image(local, sources[1])
    result = stage_images(shared, local, sources + sources)
    assert (result["copied"], result["reused"], result["images"]) == (1, 1, 2)
    assert copies == ["test-1.sif", "test-2.sif", "test-2.sif"]
    assert not (local / "test-2.sif.partial").exists()
    result = stage_images(shared, local, sources)
    assert (result["copied"], result["reused"]) == (0, 2)
    assert len(copies) == 3
    (local / "test-1.sif").write_bytes(b"truncated")
    assert stage_images(shared, local, sources)["copied"] == 1


def test_local_staging_rejects_missing_shared_image(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "index.json").write_text(json.dumps({"images": {}}))
    with pytest.raises(FileNotFoundError, match="not cached"):
        stage_images(shared, tmp_path / "local", ["docker://test/missing:latest"])
    assert not (tmp_path / "local").exists()


def test_multinode_checks_cache_before_creating_run(tmp_path, minimal_trace, monkeypatch):
    from agenttrace.experiments.multi.runtime import prepare_run

    monkeypatch.delenv("AGENTTRACE_MINISWE_IMAGE_CACHE", raising=False)
    source = "docker://test:latest"
    minimal_trace["context"].update(container_image=source, instance_id="test-1")
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps(minimal_trace))
    config = {"repo": str(tmp_path), "point": {"workload": "minisweagent", "task_cc_per_replica": 2,
        "inference_replicas": 1, "frontend_nodes": 1, "physical_nodes": 2, "layout": "balanced", "smoke": False},
        "serve": {"VLLM_MAX_NUM_BATCHED_TOKENS": "8192"}, "trace_paths": [str(trace)],
        "replay_profile": {"tool_executor": {"config": {}}}}
    root = tmp_path / "run"
    with pytest.raises(FileNotFoundError, match="not prepared"):
        prepare_run(config, root, ["front", "inference"], "front")
    assert not root.exists()
    cache = tmp_path / "runs/cache/minisweagent-images"
    cache.mkdir(parents=True)
    (cache / "test.sif").write_bytes(b"image")
    (cache / "index.json").write_text(json.dumps({"images": {source: {"file": "test.sif"}}}))
    prepare_run(config, root, ["front", "inference"], "front")
    frozen = json.loads((root / "config.json").read_text())
    assert frozen["replay_profile"]["tool_executor"]["config"]["image_cache_dir"] == str(cache)


def test_sandbox_build_slots_bound_independent_replay_processes(tmp_path):
    ctx = multiprocessing.get_context("fork")
    start, ready = ctx.Event(), ctx.Event()
    active, peak = ctx.Value("i", 0), ctx.Value("i", 0)
    mutex = ctx.Lock()

    def build():
        start.wait(5)
        with sandbox_build_slot(tmp_path / "slots", 2):
            with mutex:
                active.value += 1
                peak.value = max(peak.value, active.value)
                if active.value == 2:
                    ready.set()
            assert ready.wait(5)
            time.sleep(.1)
            with mutex:
                active.value -= 1

    workers = [ctx.Process(target=build) for _ in range(6)]
    try:
        for worker in workers:
            worker.start()
        start.set()
        for worker in workers:
            worker.join(10)
        assert all(worker.exitcode == 0 for worker in workers)
        assert peak.value == 2 and active.value == 0
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(5)


def test_sandbox_build_slot_is_released_after_process_exit(tmp_path):
    ctx = multiprocessing.get_context("fork")
    ready = ctx.Event()

    def hold():
        with sandbox_build_slot(tmp_path, 1):
            ready.set()
            time.sleep(30)

    worker = ctx.Process(target=hold)
    worker.start()
    try:
        assert ready.wait(5)
    finally:
        worker.terminate()
        worker.join(5)
    # Verify the kernel released the lock without entering an unbounded wait.
    import fcntl
    with (tmp_path / "slot-0.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_bounded_executor_records_provisioning_wait(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    expected = object()
    monkeypatch.setitem(sys.modules, "minisweagent.environments.singularity",
                        SimpleNamespace(SingularityEnvironment=lambda **kw: expected))
    config = {"max_parallel_sandbox_builds": 2, "sandbox_build_lock_dir": str(tmp_path / "slots")}
    executor = MiniSWEAgentToolExecutor(config)
    asyncio.run(executor.setup({"context": {"container_image": "/local/image.sif"}}, workspace))
    assert executor.environment is expected
    stats = json.loads((tmp_path / "sandbox-setup.json").read_text())
    assert stats["max_parallel_builds"] == 2
    assert stats["slot_wait_seconds"] >= 0 and stats["build_seconds"] >= 0
    with pytest.raises(ValueError, match="lock_dir"):
        MiniSWEAgentToolExecutor({"max_parallel_sandbox_builds": 2})
