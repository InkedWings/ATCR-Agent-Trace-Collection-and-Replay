import asyncio
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from agenttrace.experiments.multi import packed
from agenttrace.experiments.multi.config import write_json


def source(tmp_path, name, nodes):
    path = tmp_path / f"{name}.json"
    write_json(path, {
        "repo": str(tmp_path), "pbs": {"queue": "preemptable", "walltime": "03:00:00"},
        "point": {"id": name, "physical_nodes": nodes, "frontend_nodes": 1,
                  "inference_replicas": nodes - 1, "warmup_seconds": 1800, "duration_seconds": 5400,
                  "task_cc_per_replica": 32, "pbs": {"queue": "preemptable"}},
        "drain_budget_seconds": 1800, "trace_paths": ["unchanged-trace"],
        "replay_profile": {"unchanged": True}, "serve": {"VLLM_TENSOR_PARALLEL_SIZE": "4"}})
    return path


def bundle(tmp_path, counts):
    sources = [source(tmp_path, f"point{i}", n) for i, n in enumerate(counts)]
    spec = {"repo": str(tmp_path), "account": "lc-mpi",
            "packs": [{"id": "test-pack", "configs": [str(p) for p in sources]}]}
    output = tmp_path / "bundle"
    packed.render(spec, output)
    return output / "packs/test-pack.json", sources


@pytest.mark.parametrize("counts", [(5, 5), (9,)])
def test_disjoint_suballocations_preserve_topology_and_frozen_inputs(tmp_path, counts):
    manifest, originals = bundle(tmp_path, counts)
    pack = json.loads(manifest.read_text())
    # PBS nodefiles can repeat every host; the batch host need not be listed first.
    allocated = [f"n{i}.alcf" for i in range(10) for _ in range(64)]
    layout = packed.partition(pack, allocated, "n3.alcf")
    used = []
    for index, (member, original, count) in enumerate(zip(layout["members"], originals, counts)):
        assert len(member["nodes"]) == count
        assert member["coordinator"] == member["nodes"][0]
        assert member["node_mapping"]["frontends"] == [member["coordinator"]]
        assert len(member["node_mapping"]["inference"]) == count - 1
        assert not set(used) & set(member["nodes"])
        used.extend(member["nodes"])
        before, after = json.loads(original.read_text()), json.loads(Path(member["config"]).read_text())
        assert before["pbs"]["queue"] == "preemptable"
        assert after["pbs"]["queue"] == after["point"]["pbs"]["queue"] == "prod"
        assert after["allocation_pack"]["allocated_nodes"] == 10
        after.pop("allocation_pack")
        before.pop("pbs")
        after.pop("pbs")
        before["point"].pop("pbs")
        after["point"].pop("pbs")
        assert before == after
    assert len(layout["unused_nodes"]) == 10 - sum(counts)
    assert not set(used) & set(layout["unused_nodes"])
    script = manifest.parent.parent / "jobs/test-pack.pbs"
    subprocess.run(["bash", "-n", str(script)], check=True)
    assert "#PBS -q prod" in script.read_text()
    assert "#PBS -l select=10:system=polaris" in script.read_text()
    with pytest.raises(ValueError, match="allocation"):
        packed.partition(pack, [f"n{i}" for i in range(9)], "n0")


def test_reject_overlapping_or_oversized_pack_before_writing(tmp_path):
    first, second = source(tmp_path, "first", 9), source(tmp_path, "second", 5)
    spec = {"repo": str(tmp_path), "account": "lc-mpi",
            "packs": [{"id": "too-big", "configs": [str(first), str(second)]}]}
    with pytest.raises(ValueError, match="ten nodes"):
        packed.render(spec, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    spec["packs"][0]["configs"] = [str(second), str(second)]
    with pytest.raises(ValueError, match="duplicate"):
        packed.render(spec, tmp_path / "bad")


def test_member_failure_does_not_cancel_other_experiment(tmp_path, monkeypatch):
    manifest, _ = bundle(tmp_path, (5, 5))
    nodefile = tmp_path / "pbs.nodes"
    nodefile.write_text("\n".join(f"n{i}" for i in range(10)))
    monkeypatch.setenv("PBS_JOBID", "123.pbs")
    monkeypatch.setenv("PBS_NODEFILE", str(nodefile))
    monkeypatch.setattr(packed.socket, "gethostname", lambda: "n0")
    real_spawn, commands = asyncio.create_subprocess_exec, []

    async def fake_ssh(*args, **kwargs):
        # Execute only local stub processes; never SSH or start an experiment in this test.
        assert args[0] == "ssh"
        host, command = args[-2:]
        words = shlex.split(command.split(" && exec ", 1)[1])
        job_id = words[1].split("=", 1)[1]
        subset_file = words[2].split("=", 1)[1]
        config = json.loads(Path(words[-1]).read_text())
        assert job_id == "123.pbs"
        assert host == Path(subset_file).read_text().split()[0]
        commands.append((host, subset_file))
        fail = config["point"]["id"] == "point0"
        code = "import time, sys; time.sleep(" + ("0.05" if fail else "0.2") + "); sys.exit(" + ("7" if fail else "0") + ")"
        return await real_spawn(sys.executable, "-c", code, **kwargs)

    monkeypatch.setattr(packed.asyncio, "create_subprocess_exec", fake_ssh)
    result = asyncio.run(packed.run_pack(manifest))
    assert result["status"] == "failed"
    assert [r["exit_code"] for r in result["members"]] == [7, 0]
    assert {host for host, _ in commands} == {"n0", "n5"}
    assert not set(Path(commands[0][1]).read_text().split()) & set(Path(commands[1][1]).read_text().split())
    root = manifest.parent.parent / "allocations/test-pack-123.pbs"
    assert json.loads((root / "point1.status.json").read_text())["status"] == "completed"


def test_disconnect_cancels_owned_experiment_and_waits_for_cleanup():
    async def exercise():
        started, cleaned = asyncio.Event(), asyncio.Event()

        async def operation():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                cleaned.set()

        async def eof():
            await started.wait()

        with pytest.raises(RuntimeError, match="disconnected"):
            await packed.until_disconnect(operation(), asyncio.create_task(eof()))
        assert cleaned.is_set()

    asyncio.run(exercise())


def test_pack_cancellation_closes_both_child_lifelines(tmp_path, monkeypatch):
    manifest, _ = bundle(tmp_path, (5, 5))
    nodefile = tmp_path / "pbs.nodes"
    nodefile.write_text("\n".join(f"n{i}" for i in range(10)))
    monkeypatch.setenv("PBS_JOBID", "123.pbs")
    monkeypatch.setenv("PBS_NODEFILE", str(nodefile))
    monkeypatch.setattr(packed.socket, "gethostname", lambda: "n0")
    real_spawn, processes = asyncio.create_subprocess_exec, []

    async def fake_ssh(*args, **kwargs):
        process = await real_spawn(sys.executable, "-c", "import sys; sys.stdin.buffer.read()", **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(packed.asyncio, "create_subprocess_exec", fake_ssh)

    async def exercise():
        task = asyncio.create_task(packed.run_pack(manifest))
        try:
            async with asyncio.timeout(5):
                while len(processes) != 2:
                    await asyncio.sleep(0.01)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert [p.returncode for p in processes] == [0, 0]
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(exercise())
    root = manifest.parent.parent / "allocations/test-pack-123.pbs"
    assert json.loads((root / "status.json").read_text())["status"] == "interrupted"
