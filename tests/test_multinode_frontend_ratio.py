import asyncio
import copy
import json

import pytest

from agenttrace.experiments.multi.config import node_layout, write_json
from agenttrace.experiments.multi.frontend import wait_files, work


@pytest.mark.parametrize("frontends,builds", [(2, 2), (1, 8)])
def test_frontend_controls_keep_cc_seeds_and_direct_backends(tmp_path, monkeypatch, frontends, builds):
    point = {"frontend_nodes": frontends, "physical_nodes": frontends + 8, "inference_replicas": 8,
             "workload": "minisweagent", "proxy": False, "routing": "sticky", "prefix_reuse": "normal",
             "seed": 42, "task_cc_per_replica": 32, "warmup_seconds": 1800, "duration_seconds": 5400}
    mapping = node_layout(point, [f"node{i}" for i in range(frontends + 8)], "node0")
    (tmp_path / "control").mkdir()
    for host in mapping["frontends"]:
        (tmp_path / "frontends" / host).mkdir(parents=True)
    trace = tmp_path / "trace.json"
    write_json(trace, {"context": {"container_image": "docker://test:latest"}})
    config = {"point": point, "node_mapping": mapping, "serve": {"VLLM_PORT": "18000"},
              "trace_paths": [str(trace)], "preparation_timeout_seconds": 5,
              "replay_profile": {"llm_executor": {"config": {}}, "tool_executor": {"config": {
                  "image_cache_dir": str(tmp_path / "shared"), "max_parallel_sandbox_builds": builds}}}}
    original = copy.deepcopy(config)
    replays, samplers, stages, prepared = [], [], [], []

    def stage(shared, local, sources):
        stages.append((shared, local, sources))
        return {"local_cache": str(local)}

    async def precache(pool, output, profile):
        prepared.append(profile)

    async def monitor(path, *, ready, stop, **kwargs):
        samplers.append(path)
        ready.set()
        await stop.wait()

    async def replay(paths, **kwargs):
        replays.append(kwargs)
        kwargs["output"].mkdir(parents=True)
        write_json(kwargs["output"] / "ready.json", {})
        await wait_files([tmp_path / "control/start.json"], timeout=5)

    monkeypatch.setattr("agenttrace.miniswe_images.stage_images", stage)
    monkeypatch.setattr("agenttrace.experiments.multi.frontend.precache", precache)
    monkeypatch.setattr("agenttrace.experiments.multi.frontend.monitor", monitor)
    monkeypatch.setattr("agenttrace.experiments.multi.frontend.benchmark", replay)

    async def run():
        tasks = [asyncio.create_task(work(tmp_path, config, h)) for h in mapping["frontends"]]
        try:
            await wait_files([tmp_path / "frontends" / h / "ready.json" for h in mapping["frontends"]], tasks, 5)
            write_json(tmp_path / "control/start.json", {})
            await wait_files([tmp_path / "frontends" / h / "complete.json" for h in mapping["frontends"]], tasks, 5)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(run())
    assert config == original
    assert len(replays) == 8 and sum(r["concurrency"] for r in replays) == 256
    assert sorted(r["seed"] for r in replays) == list(range(42, 50))
    assert len(samplers) == len(stages) == len(prepared) == frontends
    assert len({r["output"] for r in replays}) == 8
    for replica in mapping["replicas"]:
        replay_args = next(r for r in replays if r["output"].name == replica["id"])
        assert replay_args["profile"].parent.name == replica["frontend_host"]
        assert replay_args["child_env"]["AGENTTRACE_HOME_REPLICA"] == replica["id"]
        assert replay_args["child_env"]["AGENTTRACE_LLM_BASE_URL"] == f"http://{replica['inference_host']}:18000/v1"
        assert replay_args["control_dir"] == tmp_path / "control"
        assert (replay_args["warmup_seconds"], replay_args["duration_seconds"]) == (1800, 5400)
    for host in mapping["frontends"]:
        profile = json.loads((tmp_path / "frontends" / host / "profile.json").read_text())
        tool = profile["tool_executor"]["config"]
        assert tool["max_parallel_sandbox_builds"] == builds
        # Each physical frontend has its own /local/scratch filesystem and build lock pool.
        assert tool["image_cache_dir"].startswith("/local/scratch/")
        assert tool["sandbox_build_lock_dir"].startswith("/local/scratch/")
        assert not profile["llm_executor"]["config"].get("routing_envelope", False)
