import asyncio
import copy
import json
import subprocess
import time
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from agenttrace.experiments.multi.config import check, matrix, node_layout, pbs_for_point, read_config, render, write_json
from agenttrace.experiments.multi.frontend import wait_files
from agenttrace.experiments.multi.runtime import enter_services, guarded
from agenttrace.replay.benchmark import benchmark


def test_new_pbs_job_without_usage_accounting(monkeypatch):
    from agenttrace.experiments.scale import allocation_remaining
    monkeypatch.setattr("agenttrace.experiments.scale.subprocess.run", lambda *a, **kw: SimpleNamespace(
        stdout=json.dumps({"Jobs": {"1": {"Resource_List": {"walltime": "06:00:00"}}}})))
    assert allocation_remaining("1") == 21600


@pytest.fixture
def multi_config(tmp_path, minimal_trace):
    config = read_config(Path("examples/scaling/multinode.json"))
    # Exercise the unresolved-baseline guard independently of the chosen project defaults.
    config["serve"]["VLLM_MAX_NUM_BATCHED_TOKENS"] = None
    for workload in config["workloads"].values():
        workload["task_cc_per_replica"] = None
    config["repo"] = str(tmp_path)
    for name in (".venv/bin/python", "examples/minisweagent_swebench/scripts/vllm_qwen3_32b.sh", ".tools/openclaw/bin/openclaw"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    trace = tmp_path / "trace.json"
    write_json(trace, minimal_trace)
    pool = tmp_path / "traces.txt"
    pool.write_text("trace.json\n")
    for name, workload in config["workloads"].items():
        profile = json.loads(Path(workload["profile"]).read_text())
        path = tmp_path / f"{name}.json"
        write_json(path, profile)
        workload.update(pool=str(pool), profile=str(path), expected_trace_count=1)
    return config


def test_matrix_baselines_node_layouts_and_budget(multi_config):
    points = matrix(multi_config)
    assert len(points) == len({p["id"] for p in points}) == 26
    assert len([p for p in points if p["inference_replicas"] == 1]) == 2
    assert len([p for p in points if p["proxy"]]) == 8
    assert sum(p["physical_nodes"]*(p["duration_seconds"]+p["warmup_seconds"])/3600 for p in points) == pytest.approx(405.333333)
    fixed = next(p for p in points if p["layout"] == "fixed_frontend" and p["inference_replicas"] == 4)
    mapping = node_layout(fixed, ["i0", "i0", "f0", "i1", "i2", "i3"], "f0.alcf")
    assert mapping["frontends"] == ["f0"]
    assert len(mapping["replicas"]) == 4
    assert {r["frontend_host"] for r in mapping["replicas"]} == {"f0"}
    assert [r["index"] for r in mapping["replicas"]] == list(range(4))
    with pytest.raises(ValueError, match="allocation"):
        node_layout(fixed, ["f0", "i0"], "f0")


@pytest.mark.parametrize("frontends", [1, 2, 4, 8])
def test_frontend_ratios_keep_eight_unique_backends_and_worker_ids(frontends):
    point = {"physical_nodes": frontends + 8, "frontend_nodes": frontends}
    hosts = [f"node{i}" for i in range(frontends + 8)]
    mapping = node_layout(point, [h + ".alcf" for h in hosts for _ in range(64)], "node3.alcf")
    assert mapping["frontends"][0] == "node3"
    assert len(mapping["frontends"]) == frontends
    assert len(set(mapping["inference"])) == 8
    assert not set(mapping["frontends"]) & set(mapping["inference"])
    assert [r["id"] for r in mapping["replicas"]] == [f"r{i:02d}" for i in range(8)]
    assert [r["index"] for r in mapping["replicas"]] == list(range(8))
    for host in mapping["frontends"]:
        assert sum(r["frontend_host"] == host for r in mapping["replicas"]) == 8 // frontends


def test_render_requires_baseline_and_freezes_inputs(multi_config, tmp_path):
    status = check(multi_config)
    assert not status["ready_to_render"] and len(status["pending_baseline_fields"]) == 3
    with pytest.raises(ValueError, match="baseline"):
        render(multi_config, tmp_path / "blocked")
    assert not (tmp_path / "blocked").exists()
    smoke = render(multi_config, tmp_path / "smoke", smoke=True)
    assert smoke["pbs_scripts"] == 2 and smoke["submitted_jobs"] == 0
    assert {p["physical_nodes"] for p in matrix(multi_config, smoke=True)} == {4}
    assert not list((tmp_path / "smoke/jobs").glob("*-n1.pbs"))
    frozen = json.loads(next((tmp_path / "smoke/configs").glob("*.json")).read_text())
    assert frozen["serve"]["VLLM_MAX_NUM_BATCHED_TOKENS"] == "8192"
    assert frozen["point"]["task_cc_per_replica"] == 2
    assert all(Path(p).is_absolute() for p in frozen["trace_paths"])
    multi_config["serve"]["VLLM_MAX_NUM_BATCHED_TOKENS"] = "8192"
    for w in multi_config["workloads"].values():
        w["task_cc_per_replica"] = 4
    result = render(multi_config, tmp_path / "formal")
    assert result["points"] == 26 and result["pbs_scripts"] == 26
    assert result["queue_counts"] == {"preemptable": 20, "prod": 6}
    assert not result["queue_deferred_points"]
    bundle = json.loads((tmp_path / "formal/matrix.json").read_text())
    assert len(bundle["planned_points"]) == 26 and not bundle["deferred_points"]
    assert (tmp_path / "formal/jobs/openclaw-balanced-n8.pbs").exists()
    assert (tmp_path / "formal/jobs/openclaw-fixed_frontend-n8.pbs").exists()
    for directory in ("formal", "smoke"):
        for path in (tmp_path / directory / "jobs").glob("*.pbs"):
            subprocess.run(["bash", "-n", str(path)], check=True)
            text = path.read_text()
            assert "qsub" not in text and "select=${" not in text
            frozen = json.loads((tmp_path / directory / "configs" / f"{path.stem}.json").read_text())
            nodes = frozen["point"]["physical_nodes"]
            queue = "preemptable" if nodes <= 10 else "prod"
            walltime = "03:00:00" if nodes <= 24 else "06:00:00"
            assert f"#PBS -q {queue}" in text
            assert f"#PBS -l walltime={walltime}" in text
            assert frozen["pbs"]["queue"] == queue and frozen["pbs"]["walltime"] == walltime
            assert frozen["drain_budget_seconds"] == (1800 if nodes <= 24 else 7200)
            if directory == "formal":
                workload = multi_config["workloads"][frozen["point"]["workload"]]
                assert frozen["point"]["warmup_seconds"] == workload["warmup_seconds"]
                assert frozen["point"]["duration_seconds"] == workload["duration_seconds"]
    text = (tmp_path / "formal/jobs/openclaw-fixed_frontend-n8.pbs").read_text()
    assert "select=9:system=polaris" in text
    assert "#PBS -l walltime=03:00:00" in text


def test_node_limit_does_not_change_planned_matrix(multi_config, tmp_path):
    multi_config["serve"]["VLLM_MAX_NUM_BATCHED_TOKENS"] = "2048"
    for w in multi_config["workloads"].values():
        w["task_cc_per_replica"] = 2
    multi_config["pbs"].update(queue="test_queue", max_nodes=32)
    multi_config["pbs"].pop("overflow_routes")
    result = render(multi_config, tmp_path / "full")
    assert result["pbs_scripts"] == 26 and not result["queue_deferred_points"]
    text = (tmp_path / "full/jobs/openclaw-fixed_frontend-n16.pbs").read_text()
    assert "select=17:system=polaris" in text


def test_fixed_frontend_group_renders_only_requested_jobs(multi_config, tmp_path):
    multi_config["serve"]["VLLM_MAX_NUM_BATCHED_TOKENS"] = "8192"
    for name, item in multi_config["workloads"].items():
        item["task_cc_per_replica"] = 16 if name == "openclaw" else 32
    points = matrix(multi_config, group="fixed-frontend")
    assert len(points) == 8
    assert {p["physical_nodes"] for p in points} == {3, 5, 9, 17}
    assert all(p["frontend_nodes"] == 1 and not p["proxy"] for p in points)
    bundle = tmp_path / "fixed"
    result = render(multi_config, bundle, group="fixed-frontend")
    assert result["pbs_scripts"] == result["points"] == 8
    assert result["queue_counts"] == {"preemptable": 6, "prod": 2}
    assert not list((bundle / "jobs").glob("*balanced*"))
    assert not list((bundle / "jobs").glob("*mechanism*"))
    assert len([l for l in (bundle / "submit-commands.txt").read_text().splitlines() if l.startswith("qsub ")]) == 8
    with pytest.raises(ValueError, match="smoke"):
        matrix(multi_config, smoke=True, group="fixed-frontend")


def test_queue_routing_boundaries_and_legacy_single_queue(multi_config):
    for count, queue, hours in [(10, "preemptable", "03"), (11, "prod", "03"),
                                (24, "prod", "03"), (25, "prod", "06"), (32, "prod", "06")]:
        settings = pbs_for_point(multi_config, {"physical_nodes": count})
        assert settings["queue"] == queue and settings["walltime"] == hours+":00:00"
    assert pbs_for_point(multi_config, {"physical_nodes": 497}) is None
    multi_config["pbs"].pop("overflow_routes")
    status = check(multi_config)
    assert status["renderable_points"] == 20 and len(status["queue_deferred_points"]) == 6


def test_short_queue_rejects_impossible_window_and_drain_budget(multi_config):
    multi_config["pbs"]["overflow_routes"][0]["drain_budget_seconds"] = 7200
    with pytest.raises(ValueError, match="cannot fit"):
        check(multi_config)


def test_coordinated_workers_share_window_and_keep_namespaces(tmp_path, minimal_trace):
    trace = tmp_path / "trace.json"
    write_json(trace, minimal_trace)
    control = tmp_path / "control"
    control.mkdir()
    async def run():
        workers = [asyncio.create_task(benchmark([trace], profile=None, output=tmp_path / f"r{i}",
            dry_run=True, sample_hardware=False, warmup_seconds=.05, duration_seconds=.25,
            control_dir=control, child_env={"TMPDIR": str(tmp_path)}, task_namespace=f"job/r{i}")) for i in range(2)]
        try:
            await wait_files([tmp_path / f"r{i}/ready.json" for i in range(2)], workers, 5)
            t0 = time.time()+.2
            write_json(control / "start.json", {"start_unix": t0})
            return t0, await asyncio.gather(*workers)
        finally:
            for task in workers:
                task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
    t0, results = asyncio.run(run())
    for i, result in enumerate(results):
        assert result["measurement"]["started_unix"] == t0+.05
        assert result["measurement"]["ended_unix"] == t0+.05+.25
        assert result["measurement"]["valid"]
        assert not (tmp_path / f"r{i}/metrics.jsonl").exists()
        admissions = [json.loads(line) for line in (tmp_path / f"r{i}/admissions.jsonl").read_text().splitlines()]
        assert all(r["task_instance_id"].startswith(f"job/r{i}/") for r in admissions)


def test_expired_measurement_window_fails_before_admission(tmp_path, minimal_trace):
    trace = tmp_path / "trace.json"
    write_json(trace, minimal_trace)
    write_json(tmp_path / "start.json", {"start_unix": time.time()-1})
    with pytest.raises(RuntimeError, match="measurement window ended"):
        asyncio.run(benchmark([trace], profile=None, output=tmp_path / "worker", dry_run=True,
            sample_hardware=False, duration_seconds=1, control_dir=tmp_path))
    assert not list((tmp_path / "worker/tasks").iterdir())


def test_late_barrier_keeps_common_window(tmp_path, minimal_trace):
    trace = tmp_path / "trace.json"
    write_json(trace, minimal_trace)
    start = time.time()-1
    write_json(tmp_path / "start.json", {"start_unix": start})
    result = asyncio.run(benchmark([trace], profile=None, output=tmp_path / "worker", dry_run=True,
        sample_hardware=False, warmup_seconds=2, duration_seconds=.2, control_dir=tmp_path))
    assert result["measurement"]["valid"]
    assert result["measurement"]["started_unix"] == start+2
    assert result["measurement"]["ended_unix"] == start+2+.2
    manifest = json.loads((tmp_path / "worker/manifest.json").read_text())
    assert manifest["actual_started_unix"]-manifest["started_unix"] > .5


def test_parallel_startup_failure_cleans_ready_and_pending_peers():
    events = []
    @asynccontextmanager
    async def service(name):
        try:
            if name == "slow":
                await asyncio.sleep(30)
            if name == "bad":
                await asyncio.sleep(.05)
                raise RuntimeError("startup failed")
            events.append(name+" ready")
            yield name
        finally:
            events.append(name+" closed")
    async def run():
        async with AsyncExitStack() as stack:
            await enter_services(stack, [service(n) for n in ("good", "slow", "bad")])
    with pytest.raises(ExceptionGroup):
        asyncio.run(run())
    assert set(events) == {"good ready", "good closed", "slow closed", "bad closed"}


def test_backend_crash_cancels_blocked_replay():
    closed = []
    async def blocked():
        try:
            await asyncio.sleep(30)
        finally:
            closed.append(True)
    async def run():
        process = SimpleNamespace(returncode=None)
        task = asyncio.create_task(guarded(blocked(), [process]))
        await asyncio.sleep(.02)
        process.returncode = 1
        await task
    with pytest.raises(RuntimeError, match="exited"):
        asyncio.run(run())
    assert closed == [True]


@pytest.mark.parametrize("use_proxy", [False, True, "official"])
@pytest.mark.parametrize("workload", ["openclaw", "minisweagent"])
def test_fixed_frontend_samples_once_and_keeps_replica_environments_isolated(tmp_path, monkeypatch, use_proxy, workload):
    from agenttrace.experiments.multi.frontend import work
    calls, samplers = [], []
    (tmp_path / "frontends/front").mkdir(parents=True)
    (tmp_path / "control").mkdir()
    config = {"point": {"workload": workload, "proxy": use_proxy, "routing": "sticky", "prefix_reuse": "normal",
        "seed": 42, "task_cc_per_replica": 3, "warmup_seconds": 1, "duration_seconds": 2},
        "node_mapping": {"frontends": ["front"], "replicas": [
            {"id": f"r{i:02d}", "index": i, "frontend_host": "front", "inference_host": f"backend{i}"} for i in range(2)]},
        "serve": {"VLLM_PORT": "18000"}, "trace_paths": ["trace.json"], "preparation_timeout_seconds": 5,
        "replay_profile": {"llm_executor": {"config": {}}}}
    if use_proxy == "official":
        config["point"].update(router_impl="vllm-router", routing="cache_aware")
        config["router"] = {}
    if workload == "minisweagent":
        trace = tmp_path / "trace.json"
        write_json(trace, {"context": {"container_image": "docker://test:latest"}})
        config["trace_paths"] = [str(trace)]
        config["replay_profile"]["tool_executor"] = {"config": {
            "image_cache_dir": str(tmp_path / "shared"), "max_parallel_sandbox_builds": 4}}
        def stage(shared, local, sources):
            assert shared == tmp_path / "shared"
            assert sources == ["docker://test:latest"]
            calls.append("stage")
            return {"local_cache": str(local)}
        monkeypatch.setattr("agenttrace.miniswe_images.stage_images", stage)
    async def precache(*args):
        calls.append("precache")
    async def monitor(path, *, ready, stop, **kwargs):
        samplers.append(path)
        ready.set()
        await stop.wait()
        calls.append("sampler stopped")
    async def replay(paths, **kwargs):
        calls.append(kwargs)
        output = kwargs["output"]
        output.mkdir(parents=True)
        write_json(output / "ready.json", {})
        await wait_files([tmp_path / "control/start.json"], timeout=5)
    @asynccontextmanager
    async def proxy(self, *args):
        calls.append("proxy started")
        self.failure = asyncio.create_task(asyncio.Event().wait())
        try:
            yield "http://127.0.0.1:18010/v1"
        finally:
            self.failure.cancel()
            await asyncio.gather(self.failure, return_exceptions=True)
            calls.append("proxy stopped")
    monkeypatch.setattr("agenttrace.experiments.multi.frontend.precache", precache)
    monkeypatch.setattr("agenttrace.experiments.multi.frontend.monitor", monitor)
    monkeypatch.setattr("agenttrace.experiments.multi.frontend.benchmark", replay)
    monkeypatch.setattr("agenttrace.experiments.multi.frontend.Router.serve", proxy)
    monkeypatch.setattr("agenttrace.experiments.multi.official_router.OfficialRouter.serve", proxy)
    async def run():
        task = asyncio.create_task(work(tmp_path, config, "front"))
        try:
            await wait_files([tmp_path / "frontends/front/ready.json"], [task], 5)
            write_json(tmp_path / "control/start.json", {})
            await wait_files([tmp_path / "frontends/front/complete.json"], [task], 5)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    asyncio.run(run())
    assert len(samplers) == 1 and calls.count("precache") == 1
    assert "sampler stopped" in calls
    workers = [c for c in calls if isinstance(c, dict)]
    assert [c["seed"] for c in workers] == [42, 43]
    assert all(c["concurrency"] == 3 and c["sample_hardware"] is False for c in workers)
    assert all(c["continue_on_http_error"] for c in workers)
    assert [c["child_env"]["AGENTTRACE_HOME_REPLICA"] for c in workers] == ["r00", "r01"]
    urls = {c["child_env"]["AGENTTRACE_LLM_BASE_URL"] for c in workers}
    assert len(urls) == (1 if use_proxy else 2)
    assert calls.count("proxy started") == calls.count("proxy stopped") == int(bool(use_proxy))
    profile = json.loads((tmp_path / "frontends/front/profile.json").read_text())
    assert profile["llm_executor"]["config"].get("routing_envelope", False) == (use_proxy == "official")
    assert calls.count("stage") == int(workload == "minisweagent")
    if workload == "minisweagent":
        local_profile = json.loads((tmp_path / "frontends/front/profile.json").read_text())
        local_cache = local_profile["tool_executor"]["config"]["image_cache_dir"]
        assert local_cache.startswith("/local/scratch/")
        assert local_profile["tool_executor"]["config"]["sandbox_build_lock_dir"].startswith("/local/scratch/")
        assert "sandbox_build_lock_dir" not in config["replay_profile"]["tool_executor"]["config"]
        assert config["replay_profile"]["tool_executor"]["config"]["image_cache_dir"] == str(tmp_path / "shared")
        assert calls.index("stage") < calls.index("precache")


@pytest.mark.parametrize("clock_state", ["ok", "imprecise", "timeout"])
def test_pbs_coordinator_lifecycle_with_mocked_services(multi_config, tmp_path, monkeypatch, clock_state):
    from agenttrace.experiments.multi.runtime import run
    point = matrix(multi_config)[0]
    point.update(task_cc_per_replica=1, warmup_seconds=0, duration_seconds=.1)
    multi_config.update(point=point, trace_paths=[str(tmp_path / "trace.json")], replay_profile={})
    multi_config["serve"]["VLLM_MAX_NUM_BATCHED_TOKENS"] = "2048"
    bundle = tmp_path / "bundle.json"
    write_json(bundle, multi_config)
    nodefile = tmp_path / "nodes"
    nodefile.write_text("front\nbackend\n")
    monkeypatch.setenv("PBS_JOBID", "123.fake")
    monkeypatch.setenv("PBS_NODEFILE", str(nodefile))
    monkeypatch.setattr("agenttrace.experiments.multi.runtime.socket.gethostname", lambda: "front")
    monkeypatch.setattr("agenttrace.experiments.multi.runtime.allocation_remaining", lambda job: 21600)
    probes = []
    async def clocks(*args):
        probes.append(clock_state)
        if clock_state == "timeout":
            raise TimeoutError("test clock timeout")
        return {"valid": clock_state == "ok"}
    monkeypatch.setattr("agenttrace.experiments.multi.runtime.check_clocks", clocks)
    events = []
    @asynccontextmanager
    async def backend(repo, path, config_path, host):
        events.append("backend ready")
        try:
            yield SimpleNamespace(returncode=None)
        finally:
            events.append("backend stopped")
    @asynccontextmanager
    async def frontend(repo, root, config_path, host, timeout):
        async def complete():
            await wait_files([root / "control/start.json"], timeout=5)
            window = json.loads((root / "control/start.json").read_text())
            assert window["start_unix"] > time.time()+8
            write_json(root / "frontends" / host / "complete.json", {})
        task = asyncio.create_task(complete())
        events.append("frontend ready")
        try:
            yield SimpleNamespace(returncode=None)
        finally:
            await task
            events.append("frontend stopped")
    monkeypatch.setattr("agenttrace.experiments.multi.runtime.inference_service", backend)
    monkeypatch.setattr("agenttrace.experiments.multi.runtime.frontend_service", frontend)
    def summary(root):
        assert events[-2:] == ["frontend stopped", "backend stopped"]
        return {"valid": True, "steady": False}
    monkeypatch.setattr("agenttrace.experiments.multi.report.summarize_run", summary)
    result = asyncio.run(run(bundle))
    assert result["valid"]
    root = Path(result["root"])
    assert json.loads((root / "status.json").read_text())["status"] == "completed"
    assert json.loads((root / "config.json").read_text())["sample_vllm_metrics"]
    assert len(probes) == 3
    timing = json.loads((root / "clocks.json").read_text())
    assert timing["policy"] == "diagnostic_only"
    assert timing["valid"] is (clock_state == "ok")
