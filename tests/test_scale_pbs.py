import argparse
import asyncio
import copy
import json
from pathlib import Path

import pytest

from agenttrace.experiments import pbs


@pytest.mark.parametrize("cc", [8, 16, 32])
def test_pbs_changes_only_hosts_and_task_concurrency(cc):
    base = json.loads((Path(__file__).parents[1] / "examples/scaling/single-backend.json").read_text())
    original = copy.deepcopy(base)
    config = pbs.job_config(base, "replay-new", "inference-new", cc)
    assert config["concurrency"] == [cc]
    assert config["replay_node"] == "replay-new"
    assert config["inference_node"] == "inference-new"
    for key in base.keys() - {"concurrency", "replay_node", "inference_node"}:
        assert config[key] == base[key]
    assert config["workloads"] == ["openclaw", "minisweagent"]
    assert base == original


def test_pbs_precaches_serially_before_real_sweep(tmp_path, monkeypatch):
    config = tmp_path / "base.json"
    config.write_text(json.dumps({"workloads": ["openclaw", "minisweagent"],
                                 "profiles": {"openclaw": "oc.json", "minisweagent": "ms.json"}}))
    events = []
    async def precache(pool, output, profile):
        events.append(("cache", pool.parent.name, profile.name))
        await asyncio.sleep(0)
        events.append(("cached", pool.parent.name))
    async def sweep(args):
        captured = json.loads(args.config.read_text())
        assert captured["concurrency"] == [32]
        assert captured["replay_node"] == "replay"
        assert captured["inference_node"] == "inference"
        assert args.output == tmp_path / "job/experiment"
        assert not args.output.exists()
        assert args.job_id == "123.pbs" and not args.smoke
        events.append(("sweep",))
    monkeypatch.setattr(pbs.socket, "gethostname", lambda: "replay.local")
    monkeypatch.setattr(pbs, "precache", precache)
    monkeypatch.setattr(pbs, "run", sweep)
    asyncio.run(pbs.run_job(argparse.Namespace(config=config, output=tmp_path / "job",
        inference_node="inference", concurrency=32, workloads=None,
        pools=tmp_path / "pools", job_id="123.pbs")))
    assert events == [("cache", "openclaw", "oc.json"), ("cached", "openclaw"),
                      ("cache", "minisweagent", "ms.json"), ("cached", "minisweagent"), ("sweep",)]


def test_pbs_rejects_shared_node_and_zero_cc():
    with pytest.raises(ValueError, match="separate nodes"):
        pbs.job_config({}, "node", "node", 8)
    with pytest.raises(ValueError, match="positive"):
        pbs.job_config({}, "a", "b", 0)


def test_miniswe_only_rerun_does_not_precache_or_run_openclaw(tmp_path, monkeypatch):
    config = tmp_path / "base.json"
    config.write_text(json.dumps({"workloads": ["openclaw", "minisweagent"],
                                 "profiles": {"openclaw": "oc.json", "minisweagent": "ms.json"}}))
    cached = []
    async def cache(pool, output, profile):
        cached.append(pool.parent.name)
    async def sweep(args):
        recorded = json.loads(args.config.read_text())
        assert recorded["workloads"] == ["minisweagent"]
        assert recorded["concurrency"] == [8]
    monkeypatch.setattr(pbs.socket, "gethostname", lambda: "replay")
    monkeypatch.setattr(pbs, "precache", cache)
    monkeypatch.setattr(pbs, "run", sweep)
    asyncio.run(pbs.run_job(argparse.Namespace(config=config, output=tmp_path / "job",
        inference_node="inference", concurrency=8, workloads=["minisweagent"],
        pools=tmp_path / "pools", job_id="123.pbs")))
    assert cached == ["minisweagent"]
