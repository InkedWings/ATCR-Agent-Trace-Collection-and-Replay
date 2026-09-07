import asyncio
import json

import pytest

from agenttrace.replay.benchmark import benchmark
from agenttrace.replay.window import summarize_window, timeline


def test_timeline_union_and_clipping():
    r = timeline([(0, 4), (3, 6), (9, 12)], 2, 10)
    assert r["peak"] == 2 and r["time_weighted_mean"] == 6 / 8


def test_steady_refills_stops_admission_and_drains(tmp_path, minimal_trace):
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps(minimal_trace))
    result = asyncio.run(benchmark([trace], profile=None, output=tmp_path / "run",
        concurrency=2, warmup_seconds=.2, duration_seconds=.5, dry_run=True, sample_interval=.1))
    assert result["status"] == "completed"
    window = result["measurement"]
    assert window["valid"] and window["incomplete_admission_cohort"] == 0
    assert len(result["tasks"]) >= 2
    assert all(t["started_unix"] < window["ended_unix"] for t in result["tasks"])
    assert max(t["ended_unix"] for t in result["tasks"]) >= window["ended_unix"]
    assert window["concurrency"]["tasks"]["peak"] == 2
    assert len((tmp_path / "run/admissions.jsonl").read_text().splitlines()) == len(result["tasks"])


def test_duration_rejects_repeat(tmp_path, minimal_trace):
    with pytest.raises(ValueError, match="mutually exclusive"):
        asyncio.run(benchmark([tmp_path], profile=None, output=tmp_path / "r", repeat=1, duration_seconds=1))


def test_same_basename_across_points_has_private_local_scratch(tmp_path, monkeypatch, minimal_trace):
    monkeypatch.setenv("TMPDIR", str(tmp_path / "scratch"))
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(minimal_trace))
    roots = []
    for point in ("cc1", "cc2"):
        output = tmp_path / point / "benchmark"
        r = asyncio.run(benchmark([path], profile=None, output=output, dry_run=True))
        assert r["status"] == "completed"
        roots.append(json.loads((output / "manifest.json").read_text())["local_tmp_root"])
    assert roots[0] != roots[1]


def test_latency_cohorts_include_drain_but_throughput_counts_completions(tmp_path):
    tasks = []
    for i, (start, end) in enumerate([(8, 12), (12, 25), (19, 30)]):
        root = tmp_path / "tasks" / str(i)
        root.mkdir(parents=True)
        (root / "report.json").write_text(json.dumps({"setup_seconds": 1, "replay_makespan_seconds": end-start-1}))
        (root / "events.jsonl").write_text("")
        tasks.append({"task_id": str(i), "trace_path": str(i), "status": "completed",
            "started_unix": start, "ended_unix": end, "lifecycle_seconds": end-start,
            "report": f"tasks/{i}/report.json"})
    r = summarize_window(tmp_path, tasks, 10, 20)
    assert r["completed_in_window"] == 1 and r["admitted_in_window"] == 2
    assert r["task_lifecycle_seconds"]["count"] == 2
    assert r["task_lifecycle_seconds"]["mean"] == 12
