import json

import pytest

from agenttrace.experiments.multi.config import write_json
from agenttrace.experiments.multi.plots import export
from agenttrace.experiments.multi.report import execution_breakdown, summarize_run


def jsonlines(path, rows):
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))


@pytest.fixture
def run_directory(tmp_path):
    root = tmp_path / "run"
    (root / "control").mkdir(parents=True)
    point = {"id": "openclaw-balanced-n2", "workload": "openclaw", "inference_replicas": 2,
             "frontend_nodes": 2, "physical_nodes": 4, "task_cc_per_replica": 1,
             "layout": "balanced", "routing": "sticky", "prefix_reuse": "normal", "proxy": False,
             "smoke": False, "seed": 42, "warmup_seconds": 10, "duration_seconds": 10}
    replicas = [{"id": f"r{i:02d}", "index": i, "inference_host": f"i{i}", "frontend_host": f"f{i}"} for i in range(2)]
    write_json(root / "config.json", {"point": point, "serve": {}, "replay_profile": {}, "trace_paths": ["trace.json"],
        "node_mapping": {"replicas": replicas, "inference": ["i0", "i1"], "frontends": ["f0", "f1"]}})
    write_json(root / "control/start.json", {"start_unix": 0, "measurement_start_unix": 10, "measurement_end_unix": 20})
    write_json(root / "clocks.json", {"valid": True})
    for i, replica in enumerate(replicas):
        rid = replica["id"]
        worker = root / "workers" / rid
        task_root = worker / "tasks/00001"
        task_root.mkdir(parents=True)
        began, ended = 11+i, (18 if i == 0 else 30)
        task = {"task_id": "00001", "task_instance_id": f"run/{rid}/00001", "trace_path": "trace.json",
                "started_unix": began, "ended_unix": ended, "status": "completed", "lifecycle_seconds": ended-began,
                "report": "tasks/00001/report.json"}
        call = {"event": "node_completed", "type": "llm", "node_id": "llm1", "started_unix": began,
                "timestamp_unix": began+3, "elapsed_seconds": 3, "started_seconds": 0,
                "backend_replica_id": rid, "actual_output_tokens": 4, "target_output_tokens": 4,
                "ttft_seconds": 1, "tpot_estimate_seconds": 2/3}
        write_json(worker / "summary.json", {"status": "completed", "tasks": [task],
                  "measurement": {"valid": True, "started_unix": 10, "ended_unix": 20}})
        write_json(worker / "manifest.json", {"actual_started_unix": .01, "seed": 42+i, "concurrency": 1,
                                               "trace_pool": ["trace.json"]})
        write_json(task_root / "report.json", {"setup_seconds": 1, "nodes": [call]})
        jsonlines(task_root / "events.jsonl", [call])
        backend = root / "backends" / rid
        backend.mkdir(parents=True)
        prompt, cached = (100, 100) if i == 0 else (900, 0)
        jsonlines(backend / "backend.jsonl", [
            {"event": "tokens", "hostname": f"i{i}", "request_id": "req1", "timestamp_unix": began+1,
             "output_tokens": 4, "decode_tokens": 3},
            {"event": "request_finished", "hostname": f"i{i}", "request_id": "req1", "finish_reason": "length",
             "scheduled_unix": began, "first_token_unix": began+1, "finished_unix": began+3,
             "queued_monotonic": began-.5, "scheduled_monotonic": began, "first_token_monotonic": began+1,
             "last_token_monotonic": began+3, "output_tokens": 4, "prompt_tokens": prompt, "cached_prompt_tokens": cached}])
        samples = []
        for ts in (0, 9, 10, 15, 19, 20, 30, 35):
            samples.append({"timestamp_unix": ts, "hostname": f"i{i}",
                "hardware": {"gpu_status": "ok", "cpu_busy_percent": 25, "gpus": [
                    {"index": str(g), "power_watts": 100, "gpu_busy_percent": 80} for g in range(4)]},
                "vllm_prometheus": f"vllm:generation_tokens_total {4 if ts >= 15 else 0}\n"
                    f"vllm:prompt_tokens_total {prompt if ts >= 15 else 0}\n"
                    "vllm:num_requests_waiting 0\nvllm:kv_cache_usage_perc 0.25\n"})
        jsonlines(backend / "inference-metrics.jsonl", samples)
        front = root / "frontends" / f"f{i}"
        front.mkdir(parents=True)
        jsonlines(front / "metrics.jsonl", [{k: v for k, v in s.items() if k != "vllm_prometheus"} for s in samples])
    return root


def test_global_accounting_uses_common_denominator_weighted_cache_and_pooled_latency(run_directory):
    result = summarize_run(run_directory)
    assert result["valid"], result["validation_errors"]
    assert result["output_tokens_per_second"] == .8
    assert result["actual_prefix_reuse"] == .1  # not mean([1, 0])
    assert result["tasks_per_second"] == .1
    assert result["admitted_in_window"] == 2
    assert result["task_lifecycle_seconds"]["p95"] == pytest.approx(17.45)
    assert result["task_lifecycle_seconds"]["count"] == 2  # includes task drained at t=30
    assert len(result["resources"]) == 4
    assert result["inference_gpu_energy_joules"] == 8000
    assert result["resources"][0]["lustre"]["status"] == "unavailable"
    assert not result["steady"]  # all output tokens are in first half


def test_backend_mismatch_invalidates_whole_job(run_directory):
    path = run_directory / "backends/r01/backend.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["output_tokens"] = 3
    jsonlines(path, rows)
    result = summarize_run(run_directory)
    assert not result["valid"]
    assert any("token/request mismatch" in e for e in result["validation_errors"])


def test_official_router_uses_actual_backend_ledger_and_detects_misrouting(run_directory):
    path = run_directory / "config.json"
    config = json.loads(path.read_text())
    config["point"].update(proxy=True, routing="cache_aware", router_impl="vllm-router")
    write_json(path, config)
    for i in range(2):
        rid = f"r{i:02d}"
        jsonlines(run_directory / "backends" / rid / "routing.jsonl", [{
            "task_instance_id": f"run/{rid}/00001", "node_id": "llm1",
            "destination": rid, "status": "completed"}])
    assert summarize_run(run_directory)["valid"]
    # The client must agree with the actual serving backend, not its home cohort.
    ledger = run_directory / "backends/r00/routing.jsonl"
    row = json.loads(ledger.read_text())
    row["destination"] = "r01"
    jsonlines(ledger, [row])
    result = summarize_run(run_directory)
    assert not result["valid"] and "proxy/client routing ledgers differ" in result["validation_errors"]


@pytest.mark.parametrize("relative", ["frontends/f0/metrics.jsonl", "backends/r00/inference-metrics.jsonl"])
def test_resource_sampling_gap_does_not_invalidate_complete_counts(run_directory, relative):
    path = run_directory / relative
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    jsonlines(path, [row for row in rows if row["timestamp_unix"] != 15])
    result = summarize_run(run_directory)
    assert result["valid"], result["validation_errors"]
    assert result["output_tokens_per_second"] == .8
    assert result["tasks_per_second"] == .1
    assert any("sampling has gaps" in warning for warning in result["diagnostic_warnings"])
    assert sum(not r["window_coverage_ok"] for r in result["resources"]) == 1
    if relative.startswith("backends"):
        assert result["inference_gpu_energy_joules"] is None


def test_incomplete_measurement_window_still_invalidates_job(run_directory):
    path = run_directory / "workers/r00/summary.json"
    worker = json.loads(path.read_text())
    worker["measurement"]["ended_unix"] = 19
    write_json(path, worker)
    result = summarize_run(run_directory)
    assert not result["valid"]
    assert result["validation_errors"]


def test_failed_task_preserves_full_window_but_is_not_a_successful_scaling_point(run_directory):
    path = run_directory / "workers/r00/summary.json"
    worker = json.loads(path.read_text())
    worker["status"] = "completed_with_errors"
    worker["measurement"].update(valid=False, window_complete=True)
    task = dict(worker["tasks"][0], task_id="00002", status="failed", started_unix=16, ended_unix=18,
                lifecycle_seconds=2, task_instance_id="run/r00/00002",
                failure={"kind": "http_transport", "error_type": "ReadError"})
    worker["tasks"].append(task)
    write_json(path, worker)
    events = run_directory / "workers/r00/tasks/00002/events.jsonl"
    events.parent.mkdir()
    jsonlines(events, [{"event": "task_failed", "timestamp_unix": 18}])
    result = summarize_run(run_directory)
    assert result["collection_complete"] and not result["valid"]
    assert result["failed_tasks"] == result["failed_in_window"] == 1
    assert result["completed_in_window"] == 1
    assert result["task_failure_fraction"] == .5
    assert result["tasks_per_second"] == .1 and result["output_tokens_per_second"] == .8
    assert not any("worker window" in e for e in result["validation_errors"])
    assert "r00/00002: failed" in result["validation_errors"]
    assert "ReadError" in (run_directory / "failed-tasks.csv").read_text()


@pytest.mark.parametrize("clock_state", ["imprecise", "missing", "malformed"])
def test_clock_diagnostics_and_start_lag_do_not_invalidate_data(run_directory, clock_state):
    clock_path = run_directory / "clocks.json"
    if clock_state == "missing":
        clock_path.unlink()
    elif clock_state == "malformed":
        clock_path.write_text("incomplete JSON")
    else:
        write_json(clock_path, {"valid": False})
    manifest_path = run_directory / "workers/r00/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["actual_started_unix"] = 2
    write_json(manifest_path, manifest)
    result = summarize_run(run_directory)
    assert result["valid"], result["validation_errors"]
    assert result["output_tokens_per_second"] == .8
    assert result["worker_start_lag_seconds"]["r00"] == 2
    assert any("clock probe" in warning for warning in result["diagnostic_warnings"])
    assert any("worker start" in warning for warning in result["diagnostic_warnings"])


def test_prometheus_reset_invalidates_whole_job(run_directory):
    path = run_directory / "backends/r00/inference-metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[4]["vllm_prometheus"] = rows[4]["vllm_prometheus"].replace("generation_tokens_total 4", "generation_tokens_total 2")
    jsonlines(path, rows)
    result = summarize_run(run_directory)
    assert not result["valid"]
    assert any("counter reset" in e for e in result["validation_errors"])


def test_time_breakdown_counts_overlapping_calls_once():
    result = execution_breakdown({"lifecycle_seconds": 12}, {"setup_seconds": 2, "nodes": [
        {"type": "llm", "started_seconds": 0, "elapsed_seconds": 5},
        {"type": "llm", "started_seconds": 1, "elapsed_seconds": 2},
        {"type": "tool", "started_seconds": 3, "elapsed_seconds": 4}]})
    assert result == {"setup": 2, "llm_only": 3, "tool_only": 2, "llm_tool_overlap": 2, "other_and_cleanup": 3}
    assert sum(result.values()) == 12


def test_export_tables_plots_and_exclude_unfinished_jobs(run_directory, tmp_path):
    summarize_run(run_directory)
    excluded = export([run_directory], tmp_path / "incomplete", plots=False)
    assert excluded["excluded_runs"] == 1
    write_json(run_directory / "status.json", {"status": "completed"})
    result = export([run_directory], tmp_path / "analysis")
    assert result["valid_runs"] == 1
    assert (tmp_path / "analysis/summary.csv").is_file()
    assert len(list((tmp_path / "analysis").glob("*.png"))) == 4
    assert len(list((tmp_path / "analysis").glob("*.pdf"))) == 4
