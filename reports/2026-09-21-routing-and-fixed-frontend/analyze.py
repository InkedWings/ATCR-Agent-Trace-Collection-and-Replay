"""Read-only analysis of completed Sep 21 jobs and measured historical references.

Run from any directory with the repository .venv Python. Writes only beside this
script. Failed mini-SWE RR stays failed: only its independently checked measurement
window is recovered, never the censored task/LLM latency admission cohort.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
BASE = REPO / "runs/scaling/multinode"


def read(path):
    return json.loads(path.read_text())


def records(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def percentile(values, p):
    a = sorted(values)
    if not a:
        return None
    x = (len(a) - 1) * p
    lo, hi = math.floor(x), math.ceil(x)
    return a[lo] + (a[hi] - a[lo]) * (x - lo)


def dist(values):
    return {"count": len(values), "mean": st.mean(values) if values else None,
            "p50": percentile(values, .5), "p95": percentile(values, .95),
            "max": max(values) if values else None}


def save_csv(name, rows):
    if not rows:
        return
    with (OUT / name).open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def source_patterns():
    new = [
        ("openclaw-routing-n4-round_robin-7642785.*", "routing_rr", "new"),
        ("openclaw-routing-n4-cache_aware-7642785.*", "routing_cache", "new"),
        ("minisweagent-routing-n4-round_robin-7642786.*", "routing_rr", "new"),
        ("minisweagent-routing-n4-cache_aware-7642786.*", "routing_cache", "new"),
        ("openclaw-fixed_frontend-n8-7642787.*", "fixed_sticky", "new"),
        ("minisweagent-fixed_frontend-n8-7642788.*", "fixed_sticky", "new"),
        ("openclaw-fixed_frontend-n2-7637010.*", "fixed_sticky", "earlier"),
        ("openclaw-fixed_frontend-n4-7637012.*", "fixed_sticky", "earlier"),
        ("minisweagent-fixed_frontend-n2-7637011.*", "fixed_sticky", "earlier"),
    ]
    for workload, jobs in {
        "openclaw": {1: "7608115", 4: "7616566", 8: "7618907", 16: "7618908"},
        "minisweagent": {1: "7608115", 2: "7619132", 8: "7619134", 16: "7619136"},
    }.items():
        new.extend((f"{workload}-balanced-n{n}-{job}.*", "balanced", "historical") for n, job in jobs.items())
    return new


def prom(row):
    names = {"vllm:generation_tokens_total": "output", "vllm:prompt_tokens_total": "prompt",
             "vllm:num_requests_waiting": "waiting", "vllm:num_requests_running": "running",
             "vllm:kv_cache_usage_perc": "kv_cache", "vllm:num_preemptions_total": "preemptions"}
    result = defaultdict(float)
    for line in row["vllm_prometheus"].splitlines():
        if not line.startswith("vllm:"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in names:
            result[names[name]] += float(line.split()[-1])
        elif name == "vllm:prompt_tokens_by_source_total" and 'source="local_cache_hit"' in line:
            result["cached"] += float(line.split()[-1])
    for key in ("output", "prompt", "cached", "waiting", "running", "kv_cache"):
        assert key in result, (key, row["timestamp_unix"])
    return dict(result)


def empty_bins(start, duration):
    return [{"start_unix": start + 60 * i, "duration_seconds": min(60, duration - 60 * i),
             "output_tokens": 0, "completed_tasks": 0, "prompt_tokens": 0,
             "cached_prompt_tokens": 0} for i in range(math.ceil(duration / 60))]


def recover_window(root, window, tasks):
    """Recover token events, first-token prompt cohort and full backend gauge samples."""
    start, end = window["measurement_start_unix"], window["measurement_end_unix"]
    bins = empty_bins(start, end - start)
    replicas, resources, audit = [], [], []
    for backend in sorted((root / "backends").iterdir()):
        print(f"Recovering {root.name}: {backend.name}", flush=True)
        totals, finished, window_ids, first_seen = Counter(), {}, set(), {}
        r = dict(replica=backend.name, output_tokens=0, prompt_tokens=0, cached_prompt_tokens=0)
        for row in records(backend / "backend.jsonl"):
            rid = row.get("request_id")
            if row.get("event") == "tokens":
                ts = row["timestamp_unix"]
                totals[rid] += row["output_tokens"]
                first_seen.setdefault(rid, ts)
                if start <= ts < end:
                    window_ids.add(rid)
                    r["output_tokens"] += row["output_tokens"]
                    bins[int((ts - start) // 60)]["output_tokens"] += row["output_tokens"]
            elif row.get("event") == "request_finished":
                assert rid not in finished, "duplicate request_finished"
                finished[rid] = row
        missing = window_ids - finished.keys()
        mismatches = [rid for rid in window_ids & finished.keys()
                      if totals[rid] != finished[rid]["output_tokens"]]
        assert not missing and not mismatches, (missing, mismatches)
        reasons = Counter()
        for rid, row in finished.items():
            if start <= row["first_token_unix"] < end:
                reasons[row["finish_reason"]] += 1
                assert row["finish_reason"] in ("stop", "length")
                assert 0 <= row["cached_prompt_tokens"] <= row["prompt_tokens"]
                for key in ("prompt_tokens", "cached_prompt_tokens"):
                    r[key] += row[key]
                    bins[int((row["first_token_unix"] - start) // 60)][key] += row[key]
        gauges, first, last, previous, resets = [], None, None, None, 0
        for row in records(backend / "inference-metrics.jsonl"):
            ts = row["timestamp_unix"]
            if ts < start:
                continue
            values = prom(row)
            if first is None:
                first = (ts, values)
            if previous is not None:
                resets += sum(values[k] < previous[k] for k in ("output", "prompt", "cached", "preemptions"))
            previous = values
            if ts >= end:
                last = (ts, values)
                break
            gauges.append({"timestamp_unix": ts, **{k: values[k] for k in ("waiting", "running", "kv_cache")}})
        assert first and last and not resets
        timestamps = [g["timestamp_unix"] for g in gauges] + [last[0]]
        gap = max(b - a for a, b in zip(timestamps, timestamps[1:]))
        assert first[0] - start <= 5 and last[0] - end <= 5 and gap <= 5
        deltas = {k: last[1][k] - first[1][k] for k in ("output", "prompt", "cached", "preemptions")}
        crosscheck = abs(deltas["output"] - r["output_tokens"]) / r["output_tokens"]
        assert crosscheck < .005, "event/counter cross-check differs by >0.5%"
        audit.append({"replica": backend.name, "window_token_requests": len(window_ids),
                      "unmatched_window_token_requests": len(missing), "token_total_mismatches": len(mismatches),
                      "first_token_request_finish_reasons": dict(reasons),
                      "prometheus_boundary_offsets_seconds": [first[0] - start, last[0] - end],
                      "metrics_max_gap_seconds": gap, "counter_resets_in_window": resets,
                      "prometheus_deltas": deltas, "output_event_counter_relative_difference": crosscheck})
        replicas.append(r)
        resources.append({"role": "inference", "replica": backend.name, "gauge_samples": gauges})
    for t in tasks:
        if t["status"] == "completed" and start <= t["ended_unix"] < end:
            bins[int((t["ended_unix"] - start) // 60)]["completed_tasks"] += 1
    summary = {k: sum(r[k] for r in replicas) for k in ("output_tokens", "prompt_tokens", "cached_prompt_tokens")}
    summary.update(replicas=replicas, resources=resources, measurement=window,
                   valid=False, steady=None, collection_complete=False, validation_errors=["drain timeout; censored admission cohort"],
                   diagnostic_warnings=[], duration_seconds=end - start,
                   output_tokens_per_second=summary["output_tokens"] / (end - start),
                   actual_prefix_reuse=summary["cached_prompt_tokens"] / summary["prompt_tokens"],
                   completed_in_window=sum(b["completed_tasks"] for b in bins),
                   uncached_prompt_tokens=summary["prompt_tokens"] - summary["cached_prompt_tokens"],
                   failed_tasks=sum(t["status"] == "failed" for t in tasks))
    return summary, bins, audit


def read_tasks(root):
    tasks, workers = [], []
    for worker in sorted((root / "workers").iterdir()):
        s = read(worker / "summary.json")
        workers.append({"replica": worker.name, "status": s["status"],
                        "failed_tasks": s.get("failed_tasks"),
                        "window_start": s.get("measurement", {}).get("started_unix"),
                        "window_end": s.get("measurement", {}).get("ended_unix")})
        tasks.extend({**t, "replica": worker.name} for t in s["tasks"])
    return tasks, workers


def frontend_stats(root, start, end, bins):
    stats, minute_cpu = [], defaultdict(list)
    for host in sorted((root / "frontends").iterdir()):
        samples = []
        for row in records(host / "metrics.jsonl"):
            ts = row["timestamp_unix"]
            if ts >= end:
                break
            if start <= ts and "cpu_busy_percent" in row.get("hardware", {}):
                samples.append(row)
                minute_cpu[int((ts - start) // 60)].append(row["hardware"]["cpu_busy_percent"])
        assert samples, host
        hw = [s["hardware"] for s in samples]
        cpu = [h["cpu_busy_percent"] for h in hw]
        stats.append({"host": host.name, "samples": len(samples), "cpu": dist(cpu),
                      "cpu_logical_count": hw[0]["cpu_logical_count"],
                      "cpu_sample_fraction_ge90": sum(c >= 90 for c in cpu) / len(cpu),
                      "runnable_processes_mean": st.mean(h["cpu_runnable_processes"] for h in hw),
                      "iowait_percent_mean": st.mean(h["cpu_iowait_percent"] for h in hw),
                      "memory_available_min_gib": min(h["memory_available_bytes"] for h in hw) / 2**30,
                      "scratch_free_min_gib": min(h["scratch_free_bytes"] for h in hw) / 2**30,
                      "max_sampling_gap_seconds": max(b["timestamp_unix"] - a["timestamp_unix"] for a, b in zip(samples, samples[1:]))})
    for i, b in enumerate(bins):
        b["frontend_cpu_percent"] = st.mean(minute_cpu[i]) if minute_cpu[i] else None
    return stats


def routing_stats(root, start, end):
    tasks, statuses = defaultdict(list), Counter()
    for path in sorted((root / "backends").glob("*/routing.jsonl")):
        for row in records(path):
            if start <= row["started_unix"] < end:
                statuses[row["status"]] += 1
                if row["status"] == "completed":
                    tasks[row["task_instance_id"]].append((row["started_unix"], row["destination"]))
    selected = [sorted(x) for x in tasks.values() if len(x) >= 8]
    pairs = sum(len(x) - 1 for x in selected)
    switches = sum(a[1] != b[1] for x in selected for a, b in zip(x, x[1:]))
    return {"statuses": dict(statuses), "tasks_with_at_least_8_window_calls": len(selected),
            "adjacent_call_pairs": pairs, "adjacent_call_switches": switches,
            "adjacent_call_backend_switch_fraction": switches / pairs,
            "distinct_backends_per_task_mean": st.mean(len({r[1] for r in x}) for x in selected)}


def setup_stats(root, tasks, start, end):
    rows, missing = [], 0
    for task in tasks:
        if not start <= task["started_unix"] < end:
            continue
        p = root / "workers" / task["replica"] / "tasks" / task["task_id"] / "run/sandbox-setup.json"
        if not p.exists():
            missing += 1
            continue
        rows.append({"run": root.name, "task": task["task_instance_id"],
                     "admission_minute": (task["started_unix"] - start) / 60,
                     "task_status": task["status"], **read(p)})
    return {"admissions": len(rows) + missing, "records": len(rows), "missing": missing,
            "slot_wait_seconds": dist([r["slot_wait_seconds"] for r in rows]),
            "build_seconds": dist([r["build_seconds"] for r in rows])}, rows


def analyze(root, group, batch, *, allow_warmup_failures=False,
            summary_observer=None):
    cfg = read(root / "config.json")
    p = cfg["point"]
    window = read(root / "control/start.json")
    start, end = window["measurement_start_unix"], window["measurement_end_unix"]
    tasks, workers = read_tasks(root)
    recovery = []
    if (root / "summary.json").exists():
        s = read(root / "summary.json")
        with (root / "timeseries.csv").open() as f:
            bins = [{k: float(v) if v else None for k, v in row.items()} for row in csv.DictReader(f)]
        if allow_warmup_failures:
            failed = [t for t in tasks if t["status"] == "failed"]
            assert failed and all(t["ended_unix"] < start for t in failed)
            assert set(s["validation_errors"]) == {f'{t["replica"]}/{t["task_id"]}: failed' for t in failed}
            assert s["failed_in_window"] == 0 and s["collection_complete"]
        else:
            assert s["valid"] and s.get("collection_complete", True), (root.name, s["validation_errors"])
        if allow_warmup_failures:
            assert all(w["status"] in ("completed", "completed_with_errors") for w in workers)
            assert all(t["status"] == "completed" for t in tasks if start <= t["started_unix"] < end)
        else:
            assert all(w["status"] == "completed" for w in workers)
        status = "complete_steady" if s["steady"] else "complete_nonsteady"
        if allow_warmup_failures:
            status = "window_complete_warmup_failures"
    else:
        assert root.name.startswith("minisweagent-routing-n4-round_robin-7642786.")
        s, bins, recovery = recover_window(root, window, tasks)
        status = "window_only_drain_timeout"
    assert all(w["window_start"] == start and w["window_end"] == end for w in workers)
    completed = [t for t in tasks if t["status"] == "completed" and start <= t["ended_unix"] < end]
    admissions = [t for t in tasks if start <= t["started_unix"] < end]
    assert len(completed) == s["completed_in_window"] == sum(b["completed_tasks"] for b in bins)
    assert sum(b["output_tokens"] for b in bins) == s["output_tokens"]
    assert sum(b["prompt_tokens"] for b in bins) == s["prompt_tokens"]
    assert sum(b["cached_prompt_tokens"] for b in bins) == s["cached_prompt_tokens"]
    if summary_observer is not None:
        summary_observer(s)
    # Detailed CPU/setup inspection is needed for fixed-front-end/routing points only.
    if group != "balanced":
        frontend = frontend_stats(root, start, end, bins)
        cpu_mean = st.mean(f["cpu"]["mean"] for f in frontend)
    else:
        frontend = []
        cpu_mean = st.mean(r["hardware_sample_statistics"]["cpu_busy_percent"]["mean"]
                           for r in s["resources"] if r["role"] == "frontend")
    gauge_bins = defaultdict(lambda: defaultdict(list))
    for resource in s["resources"]:
        if resource["role"] != "inference":
            continue
        for row in resource["gauge_samples"]:
            i = int((row["timestamp_unix"] - start) // 60)
            for k in ("waiting", "running", "kv_cache"):
                if row[k] is not None:
                    gauge_bins[i][k].append(row[k])
    for i, b in enumerate(bins):
        b.update(minute=(b["start_unix"] - start) / 60,
                 output_tokens_per_second=b["output_tokens"] / b["duration_seconds"],
                 tasks_per_minute=b["completed_tasks"] * 60 / b["duration_seconds"],
                 actual_prefix_reuse=b["cached_prompt_tokens"] / b["prompt_tokens"] if b["prompt_tokens"] else None)
        for k in ("waiting", "running", "kv_cache"):
            a = gauge_bins[i][k]
            b[k + "_per_replica_mean"] = st.mean(a) if a else None
    setup, setup_rows = ({}, [])
    if p["workload"] == "minisweagent" and group != "balanced":
        setup, setup_rows = setup_stats(root, tasks, start, end)
        assert setup["missing"] == 0, (root.name, setup)
    rates = [r["output_tokens"] / (end - start) for r in s["replicas"]]
    halves = []
    for half in (bins[:len(bins)//2], bins[len(bins)//2:]):
        prompt = sum(b["prompt_tokens"] for b in half)
        halves.append({"output_tokens_s": sum(b["output_tokens"] for b in half) / sum(b["duration_seconds"] for b in half),
                       "completed_tasks": sum(b["completed_tasks"] for b in half),
                       "prefix_reuse": sum(b["cached_prompt_tokens"] for b in half) / prompt,
                       "waiting_per_replica": st.mean(b["waiting_per_replica_mean"] for b in half)})
    result = {"run": root.name, "root": str(root.relative_to(REPO)), "group": group, "batch": batch,
              "workload": p["workload"], "backends": p["inference_replicas"], "frontends": p["frontend_nodes"],
              "physical_nodes_used": p["physical_nodes"], "cc_per_backend": p["task_cc_per_replica"],
              "status": status, "raw_valid": s["valid"], "raw_steady": s["steady"],
              "raw_collection_complete": s.get("collection_complete"),
              "worker_completion_verified": all(w["status"] == "completed" for w in workers),
              "duration_seconds": end - start, "measurement_start_unix": start, "measurement_end_unix": end,
              "completed_tasks_in_window": len(completed), "true_admissions_in_window": len(admissions),
              "cancelled_admissions": sum(t["status"] == "cancelled" for t in admissions),
              "task_status_counts": dict(Counter(t["status"] for t in tasks)),
              "failed_tasks": s.get("failed_tasks", sum(t["status"] == "failed" for t in tasks)), "output_tokens": s["output_tokens"],
              "output_tokens_s": s["output_tokens_per_second"], "tasks_per_minute": len(completed) * 60 / (end - start),
              "prompt_tokens": s["prompt_tokens"], "cached_prompt_tokens": s["cached_prompt_tokens"],
              "prefix_reuse": s["actual_prefix_reuse"],
              "uncached_prompt_per_output": s["uncached_prompt_tokens"] / s["output_tokens"],
              "frontend_cpu_mean_percent": cpu_mean, "frontend": frontend, "halves": halves,
              "replica_output_cv": st.pstdev(rates) / st.mean(rates),
              "replicas": [{"replica": r["replica"], "output_tokens_s": r["output_tokens"] / (end-start),
                            "prefix_reuse": r["cached_prompt_tokens"] / r["prompt_tokens"]} for r in s["replicas"]],
              "ttft_seconds": s.get("ttft_seconds"), "task_lifecycle_seconds": s.get("task_lifecycle_seconds"),
              "tool_latency_seconds": s.get("tool_latency_seconds"),
              "execution_seconds_mean": s.get("execution_seconds_mean"),
              "backend_latency": s.get("backend_latency"), "setup": setup,
              "routing": routing_stats(root, start, end) if group.startswith("routing") else {},
              "raw_validation_errors": s["validation_errors"], "raw_diagnostic_warnings": s["diagnostic_warnings"],
              "recovery_audit": recovery, "workers": workers,
              "sampling_method": "full measurement window; 60s bins; gauges averaged across samples and replicas"}
    print(json.dumps({k: result[k] for k in ("run", "status", "output_tokens_s", "tasks_per_minute", "prefix_reuse", "frontend_cpu_mean_percent")}), flush=True)
    return result, bins, setup_rows


def extract():
    results, time_rows, setup_rows, configs = [], [], [], {}
    sources = []
    for pattern, group, batch in source_patterns():
        roots = list(BASE.glob(pattern))
        assert len(roots) == 1, (pattern, roots)
        root = roots[0]
        sources.append({"root": str(root.relative_to(REPO)), "group": group, "batch": batch})
        result, bins, setups = analyze(root, group, batch)
        results.append(result)
        time_rows.extend({"run": root.name, "workload": result["workload"], "group": group, **b} for b in bins)
        setup_rows.extend(setups)
        configs[root.name] = read(root / "config.json")
    for workload in ("openclaw", "minisweagent"):
        pair = [configs[r["run"]] for r in results if r["workload"] == workload and r["group"].startswith("routing")]
        assert len(pair) == 2
        for key in ("serve", "trace_paths", "replay_profile", "router", "replay_runtime_policy"):
            assert pair[0][key] == pair[1][key], (workload, key)
        for key in ("task_cc_per_replica", "seed", "warmup_seconds", "duration_seconds", "physical_nodes", "prefix_reuse"):
            assert pair[0]["point"][key] == pair[1]["point"][key], (workload, key)
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "full measurement windows; no theoretical replacement points; no raw artifacts modified",
              "routing_matched_configuration": True, "results": results}
    (OUT / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    rows = []
    for r in results:
        row = {k: v for k, v in r.items() if not isinstance(v, (dict, list))}
        for field in ("ttft_seconds", "task_lifecycle_seconds"):
            for stat in ("mean", "p50", "p95"):
                row[f"{field}_{stat}"] = (r[field] or {}).get(stat)
        rows.append(row)
    save_csv("summary.csv", rows)
    save_csv("timeseries.csv", time_rows)
    save_csv("sandbox_setup.csv", setup_rows)
    save_csv("replicas.csv", [{"run": r["run"], **replica} for r in results for replica in r["replicas"]])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    report = read(OUT / "analysis.json") if args.plots_only else extract()
    from plot import draw
    draw(report)
