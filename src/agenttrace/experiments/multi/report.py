"""Common-window accounting. Never average replica percentiles or hit ratios."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

from agenttrace.metrics import prometheus_values, summarize_metrics
from agenttrace.replay.benchmark import distribution
from agenttrace.replay.window import timeline
from agenttrace.vllm_backend import union_seconds
from .config import write_json


def records(path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def csv_rows(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def backend_counts(path, start, end, bins):
    """Stream token events, retaining only per-request totals and latency cohorts."""
    tokens, finished, hosts = Counter(), {}, set()
    output = decoded = 0
    halves = [dict(output_tokens=0, prompt_tokens=0, cached_prompt_tokens=0) for _ in range(2)]
    for row in records(path):
        hosts.add(row["hostname"])
        if row.get("event") == "tokens":
            tokens[row["request_id"]] += row["output_tokens"]
            ts = row["timestamp_unix"]
            if start <= ts < end:
                output += row["output_tokens"]
                decoded += row["decode_tokens"]
                bins[min(int((ts-start)//60), len(bins)-1)]["output_tokens"] += row["output_tokens"]
                halves[int(ts >= (start+end)/2)]["output_tokens"] += row["output_tokens"]
        elif row.get("event") == "request_finished":
            rid = row["request_id"]
            if rid in finished:
                raise ValueError("duplicate backend request_finished")
            finished[rid] = row
    if len(hosts) != 1 or not finished:
        raise ValueError("backend must have one host and completed requests")
    if tokens.keys() != finished.keys() or any(tokens[k] != r["output_tokens"] for k, r in finished.items()):
        raise ValueError("incomplete backend log or token/request mismatch")
    active, decode, cohort, cache_cohort = [], [], [], []
    for r in finished.values():
        left, first, right = r["scheduled_unix"], r["first_token_unix"], r["finished_unix"]
        if not all(math.isfinite(t) for t in (left, first, right)) or not left <= first <= right:
            raise ValueError("invalid backend timestamps")
        if r["finish_reason"] not in ("stop", "length") or r["output_tokens"] < 1:
            raise ValueError("unsuccessful backend request")
        if not 0 <= r["cached_prompt_tokens"] <= r["prompt_tokens"]:
            raise ValueError("invalid backend prompt/cache counts")
        if right > start and left < end:
            active.append((max(start, left), min(end, right)))
        if right > start and first < end:
            decode.append((max(start, first), min(end, right)))
        if start <= left < end:
            cohort.append(r)
        if start <= first < end:
            cache_cohort.append(r)
            h = halves[int(first >= (start+end)/2)]
            h["prompt_tokens"] += r["prompt_tokens"]
            h["cached_prompt_tokens"] += r["cached_prompt_tokens"]
            b = bins[min(int((first-start)//60), len(bins)-1)]
            b["prompt_tokens"] += r["prompt_tokens"]
            b["cached_prompt_tokens"] += r["cached_prompt_tokens"]
    return {"requests": len(finished), "all_output_tokens": sum(tokens.values()),
        "all_prompt_tokens": sum(r["prompt_tokens"] for r in finished.values()),
        "output_tokens": output, "decode_tokens": decoded,
        "active_seconds": union_seconds(active), "decode_seconds": union_seconds(decode),
        "prompt_tokens": sum(r["prompt_tokens"] for r in cache_cohort),
        "cached_prompt_tokens": sum(r["cached_prompt_tokens"] for r in cache_cohort),
        "queue_seconds": [r["scheduled_monotonic"]-r["queued_monotonic"] for r in cohort],
        "prefill_seconds": [r["first_token_monotonic"]-r["scheduled_monotonic"] for r in cohort],
        "decode_latency_seconds": [r["last_token_monotonic"]-r["first_token_monotonic"] for r in cohort],
        "halves": halves}


def metric_value(values, name):
    matches = [value for key, value in values.items() if key.split("{", 1)[0] == name]
    return sum(matches) if matches else None


def resource_counts(path, start, end, inference):
    summary = summarize_metrics(path, started_unix=start, ended_unix=end)
    first = last = previous = None
    resets = set()
    samples, power, gauges = [], [], []
    gpu_count_ok = True
    for row in records(path):
        ts = row["timestamp_unix"]
        if start <= ts < end:
            samples.append(ts)
        hw = row.get("hardware", {})
        if inference and start <= ts < end and len(hw.get("gpus", [])) != 4:
            gpu_count_ok = False
        gpu_power = [g.get("power_watts") for g in hw.get("gpus", [])]
        if inference and len(gpu_power) == 4 and all(p is not None for p in gpu_power):
            power.append((ts, sum(gpu_power)))
        if "vllm_prometheus" in row:
            values = prometheus_values(row["vllm_prometheus"])
            # Request outcome/tool labels may appear only after the first call; those are not resets.
            counters = {k: v for k, v in values.items() if k.split("{", 1)[0] in (
                "vllm:prompt_tokens_total", "vllm:generation_tokens_total",
                "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total")}
            if first is None:
                first = counters
            if previous is not None:
                resets.update(previous.keys() ^ counters.keys())
                resets.update(k for k in previous.keys() & counters.keys() if counters[k] < previous[k])
            previous = last = counters
            if start <= ts < end:
                gauges.append({"timestamp_unix": ts,
                    "waiting": metric_value(values, "vllm:num_requests_waiting"),
                    "running": metric_value(values, "vllm:num_requests_running"),
                    "kv_cache": metric_value(values, "vllm:kv_cache_usage_perc")})
    energy = None
    if power and power[0][0] <= start and power[-1][0] >= end:
        energy = 0.0
        for (a, pa), (b, pb) in zip(power, power[1:]):
            left, right = max(a, start), min(b, end)
            if right > left:
                if b-a > 5:
                    energy = None
                    break
                pl, pr = pa+(pb-pa)*(left-a)/(b-a), pa+(pb-pa)*(right-a)/(b-a)
                energy += (pl+pr)/2 * (right-left)
    summary["gpu_energy_joules"] = energy
    summary["inference_gpu_count_ok"] = gpu_count_ok if inference else None
    summary["lustre"] = {"status": "unavailable", "reason": "no per-job Lustre counters; node disk/network are separate"}
    deltas = {k: last[k]-v for k, v in (first or {}).items() if k in (last or {}) and k not in resets}
    summary["whole_run_counters"] = {"deltas": deltas, "reset_or_missing_series": sorted(resets)}
    summary["window_coverage_ok"] = (bool(samples) and samples[0]-start <= 5 and end-samples[-1] <= 5
        and all(b-a <= 5 for a, b in zip(samples, samples[1:])))
    summary["gauge_samples"] = gauges
    return summary


def execution_breakdown(task, report):
    intervals = {kind: [(n["started_seconds"], n["started_seconds"]+n["elapsed_seconds"])
                        for n in report["nodes"] if n["type"] == kind] for kind in ("llm", "tool")}
    llm, tool = (union_seconds(intervals[k]) for k in ("llm", "tool"))
    union = union_seconds(intervals["llm"] + intervals["tool"])
    overlap = max(0, llm+tool-union)
    return {"setup": report["setup_seconds"], "llm_only": llm-overlap, "tool_only": tool-overlap,
            "llm_tool_overlap": overlap, "other_and_cleanup": max(0, task["lifecycle_seconds"]-report["setup_seconds"]-union)}


def summarize_run(root: Path) -> dict:
    config = json.loads((root / "config.json").read_text())
    point, mapping = config["point"], config["node_mapping"]
    window = json.loads((root / "control/start.json").read_text())
    start, end = window["measurement_start_unix"], window["measurement_end_unix"]
    duration = end-start
    bins = [{"start_unix": start+i*60, "duration_seconds": min(60, duration-i*60),
             "output_tokens": 0, "completed_tasks": 0, "prompt_tokens": 0, "cached_prompt_tokens": 0}
            for i in range(math.ceil(duration/60))]
    errors, tasks, calls, breakdowns = [], [], [], []
    diagnostic_warnings, start_lags = [], {}
    windows_complete = True
    expected_calls, expected_tokens = Counter(), Counter()
    for replica in mapping["replicas"]:
        rid = replica["id"]
        worker = root / "workers" / rid
        result = json.loads((worker / "summary.json").read_text())
        manifest = json.loads((worker / "manifest.json").read_text())
        m = result.get("measurement", {})
        if (result["status"] not in ("completed", "completed_with_errors")
                or not m.get("window_complete", m.get("valid")) or m.get("started_unix") != start
                or m.get("ended_unix") != end):
            errors.append(f"{rid}: incomplete/misaligned worker window")
            windows_complete = False
        start_lags[rid] = manifest["actual_started_unix"]-window["start_unix"]
        if abs(start_lags[rid]) > .5:
            diagnostic_warnings.append(f"{rid}: worker start differs by {start_lags[rid]:.3f}s; common window retained")
        if (manifest["trace_pool"] != config["trace_paths"] or manifest["seed"] != point["seed"]+replica["index"]
                or manifest["concurrency"] != point["task_cc_per_replica"]):
            errors.append(f"{rid}: pool/seed/concurrency differs from configuration")
        for task in result["tasks"]:
            task = {**task, "replica": rid}
            tasks.append(task)
            task_root = worker / "tasks" / task["task_id"]
            if task["status"] != "completed":
                errors.append(f"{rid}/{task['task_id']}: {task['status']}")
            if task["status"] == "completed" and start <= task["started_unix"] < end:
                breakdowns.append(execution_breakdown(task, json.loads((worker / task["report"]).read_text())))
            if not (task_root / "events.jsonl").is_file():
                errors.append(f"{rid}/{task['task_id']}: missing replay events")
                continue
            for event in records(task_root / "events.jsonl"):
                if event["event"] != "node_completed":
                    continue
                call = {**event, "task_instance_id": task["task_instance_id"], "home_replica": rid}
                calls.append(call)
                if call["type"] == "llm":
                    destination = call.get("backend_replica_id")
                    expected_calls[destination] += 1
                    expected_tokens[destination] += call["actual_output_tokens"]
                    if call["actual_output_tokens"] != call["target_output_tokens"]:
                        errors.append("client token target mismatch")
    if set(expected_calls) - {r["id"] for r in mapping["replicas"]}:
        errors.append("client call has unknown backend replica")
    # Check routing ledgers against every completed call, including warmup and drain.
    if point["proxy"]:
        routed = {}
        ledger_paths = ([root / "backends" / r["id"] / "routing.jsonl" for r in mapping["replicas"]]
                        if point.get("router_impl") == "vllm-router" else
                        [root / "frontends" / host / "routing.jsonl" for host in mapping["frontends"]])
        for ledger in ledger_paths:
            if not ledger.is_file():
                errors.append(f"missing routing ledger: {ledger}")
                continue
            for row in records(ledger):
                key = (row["task_instance_id"], row["node_id"])
                if key in routed or row["status"] != "completed":
                    errors.append(f"{ledger.parent.name}: duplicate or failed routed call")
                routed[key] = row["destination"]
        client_routes = {(c["task_instance_id"], c["node_id"]): c.get("backend_replica_id") for c in calls if c["type"] == "llm"}
        if routed != client_routes:
            errors.append("proxy/client routing ledgers differ")
    replicas, resources = [], []
    backend_latencies = {key: [] for key in ("queue_seconds", "prefill_seconds", "decode_latency_seconds")}
    halves = [dict(output_tokens=0, completed_tasks=0, prompt_tokens=0, cached_prompt_tokens=0) for _ in range(2)]
    for replica in mapping["replicas"]:
        rid = replica["id"]
        try:
            backend = backend_counts(root / "backends" / rid / "backend.jsonl", start, end, bins)
            if backend["requests"] != expected_calls[rid] or backend["all_output_tokens"] != expected_tokens[rid]:
                errors.append(f"{rid}: backend/client request or token counts differ")
            for key in backend_latencies:
                backend_latencies[key].extend(backend.pop(key))
            for i, half in enumerate(backend.pop("halves")):
                for key, value in half.items():
                    halves[i][key] += value
            prompt = backend["prompt_tokens"]
            replicas.append({"replica": rid, **backend, "output_tokens_per_second": backend["output_tokens"]/duration,
                "active_output_tokens_per_second": backend["output_tokens"]/backend["active_seconds"] if backend["active_seconds"] else None,
                "active_fraction": backend["active_seconds"]/duration,
                "actual_prefix_reuse": backend["cached_prompt_tokens"]/prompt if prompt else None})
        except (OSError, ValueError, KeyError) as error:
            errors.append(f"{rid}: {error}")
            backend = {}
        resource = resource_counts(root / "backends" / rid / "inference-metrics.jsonl", start, end, True)
        resources.append({"node": replica["inference_host"], "role": "inference", "replica": rid, **resource})
        counters = resource["whole_run_counters"]
        if counters["reset_or_missing_series"]:
            errors.append(f"{rid}: Prometheus counter reset/missing series")
        for name, key in (("vllm:generation_tokens_total", "all_output_tokens"), ("vllm:prompt_tokens_total", "all_prompt_tokens")):
            value = metric_value(counters["deltas"], name)
            if value is None or value != backend.get(key):
                errors.append(f"{rid}: {name} does not match backend totals")
        if resource["gpu_status_counts"] != {"ok": resource["samples"]} or not resource["inference_gpu_count_ok"]:
            errors.append(f"{rid}: inference GPU sampling unavailable")
    for host in mapping["frontends"]:
        resources.append({"node": host, "role": "frontend", **resource_counts(root / "frontends" / host / "metrics.jsonl", start, end, False)})
    for resource in resources:
        if resource["error_samples"]:
            errors.append(f"{resource['node']}: failed metric samples")
        if not resource["window_coverage_ok"]:
            diagnostic_warnings.append(
                f"{resource['node']}: resource metric sampling has gaps; resource statistics have limited coverage")
    try:
        clocks = json.loads((root / "clocks.json").read_text())
    except (OSError, ValueError) as error:
        clocks = {"valid": False, "error": str(error)}
    if not clocks.get("valid"):
        diagnostic_warnings.append("clock probe unavailable or imprecise; inspect clocks*.json")
    if (root / "status.json").exists() and json.loads((root / "status.json").read_text())["status"] != "completed":
        errors.append("coordinator did not complete successfully")
    cohort = [t for t in tasks if start <= t["started_unix"] < end and t["status"] == "completed"]
    finished = [t for t in tasks if start <= t.get("ended_unix", 0) < end and t["status"] == "completed"]
    failed = [t for t in tasks if t["status"] == "failed"]
    failed_in_window = [t for t in failed if start <= t.get("ended_unix", 0) < end]
    for task in finished:
        ts = task["ended_unix"]
        halves[int(ts >= (start+end)/2)]["completed_tasks"] += 1
        bins[min(int((ts-start)//60), len(bins)-1)]["completed_tasks"] += 1
    nodes = [c for c in calls if start <= c["started_unix"] < end]
    llm, tool = ([c for c in nodes if c["type"] == kind] for kind in ("llm", "tool"))
    for b in bins:
        b["output_tokens_per_second"] = b["output_tokens"]/b["duration_seconds"]
        b["tasks_per_second"] = b["completed_tasks"]/b["duration_seconds"]
        b["actual_prefix_reuse"] = b["cached_prompt_tokens"]/b["prompt_tokens"] if b["prompt_tokens"] else None
    # Compare equal wall-clock halves. No replica averages of task/token rates or reuse ratios.
    for i, half in enumerate(halves):
        half["actual_prefix_reuse"] = half["cached_prompt_tokens"]/half["prompt_tokens"] if half["prompt_tokens"] else None
        queues = []
        for resource in resources:
            if resource["role"] == "inference":
                values = [g["waiting"] for g in resource["gauge_samples"]
                          if int(g["timestamp_unix"] >= (start+end)/2) == i and g["waiting"] is not None]
                if values:
                    queues.append(sum(values)/len(values))
        half["waiting_per_replica_sample_mean"] = sum(queues)/len(queues) if len(queues) == len(replicas) and queues else None
    def relative_change(key):
        a, b = (h[key] for h in halves)
        return abs(b-a)/a if a else None
    reuse = [h["actual_prefix_reuse"] for h in halves]
    queue = [h["waiting_per_replica_sample_mean"] for h in halves]
    stability = {"halves": halves, "output_rate_relative_change": relative_change("output_tokens"),
        "task_rate_relative_change": relative_change("completed_tasks"),
        "reuse_absolute_change": abs(reuse[1]-reuse[0]) if None not in reuse else None,
        "waiting_increase_per_replica": queue[1]-queue[0] if None not in queue else None}
    stability["checks"] = {
        "output_rate_within_10_percent": stability["output_rate_relative_change"] is not None and stability["output_rate_relative_change"] <= .1,
        "task_rate_within_20_percent": stability["task_rate_relative_change"] is not None and stability["task_rate_relative_change"] <= .2,
        "reuse_within_5_percentage_points": stability["reuse_absolute_change"] is not None and stability["reuse_absolute_change"] <= .05,
        "no_queue_growth": queue[0] is not None and queue[1] is not None and queue[1]-queue[0] <= max(1, .1*queue[0]),
        "has_admission_cohort": bool(cohort)}
    prompt = sum(r["prompt_tokens"] for r in replicas)
    cached = sum(r["cached_prompt_tokens"] for r in replicas)
    output = sum(r["output_tokens"] for r in replicas)
    queries = hits = 0
    lookup_available = True
    for resource in resources:
        if resource["role"] != "inference":
            continue
        counters = resource.get("vllm", {}).get("counter_deltas", {})
        q = metric_value(counters, "vllm:prefix_cache_queries_total")
        h = metric_value(counters, "vllm:prefix_cache_hits_total")
        if q is None or h is None:
            lookup_available = False
        else:
            queries += q
            hits += h
    if not output:
        errors.append("no backend output tokens in measurement window")
    if not tasks or not calls:
        errors.append("no complete replay tasks/calls")
    energy = [r["gpu_energy_joules"] for r in resources if r["role"] == "inference"]
    joules = sum(energy) if energy and None not in energy else None
    concurrency = {"tasks": timeline([(t["started_unix"], t["ended_unix"]) for t in tasks if "ended_unix" in t], start, end),
        "llm_calls": timeline([(c["started_unix"], c["timestamp_unix"]) for c in calls if c["type"] == "llm"], start, end)}
    result = {"schema_version": 1, "root": str(root.resolve()), "point": point, "serve": config["serve"],
        "valid": not errors, "validation_errors": sorted(set(errors)),
        "diagnostic_warnings": diagnostic_warnings, "clock_diagnostics": clocks,
        "worker_start_lag_seconds": start_lags,
        "steady": not errors and not point["smoke"] and all(stability["checks"].values()), "stability": stability,
        "measurement": window, "duration_seconds": duration,
        "collection_complete": windows_complete,
        "failed_tasks": len(failed), "failed_in_window": len(failed_in_window),
        "task_failure_fraction": len(failed_in_window) / (len(failed_in_window)+len(finished)) if failed_in_window or finished else None,
        "task_failure_fraction_scope": "tasks ending in the measurement window: failed / (failed + completed)",
        "completed_in_window": len(finished), "admitted_in_window": len(cohort),
        "tasks_per_second": len(finished)/duration, "output_tokens_per_second": output/duration,
        "output_tokens": output, "prompt_tokens": prompt, "cached_prompt_tokens": cached,
        "actual_prefix_reuse": cached/prompt if prompt else None,
        "uncached_prompt_tokens": prompt-cached,
        "prefix_lookup_hit_ratio": hits/queries if queries and lookup_available else None,
        "prefix_lookup_scope": "sum hits / sum queries across backend Prometheus samples inside the window; retry lookups may repeat",
        "cache_scope": "sum cached / sum prompt for requests emitting first token in the common window; includes their full prompt",
        "latency_scope": "task admissions / node starts / backend scheduling in window; full completion through natural drain",
        "task_lifecycle_seconds": distribution([t["lifecycle_seconds"] for t in cohort]),
        "llm_latency_seconds": distribution([c["elapsed_seconds"] for c in llm]),
        "tool_latency_seconds": distribution([c["elapsed_seconds"] for c in tool]),
        "ttft_seconds": distribution([c["ttft_seconds"] for c in llm if c.get("ttft_seconds") is not None]),
        "tpot_estimate_seconds": distribution([c["tpot_estimate_seconds"] for c in llm if c.get("tpot_estimate_seconds") is not None]),
        "backend_latency": {key: distribution(values) for key, values in backend_latencies.items()},
        "execution_seconds_mean": {k: sum(b[k] for b in breakdowns)/len(breakdowns) for k in breakdowns[0]} if breakdowns else {},
        "native_tool_errors": sum(c.get("native_error", False) for c in tool),
        "completed_trace_coverage": len({t["trace_path"] for t in finished}), "trace_pool_size": len(config["trace_paths"]),
        "inference_gpu_energy_joules": joules, "inference_gpu_joules_per_task": joules/len(finished) if joules is not None and finished else None,
        "replicas": replicas, "resources": resources,
        "concurrency": {key: {k: v for k, v in value.items() if k != "points"} for key, value in concurrency.items()},
        "repetitions": 1, "interpretation": "exploratory single run; steady flag is a diagnostic, not an SLO capacity claim"}
    write_json(root / "summary.json", result)
    write_json(root / "concurrency.json", concurrency)
    csv_rows(root / "timeseries.csv", bins)
    csv_rows(root / "replicas.csv", replicas)
    csv_rows(root / "tasks.csv", tasks)
    csv_rows(root / "failed-tasks.csv", failed)
    csv_rows(root / "calls.csv", [{k: v for k, v in c.items() if k != "tool_replay"} for c in calls])
    return result
