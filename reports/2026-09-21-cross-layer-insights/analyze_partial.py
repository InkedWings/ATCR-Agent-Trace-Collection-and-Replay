"""Read fixed first-45-minute windows without scanning multi-GB backend logs."""
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
JOBS = ("7642785", "7642786", "7642787", "7642788")
DURATION = 2700


def snapshot(path, target):
    # Binary seek by monotonically recorded timestamps, then scan a bounded tail of the search.
    with path.open("rb") as stream:
        low, high = 0, path.stat().st_size
        while high - low > 1000000:
            middle = (low + high) // 2
            stream.seek(middle)
            stream.readline()
            line = stream.readline()
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                high = middle
                continue
            if row["timestamp_unix"] < target:
                low = stream.tell()
            else:
                high = middle
        stream.seek(low)
        for line in stream:
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            if row["timestamp_unix"] >= target:
                if row["timestamp_unix"] - target > 5:
                    raise ValueError(f"snapshot gap at {path}: {target}")
                return row
    raise ValueError(f"window not yet recorded: {path}")


def metric(row, name, source=None):
    values = []
    for line in row["vllm_prometheus"].splitlines():
        if not line.startswith(name + "{") and not line.startswith(name + " "):
            continue
        if source is not None and f'source="{source}"' not in line:
            continue
        values.append(float(line.split()[-1]))
    if not values:
        raise ValueError(f"missing metric {name} / {source}")
    return sum(values)


COUNTERS = {
    "output": ("vllm:generation_tokens_total", None),
    "prompt": ("vllm:prompt_tokens_total", None),
    "cached": ("vllm:prompt_tokens_by_source_total", "local_cache_hit"),
    "local_compute": ("vllm:prompt_tokens_by_source_total", "local_compute"),
    "ttft_sum": ("vllm:time_to_first_token_seconds_sum", None),
    "ttft_count": ("vllm:time_to_first_token_seconds_count", None),
    "preemptions": ("vllm:num_preemptions_total", None),
}


def analyze(root):
    config = json.loads((root / "config.json").read_text())
    start = json.loads((root / "control/start.json").read_text())["measurement_start_unix"]
    end = start + DURATION
    if time.time() < end + 5:
        raise ValueError("wait for a full first-45-minute fragment")
    replicas = []
    for backend in sorted((root / "backends").iterdir()):
        path = backend / "inference-metrics.jsonl"
        before, after = snapshot(path, start), snapshot(path, end)
        delta = {key: metric(after, *args) - metric(before, *args) for key, args in COUNTERS.items()}
        assert all(v >= 0 for v in delta.values()), "counter reset"
        elapsed = after["timestamp_unix"] - before["timestamp_unix"]
        gauges = []
        for offset in range(0, DURATION + 1, 300):
            row = before if offset == 0 else after if offset == DURATION else snapshot(path, start + offset)
            gauges.append({"timestamp_unix": row["timestamp_unix"],
                "gpu_busy": statistics.mean(g["gpu_busy_percent"] for g in row["hardware"]["gpus"]),
                "waiting": metric(row, "vllm:num_requests_waiting"),
                "running": metric(row, "vllm:num_requests_running"),
                "kv_usage": metric(row, "vllm:kv_cache_usage_perc")})
        replicas.append({"replica": backend.name, "counter_delta": delta,
            "sample_interval_seconds": elapsed,
            "start_offset_seconds": before["timestamp_unix"] - start,
            "end_offset_seconds": after["timestamp_unix"] - end,
            "output_tokens_s": delta["output"] / elapsed, "gauges_every_300s": gauges})
    sums = {key: sum(r["counter_delta"][key] for r in replicas) for key in COUNTERS}
    frontend = []
    for host in (root / "frontends").iterdir():
        samples = []
        with (host / "metrics.jsonl").open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row["timestamp_unix"] >= end:
                    break
                if row["timestamp_unix"] >= start:
                    samples.append(row)
        hw = [s["hardware"] for s in samples if "cpu_busy_percent" in s.get("hardware", {})]
        frontend.append({"host": host.name, "samples": len(hw),
            "cpu_mean": statistics.mean(h["cpu_busy_percent"] for h in hw),
            "cpu_p95": sorted(h["cpu_busy_percent"] for h in hw)[int(.95 * (len(hw)-1))],
            "cpu_iowait_mean": statistics.mean(h["cpu_iowait_percent"] for h in hw),
            "cpu_count": hw[0]["cpu_logical_count"],
            "cpu_90pct_sample_fraction": sum(h["cpu_busy_percent"] >= 90 for h in hw) / len(hw),
            "memory_available_min_gib": min(h["memory_available_bytes"] for h in hw) / 2**30,
            "scratch_free_min_gib": min(h["scratch_free_bytes"] for h in hw) / 2**30,
            "max_sampling_gap_seconds": max(b["timestamp_unix"] - a["timestamp_unix"] for a,b in zip(samples, samples[1:]))})
    rates = [r["output_tokens_s"] for r in replicas]
    gauges = [g for r in replicas for g in r["gauges_every_300s"]]
    return {"root": str(root.relative_to(REPO)), "point": config["point"],
        "provisional": True, "duration_seconds": DURATION, "start_unix": start, "end_unix": end,
        "output_tokens_s": sum(rates), "prefix_reuse": sums["cached"] / sums["prompt"],
        "uncached_prompt_per_output": sums["local_compute"] / sums["output"],
        "backend_ttft_mean_s": sums["ttft_sum"] / sums["ttft_count"],
        "prompt_tokens_per_first_token_request": sums["prompt"] / sums["ttft_count"],
        "replica_output_cv": statistics.pstdev(rates) / statistics.mean(rates),
        "preemptions": sums["preemptions"],
        "gpu_busy_sparse_sample_mean": statistics.mean(g["gpu_busy"] for g in gauges),
        "waiting_sparse_sample_mean": statistics.mean(g["waiting"] for g in gauges),
        "running_sparse_sample_mean": statistics.mean(g["running"] for g in gauges),
        "kv_usage_sparse_sample_mean": statistics.mean(g["kv_usage"] for g in gauges),
        "counter_delta": sums, "frontend": frontend, "replicas": replicas}


if __name__ == "__main__":
    roots = sorted(p for p in (REPO / "runs/scaling/multinode").iterdir() if any(job in p.name for job in JOBS))
    results = []
    for root in roots:
        result = analyze(root)
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k not in ("replicas", "counter_delta")}), flush=True)
    for workload in ("openclaw", "minisweagent"):
        configs = [json.loads((REPO / r["root"] / "config.json").read_text()) for r in results
                   if r["point"]["workload"] == workload and r["point"].get("router_impl") == "vllm-router"]
        assert len(configs) == 2
        for key in ("serve", "trace_paths", "replay_profile", "router", "replay_runtime_policy"):
            assert configs[0][key] == configs[1][key], f"routing comparison mismatch: {key}"
        for key in ("task_cc_per_replica", "seed", "warmup_seconds", "duration_seconds", "physical_nodes", "prefix_reuse"):
            assert configs[0]["point"][key] == configs[1]["point"][key], f"routing point mismatch: {key}"
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "provisional first 2700s of each measurement; not final task/cohort/count validation",
        "counters": "first Prometheus sample at/after each boundary; per-replica actual sample intervals",
        "gauges": "backend samples every 300s, including boundaries; frontend uses all samples in the window",
        "routing_matched_configuration": True, "results": results}
    (OUT / "partial_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
