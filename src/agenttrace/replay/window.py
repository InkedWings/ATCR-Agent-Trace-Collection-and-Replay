"""Window cohorts and exact client concurrency from replay events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def timeline(intervals: list[tuple[float, float]], start: float, end: float) -> dict:
    changes: dict[float, int] = {start: 0, end: 0}
    for left, right in intervals:
        left, right = max(left, start), min(right, end)
        if right > left:
            changes[left] = changes.get(left, 0) + 1
            changes[right] = changes.get(right, 0) - 1
    active = peak = 0
    area = 0.0
    previous = start
    points = []
    for timestamp, change in sorted(changes.items()):
        area += active * (timestamp - previous)
        active += change
        peak = max(peak, active)
        points.append({"timestamp_unix": timestamp, "active": active})
        previous = timestamp
    return {"peak": peak, "time_weighted_mean": area / (end - start), "points": points}


def summarize_window(root: Path, tasks: list[dict[str, Any]], start: float, end: float) -> dict:
    from agenttrace.replay.benchmark import distribution

    completed = [row for row in tasks if row["status"] == "completed"]
    reports = {row["task_id"]: json.loads((root / row["report"]).read_text()) for row in completed}
    admitted = [row for row in tasks if start <= row.get("started_unix", 0) < end]
    cohort = [row for row in admitted if row["status"] == "completed"]
    # Events retain successful calls from failed/cancelled tasks as well.
    calls = []
    for row in tasks:
        events = root / "tasks" / row["task_id"] / "events.jsonl"
        if events.exists():
            with events.open() as handle:
                for line in handle:
                    event = json.loads(line)
                    if event["event"] == "node_completed":
                        calls.append(event)
    nodes = [node for node in calls if start <= node["started_unix"] < end]
    llm = [node for node in nodes if node["type"] == "llm"]
    tool = [node for node in nodes if node["type"] == "tool"]
    finished = [row for row in completed if start <= row["ended_unix"] < end]
    concurrency = {"tasks": timeline([(r["started_unix"], r["ended_unix"])
        for r in tasks if "ended_unix" in r], start, end),
        "llm_calls": timeline([(n["started_unix"], n["timestamp_unix"])
            for n in calls if n["type"] == "llm"], start, end)}
    (root / "concurrency.json").write_text(json.dumps(concurrency, indent=2) + "\n")
    return {"started_unix": start, "ended_unix": end, "duration_seconds": end - start,
        "cohort_policy": "task admissions / node starts in [start,end); full latency after natural drain",
        "completed_in_window": len(finished), "admitted_in_window": len(admitted),
        "incomplete_admission_cohort": len(admitted) - len(cohort),
        "task_throughput_per_second": len(finished) / (end - start),
        "admitted_trace_coverage": sorted({r["trace_path"] for r in admitted}),
        "completed_trace_coverage": sorted({r["trace_path"] for r in finished}),
        "task_lifecycle_seconds": distribution([r["lifecycle_seconds"] for r in cohort]),
        "task_setup_seconds": distribution([reports[r["task_id"]]["setup_seconds"] for r in cohort]),
        "task_replay_seconds": distribution([reports[r["task_id"]]["replay_makespan_seconds"] for r in cohort]),
        "llm_latency_seconds": distribution([n["elapsed_seconds"] for n in llm]),
        "tool_latency_seconds": distribution([n["elapsed_seconds"] for n in tool]),
        "llm_ttft_seconds": distribution([n["ttft_seconds"] for n in llm if n.get("ttft_seconds") is not None]),
        "llm_tpot_estimate_seconds": distribution([n["tpot_estimate_seconds"] for n in llm if n.get("tpot_estimate_seconds") is not None]),
        "native_tool_errors": sum(n["native_error"] for n in tool),
        "concurrency": {key: {k: v for k, v in data.items() if k != "points"}
            for key, data in concurrency.items()}}
