"""vLLM 0.19.1 request intervals and output throughput excluding idle gaps.

Run this file inside the serving environment; importing it needs only stdlib.
The hook writes token-count events and request timings, never response bodies.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import socket
import sys
import time
from pathlib import Path
from typing import Any


def union_seconds(intervals: list[tuple[float, float]]) -> float:
    total = 0.0
    end = -math.inf
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def summarize_backend(path: Path, *, started_unix: float | None = None,
                      ended_unix: float | None = None, window: bool = False) -> dict[str, Any]:
    if window:
        if started_unix is None or ended_unix is None or ended_unix <= started_unix:
            raise ValueError("a positive window with both boundaries is required")
        return summarize_event_window(path, started_unix, ended_unix)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("event", "request_finished") != "request_finished":
                continue
            if started_unix is not None and row["finished_unix"] < started_unix:
                continue
            if ended_unix is not None and row["finished_unix"] > ended_unix:
                continue
            if started_unix is not None and row["scheduled_unix"] < started_unix:
                raise ValueError("backend request crosses benchmark start; use an idle dedicated endpoint")
            times = [row[key] for key in ("scheduled_monotonic", "first_token_monotonic", "last_token_monotonic")]
            if not all(math.isfinite(t) and t > 0 for t in times) or times != sorted(times):
                raise ValueError("missing or invalid backend request timestamps")
            if row["output_tokens"] < 1 or row["finish_reason"] not in ("stop", "length"):
                raise ValueError("backend throughput requires complete successful requests")
            rows.append(row)
    if not rows:
        return {"status": "unavailable", "reason": "no finished backend requests in the selected window"}
    if len({row["hostname"] for row in rows}) != 1:
        raise ValueError("backend interval union currently supports a single serving host")
    active = union_seconds([(r["scheduled_monotonic"], r["last_token_monotonic"]) for r in rows])
    decode = union_seconds([(r["first_token_monotonic"], r["last_token_monotonic"]) for r in rows])
    tokens = sum(r["output_tokens"] for r in rows)
    # vLLM's decode interval excludes the first token generated during prefill.
    decode_tokens = sum(r["output_tokens"] - 1 for r in rows)
    return {"status": "measured", "source": str(path.resolve()),
        "scope": "single_host_backend_request_interval_union",
        "requests": len(rows), "output_tokens": tokens,
        "active_seconds": active,
        "active_output_tokens_per_second": tokens / active if active else None,
        "decode_tokens": decode_tokens, "decode_seconds": decode,
        "decode_tokens_per_second": decode_tokens / decode if decode else None}


def summarize_event_window(path: Path, start: float, end: float) -> dict[str, Any]:
    requests = {}
    events = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("event") == "request_finished":
                requests[row["request_id"]] = row
            elif row.get("event") == "tokens":
                events.append(row)
    if not events:
        raise ValueError("window throughput requires incremental backend token events")
    if len({row["hostname"] for row in events}) != 1:
        raise ValueError("window throughput requires one serving host")
    # Drain must finish before summarizing. Check all events, not just window events.
    totals: dict[str, int] = {}
    for row in events:
        totals[row["request_id"]] = totals.get(row["request_id"], 0) + row["output_tokens"]
    if totals.keys() != requests.keys() or any(totals[key] != row["output_tokens"]
            for key, row in requests.items()):
        raise ValueError("incomplete backend event log or token/request count mismatch")
    active, decode, selected = [], [], []
    for row in requests.values():
        left = row["scheduled_unix"]
        first = row["first_token_unix"]
        right = row["finished_unix"]
        if not all(math.isfinite(t) for t in (left, first, right)) or not left <= first <= right:
            raise ValueError("invalid backend request timestamps")
        if right < start or left >= end:
            continue
        if row["finish_reason"] not in ("stop", "length"):
            raise ValueError("window contains unsuccessful backend request")
        active.append((max(left, start), min(right, end)))
        if min(right, end) > max(first, start):
            decode.append((max(first, start), min(right, end)))
        if start <= left < end:
            selected.append(row)
    tokens = [row for row in events if start <= row["timestamp_unix"] < end]
    generated = sum(row["output_tokens"] for row in tokens)
    decoded = sum(row["decode_tokens"] for row in tokens)
    active_time, decode_time = union_seconds(active), union_seconds(decode)
    def dist(values):
        values = sorted(values)
        def p(q):
            if not values:
                return None
            position = (len(values) - 1) * q
            i = int(position)
            return values[i] + (values[min(i + 1, len(values)-1)] - values[i]) * (position - i)
        return {"count": len(values), "mean": sum(values)/len(values) if values else None,
            "p50": p(.5), "p95": p(.95), "p99": p(.99)}
    return {"status": "measured", "source": str(path.resolve()),
        "scope": "single_host_engine_token_events_and_clipped_request_interval_union",
        "started_unix": start, "ended_unix": end,
        "output_tokens": generated, "active_seconds": active_time,
        "active_output_tokens_per_second": generated / active_time if active_time else None,
        "decode_tokens": decoded, "decode_seconds": decode_time,
        "decode_tokens_per_second": decoded / decode_time if decode_time else None,
        "window_output_tokens_per_second": generated / (end-start),
        "overlapping_requests": len(active), "event_totals_validated": True,
        "latency_cohort": "backend requests first scheduled in window; measured through completion",
        "queue_seconds": dist([r["scheduled_monotonic"]-r["queued_monotonic"] for r in selected]),
        "prefill_seconds": dist([r["first_token_monotonic"]-r["scheduled_monotonic"] for r in selected]),
        "decode_latency_seconds": dist([r["last_token_monotonic"]-r["first_token_monotonic"] for r in selected])}


def install_request_logger(path: Path):
    import vllm
    from vllm.v1.metrics.stats import IterationStats

    if vllm.__version__ != "0.19.1":
        raise RuntimeError("backend timestamp hook is tested against vLLM 0.19.1 only")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("x", encoding="utf-8", buffering=65536)
    original = IterationStats.update_from_finished_request
    original_output = IterationStats.update_from_output
    hostname = socket.gethostname()
    offset = time.time() - time.monotonic()
    request_ids = {}
    flushed = time.monotonic()

    def write(row, finish=False):
        nonlocal flushed
        handle.write(json.dumps({"hostname": hostname, **row}, separators=(",", ":")) + "\n")
        now = time.monotonic()
        if finish or now - flushed >= 1:
            handle.flush()
            flushed = now

    @functools.wraps(original_output)
    def output(self, output, engine_core_timestamp, is_prefilling, prompt_len,
               req_stats, lora_states, lora_name):
        before = req_stats.num_generation_tokens
        result = original_output(self, output, engine_core_timestamp, is_prefilling,
                                 prompt_len, req_stats, lora_states, lora_name)
        request_ids[id(req_stats)] = output.request_id
        count = len(output.new_token_ids)
        if count:
            write({"event": "tokens", "request_id": output.request_id,
                "engine_monotonic": engine_core_timestamp,
                "timestamp_unix": offset + engine_core_timestamp,
                "output_tokens": count, "decode_tokens": count - int(before == 0)})
        return result

    @functools.wraps(original)
    def finished(self, finish_reason, num_prompt_tokens, max_tokens_param,
                 req_stats, num_cached_tokens=0):
        result = original(self, finish_reason, num_prompt_tokens, max_tokens_param,
                          req_stats, num_cached_tokens)
        # EngineCore timestamps use CLOCK_MONOTONIC on this same serving host.
        # Use those for durations; wall-clock conversion only selects a run.
        row = {"event": "request_finished", "request_id": request_ids.pop(id(req_stats), None),
            "finish_reason": str(finish_reason), "queued_monotonic": req_stats.queued_ts,
            "scheduled_monotonic": req_stats.scheduled_ts,
            "first_token_monotonic": req_stats.first_token_ts,
            "last_token_monotonic": req_stats.last_token_ts,
            "scheduled_unix": offset + req_stats.scheduled_ts,
            "first_token_unix": offset + req_stats.first_token_ts,
            "finished_unix": offset + req_stats.last_token_ts,
            "output_tokens": req_stats.num_generation_tokens}
        write(row, finish=True)
        return result

    IterationStats.update_from_finished_request = finished
    IterationStats.update_from_output = output
    return handle


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backend-events", required=True, type=Path)
    args, remaining = parser.parse_known_args()
    handle = install_request_logger(args.backend_events)
    from vllm.entrypoints.cli.main import main as serve_main

    sys.argv = ["vllm", *remaining]
    try:
        serve_main()
    finally:
        handle.close()


if __name__ == "__main__":
    main()
