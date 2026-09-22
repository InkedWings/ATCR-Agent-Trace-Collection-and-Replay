"""Read-only RR request-cost diagnosis; retain request cohorts and warmup timing.

Token events are skipped without JSON decoding; no original run is changed.
Backend arrival time is reconstructed from queued_monotonic and scheduled_unix.
"""
import json
import statistics as st
from collections import Counter, defaultdict
from analyze import OUT, REPO, read, records, dist, save_csv


def metrics(rows):
    if not rows:
        return {"requests": 0}
    prompt = sum(r["prompt_tokens"] for r in rows)
    cache = sum(r["cached_prompt_tokens"] for r in rows)
    fields = ["prompt_tokens", "uncached_tokens", "output_tokens", "queue_seconds",
              "prefill_seconds", "decode_seconds", "residence_seconds"]
    return {"requests": len(rows), "prefix_reuse": cache/prompt,
            "prompt_tokens_sum": prompt, "cached_prompt_tokens_sum": cache,
            **{key: dist([r[key] for r in rows]) for key in fields}}


def main():
    result, time_rows, concise = [], [], []
    for run in read(OUT / "analysis.json")["results"]:
        if run["group"] != "routing_rr":
            continue
        root = REPO / run["root"]
        window = read(root / "control/start.json")
        start, end, warm = window["measurement_start_unix"], window["measurement_end_unix"], window["start_unix"]
        arrival_bins, request_lists = defaultdict(list), []
        for backend in sorted((root / "backends").iterdir()):
            print(f"Reading request completions: {run['workload']} / {backend.name}", flush=True)
            routes = list(records(backend / "routing.jsonl"))
            route_window = [r for r in routes if start <= r["started_unix"] < end]
            requests = []
            ids = set()
            with (backend / "backend.jsonl").open("rb") as stream:
                for line in stream:
                    if b'"request_finished"' not in line:
                        continue
                    r = json.loads(line)
                    assert r["request_id"] not in ids
                    ids.add(r["request_id"])
                    queued = r["scheduled_unix"] + r["queued_monotonic"] - r["scheduled_monotonic"]
                    item = {"queued_unix": queued, "first_token_unix": r["first_token_unix"],
                            "finish_reason": r["finish_reason"],
                            "prompt_tokens": r["prompt_tokens"], "output_tokens": r["output_tokens"],
                            "cached_prompt_tokens": r["cached_prompt_tokens"],
                            "uncached_tokens": r["prompt_tokens"] - r["cached_prompt_tokens"],
                            "queue_seconds": r["scheduled_monotonic"] - r["queued_monotonic"],
                            "prefill_seconds": r["first_token_monotonic"] - r["scheduled_monotonic"],
                            "decode_seconds": r["last_token_monotonic"] - r["first_token_monotonic"],
                            "residence_seconds": r["finished_unix"] - queued}
                    requests.append(item)
                    if warm <= queued < end and r["finish_reason"] in ("stop", "length"):
                        arrival_bins[(backend.name, int((queued-warm)//60))].append(item)
            cohort = [r for r in requests if start <= r["queued_unix"] < end]
            complete = [r for r in cohort if r["finish_reason"] in ("stop", "length")]
            entry = {"run": run["run"], "workload": run["workload"], "replica": backend.name,
                     "middleware_window_arrivals": len(route_window),
                     "middleware_status_counts": dict(Counter(r["status"] for r in route_window)),
                     "backend_window_arrival_finish_reasons": dict(Counter(r["finish_reason"] for r in cohort)),
                     "middleware_residence_seconds": dist([r["ended_unix"]-r["started_unix"] for r in route_window]),
                     "backend_arrival_cohort": metrics(complete),
                     "backend_first_token_cohort": metrics([r for r in requests if start <= r["first_token_unix"] < end and r["finish_reason"] in ("stop", "length")])}
            result.append(entry)
            c = entry["backend_arrival_cohort"]
            concise.append({"workload": run["workload"], "replica": backend.name,
                            "window_arrivals": len(route_window), "finished_backend_arrivals": c["requests"],
                            "prefix_reuse": c["prefix_reuse"],
                            **{k+"_mean": c[k]["mean"] for k in ["prompt_tokens", "uncached_tokens", "output_tokens", "queue_seconds", "prefill_seconds", "decode_seconds", "residence_seconds"]}})
            request_lists.extend(routes)
        # Actual arrival-time ordering can differ slightly from router selection ordering.
        # Counts and per-minute admission distribution are robust to this transport jitter.
        for (rid, minute), rows in sorted(arrival_bins.items()):
            m = metrics(rows)
            time_rows.append({"run": run["run"], "workload": run["workload"], "replica": rid,
                              "minute_from_warmup_start": minute, "measurement_minute": minute-(start-warm)/60,
                              "requests": m["requests"], "prefix_reuse": m["prefix_reuse"],
                              **{k+"_mean": m[k]["mean"] for k in ["prompt_tokens", "uncached_tokens", "queue_seconds", "prefill_seconds", "decode_seconds", "residence_seconds"]}})
    (OUT / "rr_hotspot_requests.json").write_text(json.dumps({
        "scope": "Completed backend requests queued in measurement, full subsequent completion; warmup timeline by queue arrival. Middleware arrivals reported separately.",
        "results": result}, indent=2)+"\n")
    save_csv("rr_hotspot_requests.csv", concise)
    save_csv("rr_hotspot_request_timeseries.csv", time_rows)
    print(json.dumps(concise, indent=2), flush=True)


if __name__ == "__main__":
    main()
