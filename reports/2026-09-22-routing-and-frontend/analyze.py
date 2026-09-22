"""Merge measured Sep 21 references with three usable Sep 22 windows.

Only reads runs. Writes beside this script. Reuses checked historical aggregates;
P2 is excluded at the user's request. No experiment launch.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import statistics as st
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
PREVIOUS = OUT.parent / "2026-09-21-routing-and-fixed-frontend"
BASE = REPO / "runs/scaling/multinode"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


old = module("previous_analysis", PREVIOUS / "analyze.py")
weak = module("weak_audit", OUT.parent / "2026-09-15-weak-scaling/analyze.py")
read, records, dist = old.read, old.records, old.dist


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def csv_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def save_csv(name, rows):
    with (OUT / name).open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def replica_gauges(run, summary):
    rows, minutes = [], []
    start = summary["measurement"]["measurement_start_unix"]
    for resource in summary["resources"]:
        if resource["role"] != "inference":
            continue
        gauges = resource["gauge_samples"]
        base = {"run": run, "replica": resource["replica"]}
        rows.append({**base, "samples": len(gauges),
                     **{f"{key}_mean": st.mean(g[key] for g in gauges)
                        for key in ("waiting", "running", "kv_cache")}})
        bins = defaultdict(list)
        for g in gauges:
            bins[int((g["timestamp_unix"] - start) // 60)].append(g)
        for minute, samples in sorted(bins.items()):
            minutes.append({**base, "minute": minute,
                            **{f"{key}_mean": st.mean(s[key] for s in samples)
                               for key in ("waiting", "running", "kv_cache")}})
    return rows, minutes


def p2_load_audit(root):
    window = read(root / "control/start.json")
    start, end = window["measurement_start_unix"], window["measurement_end_unix"]
    router = next((root / "frontends").glob("*/router"))
    cfg = read(router / "config.json")
    endpoints = {v.removesuffix("/v1"): k for k, v in cfg["endpoints"].items()}
    all_pairs, window_pairs, destinations = Counter(), Counter(), Counter()
    minutes, unparsed = defaultdict(Counter), 0
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    pattern = re.compile(r"selection: (\S+)=(-?\d+) vs (\S+)=(-?\d+) -> selected (\S+)")
    with (router / "service.log").open() as stream:
        for line in stream:
            if "Power-of-two selection:" not in line:
                continue
            line = ansi.sub("", line)
            m = pattern.search(line)
            if not m:
                unparsed += 1
                continue
            pair = f"{m[2]},{m[4]}"
            all_pairs[pair] += 1
            ts = datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
            if start <= ts < end:
                window_pairs[pair] += 1
                destinations[endpoints[m[5]]] += 1
                bucket = minutes[int((ts-start)//60)]
                bucket["decisions"] += 1
                bucket["zero_pair"] += pair == "0,0"
                bucket["nonzero_pair"] += pair != "0,0"
    assert unparsed == 0 and sum(window_pairs.values()) > 0
    probes = []
    for backend in sorted((root / "backends").iterdir()):
        count, load_route, example = 0, False, None
        with (backend / "service.log").open() as stream:
            for line in stream:
                load_route |= "Route: /load, Methods: GET" in line
                if '"GET /get_load HTTP/1.1" 404' in line:
                    count += 1
                    example = example or line.strip()
        probes.append({"replica": backend.name, "get_load_404_all_run": count,
                       "registered_load_route": load_route, "example": example})
    return {"run": root.name, "router_version": cfg["version"], "transport": cfg["transport"],
            "all_run_decisions": sum(all_pairs.values()), "all_run_load_pairs": dict(all_pairs),
            "window_decisions": sum(window_pairs.values()), "window_load_pairs": dict(window_pairs),
            "window_destinations": dict(destinations), "unparsed_decisions": unparsed,
            "timestamp_resolution_seconds": 1,
            "window_boundary_note": "router log timestamps are UTC with 1s resolution; may differ slightly from ledger",
            "backend_load_probes": probes,
            "effective_behavior": "random selection under equal zero loads; NOT an effective load-aware baseline",
            "sources": ["https://github.com/vllm-project/router/blob/v0.1.15/src/policies/power_of_two.rs",
                        "https://github.com/vllm-project/router/blob/v0.1.15/src/routers/http/router.rs"]}, [
                {"minute": minute, **counts} for minute, counts in sorted(minutes.items())]


def routing_arrivals(root, start, end):
    rows = []
    for path in sorted((root / "backends").glob("*/routing.jsonl")):
        arrivals = [r for r in records(path) if start <= r["started_unix"] < end]
        rows.append({"run": root.name, "replica": path.parent.name, "arrivals": len(arrivals),
                     "statuses": dict(Counter(r["status"] for r in arrivals)),
                     "residence_seconds": dist([r["ended_unix"]-r["started_unix"] for r in arrivals])})
    return rows


def configuration_comparisons(results):
    def get(prefix):
        return next(r for r in results if r["run"].startswith(prefix))
    comparisons = []
    baseline = get("minisweagent-fixed_frontend-n8-7642788.")
    for prefix in ("minisweagent-frontend-f2-n8-build2-", "minisweagent-frontend-f1-n8-build8-"):
        r = get(prefix)
        a, b = [read(REPO / v["root"] / "config.json") for v in (baseline, r)]
        checks = {k: a[k] == b[k] for k in ("serve", "trace_paths", "replay_runtime_policy")}
        pa, pb = json.loads(json.dumps(a["replay_profile"])), json.loads(json.dumps(b["replay_profile"]))
        slots = [p["tool_executor"]["config"].pop("max_parallel_sandbox_builds") for p in (pa, pb)]
        checks["profile_except_build_slots"] = pa == pb
        for k in ("inference_replicas", "task_cc_per_replica", "seed", "routing", "prefix_reuse", "proxy", "duration_seconds", "warmup_seconds"):
            checks[k] = a["point"][k] == b["point"][k]
        assert all(checks.values()), checks
        r["build_slots_per_frontend"] = slots[1]
        baseline["build_slots_per_frontend"] = slots[0]
        comparisons.append({"baseline": baseline["run"], "comparison": r["run"], "checks": checks,
                            "changed_frontends": [baseline["frontends"], r["frontends"]], "build_slots_per_frontend": slots})
    for base_prefix, prefix in (("openclaw-routing-n4-round_robin-", "openclaw-routing-n8-round_robin-"),):
        a, b = [read(REPO / get(v)["root"] / "config.json") for v in (base_prefix, prefix)]
        checks = {k: a[k] == b[k] for k in ("serve", "trace_paths", "replay_profile", "replay_runtime_policy")}
        for k in ("task_cc_per_replica", "seed", "prefix_reuse", "duration_seconds", "warmup_seconds"):
            checks[k] = a["point"][k] == b["point"][k]
        checks["router_runtime_settings"] = all(a["router"][k] == b["router"][k]
                                                for k in a["router"] if k not in ("policies", "inference_nodes"))
        assert all(checks.values()), checks
        comparisons.append({"baseline": get(base_prefix)["run"], "comparison": get(prefix)["run"], "checks": checks})
    return comparisons


def extract():
    results = read(PREVIOUS / "analysis.json")["results"]
    for r in results:
        r["batch"] = "previous"
    time_rows = csv_rows(PREVIOUS / "timeseries.csv")
    setup_rows = csv_rows(PREVIOUS / "sandbox_setup.csv")
    replica_rows = read(PREVIOUS / "replica_diagnostics.json")
    replica_minutes = csv_rows(PREVIOUS / "replica_timeseries.csv")
    audits = read(PREVIOUS / "completed_call_audit.json")
    arrivals = []
    patterns = [("minisweagent-frontend-f2-n8-build2-7643323.*", "frontend_f2_build2"),
                ("minisweagent-frontend-f1-n8-build8-7643324.*", "frontend_f1_build8"),
                ("openclaw-routing-n8-round_robin-7643277.*", "routing_rr")]
    metadata = None
    for pattern, group in patterns:
        roots = list(BASE.glob(pattern))
        assert len(roots) == 1, (pattern, roots)
        root = roots[0]
        cache = OUT / "extracted" / (root.name + ".json")
        if cache.exists():
            saved = read(cache)
            r, bins, setups, rs, rm, audit, arrival = [saved[k] for k in ("result", "bins", "setups", "replicas", "replica_minutes", "audit", "arrivals")]
        else:
            captured = {}
            def observe(summary):
                captured["summary"] = summary
            r, bins, setups = old.analyze(root, group, "new_20260922",
                                         allow_warmup_failures=pattern.startswith("openclaw-"), summary_observer=observe)
            summary = captured["summary"]
            rs, rm = replica_gauges(root.name, summary)
            audit, arrival = None, []
            if group.startswith("frontend"):
                cfg = read(root / "config.json")
                if metadata is None:
                    metadata = weak.tool_metadata(cfg["trace_paths"])
                audit = {"run": root.name, **weak.audit_calls(root, summary, metadata)}
            else:
                arrival = routing_arrivals(root, r["measurement_start_unix"], r["measurement_end_unix"])
            cache.parent.mkdir(exist_ok=True)
            write_json(cache.relative_to(OUT), dict(result=r, bins=bins, setups=setups, replicas=rs,
                       replica_minutes=rm, audit=audit, arrivals=arrival))
        by_rep = {v["replica"]: v for v in r["replicas"]}
        replica_rows.extend({"workload": r["workload"], "group": group, **row, **by_rep[row["replica"]]} for row in rs)
        replica_minutes.extend(rm)
        results.append(r)
        time_rows.extend({"run": root.name, "workload": r["workload"], "group": group, **b} for b in bins)
        setup_rows.extend(setups)
        if audit:
            audits.append(audit)
        arrivals.extend(arrival)
    # Historical front-end reference gauges come from the existing small summary.
    front_base = next(r for r in results if r["run"].startswith("minisweagent-fixed_frontend-n8-7642788."))
    rs, rm = replica_gauges(front_base["run"], read(REPO / front_base["root"] / "summary.json"))
    by_rep = {v["replica"]: v for v in front_base["replicas"]}
    replica_rows.extend({"workload": front_base["workload"], "group": front_base["group"], **row, **by_rep[row["replica"]]} for row in rs)
    replica_minutes.extend(rm)
    comparisons = configuration_comparisons(results)
    for r in results:
        if r["setup"]:
            rows = [v for v in setup_rows if v["run"] == r["run"]]
            half = r["duration_seconds"] / 120
            r["setup_halves"] = [dist([float(v["slot_wait_seconds"]) for v in rows
                                       if (float(v["admission_minute"]) < half) == first]) for first in (True, False)]
        r["total_cc"] = r["backends"] * r["cc_per_backend"]
        r["window_usable"] = True
    missing = []
    for prefix in ("openclaw-routing-n8-cache_aware-7643278.", "minisweagent-routing-n8-round_robin-7643279.",
                   "minisweagent-routing-n8-cache_aware-7643280."):
        root = next(BASE.glob(prefix + "*"))
        assert not (root / "control/start.json").exists()
        missing.append({"run": root.name, "root": str(root.relative_to(REPO)), "status": "startup_failed_no_measurement"})
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "20 measured windows; P2 excluded; previous aggregates reused; no imputation; raw states preserved",
              "previous_report": str(PREVIOUS.relative_to(REPO)), "results": results,
              "missing": missing, "configuration_comparisons": comparisons}
    write_json("analysis.json", report)
    write_json("sources.json", [{k:r[k] for k in ("run", "root", "group", "batch", "status")} for r in results] + missing)
    write_json("completed_call_audit.json", audits)
    write_json("replica_diagnostics.json", replica_rows)
    write_json("routing_arrivals.json", arrivals)
    summary_rows = []
    by_audit = {a["run"]: a for a in audits}
    for r in results:
        row = {k:v for k,v in r.items() if not isinstance(v,(dict,list))}
        for field in ("ttft_seconds", "task_lifecycle_seconds", "tool_latency_seconds"):
            for stat in ("mean", "p50", "p95"):
                row[f"{field}_{stat}"] = (r[field] or {}).get(stat)
        for field in ("slot_wait_seconds", "build_seconds"):
            for stat in ("mean", "p50", "p95"):
                row[f"setup_{field}_{stat}"] = r["setup"].get(field, {}).get(stat)
        row.update({"execution_" + k + "_seconds": v for k,v in (r["execution_seconds_mean"] or {}).items()})
        if r["run"] in by_audit:
            a = by_audit[r["run"]]["window_tools"]
            row.update(tool_calls=a["calls"], new_tool_errors=a.get("success_to_error",0),
                       new_tool_error_fraction=a.get("success_to_error",0)/a["calls"])
        reps = [v for v in replica_rows if v["run"] == r["run"]]
        if reps:
            row.update({k+"_mean_per_backend": st.mean(v[k+"_mean"] for v in reps) for k in ("running", "waiting", "kv_cache")})
        summary_rows.append(row)
    save_csv("summary.csv", summary_rows)
    save_csv("timeseries.csv", time_rows)
    save_csv("sandbox_setup.csv", setup_rows)
    save_csv("replica_diagnostics.csv", replica_rows)
    save_csv("replica_timeseries.csv", replica_minutes)
    assert len(results) == len({r["run"] for r in results}) == 20
    print("Wrote 20 measured windows and 3 missing points; P2 excluded.", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()
    report = read(OUT / "analysis.json") if args.plots_only else extract()
    if not args.extract_only:
        from plot import draw
        draw(report)
