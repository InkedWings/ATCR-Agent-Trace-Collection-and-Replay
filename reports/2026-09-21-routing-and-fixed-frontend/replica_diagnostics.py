"""Expose per-backend queue concentration hidden by run-level averages."""
import csv
import json
import statistics as st
from collections import defaultdict
from analyze import OUT, REPO, read, records, prom, save_csv


def main():
    rows, minute_rows = [], []
    for run in read(OUT / "analysis.json")["results"]:
        if not run["group"].startswith("routing"):
            continue
        root = REPO / run["root"]
        start, end = run["measurement_start_unix"], run["measurement_end_unix"]
        summary = read(root / "summary.json") if (root / "summary.json").exists() else None
        for replica in run["replicas"]:
            rid = replica["replica"]
            if summary:
                resource = next(r for r in summary["resources"] if r["role"] == "inference" and r["replica"] == rid)
                gauges = resource["gauge_samples"]
            else:
                gauges = []
                print(f"Reading full-window gauges: {run['run']} / {rid}", flush=True)
                for sample in records(root / "backends" / rid / "inference-metrics.jsonl"):
                    ts = sample["timestamp_unix"]
                    if ts >= end:
                        break
                    if ts >= start:
                        v = prom(sample)
                        gauges.append({"timestamp_unix": ts, **{k: v[k] for k in ("waiting", "running", "kv_cache")}})
            row = {"run": run["run"], "workload": run["workload"], "group": run["group"],
                   **replica, "samples": len(gauges),
                   **{f"{k}_mean": st.mean(g[k] for g in gauges) for k in ("waiting", "running", "kv_cache")}}
            rows.append(row)
            minutes = defaultdict(list)
            for g in gauges:
                minutes[int((g["timestamp_unix"] - start) // 60)].append(g)
            for minute, samples in sorted(minutes.items()):
                minute_rows.append({"run": run["run"], "replica": rid, "minute": minute,
                                    **{f"{k}_mean": st.mean(s[k] for s in samples) for k in ("waiting", "running", "kv_cache")}})
    save_csv("replica_diagnostics.csv", rows)
    save_csv("replica_timeseries.csv", minute_rows)
    (OUT / "replica_diagnostics.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
