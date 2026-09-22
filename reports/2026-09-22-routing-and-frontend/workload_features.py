"""Small offline calculations supporting the workload-specific interpretation.

Uses existing derived CSV/JSON only; no raw-run scan or experiment execution.
"""
import csv
import json
import math
import statistics as st
from pathlib import Path

OUT = Path(__file__).resolve().parent
TRACE = OUT.parent / "2026-09-08-expanded-trace-analysis"


def read(path):
    return json.loads(path.read_text())


def main():
    with (TRACE / "tasks.csv").open() as stream:
        tasks = [r for r in csv.DictReader(stream) if r["scale_selected"].lower() in ("true", "1")]
    trace_rows = []
    for workload in ("openclaw", "minisweagent"):
        rows = [r for r in tasks if r["workload"] == workload]
        k = math.ceil(.1 * len(rows))
        top = sorted(rows, key=lambda r: float(r["prompt_qwen36_total_tokens"]), reverse=True)[:k]
        total = lambda selected, key: sum(float(r[key]) for r in selected)
        trace_rows.append({
            "workload": workload, "traces": len(rows), "top_input_trace_count": k,
            "top_input_trace_fraction": k / len(rows),
            "top_input_trace_ids": [r["trace_id"] for r in top],
            "top_input_trace_input_share": total(top, "prompt_qwen36_total_tokens") / total(rows, "prompt_qwen36_total_tokens"),
            "top_input_trace_output_share": total(top, "output_recorded_total_tokens") / total(rows, "output_recorded_total_tokens"),
            "top_input_trace_call_share": total(top, "llm_calls") / total(rows, "llm_calls"),
            "median_sum_prompt_div_max_prompt": st.median(float(r["prompt_qwen36_total_tokens"]) / float(r["max_prompt_qwen36_tokens"]) for r in rows),
            "llm_calls": int(total(rows, "llm_calls")),
        })

    results = read(OUT / "analysis.json")["results"]
    replicas = read(OUT / "replica_diagnostics.json")
    occupancy = []
    for group in ("fixed_sticky", "frontend_f1_build8", "frontend_f2_build2"):
        r = next(r for r in results if (r["workload"], r["group"], r["backends"]) == ("minisweagent", group, 8))
        rs = [v for v in replicas if v["run"] == r["run"]]
        fraction = r["execution_seconds_mean"]["llm_only"] / r["task_lifecycle_seconds"]["mean"]
        occupancy.append({
            "run": r["run"], "group": group, "task_cc": r["total_cc"], "backends": r["backends"],
            "llm_lifecycle_fraction": fraction,
            "estimated_client_llm_occupancy_per_backend": r["total_cc"] * fraction / r["backends"],
            "observed_running_per_backend": st.mean(v["running_mean"] for v in rs),
            "observed_waiting_per_backend": st.mean(v["waiting_mean"] for v in rs),
        })

    rr = next(r for r in results if (r["workload"], r["group"], r["backends"]) == ("openclaw", "routing_rr", 8))
    rs = [r for r in replicas if r["run"] == rr["run"]]
    hot = max(rs, key=lambda r: r["running_mean"] + r["waiting_mean"])
    arrivals = [r for r in read(OUT / "routing_arrivals.json") if r["run"] == rr["run"]]
    hot_arrivals = next(r["arrivals"] for r in arrivals if r["replica"] == hot["replica"])
    hotspot = {"run": rr["run"], "replica": hot["replica"],
               "arrival_share": hot_arrivals / sum(r["arrivals"] for r in arrivals),
               "share_of_summed_mean_running_and_waiting": (hot["running_mean"] + hot["waiting_mean"]) / sum(r["running_mean"] + r["waiting_mean"] for r in rs)}
    report = {
        "scope": "Selected trace pool once per trace; occupancy and hotspot from previously checked measurement summaries. P2 excluded.",
        "trace_concentration": trace_rows, "phase_occupancy": occupancy, "rr_n8_hotspot": hotspot,
        "interpretation": {
            "trace_concentration": "Top ceil(10% of traces), ranked by cumulative submitted Qwen input. Output is recorded replay output budget. Not uncached compute or replay-window weights.",
            "sum_prompt_div_max_prompt": "Repeated input-volume descriptor, not exact shared prefix, unique information, or attainable cache savings.",
            "phase_occupancy": "Approximate stage-occupancy explanation for mostly serial calls with a full task population and stable mix; client lifecycle cohorts differ from backend wall-clock gauges. Not an exact identity or fitted predictor.",
            "hotspot": "Ratio of full-window per-backend gauge means, not share of task count and not proof of steady state.",
        },
    }
    (OUT / "workload_features.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"trace_concentration": [{k:v for k,v in r.items() if k != "top_input_trace_ids"} for r in trace_rows],
                      "phase_occupancy": occupancy, "rr_n8_hotspot": hotspot}, indent=2))


if __name__ == "__main__":
    main()
