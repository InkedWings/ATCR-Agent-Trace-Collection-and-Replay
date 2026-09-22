"""Read the completed OpenClaw cc32 budget comparison; no inference or tool calls."""
from __future__ import annotations

import csv
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/agenttrace-matplotlib")
OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
BASE = REPO / "runs/scaling/baseline-validation-7608115-20260913"
spec = importlib.util.spec_from_file_location("previous_analysis", REPO / "reports/2026-09-08-openclaw-qwen36-scale/analyze.py")
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text())


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    summaries, series, details = [], [], {}
    reference_config = reference_manifest = None
    for budget in (2048, 8192):
        run = BASE / f"openclaw-warmed-budget{budget}"
        point = run / "openclaw-cc32"
        report = read(point / "benchmark/summary.json")
        manifest = read(point / "benchmark/manifest.json")
        config = read(run / "config.json")
        m = report["measurement"]
        start, end = m["started_unix"], m["ended_unix"]
        assert report["status"] == "completed" and m["valid"] and end-start == 1800
        for key in ("backend_client_counts_match", "counter_counts_match", "inference_gpu_samples_ok", "window_has_backend_tokens"):
            assert report["validation"][key]
        assert config["serve"].pop("VLLM_MAX_NUM_BATCHED_TOKENS") == str(budget)
        if reference_config is None:
            reference_config, reference_manifest = config, manifest
        else:
            assert config == reference_config
            for key in ("trace_pool", "seed", "concurrency", "warmup_seconds", "duration_seconds", "profile_path", "llm_replay"):
                assert manifest[key] == reference_manifest[key]
        counters, bins, span = previous.raw_metrics(point / "benchmark/metrics.jsonl", start, end, 32, report["tasks"])
        series.extend({"budget": budget, **row} for row in bins)
        counts, admissions, scheduler_config, examples = Counter(), [], None, {}
        with (point / "backend-scheduler.jsonl").open() as handle:
            for line in handle:
                event = json.loads(line)
                if event["event"] == "scheduler_config":
                    scheduler_config = event
                if not start <= event["timestamp_unix"] < end:
                    continue
                if event["event"] == "scheduler_interval":
                    counts.update(event["counts"])
                    for key, value in event["examples"].items():
                        examples.setdefault(key, value)
                elif event["event"] == "cache_at_admission":
                    admissions.append(event)
        mamba_reductions = sum(any(g["manager"] == "MambaManager" and g["hit_tokens"] < g["candidate_tokens"]
            for g in row["last"]["groups"]) for row in admissions)
        gauges = {key.split("{", 1)[0]: value for key, value in m["metrics"]["vllm_gauge_sample_statistics"].items()}
        kv = []
        with (point / "benchmark/metrics.jsonl").open() as handle:
            for line in handle:
                sample = json.loads(line)
                if start <= sample["timestamp_unix"] < end:
                    values = dict(previous.PROM_RE.findall(sample["vllm_prometheus"]))
                    kv.append(float(values["kv_cache_usage_perc"]))
        summaries.append({"budget": budget, "cc": 32, "measurement_seconds": 1800,
            "output_tokens_per_s": m["backend_throughput"]["window_output_tokens_per_second"],
            "completed_tasks": m["completed_in_window"], "tasks_per_minute": 60*m["task_throughput_per_second"],
            "actual_prompt_reuse_percent": 100*counters["prompt_tokens_cached_total"]/counters["prompt_tokens_total"],
            "prefix_lookup_hit_percent": 100*counters["prefix_cache_hits_total"]/counters["prefix_cache_queries_total"],
            "kv_usage_mean_percent": 100*gauges["vllm:kv_cache_usage_perc"]["mean"],
            "kv_usage_min_percent": 100*min(kv),
            "kv_below_40_percent_sample_fraction": sum(value < .4 for value in kv)/len(kv),
            "running_mean": gauges["vllm:num_requests_running"]["mean"],
            "waiting_mean": gauges["vllm:num_requests_waiting"]["mean"],
            "queue_p95_seconds": m["backend_throughput"]["queue_seconds"]["p95"],
            "prefill_p95_seconds": m["backend_throughput"]["prefill_seconds"]["p95"],
            "ttft_p95_seconds": m["llm_ttft_seconds"]["p95"],
            "setup_p50_seconds": m["task_setup_seconds"]["p50"],
            "compute_prompt_tokens_per_s": (counters["prompt_tokens_total"]-counters["prompt_tokens_cached_total"])/span,
            "num_gpu_blocks": scheduler_config["num_gpu_blocks"],
            "alignment_zero_waiting": counts["alignment_zero_waiting"],
            "alignment_zero_waiting_per_step": counts["alignment_zero_waiting"]/counts["steps"],
            "full_isl_rejections": counts["full_isl_rejections"],
            "allocation_failures": sum(value for key, value in counts.items() if key.startswith("allocation_failures_")),
            "running_limit_steps": counts["steps_ending_with_waiters_and_running_limit"],
            "preemptions": counters["num_preemptions_total"],
            "admissions_with_mamba_candidate_reduction": mamba_reductions,
            "sample_span_seconds": span})
        details[str(budget)] = {"source": str(point.relative_to(REPO)), "validation": report["validation"],
            "scheduler_config": scheduler_config, "scheduler_counts": dict(counts), "blocking_examples": examples,
            "admissions_in_window": len(admissions), "metric_samples": len(kv)}
    write_csv("summary.csv", summaries)
    write_csv("timeseries_60s.csv", series)
    (OUT / "diagnostics.json").write_text(json.dumps(details, indent=2)+"\n")
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    for budget, color in ((2048, "#D55E00"), (8192, "#0072B2")):
        rows = [row for row in series if row["budget"] == budget]
        for ax, key, label in zip(axes,
                ("kv_mean_percent", "prompt_reuse_percent", "backend_waiting_mean", "output_tokens_per_s"),
                ("Active KV blocks (%)", "Actual prompt reuse (%)", "Waiting requests", "Output tokens / s")):
            ax.plot([r["minute_start"]+.5 for r in rows], [r[key] for r in rows], color=color, label=str(budget), linewidth=1.8)
            ax.set_ylabel(label)
            ax.grid(alpha=.2)
    axes[0].legend(title="Batch token budget")
    axes[0].set_title("OpenClaw cc32: 2048 vs 8192 (same pool and seed, 60 s bins)")
    axes[-1].set_xlabel("Minutes since measurement start")
    fig.savefig(OUT / "budget_comparison.png", dpi=170)
    fig.savefig(OUT / "budget_comparison.pdf")
    plt.close(fig)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
