"""Tables and standalone PNG/PDF figures for completed multi-node jobs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import write_json
from .report import csv_rows


def export(inputs: list[Path], output: Path, *, plots=True):
    roots = []
    for path in inputs:
        path = path.resolve()
        roots.extend([path] if (path / "config.json").is_file() else
                     sorted(p.parent for p in path.glob("*/config.json")))
    roots = list(dict.fromkeys(roots))
    if not roots:
        raise ValueError("no multi-node run directories found")
    results, excluded, identities, seen = [], [], {}, set()
    for root in roots:
        config = json.loads((root / "config.json").read_text())
        if "node_mapping" not in config:
            continue
        status = json.loads((root / "status.json").read_text()) if (root / "status.json").exists() else {}
        if status.get("status") != "completed" or not (root / "summary.json").exists():
            excluded.append({"root": str(root), "reason": status.get("error", "not completed")})
            continue
        result = json.loads((root / "summary.json").read_text())
        if not result["valid"]:
            excluded.append({"root": str(root), "reason": "; ".join(result["validation_errors"])})
            continue
        point = result["point"]
        key = (point["workload"], point["smoke"])
        identity = {"serve": config["serve"], "trace_paths": config["trace_paths"], "profile": config["replay_profile"],
                    "replay_runtime_policy": config.get("replay_runtime_policy"),
                    **{k: point[k] for k in ("task_cc_per_replica", "warmup_seconds", "duration_seconds", "seed")}}
        if point.get("router_impl") == "vllm-router":
            identity["router"] = config["router"]
        if key in identities and identities[key] != identity:
            raise ValueError(f"incompatible baseline/configuration for {key}; export separate comparisons")
        identities[key] = identity
        if point["id"] in seen:
            raise ValueError(f"multiple runs of {point['id']}; pass one chosen run explicitly (no implicit averaging)")
        seen.add(point["id"])
        results.append(result)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for result in results:
        p = result["point"]
        baseline = next((r for r in results if r["point"]["workload"] == p["workload"]
                         and r["point"]["inference_replicas"] == 1 and r["point"]["smoke"] == p["smoke"]), None)
        row = {"point": p["id"], "root": result["root"], "workload": p["workload"],
            "inference_nodes": p["inference_replicas"], "frontend_nodes": p["frontend_nodes"],
            "total_nodes": p["physical_nodes"], "task_cc_per_replica": p["task_cc_per_replica"],
            "layout": p["layout"], "routing": p["routing"], "prefix_reuse": p["prefix_reuse"],
            "proxy": p["proxy"], "smoke": p["smoke"], "valid": result["valid"], "steady": result["steady"],
            "router_impl": p.get("router_impl", "local" if p["proxy"] else "direct"),
            **{key: result[key] for key in ("tasks_per_second", "output_tokens_per_second", "actual_prefix_reuse", "prefix_lookup_hit_ratio",
                "native_tool_errors", "completed_trace_coverage", "trace_pool_size", "inference_gpu_joules_per_task")},
            "task_p95_s": result["task_lifecycle_seconds"]["p95"],
            "llm_p95_s": result["llm_latency_seconds"]["p95"], "ttft_p95_s": result["ttft_seconds"]["p95"],
            "queue_p95_s": result["backend_latency"]["queue_seconds"]["p95"],
            "prefill_p95_s": result["backend_latency"]["prefill_seconds"]["p95"],
            "output_tokens_per_s_per_total_node": result["output_tokens_per_second"]/p["physical_nodes"]}
        # Mechanism arms have a proxy not present in main baseline, so do not report their scale efficiency.
        for metric in ("tasks", "output_tokens"):
            base = baseline[f"{metric}_per_second"] if baseline else None
            speedup = result[f"{metric}_per_second"]/base if base and not p["proxy"] else None
            row[f"{metric}_speedup"] = speedup
            row[f"{metric}_efficiency"] = speedup/p["inference_replicas"] if speedup is not None else None
        rows.append(row)
    csv_rows(output / "summary.csv", rows)
    node_rows = []
    for result in results:
        for resource in result["resources"]:
            stats = resource["hardware_sample_statistics"]
            gpu_busy = [s["mean"] for k, s in stats.items() if k.startswith("gpu.") and k.endswith(".gpu_busy_percent")]
            node_rows.append({"point": result["point"]["id"], "node": resource["node"], "role": resource["role"],
                "cpu_busy_percent": stats.get("cpu_busy_percent", {}).get("mean"),
                "cpu_iowait_percent": stats.get("cpu_iowait_percent", {}).get("mean"),
                "memory_available_bytes": stats.get("memory_available_bytes", {}).get("mean"),
                "gpu_busy_percent": sum(gpu_busy)/len(gpu_busy) if gpu_busy else None,
                "gpu_energy_joules": resource["gpu_energy_joules"], "lustre_status": resource["lustre"]["status"],
                **{k+"_per_second": v["per_second"] for k, v in resource.get("node_io", {}).get("counters", {}).items()}})
    csv_rows(output / "resources.csv", node_rows)
    write_json(output / "summary.json", {"runs": results, "excluded_runs": excluded})
    lines = ["# Multi-node exploratory scaling", "", "One run per point; no across-run confidence intervals. "
             "Steady is a window diagnostic, not an SLO capacity claim.", "",
             "| Point | Nodes (inference + frontend) | Steady | Tasks/s | Output tokens/s | Actual reuse | Task p95 (s) |",
             "|---|---:|---|---:|---:|---:|---:|"]
    def fmt(value):
        return "—" if value is None else f"{value:.4g}" if type(value) in (int, float) else str(value)
    for row in rows:
        lines.append("| " + " | ".join(map(fmt, (row["point"], f"{row['inference_nodes']} + {row['frontend_nodes']}",
            row["steady"], row["tasks_per_second"], row["output_tokens_per_second"], row["actual_prefix_reuse"], row["task_p95_s"]))) + " |")
    if excluded:
        lines += ["", "Excluded incomplete/invalid jobs:", ""] + [f"- {r['root']}: {r['reason']}" for r in excluded]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    if plots and results:
        draw(results, rows, output, node_rows)
    return {"output": str(output), "valid_runs": len(rows), "excluded_runs": len(excluded), "plots": bool(plots and results)}


def draw(results, rows, output, node_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{name}.{extension}", dpi=170, bbox_inches="tight")
        plt.close(fig)
    for workload in sorted({r["workload"] for r in rows}):
        main = [r for r in rows if r["workload"] == workload and not r["proxy"]]
        if main:
            fig, axes = plt.subplots(2, 3, figsize=(13, 7))
            metrics = [("tasks_per_second", "Tasks/s"), ("output_tokens_per_second", "Output tokens/s"),
                ("output_tokens_efficiency", "Token scaling efficiency"), ("task_p95_s", "Task p95 (s)"),
                ("actual_prefix_reuse", "Actual prefix reuse"), ("ttft_p95_s", "Client TTFT p95 (s)")]
            for layout in ("balanced", "fixed_frontend"):
                selected = sorted((r for r in main if r["layout"] == layout or r["inference_nodes"] == 1),
                                  key=lambda r: r["inference_nodes"])
                if not selected:
                    continue
                for ax, (metric, label) in zip(axes.flat, metrics):
                    ax.plot([r["inference_nodes"] for r in selected], [r[metric] for r in selected], "o-", label=layout)
                    unstable = [r for r in selected if not r["steady"] and r[metric] is not None]
                    ax.scatter([r["inference_nodes"] for r in unstable], [r[metric] for r in unstable], marker="x", c="red", zorder=5)
                    ax.set(xlabel="Inference nodes (one TP4 replica each)", ylabel=label)
                    ax.set_xscale("log", base=2)
                    ax.set_xlim(min(r["inference_nodes"] for r in main)*.8,
                                max(r["inference_nodes"] for r in main)*1.25)
                    ax.set_xticks(sorted({r["inference_nodes"] for r in main}), labels=sorted({r["inference_nodes"] for r in main}))
                    ax.grid(alpha=.2)
            axes.flat[0].legend()
            fig.suptitle(f"{workload}: one run per point; red × = window not steady")
            save(fig, f"{workload}-scaling")
        arms = [r for r in rows if r["workload"] == workload and r["proxy"] and not r["smoke"]]
        if arms:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            labels = [f"{r['routing']}\n{r['inference_nodes']} backends / {r['prefix_reuse']}" for r in arms]
            for ax, (metric, label) in zip(axes, [("output_tokens_per_second", "Output tokens/s"),
                                                ("actual_prefix_reuse", "Actual prefix reuse"), ("ttft_p95_s", "TTFT p95 (s)")]):
                ax.bar(labels, [r[metric] if r[metric] is not None else float("nan") for r in arms])
                ax.set_ylabel(label)
                ax.tick_params(axis="x", labelsize=8)
            topologies = sorted({(r["inference_nodes"], r["frontend_nodes"]) for r in arms})
            topology = "; ".join(f"{n} inference + {f} frontend" for n, f in topologies)
            official = all(r["router_impl"] == "vllm-router" for r in arms)
            fig.suptitle(f"{workload}: {topology}; {'official vLLM Router' if official else 'proxy comparison'}")
            save(fig, f"{workload}-{'routing' if official else 'mechanism'}")
    for result in results:
        name = result["point"]["id"]
        with (Path(result["root"]) / "timeseries.csv").open() as handle:
            series = list(csv.DictReader(handle))
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for ax, (metric, label) in zip(axes, [("output_tokens_per_second", "Output tokens/s"),
                                              ("tasks_per_second", "Tasks/s"), ("actual_prefix_reuse", "Actual prefix reuse")]):
            ax.plot([(float(r["start_unix"])-float(series[0]["start_unix"]))/60 for r in series],
                    [float(r[metric]) if r[metric] else float("nan") for r in series])
            ax.set(xlabel="Measurement elapsed (min)", ylabel=label)
        save(fig, f"{name}-timeseries")
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        replicas = result["replicas"]
        axes[0].bar([r["replica"] for r in replicas], [r["output_tokens_per_second"] for r in replicas])
        axes[0].set_ylabel("Output tokens/s per replica")
        bottom = 0
        for key, value in result["execution_seconds_mean"].items():
            axes[1].bar(["Admission cohort"], [value], bottom=bottom, label=key)
            bottom += value
        axes[1].set_ylabel("Mean task lifecycle (s)")
        if result["execution_seconds_mean"]:
            axes[1].legend(fontsize=7)
        for resource in result["resources"]:
            if resource["role"] == "inference":
                samples = resource["gauge_samples"]
                axes[2].plot([(s["timestamp_unix"]-result["measurement"]["measurement_start_unix"])/60 for s in samples],
                             [s["kv_cache"] for s in samples], label=resource["replica"], alpha=.7)
        axes[2].set(xlabel="Measurement elapsed (min)", ylabel="KV cache usage (fraction)")
        fig.suptitle(name)
        save(fig, f"{name}-breakdown")
        resources = [r for r in node_rows if r["point"] == name]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        labels = [f"{r['role'][0]}:{r['node']}" for r in resources]
        for ax, (key, label) in zip(axes, [("cpu_busy_percent", "Node CPU busy (%)"),
                ("gpu_busy_percent", "Mean GPU busy (%)"), ("memory_available_bytes", "Available host memory (GiB)")]):
            values = [r[key] if r[key] is not None else float("nan") for r in resources]
            if key == "memory_available_bytes":
                values = [v/2**30 for v in values]
            ax.bar(labels, values, color=["C0" if r["role"] == "inference" else "C1" for r in resources])
            ax.set_ylabel(label)
            ax.tick_params(axis="x", labelrotation=90, labelsize=6)
        save(fig, f"{name}-resources")
