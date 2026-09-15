"""Read saved weak-scaling results; never start inference or replay tools.

Default: extract and check raw counts, then regenerate tables and figures.
--plots-only: regenerate from the committed analysis.json, without raw runs.
Only throughput is projected for explicitly null entries in sources.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics as stats
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/agenttrace-matplotlib")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
WORKLOADS = ["openclaw", "minisweagent"]
LABELS = {"openclaw": "OpenClaw", "minisweagent": "mini-SWE"}
COLORS = {"openclaw": "#3366AA", "minisweagent": "#D16B31"}
NODES = [2, 4, 8, 16, 32]
GAP_RUN = "openclaw-balanced-n2-7616564.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov"


def read(path):
    return json.loads(path.read_text())


def write_csv(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    values = [v for v in values if v is not None]
    return stats.mean(values) if values else None


def tool_metadata(paths):
    result = {}
    for path in paths:
        trace = read(Path(path))
        for node in trace["nodes"]:
            if node["type"] == "tool":
                result[trace["trace_id"], node["id"]] = bool(
                    (node.get("recorded_result") or {}).get("isError", False))
    return result


def audit_calls(root, summary, metadata):
    start = summary["measurement"]["measurement_start_unix"]
    end = summary["measurement"]["measurement_end_unix"]
    all_calls, window_tools = Counter(), Counter()
    with (root / "calls.csv").open() as handle:
        for call in csv.DictReader(handle):
            kind = call["type"]
            all_calls[kind] += 1
            all_calls["unfinished"] += call["status"] != "completed"
            if kind == "llm":
                output = int(call["actual_output_tokens"])
                all_calls["output_tokens"] += output
                all_calls["target_mismatches"] += output != int(call["target_output_tokens"])
            elif kind == "tool":
                before = metadata[call["trace_id"], call["node_id"]]
                after = call["native_error"].lower() == "true"
                key = ("error" if before else "success") + "_to_" + ("error" if after else "success")
                all_calls[key] += 1
                if start <= float(call["started_unix"]) < end:
                    window_tools["calls"] += 1
                    window_tools[key] += 1
                    window_tools["native_errors"] += after
    with (root / "tasks.csv").open() as handle:
        tasks = list(csv.DictReader(handle))
    checks = {
        "calls_completed": all_calls["unfinished"] == 0,
        "target_outputs_match": all_calls["target_mismatches"] == 0,
        "backend_requests_match": all_calls["llm"] == sum(r["requests"] for r in summary["replicas"]),
        "backend_outputs_match": all_calls["output_tokens"] == sum(r["all_output_tokens"] for r in summary["replicas"]),
        "tasks_completed": all(t["status"] == "completed" and t["returncode"] == "0" for t in tasks),
        "window_completed_tasks_match": sum(start <= float(t["ended_unix"]) < end for t in tasks) == summary["completed_in_window"],
        "window_admitted_tasks_match": sum(start <= float(t["started_unix"]) < end for t in tasks) == summary["admitted_in_window"],
        "tool_error_count_matches": window_tools["native_errors"] == summary["native_tool_errors"],
    }
    workers = [read(p) for p in sorted((root / "workers").glob("*/summary.json"))]
    checks["all_workers_completed"] = len(workers) == summary["point"]["inference_replicas"] and all(
        w["status"] == "completed" and w["measurement"]["valid"] for w in workers)
    checks["worker_tasks_match"] = sum(w["completed"] for w in workers) == len(tasks)
    checks["worker_tokens_match"] = sum(w["actual_output_tokens"] for w in workers) == all_calls["output_tokens"]
    resources = summary["resources"]
    checks["resource_samples_no_errors"] = all(r["error_samples"] == 0 for r in resources)
    checks["backend_counters_no_resets"] = all(not r.get("whole_run_counters", {}).get("reset_or_missing_series") for r in resources)
    checks["inference_gpu_counts_ok"] = all(r["inference_gpu_count_ok"] for r in resources if r["role"] == "inference")
    assert all(checks.values()), (root.name, checks)
    return dict(checks=checks, all_calls=dict(all_calls), window_tools=dict(window_tools),
                tasks=len(tasks), drain_seconds=max(float(t["ended_unix"]) for t in tasks)-end)


def extract():
    sources = read(OUT / "sources.json")
    result = dict(extracted_at_utc=datetime.now(timezone.utc).isoformat(), rows=[], replicas=[], timeseries=[], audits=[])
    for workload in WORKLOADS:
        baseline_cfg = None
        metadata = None
        for n_text, source in sources[workload].items():
            n = int(n_text)
            if source is None:
                continue
            root = REPO / source
            s, cfg = read(root / "summary.json"), read(root / "config.json")
            p = s["point"]
            assert (p["workload"], p["inference_replicas"], p["physical_nodes"]) == (workload, n, 2*n)
            if baseline_cfg is None:
                assert n == 1
                baseline_cfg = cfg
                metadata = tool_metadata(cfg["trace_paths"])
            for key in ["serve", "trace_paths", "replay_profile"]:
                assert cfg[key] == baseline_cfg[key], (source, key)
            for key in ["task_cc_per_replica", "warmup_seconds", "duration_seconds", "seed", "routing", "prefix_reuse", "proxy"]:
                assert p[key] == baseline_cfg["point"][key], (source, key)
            assert s["duration_seconds"] == p["duration_seconds"]
            assert s["completed_trace_coverage"] == s["trace_pool_size"] == len(cfg["trace_paths"])
            audit = audit_calls(root, s, metadata)
            gaps = [r["node"] for r in s["resources"] if not r["window_coverage_ok"]]
            gap_exception = root.name == GAP_RUN
            if gap_exception:
                assert gaps == ["x3209c0s37b0n0"]
                assert s["validation_errors"] == ["x3209c0s37b0n0: missing/failed metric samples"]
            else:
                assert s["valid"] and read(root / "status.json")["status"] == "completed", source
            audit.update(workload=workload, physical_nodes=2*n, source_run=source,
                         raw_valid=s["valid"], raw_steady=s["steady"], raw_errors=s["validation_errors"],
                         diagnostic_warnings=s["diagnostic_warnings"], resource_gap_nodes=gaps,
                         core_counts_usable=True, sampling_gap_exception=gap_exception,
                         same_serve_pool_profile_and_window=True)
            result["audits"].append(audit)
            measured = dict(workload=workload, physical_nodes=2*n, inference_replicas=n, frontend_nodes=n,
                            source_kind="measured", source_run=source, raw_valid=s["valid"], raw_steady=s["steady"],
                            quality_note="frontend sampling gap; main counts checked" if gap_exception else "passed",
                            cc_per_replica=p["task_cc_per_replica"], total_cc=n*p["task_cc_per_replica"],
                            warmup_seconds=p["warmup_seconds"], measurement_seconds=s["duration_seconds"],
                            completed_tasks=s["completed_in_window"], output_tokens=s["output_tokens"],
                            output_tokens_s=s["output_tokens_per_second"], tasks_min=60*s["tasks_per_second"],
                            per_replica_tokens_s=s["output_tokens_per_second"]/n,
                            prompt_reuse_pct=100*s["actual_prefix_reuse"],
                            ttft_p95_s=s["ttft_seconds"]["p95"], task_mean_s=s["task_lifecycle_seconds"]["mean"],
                            task_p50_s=s["task_lifecycle_seconds"]["p50"], task_p95_s=s["task_lifecycle_seconds"]["p95"],
                            queue_mean_s=s["backend_latency"]["queue_seconds"]["mean"],
                            queue_p95_s=s["backend_latency"]["queue_seconds"]["p95"],
                            token_half_change_pct=100*s["stability"]["output_rate_relative_change"],
                            task_half_change_pct=100*s["stability"]["task_rate_relative_change"],
                            reuse_half_change_pp=100*s["stability"]["reuse_absolute_change"],
                            stability_checks_pass=all(s["stability"]["checks"].values()),
                            tool_calls=audit["window_tools"]["calls"], native_tool_errors=audit["window_tools"]["native_errors"],
                            new_tool_errors=audit["window_tools"].get("success_to_error", 0),
                            recovered_tool_errors=audit["window_tools"].get("error_to_success", 0),
                            drain_seconds=audit["drain_seconds"])
            measured["new_tool_error_pct"] = 100*measured["new_tool_errors"]/measured["tool_calls"]
            measured["native_tool_error_pct"] = 100*measured["native_tool_errors"]/measured["tool_calls"]
            measured.update({"execution_"+k+"_s": v for k,v in s["execution_seconds_mean"].items()})
            assert math.isclose(sum(s["execution_seconds_mean"].values()), measured["task_mean_s"], abs_tol=.01)
            infer = [r for r in s["resources"] if r["role"] == "inference"]
            front = [r for r in s["resources"] if r["role"] == "frontend"]
            for key, needle, scale in [("kv_cache_pct", "kv_cache_usage_perc", 100), ("waiting_mean", "num_requests_waiting", 1), ("running_mean", "num_requests_running", 1)]:
                measured[key] = scale*mean(v["mean"] for r in infer for k,v in r["vllm_gauge_sample_statistics"].items() if needle in k)
            for role, resources in [("inference", infer), ("frontend", front)]:
                measured[role+"_cpu_pct"] = mean(r["hardware_sample_statistics"]["cpu_busy_percent"]["mean"] for r in resources)
                measured[role+"_iowait_pct"] = mean(r["hardware_sample_statistics"]["cpu_iowait_percent"]["mean"] for r in resources)
            measured["gpu_busy_pct"] = mean(v["mean"] for r in infer for k,v in r["hardware_sample_statistics"].items() if k.startswith("gpu.") and k.endswith(".gpu_busy_percent"))
            measured["gpu_memory_gib"] = mean(v["mean"]/1024 for r in infer for k,v in r["hardware_sample_statistics"].items() if k.startswith("gpu.") and k.endswith(".memory_used_mib"))
            # A frontend sampling gap affects that frontend's resource statistics, not inference GPU energy.
            energy = s.get("inference_gpu_energy_joules")
            measured["inference_gpu_joules_per_output_token"] = energy/s["output_tokens"] if energy is not None else None
            rates = [r["output_tokens_per_second"] for r in s["replicas"]]
            measured["replica_cv_pct"] = 100*stats.pstdev(rates)/mean(rates)
            for rep in s["replicas"]:
                result["replicas"].append(dict(workload=workload,physical_nodes=2*n,replica=rep["replica"],
                    output_tokens_s=rep["output_tokens_per_second"],prompt_reuse_pct=100*rep["actual_prefix_reuse"]))
            stages = [read(f) for f in (root/"frontends").glob("*/image-cache.json")]
            if workload == "minisweagent":
                assert len(stages) == n and all(x["images"] == 61 and x["copied"]+x["reused"] == 61 for x in stages)
                measured.update(sif_images_per_frontend=61, sif_stage_max_s=max(x["elapsed_seconds"] for x in stages))
            start = s["measurement"]["measurement_start_unix"]
            with (root/"timeseries.csv").open() as handle:
                series = list(csv.DictReader(handle))
            assert sum(int(x["output_tokens"]) for x in series) == s["output_tokens"]
            assert sum(int(x["completed_tasks"]) for x in series) == s["completed_in_window"]
            for point in series:
                result["timeseries"].append(dict(workload=workload,physical_nodes=2*n,
                    minute=(float(point["start_unix"])-start)/60+.5,
                    per_replica_tokens_s=float(point["output_tokens_per_second"])/n,
                    prompt_reuse_pct=100*float(point["actual_prefix_reuse"])))
            result["rows"].append(measured)
            print(f"Checked {workload}, {2*n} physical nodes: {measured['output_tokens_s']:.2f} tokens/s", flush=True)
        base = next(r for r in result["rows"] if r["workload"] == workload and r["inference_replicas"] == 1)
        for n_text, source in sources[workload].items():
            if source is not None:
                continue
            n = int(n_text)
            result["rows"].append(dict(workload=workload,physical_nodes=2*n,inference_replicas=n,frontend_nodes=n,
                source_kind="theoretical",source_run="",quality_note="NOT MEASURED: N=1 throughput multiplied by N",
                cc_per_replica=base["cc_per_replica"],total_cc=n*base["cc_per_replica"],
                output_tokens_s=n*base["output_tokens_s"],tasks_min=n*base["tasks_min"],per_replica_tokens_s=base["output_tokens_s"]))
        for r in result["rows"]:
            if r["workload"] == workload:
                r["token_speedup"] = r["output_tokens_s"]/base["output_tokens_s"]
                r["token_efficiency_pct"] = 100*r["token_speedup"]/r["inference_replicas"]
                r["task_speedup"] = r["tasks_min"]/base["tasks_min"]
                r["task_efficiency_pct"] = 100*r["task_speedup"]/r["inference_replicas"]
    result["rows"].sort(key=lambda r:(WORKLOADS.index(r["workload"]), r["physical_nodes"]))
    (OUT/"analysis.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def table(headers, rows):
    return "\n".join(["| "+" | ".join(headers)+" |", "| "+" | ".join(["---"]*len(headers))+" |"] + ["| "+" | ".join(map(str,r))+" |" for r in rows])


def tables(data):
    write_csv("summary.csv", data["rows"])
    write_csv("replicas.csv", data["replicas"])
    write_csv("timeseries.csv", data["timeseries"])
    sections = ["# Weak-scaling 指标表", "单位：物理节点 = 推理节点 + frontend，二者数量相同。`理论占位` 的吞吐来自同配置 N=1 基线 × 推理副本数；其他未实测指标留空。"]
    for w in WORKLOADS:
        rows = [r for r in data["rows"] if r["workload"] == w]
        sections += ["## "+LABELS[w], table(["物理节点", "数据", "输出 tokens/s", "任务/min", "Token speedup", "Token 效率", "Task 效率"],
            [[r["physical_nodes"],"理论占位" if r["source_kind"]=="theoretical" else ("实测†" if not r["raw_valid"] else "实测"),
              f"{r['output_tokens_s']:.2f}",f"{r['tasks_min']:.2f}",f"{r['token_speedup']:.3f}×",f"{r['token_efficiency_pct']:.2f}%",f"{r['task_efficiency_pct']:.2f}%"] for r in rows])]
        measured = [r for r in rows if r["source_kind"] == "measured"]
        sections += [table(["物理节点", "TTFT p95 (s)", "任务 p95 (s)", "Prompt 复用", "KV 使用率", "Waiting/副本", "GPU busy", "Frontend CPU", "新增工具错误"],
            [[r["physical_nodes"],f"{r['ttft_p95_s']:.3f}",f"{r['task_p95_s']:.2f}",f"{r['prompt_reuse_pct']:.2f}%",f"{r['kv_cache_pct']:.2f}%",f"{r['waiting_mean']:.3f}",f"{r['gpu_busy_pct']:.2f}%",f"{r['frontend_cpu_pct']:.2f}%",f"{r['new_tool_error_pct']:.3f}%"] for r in measured]),
            table(["物理节点", "全程完成任务", "全程 LLM calls", "全程 tool calls", "全程输出 tokens", "半窗口吞吐变化", "副本吞吐 CV"],
            [[r["physical_nodes"],a["tasks"],a["all_calls"]["llm"],a["all_calls"]["tool"],a["all_calls"]["output_tokens"],f"{r['token_half_change_pct']:.2f}%",f"{r['replica_cv_pct']:.2f}%"] for r in measured for a in data["audits"] if a["workload"]==w and a["physical_nodes"]==r["physical_nodes"]])]
    sections += ["† OpenClaw 4 物理节点原始 valid/steady=false，原因是单个 frontend 的一次 5.59 秒采样间隔。主计数、任务完成、后端计数及时间稳定性检查通过；保留原始状态，该 frontend 的资源均值存在采样缺口。", "实测点均为一次运行。副本差异和分钟波动不等同于重复实验的置信区间。"]
    (OUT/"tables.md").write_text("\n\n".join(sections)+"\n")


def report(data):
    measured=[r for r in data["rows"] if r["source_kind"]=="measured"]
    projected=[r for r in data["rows"] if r["source_kind"]=="theoretical"]
    largest={w:max((r for r in measured if r["workload"]==w),key=lambda r:r["physical_nodes"]) for w in WORKLOADS}
    overview=table(["Workload", "物理节点", "输出 tokens/s", "任务/min", "Token speedup", "Token 效率", "Task 效率"],
        [[LABELS[w],r["physical_nodes"],f"{r['output_tokens_s']:.2f}",f"{r['tasks_min']:.2f}",f"{r['token_speedup']:.3f}×",f"{r['token_efficiency_pct']:.2f}%",f"{r['task_efficiency_pct']:.2f}%"] for w,r in largest.items()])
    diagnostics=[]
    for w in WORKLOADS:
        rows=[r for r in measured if r["workload"]==w]
        r=largest[w];base=min(rows,key=lambda r:r["physical_nodes"])
        diagnostics.append(f"- **{LABELS[w]}**：最大规模每推理副本 {r['per_replica_tokens_s']:.2f} tokens/s，基线 {base['per_replica_tokens_s']:.2f}；"
            f"复用率 {base['prompt_reuse_pct']:.2f}% → {r['prompt_reuse_pct']:.2f}%；TTFT p95 {base['ttft_p95_s']:.3f} → {r['ttft_p95_s']:.3f} 秒；"
            f"任务 p95 {base['task_p95_s']:.1f} → {r['task_p95_s']:.1f} 秒。各档半窗口 token 吞吐变化最大 {max(x['token_half_change_pct'] for x in rows):.2f}%，"
            f"副本吞吐 CV 最大 {max(x['replica_cv_pct'] for x in rows):.2f}%。")
    if projected:
        missing="、".join(f"{LABELS[r['workload']]} {r['physical_nodes']} 物理节点" for r in projected)
        pending=f"**理论占位：{missing}，尚无实测数据。** 按同 workload 的 2 物理节点实测基线乘以推理副本数计算 token/task 吞吐。吞吐图用空心菱形区分；CSV/JSON 标记 `source_kind=theoretical`，原始 run 路径为空。没有估造延迟、cache、资源、错误率、任务计数或时间序列；占位值不参与下面的实测结论。"
    else:
        pending="所有选定档位均有实测数据，目前没有理论占位。"
    counts=table(["Workload", "物理节点", "测量期 tool calls", "Native errors", "原成功→重放错误", "新增错误率", "原错误→重放成功"],
        [[LABELS[r["workload"]],r["physical_nodes"],r["tool_calls"],r["native_tool_errors"],r["new_tool_errors"],f"{r['new_tool_error_pct']:.3f}%",r["recovered_tool_errors"]] for r in measured])
    text=f"""# Balanced weak scaling：OpenClaw 与 mini-SWE

数据提取时间：{data['extracted_at_utc']}。

**当前实测结果支持：在本次 balanced 布局和固定每副本并发下，两类任务扩到 32 个物理节点，token 吞吐仍接近线性增长，没有出现此前单节点高并发时的持续吞吐塌陷。** 每个规模只有一次运行，不把略高于 100% 的效率解读为超线性加速，也不据此宣称已找到最大容量。

{pending}

## 核心结果

{overview}

效率 = 本档吞吐 /（同配置 2 物理节点基线吞吐 × 推理副本数）；基线已使用本轮新完成的 8192 长窗口结果，未混用旧 2048 短窗口数据。完整各档数据见 [指标表](tables.md) 和 [summary.csv](summary.csv)。

![吞吐与理论线](01_throughput.png)

{chr(10).join(diagnostics)}

任务吞吐与 token 吞吐的效率并不完全相同：不同运行窗口中的任务组成、完成边界和原生工具执行结果会影响 tasks/min。不能用接近线性的 token 吞吐推断任务延迟完全不变。

## 配置和统计口径

- 横轴均为**物理节点数 2/4/8/16/32**，分别对应 1/2/4/8/16 个推理副本，加相同数量 frontend。不是 2/4/8/16/32 个推理副本。
- Qwen3.6-35B-A3B；每推理节点 TP4、4 GPU；context 262144；max_num_seqs 64；max_num_batched_tokens 8192；GPU memory utilization 0.90；prefix caching 和 thinking 开启。
- OpenClaw：每副本 cc16，110 条 trace，warmup 600 秒、measurement 3600 秒。
- mini-SWE：每副本 cc32，61 条 trace，warmup 1800 秒、measurement 5400 秒。各 frontend 均完成全部 61 个共享 SIF 到本地缓存的准备/复用检查。
- 同 workload 各档的 serve、完整 trace 路径列表、replay profile、seed 规则、warmup/measurement 与基线一致。每副本使用相同 trace pool、seed 随副本变化并循环重放；sticky 路由到对应后端，prefix cache 在各后端本地。该实验评估成比例增加 frontend 和独立推理副本的扩展性，不是单请求跨节点加速或全局共享 KV cache。
- Token 吞吐为共同测量窗口内产生的增量输出 token / 窗口秒数；task 吞吐为窗口内完成的重放数 / 时间。任务延迟使用窗口内接纳、允许自然 drain 完成的 cohort；TTFT、工具错误率按窗口内开始的调用计数。
- Prompt 复用率 = cached prompt tokens / prompt tokens，按窗口内首 token 请求加权；与 KV cache 使用率、GPU 显存占用不同。硬件和后端 gauge 指标先对节点内样本求均值、再对相同角色节点求均值。能耗只统计推理 GPU，未包含 frontend 或整机功耗。

## 数据完整性与限制

本次核对 {len(measured)} 个实测点：任务和调用均完整结束、目标输出 token 与实际输出一致，全程 LLM 请求/token 数与后端计数一致，测量窗口计数与时间序列一致；各档时间稳定性子检查均通过。检查详情和保留的原始状态在 [analysis.json](analysis.json) 的 `audits` 中。

OpenClaw **4 物理节点**是保留的例外：旧 runner 因一个 frontend 的一次 5.59 秒采样间隔将整档标成 `valid=false/steady=false`。此次确认没有失败采样、计数缺失或后端 counter reset，主吞吐可用；原始标记和错误文本保持不变，该 frontend 的资源均值存在采样缺口。其他选定实测点原始 valid/steady 均为 true。

原生工具执行不保证与录制结果逐条相同。下表将全部 native errors 与相对录制新增的错误分开；后者会影响工具耗时和端到端吞吐的解释。这里的“任务完成”指 trace 重放结束，不代表 SWE 题目修复成功。

{counts}

高 cache 复用是在固定 trace pool、循环重放和充分 warmup 下得到的；结果不能直接外推到不断出现的新任务或冷 cache 工作负载。副本差异、分钟波动均不是独立重复实验的置信区间。

## 图表

- [吞吐](01_throughput.png)：两种 workload 的 token/task 吞吐、理论参考线和明确标记的占位值。
- [效率与副本均衡](02_efficiency_and_balance.png)：实测扩展效率及每副本吞吐散点。
- [延迟与 cache](03_latency_and_cache.png)：TTFT、任务 p95、prompt 复用和 KV 使用率。
- [资源与队列](04_resources.png)：GPU busy、frontend CPU、waiting 和推理 GPU 单 token 能耗。
- [执行时间与工具错误](05_execution_and_tool_errors.png)：平均任务时间的堆叠柱图，以及原生/新增工具错误率。
- [时间稳定性](06_time_stability.png)：每分钟、每副本输出吞吐与 cache 复用曲线。
- [完整 PDF](weak_scaling.pdf)：六页图表；每张图也提供单独 PDF。

## 重现和替换占位

已提交 [analysis.json](analysis.json)、[summary.csv](summary.csv)、[replicas.csv](replicas.csv)、[timeseries.csv](timeseries.csv)，可在没有原始 runs 的机器上重画：

```bash
python -m pip install -r reports/requirements.txt
python reports/2026-09-15-weak-scaling/analyze.py --plots-only
```

8 物理节点实测结果出来后，把 [sources.json](sources.json) 中对应 workload 的键 `"4"`（4 个推理副本）从 `null` 改成新的原始 run 相对仓库路径，再在保存原始数据的机器运行：

```bash
.venv/bin/python reports/2026-09-15-weak-scaling/analyze.py
```

脚本重新核对实测计数、替换理论值，并生成全部表格、图和本 README。只读原始结果，不提交作业、不修改原始 run。GitHub 上保留的是派生指标和图表，没有上传体积很大的调用明细或运行日志。
"""
    (OUT/"README.md").write_text(text)


def axis_nodes(ax, linear=False):
    if not linear:
        ax.set_xscale("log", base=2)
    ax.set_xticks(NODES, [str(n) for n in NODES])
    ax.set_xlim(1.75, 36)
    ax.set_xlabel("Physical nodes (inference + frontend)")
    ax.grid(alpha=.2)


def draw_measured(ax, rows, metric, color, label=None, scale=1):
    measured = [r for r in rows if r["source_kind"] == "measured"]
    ax.plot([r["physical_nodes"] for r in measured], [r[metric]*scale for r in measured],
            "o-", color=color, lw=1.7, ms=5, label=label)


def plots(data):
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":180,"pdf.fonttype":42})
    groups = {w:[r for r in data["rows"] if r["workload"] == w] for w in WORKLOADS}
    with PdfPages(OUT/"weak_scaling.pdf") as pdf:
        def save(fig, name, footer="Measured points only; one run per configuration. OpenClaw 4-node frontend has a sampling gap."):
            fig.text(.5,.012,footer,ha="center",fontsize=8,color="#555555")
            fig.tight_layout(rect=(0,.045,1,.95))
            fig.savefig(OUT/(name+".png"))
            fig.savefig(OUT/(name+".pdf"))
            pdf.savefig(fig)
            plt.close(fig)

        fig,axs=plt.subplots(2,2,figsize=(12,8))
        fig.suptitle("Balanced weak scaling | Qwen3.6, TP4, token budget 8192",fontsize=15)
        for col,w in enumerate(WORKLOADS):
            rows=groups[w]; base=rows[0]
            for row,(metric,ylabel) in enumerate([("output_tokens_s","Output tokens / s"),("tasks_min","Completed tasks / min")]):
                ax=axs[row,col]; color=COLORS[w]
                draw_measured(ax,rows,metric,color,"Measured")
                ax.plot(NODES,[base[metric]*n/2 for n in NODES],"--",color="#555555",lw=1.2,label="Ideal from measured 2-node baseline")
                projected=[r for r in rows if r["source_kind"]=="theoretical"]
                if projected:
                    ax.scatter([r["physical_nodes"] for r in projected],[r[metric] for r in projected],marker="D",facecolors="white",edgecolors="#AA3377",s=80,zorder=5,label="Theoretical placeholder (not measured)")
                axis_nodes(ax,linear=True);ax.set_ylim(bottom=0);ax.set_ylabel(ylabel)
                ax.set_title(f"{LABELS[w]} | cc {base['cc_per_replica']} per replica")
                ax.legend(fontsize=8,loc="upper left")
        save(fig,"01_throughput","Hollow diamonds are NOT measured (see sources.json). OpenClaw 4-node frontend has a sampling gap; main counts checked.")

        fig,axs=plt.subplots(2,2,figsize=(12,8))
        fig.suptitle("Scaling efficiency and replica balance",fontsize=15)
        for col,w in enumerate(WORKLOADS):
            rows=groups[w];ax=axs[0,col]
            draw_measured(ax,rows,"token_efficiency_pct",COLORS[w],"Token throughput")
            draw_measured(ax,rows,"task_efficiency_pct","#228866","Task throughput")
            ax.axhline(100,color="#777777",ls="--",lw=1)
            ax.set_ylim(90,110);axis_nodes(ax);ax.set_ylabel("Efficiency vs 2-node baseline (%)");ax.set_title(LABELS[w]);ax.legend(fontsize=9)
            ax=axs[1,col]
            for node in [r["physical_nodes"] for r in rows if r["source_kind"]=="measured"]:
                reps=[r for r in data["replicas"] if r["workload"]==w and r["physical_nodes"]==node]
                jitter=[node*2**((i-(len(reps)-1)/2)*.2/max(1,len(reps)-1)) for i in range(len(reps))]
                ax.scatter(jitter,[r["output_tokens_s"] for r in reps],color=COLORS[w],s=22,alpha=.7)
            draw_measured(ax,rows,"per_replica_tokens_s","#333333","Replica mean")
            axis_nodes(ax);ax.set_ylabel("Output tokens / s / inference replica");ax.legend(fontsize=9)
        save(fig,"02_efficiency_and_balance")

        fig,axs=plt.subplots(2,2,figsize=(12,8))
        fig.suptitle("Latency and cache behavior",fontsize=15)
        for w in WORKLOADS:
            for ax,key,label,scale in [(axs[0,0],"ttft_p95_s","TTFT p95 (s)",1),(axs[0,1],"task_p95_s","Task lifecycle p95 (min)",1/60),(axs[1,0],"prompt_reuse_pct","Prompt token reuse (%)",1),(axs[1,1],"kv_cache_pct","Mean KV cache usage (%)",1)]:
                draw_measured(ax,groups[w],key,COLORS[w],LABELS[w],scale)
                axis_nodes(ax);ax.set_ylabel(label);ax.legend(fontsize=9)
        axs[1,0].set_ylim(90,100);axs[1,1].set_ylim(0,70)
        save(fig,"03_latency_and_cache")

        fig,axs=plt.subplots(2,2,figsize=(12,8))
        fig.suptitle("Resource use and backend queue",fontsize=15)
        for w in WORKLOADS:
            for ax,key,label in [(axs[0,0],"gpu_busy_pct","Mean inference GPU busy (%)"),(axs[0,1],"frontend_cpu_pct","Mean frontend CPU busy (%)"),(axs[1,0],"waiting_mean","Mean waiting requests / replica"),(axs[1,1],"inference_gpu_joules_per_output_token","Inference GPU energy (J / output token)")]:
                draw_measured(ax,groups[w],key,COLORS[w],LABELS[w]);axis_nodes(ax);ax.set_ylabel(label);ax.legend(fontsize=9)
        axs[0,0].set_ylim(80,100);axs[0,1].set_ylim(bottom=0);axs[1,0].set_ylim(bottom=0)
        save(fig,"04_resources")

        fig,axs=plt.subplots(2,2,figsize=(12,8))
        fig.suptitle("Task execution breakdown and tool result changes",fontsize=15)
        keys=[("setup","Setup","#B6BDC7"),("llm_only","LLM","#3366AA"),("tool_only","Tools","#D6A341"),("llm_tool_overlap","Overlap","#AA3377"),("other_and_cleanup","Other / cleanup","#66AA88")]
        for col,w in enumerate(WORKLOADS):
            rows=[r for r in groups[w] if r["source_kind"]=="measured"];ax=axs[0,col];bottom=[0.]*len(rows)
            for key,label,color in keys:
                values=[r["execution_"+key+"_s"] for r in rows]
                ax.bar(range(len(rows)),values,bottom=bottom,label=label,color=color,width=.6)
                bottom=[a+b for a,b in zip(bottom,values)]
            ax.set_xticks(range(len(rows)),[str(r["physical_nodes"]) for r in rows]);ax.set_xlabel("Physical nodes (measured)");ax.set_ylabel("Mean task lifecycle (s)");ax.set_title(LABELS[w])
            ax.set_ylim(0,max(bottom)*1.32);ax.legend(fontsize=8,ncol=3,loc="upper left")
            ax=axs[1,col]
            draw_measured(ax,rows,"native_tool_error_pct","#888888","All native tool errors")
            draw_measured(ax,rows,"new_tool_error_pct","#AA3377","Original success -> replay error")
            axis_nodes(ax);ax.set_ylim(bottom=0);ax.set_ylabel("Share of tool calls (%)");ax.legend(fontsize=8)
        save(fig,"05_execution_and_tool_errors")

        fig,axs=plt.subplots(2,2,figsize=(12,8))
        fig.suptitle("Measurement-window stability | 1-minute bins",fontsize=15)
        node_colors={2:"#4477AA",4:"#228833",8:"#66CCEE",16:"#CC9933",32:"#AA3377"}
        for col,w in enumerate(WORKLOADS):
            for node,color in node_colors.items():
                points=[r for r in data["timeseries"] if r["workload"]==w and r["physical_nodes"]==node]
                if not points:
                    continue
                for ax,key in [(axs[0,col],"per_replica_tokens_s"),(axs[1,col],"prompt_reuse_pct")]:
                    ax.plot([r["minute"] for r in points],[r[key] for r in points],color=color,lw=1.2,alpha=.85,label=f"{node} physical nodes")
                    ax.set_xlabel("Minutes into measurement");ax.grid(alpha=.2)
            axs[0,col].set_title(LABELS[w]);axs[0,col].set_ylabel("Output tokens / s / replica");axs[0,col].legend(fontsize=8,ncol=2)
            axs[1,col].set_ylabel("Prompt token reuse (%)");axs[1,col].set_ylim(80,100)
        save(fig,"06_time_stability")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots-only",action="store_true")
    args=parser.parse_args()
    data=read(OUT/"analysis.json") if args.plots_only else extract()
    tables(data)
    plots(data)
    report(data)
    print("Wrote summary tables, six PNG/PDF figures, and weak_scaling.pdf",flush=True)


if __name__ == "__main__":
    main()
