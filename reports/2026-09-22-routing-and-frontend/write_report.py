"""Render the merged narrative/tables from the derived results, without raw reads."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
report = json.loads((OUT / "analysis.json").read_text())
results = report["results"]
audits = {r["run"]:r for r in json.loads((OUT / "completed_call_audit.json").read_text())}
replicas = json.loads((OUT / "replica_diagnostics.json").read_text())


def point(w,g,n):
    return next(r for r in results if r["workload"] == w and r["group"] == g and r["backends"] == n)


fronts = [point("minisweagent",g,8) for g in ("fixed_sticky","frontend_f2_build2","frontend_f1_build8")]
base, two, eight = fronts
names = ["1+8，build=4（旧对照）","2+8，build=2/前端（新）","1+8，build=8（新）"]
front_table, setup_table, time_table, error_table = [], [], [], []
for name,r in zip(names,fronts):
    front_table.append(f'| {name} | {r["output_tokens_s"]:,.2f} | {r["tasks_per_minute"]:.2f} | {r["prefix_reuse"]:.2%} | {r["frontend_cpu_mean_percent"]:.2f}% |')
    s=r["setup"]
    setup_table.append(f'| {name} | {s["slot_wait_seconds"]["mean"]:.2f} / {s["slot_wait_seconds"]["p95"]:.2f} | {s["build_seconds"]["mean"]:.2f} / {s["build_seconds"]["p95"]:.2f} | {r["setup_halves"][0]["mean"]:.2f} → {r["setup_halves"][1]["mean"]:.2f} |')
    t=r["execution_seconds_mean"]
    time_table.append(f'| {name} | {t["setup"]:.2f} | {t["llm_only"]:.2f} | {t["tool_only"]:.2f} | {t["other_and_cleanup"]:.2f} | {r["task_lifecycle_seconds"]["mean"]:.2f} |')
    a=audits[r["run"]]["window_tools"]
    error_table.append(f'| {name} | {a["calls"]:,} | {a["success_to_error"]:,} | {a["success_to_error"]/a["calls"]:.2%} |')

rr4,rr8,ca4,sticky8 = [point("openclaw",g,n) for g,n in (("routing_rr",4),("routing_rr",8),("routing_cache",4),("fixed_sticky",8))]
route_table=[]
for w,title in (("openclaw","OpenClaw"),("minisweagent","mini-SWE")):
    for g,policy in (("routing_rr","RR"),("routing_cache","cache-aware")):
        for n in (4,8):
            matches=[r for r in results if (r["workload"],r["group"],r["backends"]) == (w,g,n)]
            if not matches:
                route_table.append(f'| {title} 1+{n} {policy} | 缺测 | — | — | — |')
                continue
            r=matches[0]
            note={"complete_steady":"完整", "complete_nonsteady":"完整、非稳态", "window_only_drain_timeout":"窗口恢复、drain 超时",
                  "window_complete_warmup_failures":"窗口完整、warmup 失败且非稳态"}[r["status"]]
            route_table.append(f'| {title} 1+{n} {policy} | {note} | {r["output_tokens_s"]:,.2f} | {r["tasks_per_minute"]:.2f} | {r["prefix_reuse"]:.2%} |')

new_replicas=sorted([r for r in replicas if r["run"] == rr8["run"]],key=lambda r:r["replica"])
arrivals={r["replica"]:r for r in json.loads((OUT / "routing_arrivals.json").read_text())}
hot_table=[]
for r in new_replicas:
    a=arrivals[r["replica"]]
    hot_table.append(f'| {r["replica"]} | {a["arrivals"]:,} | {r["prefix_reuse"]:.2%} | {r["running_mean"]:.2f} | {r["waiting_mean"]:.2f} | {a["residence_seconds"]["mean"]:.2f} |')

text=f'''# 新结果合并分析：前端资源对照与 OpenClaw RR 扩展

2026-09-22。将 **3 个新测量窗口**与上一版 **17 个实测点**合并，更新前端配比和 routing 对比。按要求，**P2 暂不进入主数据表、图表或性能结论**；启动失败点保持缺测，没有按零吞吐或理论值填充。

[全部图表 PDF](results.pdf) · [汇总 CSV](summary.csv) · [完整统计 JSON](analysis.json) · [运行索引](sources.json)

跨 trace、CC、weak scaling、前端配比与 routing 的综合结论及归因见 [Agentic benchmark insights](insights.md)。

本次最有价值的增量结论是：mini-SWE 的前端受限不能靠单纯放宽 build 名额解决；增加到两个前端后，等待与工具耗时同时下降，吞吐恢复。OpenClaw RR 4→8 的相对吞吐近线性，但同一种单后端热点仍存在，不能把扩展比例理解为系统效率已经合理。

## 1. 新数据的使用范围

| 新实验 | PBS Job | 使用范围 |
|---|---|---|
| mini-SWE 2+8，build=2/前端 | 7643323 | 完整 90 min；valid/steady/collection_complete 均为 true；0 失败任务 |
| mini-SWE 1+8，build=8 | 7643324 | 完整 90 min；valid/steady/collection_complete 均为 true；0 失败任务 |
| OpenClaw 1+8 RR | 7643277 | 完整 60 min；2 次 HTTP ReadTimeout 均在 warmup 结束前发生并结束；测量窗口任务失败为 0 |

两个 mini-SWE 点已逐项核对任务/调用/token、输出长度、工具错误计数、worker 和资源记录，检查全部通过，见 [completed_call_audit.json](completed_call_audit.json)。OpenClaw 的 raw `valid=false/steady=false` 保留；除 warmup 两个失败外，原汇总没有其它验证错误。它的窗口 admission cohort 已全部完成，但**窗口本身也存在吞吐下滑与队列增长**，不是仅有 warmup 标记问题。

旧 mini-SWE 1+4 RR 继续只用于已核验的完整窗口计数与资源分析，drain 截断的任务/LLM 尾延迟不使用。详情和失败点索引见 [收集检查](../2026-09-22-collection-check/README.md)。

## 2. 前端对照：增加资源有效，单纯增加 build 名额无吞吐收益

三点均为 8 个 TP4 后端、总 cc256、sticky、相同 trace pool/seed、模型配置、HTTP policy 和 90 min 窗口；配置差异检查见 `analysis.json.configuration_comparisons`。2+8 每前端两个 build 名额，**总名额仍为 4**。

| 配置 | 输出 tokens/s | tasks/min | 实际前缀复用 | 每前端 CPU 均值 |
|---|---:|---:|---:|---:|
{chr(10).join(front_table)}

![Frontend controls](figures/01_frontend_controls.png)

**2+8 相对 1+8/build4：输出 +{100*(two["output_tokens_s"]/base["output_tokens_s"]-1):.2f}%，任务 +{100*(two["tasks_per_minute"]/base["tasks_per_minute"]-1):.2f}%。** 两个前端 CPU 分别为 44.83%、45.13%，均未接近饱和。实际复用略低于旧对照，吞吐却更高，所以本次提升不需要用“cache hit 变好”解释。

**1+8/build8：输出变化 {100*(eight["output_tokens_s"]/base["output_tokens_s"]-1):.2f}%，未看到改善。** CPU 升至 98.09%，平均 runnable processes 从 124 增至 168；放宽创建名额后，前端仍没有足够资源让创建和工具执行同时更快。

| 配置 | 等待 build 名额 mean / p95，s | 实际 build mean / p95，s | 前/后 45 min 的等待均值，s |
|---|---:|---:|---:|
{chr(10).join(setup_table)}

这组对照修正了“4 个 build 名额可能就是主要硬上限”的简单解释：build8 将名额等待从 97.29 s 降至 11.09 s，但实际 build 从 12.72 s 增至 21.04 s，工具阶段也更慢。**等待减少不等于总服务能力增加；增加并行创建把更多工作移入已饱和的前端。** 2+8 在总名额不变时把实际 build 降到 5.32 s、名额等待降到 0.94 s，更支持前端资源竞争及其与创建限流的耦合。

| 配置 | setup，s | LLM，s | tools，s | 其他/清理，s | 总 lifecycle，s |
|---|---:|---:|---:|---:|---:|
{chr(10).join(time_table)}

![Lifecycle and build distributions](figures/02_frontend_time_breakdown.png)

2+8 的工具阶段约缩短 {100*(1-two["execution_seconds_mean"]["tool_only"]/base["execution_seconds_mean"]["tool_only"]):.1f}%；build8 则增长约 {100*(eight["execution_seconds_mean"]["tool_only"]/base["execution_seconds_mean"]["tool_only"]-1):.1f}%。2+8 每后端平均 running 从 18.89 增至 26.45；build8 为 19.17。2+8 的 LLM 墙钟阶段增长，符合更多并发从 setup/tools 转入后端的现象，不能把它单独解释为模型变慢。任务整体生命周期从 798.55 s 降至 737.05 s。

![Frontend trajectories](figures/03_frontend_timeseries.png)

2+8 前后半段输出约 5696 / 5685 tokens/s，几乎不变；build8 约 5333 / 5076 tokens/s，下降 4.83%，名额等待也从 2.73 s 增至 19.56 s。后者虽然通过原 runner 的 steady 阈值，仍不能称所有前端延迟都已稳定。

三点 iowait 都低于 0.12%，最低可用内存仍超过 442 GiB；目前没有内存容量耗尽或高 iowait 的证据。增加前端还会增加内存带宽、本地 I/O 等资源，不能把全部提升精确归为 CPU，或用这一个重复点给出各因素的因果百分比。

工具行为也有同方向变化，但尚未对错误内容做逐项因果归类：

| 配置 | 窗口工具调用 | 录制成功→回放 error | 占全部工具调用 |
|---|---:|---:|---:|
{chr(10).join(error_table)}

原生工具错误不等于 task 失败；这些点的 0 失败任务不能替代工具层检查。不同闭环运行的任务混合不完全相同，错误比例差异不能全部归为 CPU。

## 3. Routing：RR 的相对扩展仍近线性，热点与非稳态并未消失

| 配置 | 数据状态 | 输出 tokens/s | tasks/min | 实际前缀复用 |
|---|---|---:|---:|---:|
{chr(10).join(route_table)}

![Routing comparison](figures/04_routing_comparison.png)

OpenClaw RR 4→8：输出 **{rr8["output_tokens_s"]/rr4["output_tokens_s"]:.2f}×**，任务 **{rr8["tasks_per_minute"]/rr4["tasks_per_minute"]:.2f}×**。但 1+8 RR 的 {rr8["output_tokens_s"]:.2f} tokens/s 只有 1+8 sticky 实测的 **{rr8["output_tokens_s"]/sticky8["output_tokens_s"]:.2%}**，也低于 1+4 cache-aware 的 {ca4["output_tokens_s"]:.2f} tokens/s。Sticky 走直连、routing 走官方 router，二者作为部署性能参考，不能将差异全归于路由算法。

RR N8 的实际前缀复用为 61.77%，与 N4 的 63.48% 同处低位；相邻调用切换后端比例为 86.10%，N4 为 73.91%。这说明增加后端数量没有修复请求局部性或热点状态。切换率是窗口内至少 8 次已完成调用的 task，按相邻调用对加权。

| N8 后端 | 窗口到达请求 | 实际复用 | running 均值 | waiting 均值 | 请求滞留均值，s |
|---|---:|---:|---:|---:|---:|
{chr(10).join(hot_table)}

![Per-backend imbalance](figures/05_openclaw_replica_hotspot.png)

**每个后端都收到 2006 个窗口请求，热点不是因为 RR 给 r07 分发了更多请求。** r07 的请求平均滞留 188.26 s，其余只有 4.46–5.18 s；服务滞留差异使它积累大量在途请求。低复用与高排队共同出现在 r07，其余后端后续请求又受 agent 串行依赖约束，仍与此前发现的闭环反馈解释一致。这里的请求滞留来自 middleware 的 started→ended cohort，允许自然完成，包含服务/传输时间，不是独占 GPU 计算时间。

新点再次观察到“单热点”，但热点副本编号已由 N4 的 r03 变为 N8 的 r07。它不证明特定编号必然有问题，也没有单独确定最初分化是节点速度、缓存状态还是调度相位触发；此前请求级核对见 [热点归因报告](../2026-09-21-rr-hotspot-cause/README.md)。

![RR normalized scaling trajectories](figures/06_openclaw_rr_scaling.png)

N8 前/后半窗口输出 **1327.79 → 1166.27 tokens/s（−12.16%）**，完成任务 **626 → 496（−20.77%）**，平均每后端 waiting **7.16 → 8.42**。因此当前值只代表给定 60 min 窗口，不代表稳定容量；即使相对 N4 接近 2×，也不能据此关闭 routing 效率问题。缺少 N8 cache-aware 有效点，尚不能判断 cache-aware 的收益是否随规模扩大。

## 4. 与既有前后端比例、weak scaling 的合并

![Updated frontend scaling](figures/07_frontend_scaling_updated.png)

保留既有实测 weak-scaling 和 1 前端 sticky 点，新补 2+8/build2 与 1+8/build8。2+8 输出 5690.48 tokens/s，接近历史 8+8 的 5699.78（约 99.84%）；使用节点从 16 减至 10。历史 8+8 使用旧 HTTP/build policy，此处只作容量参考，不能视为严格的仅前端数 A/B。mini-SWE 1+4 sticky 仍保持缺测，不用连线插值代替结果。

当前可以支持的文章表述是：**端到端 agent 吞吐同时受前端工具/环境供给和后端请求局部性与排队影响；增加并行度或得到近线性 scale ratio，都不足以证明资源使用高效。** 前端实验给出了资源竞争与限流相互作用的对照；routing 实验则区分了均分请求数与均衡服务压力。

## 5. 口径、限制与复现

模型 Qwen3.6-35B-A3B，后端 TP4，context 262144，batch budget 8192，max sequences 64，APC/thinking 开启。OpenClaw：110 条 trace，cc16/逻辑 worker，warmup 600 s、measurement 3600 s；mini-SWE：61 条 trace，cc32/worker，warmup 1800 s、measurement 5400 s。本次两个前端点的 SIF 均已本地 staging，`web_search` 仍为 recorded delay。

- 输出吞吐按测量窗口内 token 事件统计；tasks/min 按窗口内完成任务统计。
- 实际复用按首 token 在窗口的请求汇总 cached/prompt；不平均各后端命中率，也不用 KV occupancy gauge 替代命中率。
- 完整点的任务/setup 统计取窗口 admission cohort，包括自然 drain。工具错误按窗口内 tool-start cohort。它们不是同一个样本集合。
- 时间曲线为不重叠 5 min 均值，复用率按该 5 min 的 token 总量加权；底层 CSV 保留 60 s 分箱。箱线图为 p25–p75、须 p5–p95、均值菱形。
- 每点只有一次运行，没有跨 run 置信区间；固定 trace pool 循环回放，不直接代表不断到来的全新任务，也不评估 SWE-bench 解题正确率。

报告包含 20 个有效实测窗口的索引和 3 个非 P2 启动失败点索引；P2 不进入此汇总。7 张图同时提供 PNG、矢量 PDF，以及合并 `results.pdf`。历史汇总复用上一报告，新点的提取缓存位于 `extracted/`，不修改任何原始 run。

```bash
.venv/bin/python reports/2026-09-22-routing-and-frontend/analyze.py
.venv/bin/python reports/2026-09-22-routing-and-frontend/write_report.py
# 仅重绘，避免重复读取原始记录：
.venv/bin/python reports/2026-09-22-routing-and-frontend/analyze.py --plots-only
```

后续新补跑须以新的 run ID 加入，保留失败及排除记录。此报告替代上一版中关于本次两项前端实验的待验证判断；旧 [09-21 报告](../2026-09-21-routing-and-fixed-frontend/README.md) 保留作历史快照。
'''
(OUT / "README.md").write_text(text)
print("Wrote README.md with tables from analysis.json.")
