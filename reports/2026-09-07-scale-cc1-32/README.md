# 单推理节点 scale 趋势：cc=1、2、4、8、32

汇总日期：2026-09-07。当前两种负载各有五个有效点；cc=16 尚缺有效结果。全部指标见 [tables.md](tables.md)，可直接使用的数值见 [summary.csv](summary.csv)。

## 本次选用的结果

| 负载与并发 | 结果目录（相对 trace_framework/runs/scaling） | 推理节点 |
|---|---|---|
| 两种负载 cc=1/2/4 | single-backend-cc124-20260907-2 | x3005c0s1b1n0 |
| OpenClaw cc=8 | preemptable-cc8-7596639/experiment | x3211c0s13b1n0 |
| mini-SWE cc=8 | debug-cc8-7597420/experiment | x3001c0s19b0n0 |
| 两种负载 cc=32 | preemptable-cc32-7596641/experiment | x3212c0s1b0n0 |

服务配置、各负载 trace pool、seed、预热/测量时长、LLM 回放设置一致：Qwen3-32B BF16、TP4、4×A100；上下文 32768、max_num_seqs=32、max_num_batched_tokens=2048、显存比例 0.90、prefix caching 开；120s warmup + 600s measurement，随后自然 drain。每点一次独立运行；不同点使用不同物理节点。

mini-SWE cc=8 重跑前已有启动隔离及清理修复。这些修复与其他节点差异应保留在实验说明里；本次比较的是已记录的服务与回放参数，不宣称全部软件文件均由不可变版本快照固定。

## 核对范围

十个点均为 completed、measurement.valid=true，既有 backend/client/counter 校验通过，采样 error_samples 均为零。本次读取任务报告重新核对任务窗口计数、节点数、调用数及指定输出 token 长度：共 447 个任务、15,515 个节点、1,993,156 个输出 tokens，与保存的 vLLM generation counter 校验一致。未重复扫描此前已校验的所有后端 token 事件。

所有点的原生工具错误对照均没有“原 trace 成功→回放失败”；mini-SWE 中原失败转成功的提交命令仍单独记录在 checks.json，不能将原生工具错误率当作回放框架失败率。

## 指标口径

- 主表使用 summary.json.measurement，不使用包含 warmup/drain 的顶层吞吐。
- 任务吞吐：600 秒内完成的任务数 / 10，单位 tasks/min。任务延迟：600 秒内启动的任务的完整生命周期，含 drain 部分。两组任务数在本轮恰好相同，但不能据此认为是同一组任务。
- Setup/replay/其他生命周期均在同一任务入场队列上取平均。其他生命周期由总时间减去 setup 和 replay 得到，包含 cleanup、进程启动/退出等，不能直接改名为“纯编排开销”。
- LLM/tool 调用开始率：窗口内启动的调用数 / 600；延迟包含这些调用在窗口外完成的时间。mini-SWE cc=8 的 LLM/tool 数是 740/743，边界跨越不同造成差值，非漏记。
- 窗口输出 token/s：窗口内 token 事件计数 / 600。后端活跃输出 token/s：同一计数除以窗口内有活跃请求的时间。低 cc 下后者较高是分母差异；cc≥8 活跃时间为 600s，两者相同。
- 请求输入 token/s：Prometheus prompt counter 的增量 / 首尾采样间隔（约 599s），包含命中 prefix cache 的逻辑输入。它不等于 GPU 实际未缓存 prefill tokens/s，也不与严格事件窗口完全同边界。
- TTFT 与 TPOT 是客户端统计。TPOT 为流输出时间 / (输出 tokens−1) 的每调用估计值，再按调用取平均。Backend prefill/decode/queue 使用引擎事件定义，queue 不包含客户端到服务端的全部等待与网络时间。
- GPU busy 是推理节点四卡均值；功率是四卡平均功率之和，不是节点整机功率。每卡已分配显存包括模型和预分配 KV 等，不能与 KV-cache 使用百分比等同。
- CPU 百分比是整节点平均，低平均值不能排除单核/线程瓶颈。GPU memory busy 也不能直接视为显存容量使用率或实际内存带宽占比。
- Waiting 是约一秒采样值；采样为零不能证明没有任何短暂排队。原始 backend queue mean 有时高于 p95，可能由少数排队离群请求造成，不是统计上不可能。
- 只有块设备和网络计数，缺少可靠的 Lustre 文件系统 I/O 测量；本次不把磁盘计数改称为文件系统压力。

## 趋势判断

**主趋势合理：**两种负载的任务吞吐与窗口输出 token/s 均随 cc 单调增长。cc≈4 时 GPU busy 已接近 100%，之后仍能通过更大的并发批处理提升总吞吐；GPU busy=100% 不是吞吐上限。

| cc=8 → 32 | OpenClaw | mini-SWE |
|---|---:|---:|
| 任务吞吐倍数 | 2.39× | 2.31× |
| 窗口输出 token/s 倍数 | 2.50× | 2.56× |
| 任务 p95 增幅 | 53.5% | 63.5% |
| 每 token 时间（TPOT mean） | 22.77→39.80 ms | 23.75→43.46 ms |
| LLM 延迟 p95 | 46.49→87.85 s | 16.48→29.55 s |
| Tool 延迟 p95 | 0.771→0.740 s | 2.589→2.559 s |
| Setup mean | 11.07→11.52 s | 27.88→30.09 s |

并发提高四倍，吞吐增长约 2.3–2.6 倍，延迟明显增加，符合批处理提高总吞吐、同时延长单请求 decode 的权衡。工具 p95 和 setup 的变化小得多，数据支持当前高并发下新增延迟主要与 LLM 阶段有关；这是观察性判断，仍受到任务组合影响。

**尚不能确定容量上限或准确转折点：**cc=32 时 KV 使用峰值为 OpenClaw 56.72%、mini-SWE 54.88%；waiting 的采样最大值均为零。当前表现出吞吐收益低于并发倍数，但没有吞吐平台/回落证据；缺少 cc=16，无法定位转折区。

**需要保留的非单调变化：**

1. mini-SWE cc=2 的任务 mean/p95 低于 cc=1，以及 OpenClaw cc=4/8 的 mean 低于 cc=2：低 cc 的入场样本少且覆盖不同 trace。不能解释为并发本身降低了任务耗时。任务延迟样本数 OpenClaw 为 8/11/28/56/134，mini-SWE 为 2/5/9/16/37。
2. OpenClaw TTFT p95 在 cc=1/2 时约 803/815ms，到 cc=8 降为 153ms：prefix hit 同期从 83.67%/90.98% 上升到 99.56%，任务和缓存热度存在混杂。cc=32 时再升到 301ms，仍低于低 cc，不矛盾。
3. mini-SWE 四卡功率 cc=4→8 从 1104.6W 降到 1044.9W，但吞吐上升。两个点来自不同推理节点，各只有一次；当前未采集足够 GPU 时钟/功率上限/温度状态，不能断言此下降来自某个具体机制，也不能据此推翻 token/任务计数。
4. 工具原生错误率约 10%，高 cc 没有明显升高；错误绝对数上升主要伴随更多调用。mini-SWE cc=1 的较高错误率还受到少量任务组合影响。
5. Prefix hit 在 cc=32 达到 99.78%/99.21%，属于有限 trace 池循环回放条件。部署到大量不同新任务时，不能直接假设仍有相同命中率。

**当前可采用的表述：**在固定 trace 池和默认服务配置下，提高任务并发到 32 仍提升吞吐，但相对 cc=8 已付出明显的单请求 decode 和任务尾延迟代价。cc=32 是当前已测点中吞吐最高的点；cc=8 是更低延迟的参考点；待 cc=16 与重复实验补齐后再判断合适运行区间。

在仓库根目录复现：`.venv/bin/python reports/2026-09-07-scale-cc1-32/aggregate.py`。原始输入位置及其他报告见 [报告索引](../README.md)。
