# OpenClaw 16 物理节点结果核对

2026-09-15 核对 job `7618907`：**8 个推理节点＋8 个 frontend 正常完成，吞吐接近线性扩展；native 工具仍有少量相对录制结果新增的错误，需保留这一限制。**

## 完成状态与数据

- PBS `Exit_status=0`，运行时长 1:56:10，2026-09-14 22:15 UTC 结束。
- Run 为 `completed`、`valid=true`、`steady=true`，validation errors 和 diagnostic warnings 均为空。
- 8 个 worker 均完成同一个 3600 秒测量窗口及自然 drain；16 个物理节点采样均通过，没有采样报错或覆盖缺口，每个推理节点均有 4 张 GPU。
- 测量窗口为 2026-09-14 21:01:39–22:01:39 UTC；完成 3,085 个任务，输出 12,638,470 tokens。最后任务在窗口结束约 430 秒后完成。
- 全程（含 warmup/drain）3,777 次重放全部完成；54,369 次 LLM call、51,023 次 tool call、15,490,788 个输出 token。调用表与后端全程请求/token 计数一致，输出目标不匹配及未完成调用均为 0。

## 与匹配基线比较

新完成的 capacity N=1、既有 N=2 和本次 N=8 的 serve 配置、110 条 trace pool、replay profile、cc16、seed 规则、600 秒 warmup 和 3600 秒 measurement 均一致。

| 物理节点（推理＋frontend） | 输出 tokens/s | 任务/min | 实际 prompt token 复用 | TTFT p95 | 任务 p95 |
|---|---:|---:|---:|---:|---:|
| 2（1＋1） | 433.58 | 6.10 | 93.95% | 1.03 s | 522.11 s |
| 4（2＋2） | 869.04 | 12.78 | 93.89% | 0.98 s | 483.77 s |
| 16（8＋8） | 3510.69 | 51.42 | 93.99% | 1.00 s | 483.21 s |

16 节点相对 2 节点的 token 吞吐为 **8.097 倍**，扩展效率 **101.21%**；相对 4 节点为 **4.040 倍**。任务吞吐相对 2 节点为 8.429 倍，相对 4 节点为 4.022 倍。单次运行中略高于理想线的结果不作为超线性加速结论，任务组成、完成窗口与工具行为均可能影响任务吞吐。

8 个推理副本的输出吞吐为 435.06–447.79 tokens/s，均值 438.84，变异系数 0.90%。测量前后半段 token 吞吐变化仅 0.47%；平均 waiting 约 0.0104 requests/副本。前缀复用和 TTFT 未随节点数增加而恶化，本次没有观察到整体吞吐塌陷或明显负载不均衡。

旧 4 节点结果原始 `valid=false` 来自单个 frontend 的一次 5.59 秒采样间隔；其主计数已单独核对可用。本报告沿用这些主数据，保留原始状态，不修改或重新生成旧 run 的结果。

## 工具错误限制

按每次工具调用对应的原 trace 节点 `recorded_result.isError` 核对：

| 范围 | Tool calls | Native errors | 原成功→重放错误 | 其中 web_fetch / exec |
|---|---:|---:|---:|---:|
| 全程 | 51,023 | 4,022 | 526（1.03%） | 444 / 82 |
| 测量窗口内开始的调用 | 41,754 | 3,273 | 397（0.95%） | 329 / 68 |

测量窗口还有 183 次原报错→重放成功（web_fetch 10、exec 173）。因此不能把全部 native errors 都解释为录制时已有错误。这些变化未中断任务或造成 LLM 计数异常，但工具耗时和端到端任务吞吐的解释应附带错误率；本次未逐条诊断新增工具错误的具体原因。

## 来源

- [comparison.csv](comparison.csv)：三档对照及原始 run 路径。
- 本次 run：`runs/scaling/multinode/openclaw-balanced-n8-7618907.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/`。
- 读取各 run 的 config、summary、worker summary、calls/tasks CSV，结合原始 trace 工具节点核对。没有启动、停止或重跑实验，没有修改原始结果。
