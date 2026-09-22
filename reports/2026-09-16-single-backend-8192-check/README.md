# 单推理后端 8192 补跑结果

2026-09-16 核对 capacity job 7608115 中的五点序列：**4 档完成，mini-SWE cc64 在环境创建阶段失败，没有正式测量结果。OpenClaw cc32/64 长窗口下仍明显降吞吐，不支持“改成 8192 后高并发进入稳定平台”的假设。**

所有点均为 1 个推理节点＋1 个 frontend，Qwen3.6-35B-A3B、TP4、context 262144、batch token budget 8192、APC/thinking 开启。同 workload 的 serve、trace pool、replay profile 均核对一致。OpenClaw 使用 600 秒 warmup＋3600 秒 measurement，mini-SWE 使用 1800＋5400 秒。

| Workload | cc | 输出 tokens/s | tasks/min | Prompt token 复用 | 平均 KV 使用率 | TTFT p95 | 状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| OpenClaw | 16 | 433.58 | 6.10 | 93.95% | 33.44% | 1.03 s | valid / steady |
| OpenClaw | 32 | 282.41 | 4.10 | 68.87% | 73.46% | 10.23 s | valid / 非稳态 |
| OpenClaw | 64 | 167.29 | 2.55 | 9.24% | 92.36% | 50.84 s | valid / 非稳态 |
| mini-SWE | 32 | 722.14 | 2.62 | 95.62% | 44.62% | 0.71 s | valid / steady |
| mini-SWE | 64 | — | — | — | — | — | warmup 启动失败 |

## OpenClaw 高并发结果

- cc32、cc64 输出吞吐相对 cc16 分别下降 **34.87% 和 61.42%**。两个高并发点的任务与调用均正常完成，LLM 全程请求数和输出 token 数与后端一致，目标输出不匹配为 0；资源采样没有错误或覆盖缺口。`valid=true` 表示数据完整，不能替代稳态判断。
- cc32 前后半段吞吐为 **337.54 → 227.28 tokens/s**，prompt 复用率 **77.61% → 56.24%**。测量第 40–50 分钟更低至 **124.95 tokens/s、12.08% 复用率**，最后 10 分钟又回升至 329.92 tokens/s，因此也不是单调下降。平均 KV 使用率 73.46%，峰值 99.83%；平均 waiting 0.91，初始排队 p95 6.60 秒。
- cc64 复用率一直很低；平均 KV 使用率 92.36%，峰值 100%。平均 waiting **17.36**，前后半段从 **13.27 增至 21.45**；初始排队 p95 **47.61 秒**。虽然两半吞吐只差 5.55%，队列继续增长，因此不能称为稳定容量点。最后任务在测量结束约 2595 秒后完成。
- 这些结果直接显示高 KV 活跃占用、前缀复用损失和排队同时存在；仅凭这些汇总不能区分前缀淘汰、hybrid/Mamba state 匹配限制等具体机制，也不能把“高 KV 使用率”当作“高 cache hit”。

此前短窗口 A/B 中 8192 比 2048 吞吐提高 46.9%，但当时已经记录到复用率低谷。本轮长窗口说明缓解调度阻塞并未消除高并发性能下降。两轮 warmup/measurement 不同，本轮没有匹配的长窗口 2048 对照，不能用跨轮平均值声称 8192 比 2048 更慢。

## mini-SWE cc64 失败原因

2026-09-15 05:47:01 UTC 整档结束。61 个 SIF 全部从已有本地缓存复用，`copied=0, reused=61`。64 个任务同时启动后，任务 `00032`（`django__django-11999`）在约 9.97 秒时失败，其他 63 个任务被取消；完成任务为 0，尚未进入正式测量。

该任务的 `console.log` 显示本地 SIF 解包创建 writable sandbox 时发生：

```text
FATAL ERROR: Failed to create thread
fork/exec .../apptainer: resource temporarily unavailable
BlockingIOError: [Errno 11] Resource temporarily unavailable
RuntimeError: can't start new thread
```

错误发生在本地环境并发创建阶段；不能将这一档作为推理吞吐结果。日志提示进程/线程创建资源不足，尚未确认具体是哪一项系统限制。补跑前需要限制 sandbox 构建并发或解包线程；推理阶段的目标任务并发仍应保持 cc64。

## 对现有 weak scaling 的影响

OpenClaw cc16、mini-SWE cc32 的 8192 基线均稳定，已用于同配置 weak-scaling 报告。本次发现不推翻其多节点扩展结果，但不能把多副本扩展接近线性推导成单副本 cc32/64 也会进入平台。mini-SWE 的 cc64 性能结论仍缺失。

## 来源与核对范围

- 五点序列与路径：`runs/single-node-8192-capacity-20260914/sequence.json`；序列状态为 `failed (exit 1)`，完成列表有前四点。
- 完成点的 summary/status/config、timeseries 和资源汇总；本次另核对 OpenClaw cc32/64 的 calls/tasks CSV 全程计数。两个稳定基线在 weak-scaling 分析中已完成同类计数检查。
- 失败点：`runs/scaling/multinode/minisweagent-calibration-cc64-n1-8192-20260914-7608115.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/` 下的 frontend 日志、image-cache 记录和 `workers/r00/tasks/00032/console.log`。
- 本次没有提交或重启实验，没有修改原始结果。
