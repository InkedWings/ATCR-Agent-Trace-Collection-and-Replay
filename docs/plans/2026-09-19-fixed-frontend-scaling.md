# 固定一个 frontend 的多后端 scaling

目标：测量一个 frontend/tool 节点能够驱动多少个独立推理节点，区分整节点 CPU 饱和、单核瓶颈、环境创建等待、内存/本地盘压力和后端限制。

2026-09-21 提交更新：两个 1+8 实验改为申请 10 节点 prod / 3h，只使用其中 9 个节点。mini-SWE 1+4（7637013）保留原来的抢占式排队。新的脚本与迁移命令见 [prod 打包提交计划](2026-09-21-prod-packed-submissions.md)；下表记录最初的资源申请方案。

## 实验矩阵

沿用 Qwen3.6-35B-A3B、vLLM 0.19.1、TP4、context 262144、batch budget 8192、max_num_seqs 64、thinking/APC 开启。每个任务固定访问自己的后端，不引入路由 proxy。

这里的路由定义是 **task-sticky / direct**，不能用作逐 LLM call 的 RR 数据，也不是读取实际 KV 状态的路由。独立的 RR、load-aware、sticky、KV-aware 对照见 [routing baseline 计划](2026-09-19-routing-baselines.md)。

| 前端 : 后端 | 物理节点 | OpenClaw 总 task cc | mini-SWE 总 task cc | 队列 / walltime |
|---|---:|---:|---:|---|
| 1 : 1 | 2 | 16 | 32 | 复用已完成基线 |
| 1 : 2 | 3 | 32 | 64 | preemptable / 3h |
| 1 : 4 | 5 | 64 | 128 | preemptable / 3h |
| 1 : 8 | 9 | 128 | 256 | preemptable / 3h |
| 1 : 16 | 17 | 256 | 512 | prod / 3h |

每后端并发固定为 OpenClaw 16、mini-SWE 32，只增加前端驱动的逻辑副本数。每副本仍读取整个 trace pool，使用 seed 42 + replica index。cc 包含 setup 和 cleanup，不能理解为同时在后端 decode 的请求数。

OpenClaw 110 条 trace、warmup 10 分钟、measurement 60 分钟；mini-SWE 61 条、warmup 30 分钟、measurement 90 分钟。仅 web_search 使用 recorded delay，其他工具真实执行。每档与已有相同后端数的 balanced 结果对比。

正式测量窗口合计 12 小时，warmup 合计 2 小时 40 分钟，准备与 drain 另计；8 点 warmup + measurement 共约 107.7 node-hours，不包括排队。按 N=2 → 4 → 8 → 16，每档先 OpenClaw 再 mini-SWE，先确认低档可以运行再继续高档。

## 本轮运行准备

- 共享 SIF 已完成 61/61，总计约 64.66 GiB。继续在测量前复制到 frontend 的节点本地缓存，不访问 Docker Hub。
- mini-SWE 在一个 frontend 上使用跨进程文件锁，最多同时创建 **4 个 sandbox**。锁位于 `/local/scratch/$USER/agenttrace-multinode/sandbox-build-slots`；环境创建完即释放，进程退出也会释放锁。它不限制 LLM 或工具执行并发。
- 每个任务的 `run/sandbox-setup.json` 记录 `slot_wait_seconds`、`build_seconds`。等待仍计入 task setup 和端到端时间。如果该限制先成为瓶颈，结果应标为环境创建瓶颈，不能宣称 CPU 饱和。旧 balanced 结果没有该限制，解释 setup 差异时必须说明。
- 继续每秒采样节点 CPU、iowait、内存、磁盘和网络，新增逐逻辑 CPU busy、最忙核、忙碌逻辑 CPU 等效数、运行/阻塞进程数、load average、sampler CPU affinity、本地 scratch 剩余空间。
- 所有变更只涉及前端准备和观测；后端配置及测量窗口保持一致。新建 bundle，不覆盖之前的 weak-scaling 配置或结果。

## 怎样判断 frontend 饱和

主要输出表：后端数、task/min、output tokens/s、相对 balanced 吞吐、frontend CPU mean/p95、60 秒平均 CPU、最忙核、运行队列、iowait、可用内存、本地盘最小余量、setup/工具延迟、sandbox 创建等待，以及每后端 running/waiting、GPU busy、TTFT、prefix reuse。

整节点 CPU 饱和需要联合证据：持续高 CPU（例如连续多个 60 秒窗口接近 90%）、可运行队列压力、吞吐增长放缓，并且相同后端数的 balanced 吞吐更高。90% 是分析参考，不是启动或数据验收条件。

- 总 CPU 较低但某核持续满载：检查串行调度/事件循环，不称整节点 CPU 饱和。
- iowait、磁盘占用或 sandbox 名额等待先上升：区分 I/O、存储容量和环境创建限制。
- 后端 running/GPU busy 相对 balanced 下降且前端 setup/tool 时间增加：支持前端供给不足。
- 后端 cache hit 本身显著下降：单独报告，不能全部归因于前端。
- 若 1:16 仍能继续扩展，只能报告“该范围未达到 frontend CPU 饱和”，后续再扩展后端数；不提高单后端 cc 来人为制造不同瓶颈。

N=1 的已有 frontend 平均 CPU 为 OpenClaw 3.20%、mini-SWE 10.61%。这提示两类任务可能在不同配比遇到限制，但不能将该均值直接线性外推作为实验结论。

## 配置与提交

配置：`examples/scaling/fixed-frontend.json`。生成只含本组的 8 个任务：

```bash
.venv/bin/python -m agenttrace.experiments.multinode render \
  --config examples/scaling/fixed-frontend.json --group fixed-frontend \
  --output runs/fixed-frontend-jobs-20260919
```

本轮 bundle 已生成。首档命令（先看 OpenClaw 的运行状态，再交 mini-SWE）：

```bash
qsub runs/fixed-frontend-jobs-20260919/jobs/openclaw-fixed_frontend-n2.pbs
qsub runs/fixed-frontend-jobs-20260919/jobs/minisweagent-fixed_frontend-n2.pbs
```

其他档位见 bundle 的 `submit-fixed-frontend.txt`。N 表示后端数；N=2 实际申请 3 个物理节点。前端资源集中后风险与 balanced 不同，首档是完整窗口的正式点，同时用于核对多后端接入和环境创建控制。

当前没有运行中的 allocation；查询到的 2 节点 capacity 作业 `7636974` 仍在排队，且不足以容纳本组最低的 3 节点拓扑。真实多节点验证需等首档获得资源；本地模拟测试不替代该验证。

准备检查：47 项本地测试通过（资源统计、跨进程构建并发及进程退出后释放、独立副本与节点映射、报告计数），8 个 PBS 脚本全部通过 `bash -n`。池和引用路径检查为 OpenClaw 110/110、mini-SWE 61/61；共享缓存 61 个 SIF 均存在且非空。保留之前由用户手动提交的安排，目前未执行 `qsub`。
