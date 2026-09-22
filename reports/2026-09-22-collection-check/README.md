# 2026-09-22：两批补充实验的数据收集检查

后续分析发现 P2 负载信号未生效，按用户要求暂不进入性能对比，见 [排除原因与证据](p2-exclusion.md)。以下“窗口可用”仅指计数完整性，不代表它是有效的负载感知消融。[新旧结果合并分析](../2026-09-22-routing-and-frontend/README.md) 已完成。

检查 7 个已结束 PBS 作业、8 个实验点：**2 点完整成功，2 点测量窗口可用但带异常，4 点在启动阶段失败、没有测量数据。** 原始状态和数据保持不变。

| 实验 | Job | 数据状态 | 测量分钟 | 窗口完成任务 | 输出 tokens/s | tasks/min |
|---|---|---|---:|---:|---:|---:|
| mini-SWE 2+8，build=2/frontend | 7643323 | 完整成功 | 90 | 1,835 | 5,690.48 | 20.39 |
| mini-SWE 1+8，build=8 | 7643324 | 完整成功 | 90 | 1,664 | 5,204.49 | 18.49 |
| OpenClaw 1+8 RR | 7643277 | 窗口完整；warmup 两次失败 | 60 | 1,122 | 1,247.03 | 18.70 |
| mini-SWE 1+4 P2 | 7643321 | 窗口已核验恢复；drain 超时 | 90 | 237 | 762.58 | 2.63 |
| OpenClaw 1+8 cache-aware | 7643278 | 启动失败，需补跑 | — | — | — | — |
| mini-SWE 1+8 RR | 7643279 | 启动失败，需补跑 | — | — | — | — |
| mini-SWE 1+8 cache-aware | 7643280 | 启动失败，需补跑 | — | — | — | — |
| OpenClaw 1+4 P2 | 7643321 | 启动失败，需补跑 | — | — | — | — |

## 两个完整前端对照

PBS Exit_status=0；两点均 `valid=true`、`steady=true`、`collection_complete=true`，任务失败为 0，验证错误及采样警告为空。各 8 个 worker 的窗口、seed 42–49、trace pool、cc32 与全局完成任务数对齐；真实前端 profile 和 SIF staging 记录符合计划。

- **2+8**：每前端 4 个 worker、cc128、2 个 build 名额，合计 cc256、4 个名额；两个前端均已准备 61 个本地 SIF。CPU 均值分别为 44.83%、45.13%。
- **1+8 / build=8**：cc256、8 个创建名额，61 个本地 SIF；CPU 均值 98.09%。

历史 1+8 / build=4 的输出为 5,259.45 tokens/s、18.66 tasks/min、CPU 95.79%。本次增加前端资源后输出提高约 8.2%；只放宽创建名额的输出约低 1.0%，未看到吞吐改善。这是单次运行的初步趋势，后续需结合 setup/tool 时间和资源变化分析；不能只凭这几个均值精确分配 CPU 与构建限流的因果贡献。

## 两个带异常的可用窗口

**OpenClaw 1+8 RR**：8 个 worker 跑满同一个 60 min 窗口。两次失败为 r06/00008、r07/00009 的 HTTP `ReadTimeout`，都发生并结束在 warmup 内；测量窗口内失败任务为 0、admission cohort 无未完成任务。原主汇总的全部验证错误只有这两条失败任务记录，没有请求/token 对账或资源覆盖错误。因此窗口吞吐与缓存统计可以使用，但保留 warmup 异常及原 `valid=false/steady=false` 标记，不宣称整轮无异常或已达到稳态。

**mini-SWE 1+4 P2**：完成 90 min 测量后，30 min drain 到期仍有任务未结束。全部任务共 431 个完成、38 个取消；窗口接纳的 237 个任务中 200 个完成、37 个被取消。窗口内完成的 237 个任务包括 warmup 接纳的任务，不能与 admission cohort 混为一谈。

独立读取后端 token 事件及监控记录，恢复出窗口输出 **4,117,950 tokens**：涉及窗口输出的请求均有完成记录，token 总数无不一致；首 token 在窗口内的请求全部正常按长度结束。四个后端监控覆盖完整窗口，无计数器重置，最大采样间隔 1.123 秒；事件与 Prometheus 边界增量的最大相对差约 0.0138%。实际 prompt 复用率为 57.43%。

因此 P2 的固定窗口吞吐、缓存和后端负载数据可用；完整任务/调用延迟 cohort 被截断，不报告为完整尾延迟。没有重写原失败状态，也没有向原 run 补写成功 summary。恢复数据在 [miniswe-p2-window.json](miniswe-p2-window.json)。

## 四个启动失败点

这些点均没有 `control/start.json`，没有正式测量数据，不能按零吞吐计入曲线。

| 实验 | 后端 / 节点 | 直接错误 |
|---|---|---|
| OpenClaw 1+8 cache-aware | r07 / `x3003c0s25b0n0` | CUDA 初始化断言：`device=1, num_gpus=1` |
| mini-SWE 1+8 RR | r00 / `x3208c0s37b1n0` | CUDA device busy or unavailable |
| mini-SWE 1+8 cache-aware | r00 / `x3208c0s37b1n0` | 同上 |
| OpenClaw 1+4 P2 | r00 / `x3208c0s37b1n0` | 同上 |

三个作业在 06:21–06:25 UTC 先后分到 `x3208c0s37b1n0` 并出现相同错误。两个节点也出现在此前失败诊断中。当前证据将失败定位到特定节点的 CUDA 初始化/设备可用性，但不足以确定是硬件、驱动、容器可见性或残留占用；没有将其直接归为 routing 算法问题。补跑前需处理或避开该启动问题，避免原样重交后再次分到相同节点。

P2 包内 OpenClaw 的启动失败没有中断另一半 mini-SWE 的测量。其父 PBS 作业 Exit_status=1 不代表两个实验均无可用数据。

## 检查范围与证据

- [collection-check.json](collection-check.json)：逐实验分类、原状态、窗口指标、worker 对账、实际 profile、异常位置。
- [pbs-status.json](pbs-status.json)：只读查询的 PBS 退出码与运行时间。
- [miniswe-p2-window.json](miniswe-p2-window.json)：收尾失败点的独立窗口恢复及逐后端核验。

完整点复用 runner 已生成的全局请求/token 校验结论，并核对 worker 窗口、计数、配置及必要文件存在性；没有重复扫描其多 GB 原始日志。仅对缺主汇总的 mini-SWE P2，复用了前次分析脚本的 `recover_window`，读取原始后端事件/采样进行恢复。本次没有连接计算节点，没有修改运行脚本，也没有提交或取消作业。
