# 2026-09-21：fixed-frontend / routing 结果检查

检查时间：2026-09-21T03:45:08.832599+00:00。本轮 3 项完整有效、最新 2 项 n16 启动失败、7 项排队。

以下 1+N 为 1 个物理 frontend + N 个 TP4 推理节点。配置为 Qwen3.6-35B-A3B、context 262144、batch token budget 8192、thinking/APC 开启。OpenClaw 每副本 task cc16、测量 60 分钟；mini-SWE 每副本 cc32、测量 90 分钟。

| Job | 任务 | 前端+后端 | 总 cc | 完成 task | Tasks/min | Output tokens/s | 前端 CPU 均值 | 实际 prefix 复用率 | TTFT p95(s) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 7637010 | OpenClaw | 1+2 | 32 | 767 | 12.78 | 868.97 | 6.77% | 93.97% | 0.971 |
| 7637012 | OpenClaw | 1+4 | 64 | 1534 | 25.57 | 1747.21 | 14.72% | 93.95% | 0.994 |
| 7637011 | mini-SWE | 1+2 | 64 | 456 | 5.07 | 1425.35 | 21.24% | 95.42% | 0.830 |

## 有效结果的含义

三项 PBS Exit_status=0，summary valid=true、steady=true；所有 worker 完成且共同测量窗口一致，failed task=0。读取了已有程序生成的全局请求/token 校验结论，本次没有重新扫描完整后端 token 日志。

OpenClaw 从 2 后端扩到 4 后端：任务吞吐 2.000 倍，输出 token 吞吐 2.011 倍。两档 serve、trace pool、profile、HTTP/失败策略、每副本并发及测量时长相同，支持在这两个点间做比较。TTFT p95 仅增长 2.38%，复用率保持约 94%。这是单次运行的近线性趋势，没有重复实验的置信区间。

OpenClaw 前端 CPU 均值从 6.77% 到 14.72%，后者最大采样值为 33.18%；当前 1+4 没有观察到前端 CPU 饱和。mini-SWE 1+2 均值 21.24%，5381 个测量期样本中只有 2 个达到 90%，不能凭短暂峰值判断持续饱和。mini-SWE 尚缺更高后端数的完整点。

mini-SWE frontend 采样有一处 6.77 秒间隔，summary 标注资源覆盖警告；任务与 token 完整计数校验仍通过。CPU 数值是现有样本均值。

失败 task=0 不等于原生工具返回值全部成功。OpenClaw 1+2 / 1+4 及 mini-SWE 1+2 的窗口内原生 tool error 分别为 743 / 1560 / 4397；本次未逐条与录制 trace 对照，不能把这些数直接解释为新增错误。

## 最新 n16 重交

7638384（OpenClaw）运行 1 分 17 秒，7638385（mini-SWE）运行 42 秒，均 Exit_status=1。两者在同一个 x3003c0s25b0n0 上报 CUDA 初始化断言 `device=1, num_gpus=1`；分别为 r05 和 r10。未生成测量窗口，没有可用 scaling 数据。

这次故障节点与前一轮 x3208c0s37b1n0 不同。日志明确显示已使用本地 snapshot、HF_HUB_OFFLINE=1。现有日志不足以区分设备/驱动异常和容器内 CUDA 可见性问题，也不能据此断言机器物理上只有一张 GPU。

- [7638384 首个 CUDA 错误](../../runs/scaling/multinode/openclaw-fixed_frontend-n16-7638384.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/backends/r05/service.log)：第 53 行起。
- [7638385 首个 CUDA 错误](../../runs/scaling/multinode/minisweagent-fixed_frontend-n16-7638385.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/backends/r10/service.log)：第 53 行起。

## 仍在排队

- 7637013：mini-SWE 1+4。
- 7637014：OpenClaw 1+8。
- 7637015：mini-SWE 1+8。
- 7637645：OpenClaw official RR n4。
- 7637646：OpenClaw official cache-aware n4。
- 7637647：mini-SWE official RR n4。
- 7637650：mini-SWE official cache-aware n4。

PBS 当前备注为可用 queue_tags 资源不足，这些作业尚未启动，不是应用程序报错。mini-SWE 8 物理节点 weak scaling 的最新可见记录仍是被调度器终止的 7619133，没有看到新的补交作业。

原始有效结果：

- [7637010 summary](../../runs/scaling/multinode/openclaw-fixed_frontend-n2-7637010.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/summary.json)
- [7637012 summary](../../runs/scaling/multinode/openclaw-fixed_frontend-n4-7637012.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/summary.json)
- [7637011 summary](../../runs/scaling/multinode/minisweagent-fixed_frontend-n2-7637011.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/summary.json)

本次只读检查调度/运行数据并生成本报告；没有启动、取消、重交作业或修改运行配置。
