# 多节点脚本实际 smoke 验证

2026-09-14，使用已有 allocation `7608115`，两类任务均完整通过，总耗时约 12 分钟。没有提交新 job。

前端节点 `x3005c0s31b0n0`，推理节点 `x3005c0s7b0n0`。调用与正式 PBS 脚本相同的 multi-node coordinator、inference service、frontend worker 和结果汇总代码。Qwen3.6-35B-A3B、TP4、context 262144、batch token budget 2048、thinking/APC 及 replay profile 均与测试时的待提交配置一致。

每类选择两条完整短 trace，cc2，30 秒 warmup、60 秒 measurement，窗口结束后已开始的任务自然完成。各自使用新的推理服务，完整测试启动、重放、采样、汇总与清理。

| 项目 | OpenClaw | mini-SWE |
|---|---:|---:|
| 完整 trace 重放次数 | 8 | 3 |
| LLM calls | 16 | 82 |
| Tool calls | 8 | 82 |
| 全程输出 tokens（含 warmup/drain） | 11,580 | 15,975 |
| 60 秒窗口内输出 tokens | 6,469 | 10,630 |
| 客户端 / 后端 / Prometheus 计数校验 | 通过 | 通过 |
| 两节点硬件与后端采样、窗口覆盖校验 | 通过 | 通过 |
| 汇总 validation errors / diagnostic warnings | 0 / 0 | 0 / 0 |
| 协调器与自有服务正常退出 | 通过 | 通过 |

mini-SWE 中有 7 次命令返回 error，逐节点核对均与原始 trace 的 error 状态一致，没有新增 error 转换；全部 replay 进程成功完成。`steady=false` 是 smoke 的预期标记，不表示验证失败。

本次实机覆盖 **1 个推理节点 + 1 个前端节点**。多个推理副本同时启动、正式并发与全量 trace 的长时间运行，以及排队系统抢占不在这次短测覆盖范围内。

验证数据见 [summary.json](summary.json)。完整运行目录在该文件 `run_directory` 字段中；协调器日志为 `runs/multinode-smoke-existing-20260914/coordinator-c.log`。此前 a/b 两次由用户调整工作顺序而中断的记录没有纳入通过结果。

待提交的 weak scaling 列表为 `runs/multinode-jobs-formal-20260913/submit-weak-scaling.txt`，共 10 个任务；正式单副本并发为 OpenClaw 16、mini-SWE 32。Smoke 完成后，按用户决定将待提交配置的预算统一改为 8192；本报告的实机 smoke 使用 2048，未重新运行。
