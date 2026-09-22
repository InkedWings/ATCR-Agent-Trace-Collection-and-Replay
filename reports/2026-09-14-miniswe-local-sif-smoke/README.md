# mini-SWE 节点本地 SIF 与完整流程 smoke

2026-09-14，使用现有 capacity allocation `7608115` 完成验证，未提交新 job。前端为 `x3005c0s31b0n0`，推理节点为 `x3005c0s7b0n0`。

**完整流程通过：`status=completed`、`valid=true`，validation errors 和 diagnostic warnings 均为空。**

通过正式 multi-node coordinator、frontend、inference service 和汇总代码执行。配置为 Qwen3.6-35B-A3B、TP4、context 262144、batch token budget 8192、thinking/APC 开启；cc2、warmup 30 秒、measurement 60 秒，之后自然 drain。

| 检查 | 结果 |
|---|---|
| 共享 SIF → 节点本地缓存 | 61/61，64.66 GiB，复制耗时 73.11 秒 |
| 再次调用缓存准备 | 61 个全部复用，复制 0 个 |
| 本地镜像路径 | `/local/scratch/zye25/agenttrace-multinode/minisweagent-images` |
| 短 trace | `django__django-11848`、`django__django-11099`，均完整重放 |
| 完整任务 / LLM calls / tool calls | 4 / 106 / 106 |
| 全程输出 tokens（含 warmup/drain） | 22,480；客户端与后端一致，所有输出目标满足 |
| 60 秒测量窗口内输出 tokens | 9,870；窗口内完成任务 2 个 |
| 客户端、后端、Prometheus 计数及采样校验 | 通过 |
| 全程 native tool errors | 10 次，全部与原 trace 的 error 状态一致，新增错误 0 次 |
| 任务节点集合与结果文件 | 与原 trace 一致，report/events/console/status 均存在 |
| Sandbox 构建与清理 | 2 次预热＋4 次任务，共 6 个独立本地路径；结束后均已删除 |
| 协调器和自有推理服务退出 | 正常，退出码 0 |
| 本地缓存与 multi-node 单元测试 | 27 passed |

短测使用一个 Apptainer wrapper，在 `build` 阶段拒绝 registry URI，并将外网代理设为不可达地址。6 次构建的镜像源均为节点本地 SIF。正式 job 继续使用原 Apptainer executable；共享镜像复制和本地路径解析已接入正式 frontend，无需修改旧 PBS 提交脚本。

本次实机覆盖 **1 个推理节点＋1 个 frontend、cc2**。完整 61 个镜像已复制，但仅对选定的两条完整短 trace 执行重放；正式 cc32、多 frontend 同时运行和长时间性能不由这次 smoke 代替。`steady=false` 是短 smoke 的预期标记，不表示流程失败。

## 提交 mini-SWE weak scaling

沿用既有配置：每推理副本 cc32，warmup 1800 秒、measurement 5400 秒。`n` 表示推理节点数，物理节点总数为 `2n`。

```bash
cd /lus/eagle/projects/lc-mpi/ZhijingYe/SIGMETRICS_2027/trace_framework
qsub runs/multinode-jobs-formal-20260913/jobs/minisweagent-balanced-n2.pbs
qsub runs/multinode-jobs-formal-20260913/jobs/minisweagent-balanced-n4.pbs
qsub runs/multinode-jobs-formal-20260913/jobs/minisweagent-balanced-n8.pbs
qsub runs/multinode-jobs-formal-20260913/jobs/minisweagent-balanced-n16.pbs
```

前两项申请 4/8 个物理节点，preemptable 3 小时；后两项申请 16/32 个物理节点，prod 3/6 小时。N=1 按已确认计划留给单节点 cc32 的正式长窗口补测，不能用本次 smoke 代替。

提交前查询时，OpenClaw N=4 和 N=16 仍在排队，N=8 正在运行；没有 mini-SWE job 在队列中。这里仅提供提交命令，没有执行 `qsub`。

## 记录

- [summary.json](summary.json)：配置、计数、缓存复制与复用、工具错误复核。
- [短测配置和启动脚本](../../runs/multinode-sif-smoke-20260914/)：`config.json`、`run.sh`、`coordinator.log`、`offline-apptainer.sh`、`local-builds.log`、`cache-reuse-check.json`。
- 原始运行：`runs/scaling/multinode/minisweagent-sif-smoke-n1-20260914-7608115.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/`。
