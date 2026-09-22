# Routing 与前端瓶颈：三份后续提交脚本

本批为 4 个实验、3 个 PBS 作业，均申请 **10 节点、prod、3 小时**，由用户手动提交。独立于已准备的 1+8 RR/cache-aware 四点。

| PBS 脚本 | 包含实验 | 实际节点分配 | 总 task cc | sandbox 创建名额 |
|---|---|---|---|---|
| `at-p2-n4.pbs` | OpenClaw 与 mini-SWE，官方 `power_of_two` | 各 1+4，两个实验各占独立 5 节点并行 | 64 / 128 | mini-SWE 为 4 |
| `at-ms-f2-n8-b2.pbs` | mini-SWE，增加前端资源 | 2 前端 + 8 推理后端 | 256 | 每前端 2，总计 4 |
| `at-ms-f1-n8-b8.pbs` | mini-SWE，放宽环境创建限制 | 1 前端 + 8 推理后端，闲置 1 节点 | 256 | 8 |

## 配对与控制变量

P2 从已完成 1+4 RR 的冻结配置复制，仅改变路由策略为官方 `power_of_two`，与已有 RR/cache-aware 比较。继续使用 vllm-router 0.1.15 和相同聊天协议适配、请求超时、重试策略及后端配置。它用于观察无前缀亲和的负载反馈效果，不能直接把策略间差值当作缓存贡献百分比。

两个 mini-SWE 前端控制点从已完成的 1+8 sticky/direct 配置复制，保留 8 个独立 TP4 后端、每逻辑 worker cc32、总 cc256、seed 42–49 和完整 trace pool：

- **2+8 / build=2**：每前端带 4 个后端对应的 worker、cc128；构建锁位于各自节点的 `/local/scratch`，因此合计 4 个创建名额。
- **1+8 / build=8**：同一个前端带 8 个 worker，只把该前端的创建名额从 4 改为 8。
- 两点都采用原来的直接 task-sticky 路径，不引入 router。增加前端会同时增加 CPU、内存和本地 I/O 等资源，应结合观测解释，不能把所有收益自动归为 CPU。

Qwen3.6-35B-A3B、TP4、context 262144、batch token budget 8192、max sequences 64、APC/thinking 沿用基线。OpenClaw 为 110 条 trace、10 min warmup + 60 min measurement；mini-SWE 为 61 条、30 + 90 min。mini-SWE 各前端在测量前把共享 SIF 复制到本地，继续记录创建等待与实际 build 耗时。

保留 30 min drain。主要比较完整测量窗口；收尾不计入窗口吞吐。收尾超时可能仍被现有 runner 标记为失败并跳过主汇总，需按原始日志验证窗口可用性，不能用被截断的任务集合报告完整尾延迟。本批不修改测量/收尾判定逻辑。

## 提交命令

```bash
cd /lus/eagle/projects/lc-mpi/ZhijingYe/SIGMETRICS_2027/trace_framework
qsub runs/mechanism-followups-20260921/jobs/at-p2-n4.pbs
qsub runs/mechanism-followups-20260921/jobs/at-ms-f2-n8-b2.pbs
qsub runs/mechanism-followups-20260921/jobs/at-ms-f1-n8-b8.pbs
```

三份都是新增作业，不需要取消已有任务。包内实验结果目录独立；P2 包里一个实验失败不会自动取消另一个。

## 文件与验证

- 冻结配置与提交脚本：`runs/mechanism-followups-20260921/`。
- 原始基线：`runs/prod-packs-20260921/configs/`。新配置的 `baseline_config` 记录具体来源。
- 准备检查：bundle 中的 `validation.json`，包含 trace 数量、拓扑、worker 分配、总并发和 build 名额。
- 包级日志：bundle 中的 `pack-<PBS_JOBID>-<pack-id>.log`、`allocations/<pack-id>-<PBS_JOBID>/`。
- 实验数据：`runs/scaling/multinode/<point-id>-<PBS_JOBID>/`，四个 point ID 为 `openclaw-routing-n4-power_of_two`、`minisweagent-routing-n4-power_of_two`、`minisweagent-frontend-f2-n8-build2`、`minisweagent-frontend-f1-n8-build8`。

重新准备到新目录：

```bash
.venv/bin/python examples/scaling/prepare-mechanism-followups.py \
  --output runs/mechanism-followups-new
```

准备程序复用 runner 的本地输入检查和已有 prod 包装器，只检查必要的 schema、记录数、引用路径、SIF 存在性、官方 router 安装、节点分配和 PBS 语法。不会覆盖旧 bundle。

本次运行代码仅扩展两处：节点映射支持把多个推理 worker 均匀分给多个前端，官方 router 启动封装允许 `power_of_two`。原来 1 个前端或前后端等数量时的映射保持相同。验证使用本地模拟工具/后端及真实官方 router，不包含计算节点上的 Qwen + 工具 smoke。
