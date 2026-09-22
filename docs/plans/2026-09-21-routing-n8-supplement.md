# Routing 最小补充实验：1 前端 + 8 推理后端

只补 OpenClaw、mini-SWE 各自的官方 RR 与 cache-aware，共 4 点，与已有的 1+4 routing 结果比较。暂不增加 1+2、1+16、2+8 或其他策略。

| 提交脚本 | 任务 | 路由 | 实验节点 | PBS 申请 | 总 task cc | warmup / measurement |
|---|---|---|---|---|---|---|
| `at-r-oc-rr-n8.pbs` | OpenClaw | round_robin | 1+8 | 10 / prod / 3h | 128 | 10 / 60 min |
| `at-r-oc-ca-n8.pbs` | OpenClaw | cache_aware | 1+8 | 10 / prod / 3h | 128 | 10 / 60 min |
| `at-r-ms-rr-n8.pbs` | mini-SWE | round_robin | 1+8 | 10 / prod / 3h | 256 | 30 / 90 min |
| `at-r-ms-ca-n8.pbs` | mini-SWE | cache_aware | 1+8 | 10 / prod / 3h | 256 | 30 / 90 min |

每个作业使用 9 个节点，额外 1 个节点闲置，以满足 prod 的最小申请规模。沿用现有子分配包装器，把 9 节点 nodefile 交给原 runner，不把第 10 个节点算作后端。四个作业独立提交。

模型为 Qwen3.6-35B-A3B，每后端 TP4、context 262144、batch token budget 8192、max sequences 64、APC/thinking 开启。沿用官方 vllm-router 0.1.15、原有 router 参数和 seed；OpenClaw 为 110 条 trace、cc16/backend，mini-SWE 为 61 条、cc32/backend。mini-SWE 每前端仍有 4 个环境创建名额，并在测量前把共享 SIF 缓存复制到前端本地盘。web_search 仍按 recorded delay 回放。

## 收尾与结果口径

本轮优先比较固定窗口内的 token/task 吞吐、前缀复用、逐后端 running/waiting、CPU/GPU 使用和时间序列。收尾阶段不计入固定窗口吞吐；保留原来的 30 分钟 drain，不为完整任务尾延迟扩大到 25 节点或 6 小时。

窗口跑满还需核对日志覆盖、计数和异常。Token 吞吐按窗口内的 token 事件统计，任务吞吐按窗口内完成的任务统计，不能只统计最终成功任务中的调用或把窗口后完成的任务计入吞吐。请求级缓存统计如依赖完成记录，也需检查窗口内请求是否有缺失；不能因为时长足够就直接认定每个指标完整或性能已稳态。

收尾超时会导致未完成任务被取消，完整任务/调用延迟 cohort 被截断；不可把剩余成功样本的 p95 当作完整尾延迟。现有 runner 此时仍标记失败并跳过主 `summary.json`，但 worker 汇总、后端 token 日志和监控采样保留。分析时单独验证并恢复完整测量窗口，保留原失败状态；不能仅凭 PBS/runner 的失败标签作废已收集窗口，也不能把窗口有效等同于整轮完全成功。已有 mini-SWE 1+4 RR 的恢复分析可参考 `reports/2026-09-21-routing-and-fixed-frontend/`。

mini-SWE 的 1+8 已有前端接近饱和的证据，因此 routing 增益需要与前端 CPU、sandbox 等待和后端供给共同解释；本批不同时修改 build 名额或前端数量。

## 提交

```bash
cd /lus/eagle/projects/lc-mpi/ZhijingYe/SIGMETRICS_2027/trace_framework
qsub runs/routing-n8-prod-20260921/jobs/at-r-oc-rr-n8.pbs
qsub runs/routing-n8-prod-20260921/jobs/at-r-oc-ca-n8.pbs
qsub runs/routing-n8-prod-20260921/jobs/at-r-ms-rr-n8.pbs
qsub runs/routing-n8-prod-20260921/jobs/at-r-ms-ca-n8.pbs
```

本批不替代已有排队任务，没有需要随之取消的旧作业。只提交以上 `prod` bundle 中的四份脚本；旁边 `-sources` 目录是中间冻结输入。

- 冻结配置、提交脚本：`runs/routing-n8-prod-20260921/`。
- 作业日志及节点映射：该目录的 `pack-<PBS_JOBID>-<pack-id>.log` 和 `allocations/`。
- 实验数据：`runs/scaling/multinode/<workload>-routing-n8-<policy>-<PBS_JOBID>/`。
- 重新准备到新目录：`bash examples/scaling/prepare-routing-n8.sh runs/routing-n8-prod-new`。只准备文件，不提交作业，不覆盖旧 bundle。

准备阶段检查 trace 数量、schema、引用路径、SIF 缓存存在性、router 安装和 walltime 预算；再检查生成脚本语法、9+1 节点映射与冻结参数。没有启动计算节点实验。
