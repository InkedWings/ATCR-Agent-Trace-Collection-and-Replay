# 将待跑实验打包到 10 节点 prod 作业

2026-09-21：用户决定把两个 5 节点实验合并申请 10 节点、同时运行；9 节点实验申请 10 节点、使用其中 9 个。mini-SWE fixed-frontend 1+4（7637013）继续保留原来的 preemptable 排队。当前 prod/small 的最小节点数为 10，small 最大 walltime 为 3 小时，已通过 `qstat -Qf prod small` 核对。

## 新的提交清单

| PBS 脚本 | 包含的实验 | 节点使用 | 替代的旧排队作业 |
|---|---|---|---|
| `at-prod-oc-route.pbs` | OpenClaw 官方 RR 与 cache-aware | 5+5，各自 1 前端+4 后端 | 7637645、7637646 |
| `at-prod-ms-route.pbs` | mini-SWE 官方 RR 与 cache-aware | 5+5，各自 1 前端+4 后端 | 7637647、7637650 |
| `at-prod-oc-n8.pbs` | OpenClaw fixed-frontend 1+8 | 使用 9，空闲 1 | 7637014 |
| `at-prod-ms-n8.pbs` | mini-SWE fixed-frontend 1+8 | 使用 9，空闲 1 | 7637015 |

四个 PBS 作业全部为 **prod / select=10 / 03:00:00**，由用户手动提交。

两个 routing 实验在不同的 5 节点子集同时运行；每个子集的第一个节点担任自己的 frontend 和 coordinator。分别设置子集 `PBS_NODEFILE`，保留真实父作业 `PBS_JOBID` 用于查询剩余 walltime。节点、router、vLLM、工具 sandbox、测量窗口与结果目录彼此独立。同一包中的一个实验失败后，另一个继续收集；父作业被停止或连接断开时，子实验会走现有清理流程。

9 节点实验只收到 9 节点子集，额外节点记入 `unused_nodes`，不会成为额外的后端。结果中的 `physical_nodes` 仍表示实验实际使用的 5 或 9 个节点；`allocation_pack` 记录整个 PBS 分配为 10 节点。计算申请资源成本时应使用整个包的 10 节点，不能把空闲节点成本隐藏到实验吞吐效率中。

模型、TP4、context 262144、budget 8192、trace pool、每副本 cc、warmup/measurement、drain 和官方 router 设置均从原冻结配置复制。两个 mini-SWE routing 实验各有自己的 frontend，分别使用已有的 4 个 sandbox 创建名额。共享 SIF 继续在测量前复制到各前端本地盘；测量期间工具依旧真实执行，只有 web_search 为 recorded delay。模型/SIF 启动读取共享文件系统可能相互影响准备时间，因此完整窗口仍由各子实验按原 runner 规则判断；不因打包缩短窗口。

## 提交命令

新 bundle：`runs/prod-packs-20260921/`。准备过程没有修改旧 bundle，也没有调用 qsub、qdel 或 qmove。

先核对下列旧作业仍为 Q，再取消这 6 个被替代的作业。**不要取消 7637013（mini-SWE 1+4）或 7636974（capacity）**。

```bash
cd /lus/eagle/projects/lc-mpi/ZhijingYe/SIGMETRICS_2027/trace_framework
qstat -u zye25
qdel 7637645 7637646 7637647 7637650 7637014 7637015

qsub runs/prod-packs-20260921/jobs/at-prod-oc-route.pbs
qsub runs/prod-packs-20260921/jobs/at-prod-ms-route.pbs
qsub runs/prod-packs-20260921/jobs/at-prod-oc-n8.pbs
qsub runs/prod-packs-20260921/jobs/at-prod-ms-n8.pbs
```

不要再重复提交原来的 6 个独立脚本。如果取消前发现旧任务已经开始运行，先核对该任务再决定是否替换。

## 查看结果与复现准备

- 包级布局、子集 nodefiles、coordinator 日志与退出状态：`runs/prod-packs-20260921/allocations/<pack-id>-<PBS_JOBID>/`。
- 各实验的主数据位置保持为 `runs/scaling/multinode/<point-id>-<PBS_JOBID>/`。同包的两个实验共享父 Job ID，但 point ID 不同，不会覆盖结果；分析时分别读取各自 summary。
- 源清单：`examples/scaling/prod-packs-20260921.json`。
- 包装程序：`src/agenttrace/experiments/multi/packed.py`。现有多节点 runner 与旧提交脚本保持原样。

如需重新生成到新的目录：

```bash
.venv/bin/python -m agenttrace.experiments.multi.packed render \
  --spec examples/scaling/prod-packs-20260921.json \
  --output runs/prod-packs-new
```

6 项本地测试全部通过，覆盖 5+5 / 9+1 节点映射、PBS 重复主机行、原冻结实验参数保留、独立子任务退出状态、单个失败不取消另一个，以及父进程取消后的子进程清理。生成的 4 份 PBS 脚本全部通过 `bash -n`；6 个实验的冻结参数、trace 数量与引用路径检查通过，记录见 `runs/prod-packs-20260921/validation.json`。此验证使用本地子进程和模拟主机，不启动 GPU 实验，不能替代实际计算节点启动验证。
