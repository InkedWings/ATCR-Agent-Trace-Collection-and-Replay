# 多节点 scaling：配置、提交与分析

入口：`python -m agenttrace.experiments.multinode`。在 `trace_framework` 根目录使用 `.venv/bin/python`。
`check`、`matrix`、`render`、`summarize` 都在登录节点运行；只有生成的 PBS 脚本会调用 `run`。
生成器不执行 `qsub`，也不会启动推理服务。每个实验点是一个独立 job，首次 sweep 每点只跑一次。

官方 `vllm-router` 的 RR / cache-aware 对照使用独立的 `--group routing`，配置为
`examples/scaling/routing.json`，详见 [实验方案与提交命令](plans/2026-09-19-routing-baselines.md)。
生成命令为 `bash examples/scaling/prepare-routing.sh <新 bundle 目录>`；原有默认 26 点矩阵保持不变。

## 与文章大纲对应的实验问题

对应 [文章大纲](../../sigconf.tex) 的 **End-to-End Benchmarking / Resource Scaling**、
**Cross-layer Profiling** 和 **Microbenchmarking / Inference-Heavy Workflows**：

1. **整体资源一起增加时能否保持单副本效率？** 推理节点和 frontend/tool 节点都从 1 增加到 16，
   每个逻辑副本保持已确定的 task concurrency。比较 task throughput、token throughput、延迟和资源效率。
2. **只加推理节点，frontend 会不会成为瓶颈？** 固定一个 frontend/tool 节点，增加推理副本，
   对照 frontend CPU、I/O、task setup、tool latency、请求并发以及推理侧等待和空闲情况。
3. **路由导致的 prefix locality 变化贡献了多少？** 在 4 个推理节点上交叉比较 task-sticky 与
   round-robin，以及正常复用与每次 LLM call 独立的 cache salt。

本轮固定 TP4，增加独立推理副本；不引入跨节点 tensor parallel。Sticky 是静态 task affinity，
不等同于读取 KV 状态的动态 cache-aware router。大纲中的 prefill/decode disaggregation、
不同工具部署、Parsl、DAG 改造留作后续独立实验。

## 固定条件和实验矩阵

配置模板：[examples/scaling/multinode.json](../examples/scaling/multinode.json)。

| 项目 | 设置 |
|---|---|
| 推理服务 | 每推理节点一个 Qwen3.6-35B-A3B vLLM 0.19.1 实例，TP4，BF16，4 张 GPU |
| Context / scheduler | context 262144；max_num_seqs 64；GPU memory utilization 0.90 |
| Thinking / APC | 新生成开启 thinking；所有实验 APC 均开启 |
| 模型缓存 | 继续使用 launcher 的 `/lus/eagle/projects/lc-mpi/ZhijingYe/Models`，含 huggingface、vllm-cache |
| OpenClaw | 固定 110 条；warmup 600 秒，measurement 3600 秒 |
| mini-SWE-agent | 固定 61 条；warmup 1800 秒，measurement 5400 秒 |
| 工具 | 仅 `web_search` 使用 recorded delay；其他工具使用当前 native executor |
| Replay | 保留完整 DAG、原始输入和输出 token 目标；`ignore_eos=true`；自然 drain |
| 随机序列 | 每个逻辑副本读取整个 trace pool，逐轮 shuffle；副本 j 的 seed 为 42+j |
| 统计 | 每点一次，探索性趋势；不画跨 run 置信区间，不据此声称 SLO capacity |

保留现有 vLLM 输入行为：历史消息独立的 `reasoning_content` 不会自动变成历史 prompt 文本。
这次不做 reasoning remap。开启新生成的 thinking 与保留历史 reasoning 是两件事。

| 类型 | 推理节点 N | Frontend/tool 节点 | 总节点 | 两个 workload 共计 |
|---|---|---|---|---|
| balanced | 1 / 2 / 4 / 8 / 16 | N | 2N | 10 点 |
| fixed_frontend | 2 / 4 / 8 / 16 | 1 | N+1 | 8 点 |
| mechanism | 4 | 4 | 8 | 2 routing × 2 reuse × 2 workloads = 8 点 |

N=1 的 balanced 和 fixed_frontend 共用基线，不重复跑。总计 **26 个正式 job**，最大 **32 个节点**。
仅 warmup + measurement 合计约 **405.3 node-hours**；缓存准备、模型启动、drain、smoke 和单节点校准另计。
PBS account 为 `lc-mpi`，根据每个 job 的**总物理节点数**自动选队列。2026-09-13 已用
`qstat -Qf preemptable`、`qstat -Qf prod small medium` 核对，并参考
[ALCF 队列说明](https://docs.alcf.anl.gov/polaris/running-jobs/#queues)：

| 总物理节点数 | 提交队列 | 当前矩阵中的点数 | Walltime | Drain 预算 |
|---|---|---|---|---|
| ≤10 | preemptable | 20 | 3 小时 | 30 分钟 |
| 16 / 17 | prod（路由到 small 档） | 4 | 3 小时 | 30 分钟 |
| 32 | prod（路由到 medium 档） | 2 | 6 小时 | 2 小时 |

26 个点都有队列安排。`pbs.max_nodes=10` 与 `pbs.overflow_routes` 保存选队列规则：
11–24 节点通过 prod 申请 3 小时，25–496 节点通过 prod 申请 6 小时。
脚本只提交到 `prod`，不直接指定 small 或 medium；也不补空闲节点来改变路由档位。
每点的 queue、walltime 和 drain 预算都会冻结到 job 配置及 `matrix.json` 中。

3 小时任务（包括 preemptable smoke）保持相同的 warmup/measurement 窗口，drain 上限为 30 分钟。
mini-SWE 的窗口合计 2 小时，再预留 30 分钟 drain，因此模型启动、环境准备和检查必须在约 30 分钟内完成。
若准备后剩余时间不足，程序退出；若 drain 超时，该点无效，不拼接、不截短窗口当作成功。

Balanced 中每个 frontend 驱动一个逻辑副本。Fixed_frontend 中同一个 frontend 驱动 N 个逻辑副本，
总 task concurrency 是 `N × task_cc_per_replica`。每个副本都使用完整 trace pool，避免 N 增长时
每个副本的 working set 缩小而人为改善缓存命中。相同 seed 固定每个副本的 trace 序列，
不同机器的完成速度仍会改变跨副本的全局交错顺序。

主 scaling 点直接访问分配的推理 endpoint。4 个 mechanism 组合都经过每个 frontend 上的同一种
流式 HTTP proxy，避免只给某个组合额外增加 proxy。Round-robin 按每个 frontend 的 LLM call 轮转；
sticky 将同一 task 的调用送回 home replica。`prefix_reuse=none` 给每次调用单独的 `cache_salt`，
隔离跨 call 的前缀复用，同时保持 APC 和 KV 配置一致；不删除或改变消息、工具 schema 和输出目标。
Proxy 记录目的副本与状态，不记录请求或生成文本，不重试、不切换到备用副本。

## 本轮使用的单节点参数

本轮 weak scaling 保留上一轮的单副本并发，按 2026-09-14 的决定将 batch token budget 改为 8192，模板已经填好：

- `serve.VLLM_MAX_NUM_BATCHED_TOKENS = 8192`
- `workloads.openclaw.task_cc_per_replica = 16`
- `workloads.minisweagent.task_cc_per_replica = 32`

这三个值在整个矩阵中固定，N=1 仍使用本轮更长的 warmup/measurement 窗口重新测量，作为 scaling 参照。
这些默认值用于探索性 weak scaling，尚不代表已确认的稳态最大容量。
后续如果改做容量基线，应先完成充分 warmup 的单节点验证，再选择
**稳定测量点中，达到峰值 task throughput 95% 的最小已测 concurrency**，并生成独立的一套配置与结果。

```bash
.venv/bin/python -m agenttrace.experiments.multinode check
.venv/bin/python -m agenttrace.experiments.multinode matrix
```

`check` 检查 trace 数量、schema、必要的引用路径、profile 和运行依赖路径；不会修改数据集或执行推理。
当前池路径和 profile 被保存在模板中。`render` 会把完整 trace 路径列表和 profile 内容冻结进每个 job 配置，
trace artifact 留在原位。改变基线或 pool 后应生成新 bundle，避免手工改动已提交的配置。

## mini-SWE 镜像先顺序缓存

mini-SWE 的正式和 smoke job 均从 `runs/cache/minisweagent-images/index.json` 查找完整共享 SIF，
在启动推理服务前检查本次 pool 引用的镜像是否齐全。每个 frontend 在测量前将镜像逐个复制到
`/local/scratch/$USER/agenttrace-multinode/minisweagent-images`，已登记且大小一致的本地镜像直接复用。
复制使用临时文件，完成后再发布；中断后重新启动会跳过已完成镜像。
每个 replay 从节点本地 SIF 创建独立可写 sandbox，不共享可写环境，也不在运行时回退到 Docker Hub。
任务结束删除 sandbox，保留本地 SIF。Smoke 同样复制完整来源 pool，但只重放选定的两条短 trace。
各 frontend 的 `image-cache.json` 记录复制/复用数量、路径、字节数和耗时，`profile.json` 指向实际本地缓存。

在一个已分配节点上顺序准备一次：

```bash
bash examples/scaling/cache-miniswe-images.sh \
  runs/scaling/qwen36-expanded-7594555-20260908/pools/minisweagent/traces.txt \
  "$PWD/runs/cache/minisweagent-images"
```

缓存目录必须是所有 frontend 可访问的共享路径。要换位置，在提交环境导出
`AGENTTRACE_MINISWE_IMAGE_CACHE` 并确保 PBS 传入，或在 replay profile 的 tool executor config 设置
`image_cache_dir`。协调器将最终路径写入运行配置。默认目录不需要另加 PBS 环境变量。

准备程序逐个拉取，已完成镜像直接复用；Docker Hub 限流时等待 5 分钟后重试同一个镜像，
不会并行拉取。`status.json` 给出 completed/total 和当前状态；`index.json` 只登记可复用的镜像路径，
`*.sif.log` 保存单个拉取日志。镜像准备时只检查镜像能启动且含 `/testbed`，不执行 trace 或 LLM。

下载层仍在各节点 `/local/scratch/$USER/agenttrace-multinode/apptainer-cache`，
解压临时文件在 `/local/scratch/$USER/agenttrace-multinode/image-preparation/`。
这些节点本地目录不替代共享 SIF 缓存；仅有 OCI layer cache 时，`docker://` 构建仍可能访问 registry。

## Smoke 与正式提交

单推理节点已经验证过，跳过 2 个物理节点的 N=1 smoke；这不删除正式矩阵中的 N=1 测量基线。
现在只生成 2 个 smoke job：两个 workload，各跑 N=2（2 个推理节点 + 2 个 frontend，共 4 节点）。
每个使用两个最短的、同时包含 LLM
与非 `web_search` native tool 的完整 trace；cc=2，warmup 30 秒，measurement 60 秒，之后等待完整任务结束。
两个 job 均检查 round-robin/独立 cache salt，验证跨推理副本的请求路由与缓存隔离。
Smoke 沿用配置中的 token budget，未填写时默认使用 8192；短窗口 **不能当作 capacity 基线**。
完成时间可能因完整 trace 的工具运行和 drain 明显超过 90 秒。

```bash
.venv/bin/python -m agenttrace.experiments.multinode render \
  --smoke --output runs/multinode-jobs-smoke-preemptable

# 以下命令由你手动执行；按 submit-commands.txt 逐个提交并检查结果。
qsub runs/multinode-jobs-smoke-preemptable/jobs/openclaw-smoke-n2.pbs
# OpenClaw 完成并通过验证后，再执行下面一条。
qsub runs/multinode-jobs-smoke-preemptable/jobs/minisweagent-smoke-n2.pbs
```

每个 smoke 必须 `status.json` 为 `completed`、`summary.json` 为 `valid: true`，
且计数、路由、完整 task、native 工具和采样都无异常，才继续正式 sweep。
`steady: false` 在短 smoke 中是预期行为，不用 smoke 的短窗口判断稳态。

正式提交配置可另存一份；当前默认值已经固定，可直接生成：

```bash
cp examples/scaling/multinode.json runs/multinode-config.json
# 本轮默认 budget=8192，OpenClaw cc=16，mini-SWE cc=32。
.venv/bin/python -m agenttrace.experiments.multinode check \
  --config runs/multinode-config.json
.venv/bin/python -m agenttrace.experiments.multinode render \
  --config runs/multinode-config.json --output runs/multinode-jobs-formal
```

未填定基线时，正式 `render` 会明确报错，不生成半套文件。输出目录必须尚不存在，避免覆盖。
`jobs/*.pbs` 包含具体的 `select=2/3/4/5/.../32`，不是运行时 shell 变量。
`matrix.json` 提供完整计划及各点的队列和时限；`submit-commands.txt` 提供手动提交命令，
生成后提交数始终为 0。当前默认生成 26 个正式脚本：20 个 preemptable、6 个 prod。
如果以后修改矩阵使节点数超出配置中的全部路由范围，仍会在 `deferred_points` 中明确列出。

默认 weak scaling 对应 `balanced`，单独的 **`submit-weak-scaling.txt`** 只包含它的 10 个 job：
`openclaw-balanced-n{1,2,4,8,16}.pbs` 和 `minisweagent-balanced-n{1,2,4,8,16}.pbs`。
每个逻辑副本保持固定 task cc，总 task cc 随 N 成比例增长。
另有 `submit-fixed-frontend.txt`（8 个点，共用 balanced N=1 基线）和 `submit-mechanism.txt`（8 个点）。
这三份是完整提交清单的分组视图，不要把同一个点在两个清单中重复提交。

2026-09-19 的单 frontend 实验使用独立配置 `examples/scaling/fixed-frontend.json`，
通过 `render --group fixed-frontend` 只生成该组 8 个任务。当前已生成
`runs/fixed-frontend-jobs-20260919/submit-fixed-frontend.txt`，包含 mini-SWE 跨进程环境创建限制和额外 CPU 观测。
矩阵、判断口径与首档命令见 [固定前端计划](plans/2026-09-19-fixed-frontend-scaling.md)。

建议按生成顺序依次完成：每档 N 先 balanced 两个 workload，再 fixed_frontend 两个 workload，
N 递增；最后跑 4-node mechanism。不要一次性执行全部提交命令造成测量 job 重叠。
这样更容易区分共享文件系统和其他同时运行任务对结果的影响。

## 每个 job 的执行流程与日志

1. 从 `PBS_NODEFILE` 去重读取物理节点，把 batch host 作为第一个 frontend，剩余节点按配置映射。
   节点数量不符时直接退出，不到其他 allocation 查找机器。
2. 检查剩余 walltime、输入和节点时钟。只启动并管理本 job 创建的服务进程组；端口被占用则报错。
3. 并行启动 N 个新 vLLM 实例；每个 frontend 在 measurement 外逐条准备 native 环境缓存。
   mini-SWE 先将共享 SIF 复制到节点本地缓存，再从本地镜像构建环境；task scratch 使用 job 专属目录。
4. 所有 frontend 的逻辑 worker 都 ready 后，复查剩余时间和时钟，原子发布共同的未来起始时间
   （提前 10 秒）。每个 worker 使用完全相同的 warmup/measurement 起止时间。
5. 在 measurement 截止时停止 admission，所有已经开始的 task 自然结束。
   Drain 上限按上述队列配置：3 小时任务为 1800 秒，6 小时 prod 任务为 7200 秒。
   超时或 allocation 不足则该点失败，不把残缺窗口标成成功。
6. 后端/前端异常退出时停止其他 owned worker，保留部分结果；成功 drain 后停止采样和服务，做全局校验。

时钟检查先建立每个节点的 SSH 连接并等待远端时钟响应程序 ready，再在同一连接上进行三次请求/响应采样。
SSH 建连、shell 和 Python 启动耗时不计入 RTT；取最小 RTT 样本，按
`|estimated offset| + RTT/2` 记录估计精度。时钟采样仅用于诊断，超过 0.5 秒、探针超时或不可用
只记录警告，不中断 job，也不因此把结果标记无效。每轮探针限时 10 秒，另留连接清理时间。
启动前、测量前和 drain 后都会采样。worker 轻微迟到时仍沿用共同的测量起止时间，
在 manifest 和 summary 中保留实际启动时间及偏移；只有整个测量窗口已过去才拒绝启动。
默认 cache preparation 超时 7200 秒。每个物理节点采样一次硬件，每个后端采样一次 Prometheus；
fixed_frontend 不会因有 N 个逻辑 worker 而重复采样该节点。

结果目录：`runs/scaling/multinode/<point-id>-<PBS_JOBID>/`。

| 文件 | 用途 |
|---|---|
| bundle 中 `coordinator-<jobid>-<point>.log` | 实时 coordinator 日志，不依赖 PBS 输出延迟 |
| `config.json` / `node_mapping.json` / `control/start.json` | 冻结配置、物理节点映射、共同窗口 |
| `clocks*.json` / `allocation.json` | 时钟和 allocation 检查 |
| `backends/rXX/service.log`、`serve-logs/` | 模型启动、vLLM 服务与清理日志 |
| `backends/rXX/backend.jsonl` | 后端 incremental token events、请求时序与缓存 token 计数 |
| `backends/rXX/inference-metrics.jsonl` | 该推理节点的硬件及 Prometheus 原始采样 |
| `frontends/<host>/service.log`、`precache/` | 环境准备与 frontend 执行日志 |
| `frontends/<host>/metrics.jsonl`、`routing.jsonl` | frontend 硬件指标；proxy 模式另有逐 call 路由记录 |
| `workers/rXX/tasks/<id>/` | 完整 task 的 report、events、console、status |
| `status.json` / `summary.json` | job 状态、验证结果、稳态诊断及统计 |
| `tasks.csv` / `calls.csv` / `replicas.csv` / `timeseries.csv` | 原始聚合样本、逐副本计数、60 秒窗口趋势 |

正常结束的 job 自动汇总；作图在登录节点通过下面的 `summarize` 完成。
如果失败，先看 coordinator，再看对应 `service.log` 和 task `console.log`。记录失败原因后修复并单独
重新 `qsub` 同一个点；新的 PBS job ID 生成独立输出，不覆盖旧 run。程序不自动重试整个 task 或 LLM call。
如果同一点有多个成功 run，汇总时显式指定所选目录；工具不会自动挑选“最好”的结果或把重跑当作重复实验。
抢占可能直接终止 job，因而可能来不及写 `status.json`。保持 `#PBS -r n`，不自动重启；
没有 completed 状态的结果会被汇总排除。被抢占的点需要手动重新提交，不能拼接两个窗口作为一次测量。

## 汇总口径与图表

```bash
.venv/bin/python -m agenttrace.experiments.multinode summarize \
  --runs runs/scaling/multinode --output reports/multinode-first-sweep

# 也可只传明确选中的几个 run；--no-plots 只输出 JSON/CSV/Markdown。
```

输出目录必须尚不存在。未完成或无效的 job 在 `excluded_runs` 中列出，不进入曲线。
不同 budget、cc、pool、profile、seed 或窗口配置不会被悄悄合并。
缺少 N=1 时仍可汇总其余点，speedup/efficiency 留空。

| 指标 | 口径 |
|---|---|
| Task throughput | 所有副本在共同窗口内完成的 task 数 / 窗口秒数 |
| Token throughput | 所有后端在共同窗口内产生的 incremental output tokens / 窗口秒数 |
| Scaling efficiency | `throughput(N) / (N × throughput(1))`，只比较配置相同的主 scaling 点 |
| Actual prefix reuse | 首 token 落在窗口内的请求：`Σ cached_prompt_tokens / Σ prompt_tokens` |
| Prefix lookup hit ratio | 各后端窗口内 Prometheus 的 `Σ hits / Σ queries`；重复查询可能被反复计入，与 actual reuse 分开 |
| Task / call p50、p95、p99 | 合并全部原始 latency，再取分位数；不平均各节点 p95 |
| Latency cohort | 窗口内 admission 的 task、窗口内开始的 call；保留直到 drain 完成的全部 latency |
| Backend queue / prefill / decode latency | 窗口内 first scheduled 的请求，沿用后端单调时钟，统计至完成 |
| Task execution breakdown | setup、LLM only、tool only、LLM/tool overlap、其余调度/进程/cleanup；并发区间不重复相加 |
| 并发 | 从真实 task 和 LLM call 区间得到时间加权平均和峰值 |
| CPU/GPU/memory/power | 原始每节点采样；硬件均值是采样均值 |
| Energy/task | 测量窗口内推理 GPU 功率的梯形积分 / 完成 task 数；不代表整机能耗 |
| I/O | 保留每设备 block I/O、每接口网络计数；不能把它们当成 Lustre 文件系统流量 |

后端完整请求/token 计数会与客户端记录的真实目的副本核对，同时检查 Prometheus 全程增量。
原始 token events 与 finished request 计数不符、目标输出不符、测量窗口不完整或不一致、采样报错、
GPU 数量不符都会标记无效。资源采样超过 5 秒的空洞仅记入 `diagnostic_warnings`，对应资源保留
`window_coverage_ok=false`，不因此否定完整的任务和 token 计数；能耗积分不会跨过超过 5 秒的空洞。
Prometheus 首末采样处于请求前和 drain 后，
不会用仅覆盖窗口内部的 counter 差值去强行匹配全程输出。

当前没有可用的 per-job Lustre 指标，报告明确标为 `unavailable`，不填 0。
Native tool 自身返回 error 会单独统计；它可能也是原始固定执行路径的一部分，不等同于 replay 进程失败。

稳态诊断比较 measurement 的前后两半：output rate 差异 ≤10%、task rate ≤20%、actual reuse
差异 ≤5 个百分点、每副本平均 waiting 增长 ≤`max(1, 前半均值×10%)`，且存在 admission cohort。
任一不满足则 `steady=false`。有效但非稳态的点保留，并在主曲线上用红色叉标记；先检查时间序列，
再决定是否统一延长窗口重跑。不要把 `valid=true` 直接解释为达到可持续最大容量。

汇总提供 `summary.csv/.json/.md`、逐节点 `resources.csv`，以及 PNG/PDF 图：主 scaling 曲线、
4-node mechanism 对照、逐点 60 秒 throughput/reuse 时间序列、逐副本负载、任务时间 breakdown、
KV cache 时间序列和 CPU/GPU/host-memory 对照。

## 本地验证与运行依赖

在现有环境中使用 `.venv`、现有 OpenClaw wrapper 和 Apptainer。
新环境可安装项目的 `scaling` extra（aiohttp、matplotlib）和 `test` extra。

```bash
.venv/bin/python -m pytest tests/test_multinode.py tests/test_multinode_proxy.py tests/test_multinode_report.py -q
bash -n examples/scaling/multinode-node.sh
```

测试使用本机模拟 HTTP backend 与 mocked PBS/service lifecycle，覆盖统一计时、多个逻辑副本只采样一次、
流式转发、真实目的副本记录、cache salt 隔离、断连取消、失败清理、加权缓存率、合并分位数和计数错误。
这些测试不提交 job。真实多节点 vLLM、跨节点网络与 native 工具环境仍须由上述 smoke job 验收。
