# 官方 vLLM Router：RR 与 cache-aware 对照

采用未经修改的官方 `vllm-router==0.1.15`，复用现有多节点启动、trace replay、SIF staging 和结果核对流程。路由算法由官方组件执行。本轮准备了 4 个正式任务及 4 个可选的短 smoke 任务，由用户手动提交。

2026-09-21 提交更新：四个正式实验改为按 workload 将 RR 与 cache-aware 两两打包，每包申请 10 节点 prod / 3h，各实验仍使用独立的 5 个节点。新的替代脚本与旧排队作业取消清单见 [prod 打包提交计划](2026-09-21-prod-packed-submissions.md)。下文保留最初的独立提交方案作记录，迁移后不要重复提交这些旧脚本。

## 要回答的问题

在相同 GPU、前端资源、任务并发和 trace pool 下，基于前缀缓存的路由能否比逐 LLM call 的 RR 减少 prefill、降低 TTFT，并提升任务和 token 吞吐？同时观察缓存亲和是否造成后端负载不均衡。

已有 balanced / fixed-frontend 数据是 task-sticky / direct，不能重新标成 RR。本组使用统一的官方 router 路径，独立启动后端测量。

当前部署为 N 个独立推理服务，每个 TP4、DP1；这是多副本服务的外部路由对照。vLLM 0.19.1 原生 `--data-parallel-size=N` 内部采用 running/waiting 负载评分，MoE 通信也可能不同，因此不把本组称为原生 DP 默认策略对照。[原生 DP 文档](https://docs.vllm.ai/en/v0.19.1/serving/data_parallel_deployment/)

## 正式实验矩阵

固定 **1 个 frontend/tool 节点 + 4 个推理节点**，router 与 replay 位于同一个前端节点。每个点独立申请 5 个物理节点。

| Workload | 官方 policy | 全局 task cc | Warmup | Measurement | 队列 / walltime |
|---|---|---:|---:|---:|---|
| OpenClaw | round_robin | 64 | 10 min | 60 min | preemptable / 3h |
| OpenClaw | cache_aware | 64 | 10 min | 60 min | preemptable / 3h |
| mini-SWE | round_robin | 128 | 30 min | 90 min | preemptable / 3h |
| mini-SWE | cache_aware | 128 | 30 min | 90 min | preemptable / 3h |

- Qwen3.6-35B-A3B、vLLM 0.19.1、TP4、context 262144、batch token budget 8192、max_num_seqs 64、GPU memory utilization 0.90，thinking/APC 开启。
- OpenClaw 使用 110 条 trace；mini-SWE 使用 61 条。每个逻辑 worker 读取完整 pool，seed 为 42 + worker index。
- 4 个逻辑 worker 各持有 cc16/32，全都向同一个 router 发请求。这里的 cc 是全局任务预算，不限制每个物理后端恰好处理 cc16/32；一次 task 的多次 LLM call 可以被分配到不同后端。
- 每个任务新启动全部后端和 router，缓存从冷状态开始，经过相同 warmup 后进入统一测量窗口。
- 两组均保留 APC，不使用 per-call cache salt，不改变原始 prompt、工具 schema、输出 token 目标或 thinking 参数。
- 仅 web_search 重放 recorded delay；其他工具真实执行。mini-SWE 从共享 SIF 缓存复制到前端本地盘，任务期间不访问 Docker Hub；同一前端最多同时创建 4 个 sandbox，记录创建等待与耗时。
- Router 禁用额外重试，后端错误直接返回并计入失败。健康检查与所有其余设置在两组保持一致。

四点的 warmup + measurement 合计 6 小时 20 分钟，约 31.7 node-hours；模型启动、SIF staging、drain 和排队另计。每点最多预留 30 分钟 drain。mini-SWE 的 warmup + measurement 为 2 小时，3 小时 allocation 留下约 30 分钟用于准备、30 分钟用于 drain。不会因等待资源而自动缩短测量窗口。

若一个 frontend 已经限制供给，应结合 frontend CPU / setup 等待和 backend 队列解释，不能把前端瓶颈下两策略接近解释为路由无收益。先完成这组，再决定是否扩展后端数量或增加前端资源。

## 官方实现与聊天协议适配

官方组件支持 `round_robin`、`cache_aware` 等策略，可连接静态 worker URLs，无需 Kubernetes。[官方仓库](https://github.com/vllm-project/router)、[版本页面](https://pypi.org/project/vllm-router/0.1.15/)

本次核对 0.1.15 发布源码发现两个与本任务相关的问题：

1. `ChatCompletionRequest.extract_text_for_routing()` 只读取 `session_params.session_id`；普通 messages 请求返回空字符串，直接设置 cache_aware 无法比较对话前缀。
2. ChatMessage 的类型转换不能完整保留所有输入形式，例如 system 的数组 content、assistant 的 reasoning_content。

因此两组采用相同的 **lossless_chat_via_completions** 适配：

```text
replay client
  → 官方 /v1/completions：prompt 为对话前缀表示，扩展字段携带完整原始 chat payload
  → 官方 round_robin 或 cache_aware 选择后端
  → 被选后端的 ASGI middleware 恢复 /v1/chat/completions 和原始 payload
  → 原来的 vLLM 推理路径
```

这是客户端和后端的协议适配，不额外增加 HTTP 转发进程，不修改官方 wheel，也不自行实现路由算法。官方 CompletionRequest 的扩展字段透传保留原始请求；恢复发生在 vLLM 解析聊天请求之前。响应仍直接流式转发。

路由前缀由 prompt 相关配置、工具定义和完整 messages 的确定性 JSON 表示组成，排除 max_tokens 等生成长度参数。前缀基于文本字符，不是 Qwen chat template 渲染后的精确 token 序列。完整历史随请求增长，可以参与前缀匹配；没有把 task ID 冒充 KV 内容。包裹请求会增加序列化和传输体积，两策略承担相同成本，解释 frontend CPU 时需要说明。

官方 cache_aware 根据历史请求维护近似前缀树，结合负载选择副本，不读取 GPU 当前完整 KV/GDN 驻留状态。对 Qwen hybrid 模型，预计前缀匹配不保证 attention KV 与 GDN 状态都能命中；通过后端实际 cached prompt tokens 验证收益。[策略源码](https://github.com/vllm-project/router/blob/main/src/policies/cache_aware.rs)、[精确 KV 观测 roadmap](https://github.com/vllm-project/router/issues/244)

本组论文标签建议为 **vLLM Router cache_aware（历史前缀估计，chat transport adapter）**，不标为精确 KV-aware 或未经适配的原生 Chat Completions 路径。

## 固定的 router 参数

| 参数 | 值 | 说明 |
|---|---:|---|
| version | 0.1.15 | 独立安装，Rust 实现 |
| cache_threshold | 0.3 | 官方该版本 CLI 默认值 |
| balance_abs_threshold | 8 | 本实验显式配置；默认 64 对 OpenClaw 全局 cc64 无法触发严格大于阈值的失衡判断 |
| balance_rel_threshold | 1.5 | 与绝对差阈值同时满足时按负载选择 |
| eviction_interval_secs | 120 | 路由估计树维护周期，不代表 GPU 缓存实际淘汰周期 |
| max_tree_size | 67108864 | 官方该版本 CLI 默认值 |
| request_timeout_secs | 10800 | 不引入比任务 walltime 更短的额外请求超时 |
| max_concurrent_requests | 32768 | 高于本组请求并发，避免 router 自身限流 |
| TOKIO_WORKER_THREADS | 4 | 固定 router 的异步运行线程数 |
| inference DP per worker URL | 1 | 每 URL 为一个独立 TP4 服务 |
| HTTP / metrics port | 18010 / 18011 | 仅绑定 frontend 的 loopback |
| retries / circuit breaker | disabled | 不通过额外重试隐藏实验失败 |

这是固定参数的官方策略对照，不声称所有参数都是 upstream 默认值。后续若需要把收益拆为负载均衡和缓存亲和两部分，再增加官方 power_of_two、consistent_hash 和阈值敏感性实验；本轮提交列表只包含 RR / cache-aware。

## 已准备的文件与提交命令

配置：`examples/scaling/routing.json`。

正式 bundle：`runs/routing-jobs-20260919/`；已冻结 profile、trace 路径列表、模型参数、router 参数和 PBS 资源配置。

在仓库目录运行：

```bash
cd /lus/eagle/projects/lc-mpi/ZhijingYe/SIGMETRICS_2027/trace_framework

qsub runs/routing-jobs-20260919/jobs/openclaw-routing-n4-round_robin.pbs
qsub runs/routing-jobs-20260919/jobs/openclaw-routing-n4-cache_aware.pbs
qsub runs/routing-jobs-20260919/jobs/minisweagent-routing-n4-round_robin.pbs
qsub runs/routing-jobs-20260919/jobs/minisweagent-routing-n4-cache_aware.pbs
```

建议按上述顺序核对每组状态与数据。提交命令也保存在 bundle 的 `submit-routing.txt`；脚本不会自动提交其他实验。

可选 smoke 位于 `runs/routing-jobs-20260919/smoke/jobs/`，同样使用 1 前端 + 4 后端、官方组件、完整工具环境和相同协议适配，选取两条较短完整 trace，每逻辑 worker cc2、warmup 30 秒、measurement 60 秒：

```bash
qsub runs/routing-jobs-20260919/smoke/jobs/openclaw-routing-smoke-n4-round_robin.pbs
qsub runs/routing-jobs-20260919/smoke/jobs/openclaw-routing-smoke-n4-cache_aware.pbs
qsub runs/routing-jobs-20260919/smoke/jobs/minisweagent-routing-smoke-n4-round_robin.pbs
qsub runs/routing-jobs-20260919/smoke/jobs/minisweagent-routing-smoke-n4-cache_aware.pbs
```

90 秒是 warmup + measurement，不包含模型启动、缓存复制和完整任务 drain，不能把 smoke 总时间理解为 90 秒。Smoke 只验证流程，不用于吞吐结论。

项目内已安装组件，新环境才需要执行安装脚本：

```bash
bash examples/scaling/install-vllm-router.sh
```

修改配置后生成新的 bundle，避免覆盖已提交配置：

```bash
bash examples/scaling/prepare-routing.sh runs/routing-jobs-new
```

上述脚本只安装或生成文件，不调用 qsub；计算节点运行时不安装包。旧 weak-scaling 和 fixed-frontend bundle 保持原设置。

## 采集与分析

每个运行目录为 `runs/scaling/multinode/<point-id>-<PBS_JOBID>/`，其中：

- `config.json`、`node_mapping.json`、`control/start.json`：冻结配置、物理映射和共同测量窗口。
- `frontends/<host>/router/config.json`、`service.log`、`metrics.jsonl`：官方 router 版本、完整启动参数、日志及每秒 Prometheus 采样。
- `backends/rXX/routing.jsonl`：实际接收请求的后端记录 task/call、目的副本、HTTP 状态和时间，不记录 prompt 或生成文本。后端返回 `X-Agenttrace-Replica`，客户端必须取得真实目的副本。
- `backends/rXX/backend.jsonl`、`inference-metrics.jsonl`：真实输出 token、prompt/cached prompt token、排队和 prefill/decode 时间、GPU/KV/队列指标。
- `frontends/<host>/metrics.jsonl`、`workers/rXX/`：前端资源、任务与 LLM/tool 时间、mini-SWE sandbox 创建等待。
- `summary.json`、`replicas.csv`、`timeseries.csv`、`status.json`：统一窗口统计及有效性核对。

正式主表：task/min、output tokens/s、task p50/p95、TTFT p50/p95、实际 prefix reuse、未复用 prompt tokens/s、各后端负载与吞吐差异、frontend CPU、GPU busy、sandbox 创建等待。用后端计数加权计算 cache ratio，不平均各副本比率。KV usage 仅作为辅助容量指标。

客户端完成调用、后端路由 ledger、后端 token/request 计数继续交叉核对。router 进程提前退出会使当前任务失败；不静默回退为自写 router。时钟精度、资源采样间隙和稳态诊断仍作为解释信息，不新增精确时钟门槛。

固定 seed 不保证不同策略在测量窗口内完成相同 trace 集合，需报告 trace coverage 与组成。第一轮每点一次，没有跨 run 置信区间；若形成论文主结论，再重复关键点并交替策略顺序。

用实际完成的四个运行目录汇总：

```bash
.venv/bin/python -m agenttrace.experiments.multinode summarize \
  --runs <OpenClaw-RR目录> <OpenClaw-cache-aware目录> <mini-SWE-RR目录> <mini-SWE-cache-aware目录> \
  --output reports/routing-rr-vs-cache-aware
```

## 验证范围

本地 60 项相关测试通过，包括真实官方 Rust router 的 RR、前缀亲和、负载回退、流式输出、错误不重试，聊天请求完整恢复、replay executor 接入、前端服务编排、后端归属核对，以及旧多节点流程回归。

生成时检查 trace 数量、schema、引用路径、61 个共享 SIF 的存在/非空，以及独立环境中的官方组件版本。PBS/shell 脚本执行 bash -n 检查。

尚未在计算节点上运行本组真实 Qwen + 工具的完整 smoke；本地模拟测试不能替代该验证。本轮没有执行 qsub。
