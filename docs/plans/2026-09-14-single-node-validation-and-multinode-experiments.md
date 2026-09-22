# 单节点补测与多节点实验计划

确认日期：2026-09-14。文中任务状态为制定计划时的记录，实际提交前需重新核对，避免重复提交。

## 1. 固定实验条件

目标是确认增加并发后实际出现的是增长、平台还是回落，并据此解释多节点扩展效率，不预设 8192 一定消除下降。

- 固定 Qwen3.6-35B-A3B、TP4、context 262144、batch token budget 8192、max_num_seqs 64，开启 thinking 和 APC。
- 沿用 OpenClaw 110 条、mini-SWE 61 条 trace；保持请求处理、输出 token 目标和 seed 规则一致。
- 仅 `web_search` 使用 recorded delay，其他工具继续真实执行。
- 本轮 weak scaling 保持每副本 OpenClaw cc16、mini-SWE cc32；不自动更改并发或扩大参数搜索。

## 2. 单节点：核对历史数据，补五个点

先整理已有 2048 曲线及 OpenClaw cc32 的 2048/8192 对照，标明配置、窗口和结果可用性。旧短窗口结果作为历史与调度预算对照，单独展示。

新补测统一使用现有多节点 runner 的 N=1 布局，即 **1 个推理节点＋1 个 frontend**：

| 工作负载 | 补测 cc | Warmup | Measurement | 用途 |
|---|---|---:|---:|---|
| OpenClaw | 16、32、64 | 10 分钟 | 60 分钟 | 验证旧拐点及后续趋势；cc16 复用为 weak scaling N=1 |
| mini-SWE | 32、64 | 30 分钟 | 90 分钟 | 验证旧峰值至下降区间；cc32 复用为 weak scaling N=1 |

- 缓存准备完成后，在现有 capacity 分配内顺序执行：OpenClaw 16→32→64，再 mini-SWE 32→64；每点启动新后端。
- 五点的 warmup＋measurement 共 **7.5 小时**，启动与 drain 另计。两个 cc64 点允许最多 2 小时自然 drain，避免旧实验中约一小时的长尾被提前截断。
- 新增独立配置与结果目录，保留已提交配置和旧结果。首轮每点一次。

每点汇总 task throughput、output tokens/s、任务 p50/p95、TTFT、排队、实际 token 复用率、未复用 prompt tokens/s、KV 使用、running/waiting、GPU 利用率，以及 setup/tool 时间和工具错误。

同时画并发曲线与 60 秒时间序列，检查测量前后半段和 trace 组成。吞吐接近但排队、时延持续上升，只能说明吞吐趋于饱和，不能称为可持续容量。若仍有回落，先报告证据，不自动追加新参数实验。

## 3. 环境就绪后完成 weak scaling

**先完成共享 SIF，再接入节点本地缓存。**

- 共享缓存达到 61/61，核对索引、文件存在性及已有启动检查。
- 每个 frontend 在测量前串行复制所需 SIF 到 `/local/scratch/$USER/agenttrace-multinode/minisweagent-images`，复用已完成副本；临时文件复制完成后再发布。
- 沿用 `image_cache_dir`：节点生成的 replay profile 指向本地缓存。每个任务从本地 SIF 创建独立可写 sandbox；任务结束删除 sandbox，保留 SIF。
- 使用现有已分配节点验证离线构建、工具执行和两个 sandbox 的隔离，不另交重复的两节点 smoke job。

Weak scaling 固定 **N 个推理节点＋N 个 frontend**，N＝1、2、4、8、16：

| 工作负载 | 当前处理方式 |
|---|---|
| OpenClaw | N=1 使用新补测；N=2 保留已完成主数据并附采样、工具错误说明；N=4/8/16 已排队，不重复提交 |
| mini-SWE | N=1 使用新补测；环境验证后提交 N=2/4/8/16，其中 N=16 是失败补跑 |

沿用旧 weak scaling 提交脚本，只生成仍需提交的清单，由用户手动提交。≤10 个物理节点使用 preemptable 3 小时；16/17 节点使用 prod 3 小时；32 节点使用 prod 6 小时。

补测中 cc64 回落不自动阻止较低并发的 weak scaling；若 cc16/cc32 基线本身仍明显不稳定，则先说明原因，再调整该 workload 的后续安排。

## 4. Weak scaling 后的两组实验

对应文章大纲中的 Resource Scaling、Cross-layer Profiling 和 Inference-Heavy Workflows。

| 实验组 | 配置 | 点数 | 要回答的问题 |
|---|---|---:|---|
| 固定 frontend | frontend 固定为 1；推理节点 N=2/4/8/16；总 cc=N×每副本 cc | 两类任务共 8 点 | 只增加推理资源时，frontend、工具执行或本地 I/O 是否限制吞吐？ |
| 路由 × 前缀复用 | 4 个推理＋4 个 frontend；sticky/round-robin × normal/独立 cache salt | 两类任务共 8 点 | 路由改变前缀复用后，对吞吐、prefill 和排队有多大影响？ |

执行顺序：先完成 fixed-frontend 的 N=2→4→8→16，再做四种机制组合。

- Fixed-frontend 与相同 N 的 balanced 对照，重点检查 frontend CPU、内存、磁盘、sandbox 数量、setup/tool latency，以及推理侧是否缺请求。
- 四个机制组合均经过相同 proxy；APC 始终开启，独立 cache salt 隔离跨调用复用。Sticky 明确标为静态任务亲和，不称为动态 KV-aware routing。
- 本轮不扩展 TP/DP、prefill/decode 分离、MCP/Parsl 或 DAG 实验。

## 5. 数据验收与交付

- 完整测量窗口、自然 drain、客户端／后端请求和 token 计数对齐，作为主数据验收依据。
- 资源采样缺口单独标注质量；不因一次超过 5 秒的采样间隔废弃完整吞吐数据。真实窗口中断、计数不一致仍判为失败。
- 旧 OpenClaw N=2 的原始失败标记保持不变，离线复核报告明确列出哪些指标可用；额外工具错误单独分析。
- 测试覆盖本地缓存复用、复制中断恢复、缺失镜像不联网回退、sandbox 隔离，以及采样缺口与真实计数错误的区别。
- 输出每阶段的指标表、PNG/PDF 图、结果选择清单和简短结论；weak scaling 效率统一计算为 `T(N)/(N×T(1))`，分位数从原始样本合并计算。
- 单次运行只给探索性趋势，不生成跨 run 置信区间；不新增 checksum 或哈希检查。
