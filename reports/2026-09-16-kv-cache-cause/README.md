# KV/cache drop：请求级证据与机制解释

2026-09-16。只读取已有运行结果与保存的容器源码；本次未启动实验或修改后端。

**已确认 vLLM 的 hybrid prefix-cache 匹配在丢弃原本可用的 attention 前缀，导致额外 prefill；这不是仅凭 KV 曲线作出的猜测。** 旧状态的保留/回收策略是重点原因，但现有日志没有逐块生命周期，不能给长窗口 cc64 的全部下降指定一个唯一 bug。

## 1. 先区分三个指标

- GPU 显存使用量：模型、预留缓存、运行时缓冲等实际分配的显存。
- `kv_cache_usage_perc`：vLLM block pool 中不可回收的活跃块比例。实际代码是 `1 - free_blocks / (num_gpu_blocks - 1)`。`free_blocks` 包含引用计数已归零、仍带历史前缀缓存的块，因此“free”不等于没有缓存内容。
- 实际 prompt 复用率：请求最终使用的 cached prompt tokens / prompt tokens。它不同于可能包含重复查询的 prefix lookup hit ratio。

`free_blocks()` 使块可回收，但不立刻删除缓存映射；`get_new_blocks()` 从 free queue 取块时才移除其旧映射。于是 **KV 活跃占用低于 100% 时也能淘汰历史前缀**。这个 gauge 无法回答历史缓存占了多少块、是否正在发生淘汰。[v0.19.1 BlockPool 源码](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/v1/core/block_pool.py)

## 2. 我们实际用的 hybrid cache

后端版本 vLLM 0.19.1，Qwen3.6-35B-A3B 使用 Qwen3.5-MoE 架构实现；TP4、context 262144、APC 开启，Mamba cache mode 默认 `align`，无 speculative decoding。启动日志明确提示 Mamba prefix caching 仍是 experimental。

实际配置有 1 个 full-attention cache group 和 3 个 GDN/Mamba state groups，attention block size 为 528。GDN 在这里由 vLLM 的 `MambaManager` 管理，并不表示模型本身就是 Mamba。

复用必须同时满足：attention 前缀连续可用，并且三个 state groups 在同一边界都有可用状态。协调器先检查 attention，再由 state 命中位置截短最终复用长度。

`align` 的 prefill 状态只保存在被调度 chunk 的末端，不是每个 attention block 都存一个完整状态。旧状态在不再服务当前请求时会归还公共 block pool，可被后续分配覆盖。attention 连续 KV 仍可能存在，但缺少匹配位置的状态会迫使整个模型从更早的位置重新计算。[v0.19.1 MambaManager 源码](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/v1/core/single_type_kv_cache_manager.py)、[协调器源码](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/v1/core/kv_cache_coordinator.py)

本地对应源码在 `runs/scaling/qwen36-expanded-7594555-20260908/diagnostics/throughput-cause/container-source/`；关键函数为 `MambaManager.find_longest_cache_hit/remove_skipped_blocks/allocate_new_blocks`、`HybridKVCacheCoordinator.find_longest_cache_hit`、`BlockPool.free_blocks/get_new_blocks/get_usage`。

## 3. 相同 trace、相同节点，直接观察到状态限制

来自之前 cc32、budget8192 的 30 分钟诊断实验。同一 OpenClaw trace `676e5e31-a554-4acc-9286-b60d90a92d26`、同一节点 `llm-024`，分别对应任务 `00067` 和 `00184`：

| 指标 | 较早一次 | 较晚一次 |
|---|---:|---:|
| Prompt tokens | 168,814 | 168,814 |
| FullAttentionManager 命中 | 167,904 | 167,904 |
| MambaManager 命中 / 最终复用 | 167,904 | 4,752 |
| 未复用 prompt tokens | 910 | 164,062 |
| 输出 tokens | 71 | 71 |
| 后端 scheduled → first token | 0.426 s | 9.275 s |
| 客户端 TTFT | 0.744 s | 9.639 s |

两次后端初始排队都约 3 微秒，说明该例的首 token 延迟增长主要发生在调度开始之后。它是已有运行中的观察对照，不是控制其他请求负载后的因果 A/B。

较晚一次的前序 `llm-023` 在 `1789321390.000769` 时刚命中过 **143,088 tokens**（attention 和 state 都命中），下一轮在 `1789321419.951616` 查询时，attention 能命中 167,904，state 却只有 4,752。之前的精确前缀核对显示这两个相邻节点共有 **168,330 tokens** 的相同输入前缀，所以 143,088 位置的前缀本身没有改变。已存在且命中过的状态在约 30 秒后变得不可查到。

这是状态缓存可用性损失的直接证据；结合释放和分配代码，**旧状态变为可回收后被覆盖**是最符合证据的解释。现有日志未记录每个块的创建、释放、覆盖，因此不能逐块证明究竟何时、由哪个请求覆盖，也不能直接判定所有缺失边界都曾创建过状态。

请求 ID、时间、命中长度和后端时间差保存在 [evidence.json](evidence.json)。原始来源为 `runs/scaling/baseline-validation-7608115-20260913/openclaw-warmed-budget8192/openclaw-cc32/` 下的 `backend-scheduler.jsonl`、`backend.jsonl`、`benchmark/tasks/{00067,00184}/report.json`。前缀内容核对来源：[prefix_checks.json](../2026-09-13-single-node-baseline/prefix_checks.json)。

## 4. 不只是孤立样例

按两个短 A/B 诊断实验的 measurement window，取每个实际 admission 的最后一次 cache lookup；避免把未入场的重复查询计入。两档都为 OpenClaw cc32，窗口 1800 秒。

| Budget | 请求数 | 被 state 截短的请求 | Attention 单独可命中比例 | 最终复用比例 | 被 state 截掉的 tokens | 占全部未复用 prompt tokens |
|---|---:|---:|---:|---:|---:|---:|
| 2048 | 1,993 | 303 | 92.98% | 81.71% | 9,247,392 | 61.62% |
| 8192 | 2,879 | 410 | 93.88% | 85.61% | 10,500,864 | 57.43% |

计算：`state 截短量 = attention_hit_tokens - final_cached_tokens`；分母为 `prompt_tokens - final_cached_tokens`。这里量化的是 admission 时的 token 复用损失，不是 GPU 时间占比，也不是“修好就能恢复这么多吞吐”的预测。两档短实验 preemption 都为 0，说明这种损失不需要通过请求抢占触发。

**这两个比例不能套用到新的长窗口 cc64**：新长跑没有相同的逐组 cache lookup 日志。

## 5. 新长跑的吞吐损失与 prefill 负担一致

OpenClaw，全部 budget8192，600 秒 warmup + 3600 秒 measurement：

| cc | 输出 tokens/s | 实际 prompt 复用 | 平均活跃 KV | 未复用 prompt tokens/s | 未复用 prompt / 输出 token | 窗口内 preemptions |
|---|---:|---:|---:|---:|---:|---:|
| 16 | 433.58 | 93.95% | 33.44% | 4,476 | 10.32 | 0 |
| 32 | 282.41 | 68.87% | 73.46% | 14,336 | 50.76 | 1 |
| 64 | 167.29 | 9.24% | 92.36% | 22,162 | 132.48 | 8 |

cc64 的活跃 KV 峰值达到 100%；cc32 达到 99.83%。新增并发伴随更高活跃占用、更低复用、更大 prefill 负担和排队，不能按固定 cache hit 的纯 decode 容量平台解释。长跑每单位输出对应的未复用输入在 cc64 达到 cc16 的约 **12.8 倍**；这是窗口吞吐比值，不是每个请求等权平均或 GPU 时间分解。

原始 Prometheus `prompt_tokens_by_source_total{source="local_compute"}` 的窗口增量与上述未复用输入吻合（边界采样有很小差别），因此并非仅客户端绘图口径导致。原始 `num_preemptions_total` 增量为 0/1/8，`prompt_tokens_recomputed_total` 增量均为 0；这里把“未命中历史前缀产生的 prefill”与“抢占后 recompute 计数”分开，不能用后者为 0 声称前者不存在。

mini-SWE cc32 的复用 95.62%，每输出 token 对应未复用输入约 6.70，仍稳定。mini-SWE cc64 在环境创建时失败，不能据此判断其缓存表现。

旧 budget2048 的低活跃 KV 还有一个独立原因：剩余 token budget 经 Mamba block 对齐后变成 0，阻止 waiting 请求入场。此前匹配 A/B 中该事件占 scheduler step 的比例由 34.95% 降到 3.00%，吞吐提升 46.9%。**8192 缓解了调度对齐限制，没有消除混合状态缓存损失**。详见 [budget 对照报告](../2026-09-14-token-budget-comparison/README.md)。

## 6. Replay 的相关边界与上游进展

我们的 replay 每次发送 trace 中记录的请求 payload；新 Qwen 生成的正文不会替换后续记录中的历史。这符合固定 trace 的回放方式，但意味着新生成的 decode-tail 状态未必匹配下一轮 recorded prompt。因而更依赖共享输入前缀处的 prefill checkpoint 保留。它可能放大上述问题，但目前没有量化其独立影响，不能把它当成已确认的全部原因。上述样例已经核对相邻请求的真实公共前缀，不能简单归因于 prompt 变了。

vLLM 后续已合入 [PR #37898](https://github.com/vllm-project/vllm/pull/37898)，专门在 attention 命中但 state 落后的共享前缀边界增加缓存机会。另有 [PR #45845](https://github.com/vllm-project/vllm/pull/45845) 调整 Mamba/linear-attention 状态保留间隔。它们说明共享前缀的 checkpoint 创建与保留策略确有已知局限；两者的测试配置并不等于我们的 0.19.1/528-block 配置，不能宣称升级或设置某个新参数就一定解决。

## 7. 结论与最小后续验证

可以确认：**vLLM 0.19.1 的混合状态缓存匹配是实际损失来源，长上下文并发压力会与其状态保留策略相互影响。** “活跃 KV 未满就不应掉 hit”这一前提不成立；现在也不能把 cc32/64 的大跌解释成已经验证过的正常饱和平台。

尚缺的一步是长窗口中状态创建/回收的归因：在固定 cc32、相同 trace/config/window 下，记录每组 attention/state 的命中长度、实际缓存块数量、状态 checkpoint 创建/回收和前后请求关联，然后才比较旧版本与候选缓存策略。不要直接开启会改变 hybrid manager 行为的通用 KV events 开关；当前版本该路径有兼容性限制。无需先重新提交整组 scale 实验。

离线复核：在仓库目录运行 `python3 reports/2026-09-16-kv-cache-cause/analyze.py`，只读取已有日志并更新本报告目录的 `evidence.json`。
