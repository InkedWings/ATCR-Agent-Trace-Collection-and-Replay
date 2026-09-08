# 为什么 KV usage 没到 100%，缓存复用却下降？

2026-09-08 复核实际使用的 vLLM 0.19.1 源码与原始 Prometheus 后，确认需要区分两个缓存指标，并修正对重计算计数器的解释。任务吞吐和延迟数据不变。

## 直接测到的数据

| cc | KV usage 均值 / 峰值 | 原查询命中率 | 实际 prompt token 复用 | 查询 token / 处理 prompt token | 本地 prefill 计算 token/s |
|---:|---:|---:|---:|---:|---:|
| 16 | 30.2% / 45.2% | 93.3% | 93.8% | 1.30× | 4,254 |
| 32 | 50.3% / 74.5% | 61.5% | 81.1% | 4.45× | 8,371 |
| 64 | 28.2% / 61.8% | 19.3% | 21.8% | 16.48× | 14,971 |

完整精度见 [summary.csv](summary.csv)。全部七档还核对了原始 `prompt_tokens_by_source_total`：external transfer 和特殊 recomputed 计数均为 0，local_compute + local_cache_hit = prompt 总量，local_cache_hit = cached prompt 总量。

## KV usage 不包括全部历史前缀

vLLM 的 KV usage 是 `1 - free_blocks / usable_blocks`。请求释放一个块后，引用计数可以降为 0，块进入 free queue，但仍保留可复用的前缀内容。新请求从队首分配块时，可以直接淘汰其中的旧缓存。因此 free 表示可重新分配，不代表里面没有历史缓存。见 [vLLM 0.19.1 BlockPool 源码](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/v1/core/block_pool.py)。

例如有 100 个块，30 个正在被请求占用，70 个空闲块还存着历史前缀：KV usage 显示 30%，但全部块都可以已经写过数据。新请求持续到来时，历史缓存会被替换，不需要先让 usage 达到 100%。这是解释指标的示例，不是本次测得的块数量。

本系列每卡总分配显存约 37.7 GiB；启动日志显示每卡 KV 预算约 18.28 GiB。总分配显存、正在占用的 KV 块、历史前缀是否仍可复用，是三个不同概念。

## 原查询命中率受重复调度查询影响

`get_computed_blocks()` 每次查询都会累加 prefix-cache queries/hits。调度器先查询缓存，再检查 Mamba 对齐、准入和块分配条件；请求如果未能进入运行，下一轮可能再次查询。因此同一个等待请求会重复贡献分母和分子，且等待更久的请求权重更大。见 [KVCacheManager](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/v1/core/kv_cache_manager.py) 和 [Scheduler](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/v1/core/sched/scheduler.py)。

本次 cc32 的查询口径为 61.5%，但实际复用是 81.1%；cc64 两者为 19.3% 和 21.8%。查询口径放大了 cc32 的下降，cc64 仍有真实的缓存复用下降，不能全部归为统计现象。

实际复用比例采用 `Δprompt_tokens_cached_total / Δprompt_tokens_total`，按 token 加权，不是命中请求百分比。两项在 prefill 输出时记账；查询计数在调度尝试时记账，所以有限窗口的两种事件边界也不同。本地计算量采用原始 source=local_compute 计数交叉核对，报告中等价计算为 prompt − cached；它不是精确的 GPU 计算耗时或 FLOPs。

## 混合模型的状态缓存也要考虑

实际启动日志确认 Qwen3.6 走 `Qwen3_5MoeForConditionalGeneration`，使用 GDN 线性注意力与全注意力混合结构；Mamba cache mode 为 `align`，attention block size 自动设置为 528 tokens。其前缀复用还需要匹配可用的状态 checkpoint，而不是只看 attention KV 是否还在。该管理器会把不再需要的旧状态块归还 free queue。见 [MambaManager 源码](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/v1/core/single_type_kv_cache_manager.py)。

高并发让更多会话交错执行，可能扩大需要保留的前缀集合、增加同一前缀两次使用之间的其他块分配，从而影响历史前缀或状态 checkpoint 的保留。这个机制与现象相容，但本次没有逐块淘汰事件或每请求 miss 原因，尚不能确定它与请求组成、预热、checkpoint 匹配等因素各占多少。

## 零抢占不代表没有历史输入被重新计算

本次 `num_preemptions_total` 增量为 0，说明没有记录到引擎抢占。`prompt_tokens_recomputed_total` 在此版本主要由“全部命中时仍强制计算最后一个 token”这一特殊情况更新，不能用它判断普通缓存 miss 是否让历史输入再次进入 prefill。见 [PromptTokenStats 源码](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/v1/metrics/stats.py)。

可确认的是：cc64 的实际缓存复用明显下降，本地 prefill 计算 token 计数率约为 cc16 的 3.52 倍，同时输出率从 410.9 降到 184.9 tokens/s。进一步定位应记录请求实际 cached tokens、可淘汰缓存的驻留和淘汰事件，并在固定 trace 子集下对比；当前数据不足以把某一种缓存淘汰机制写成已证实的唯一根因。
