# 单节点吞吐下降：已证实的路径与基线待验证项

2026-09-13。基于同一轮 Qwen3.6 TP4 的 OpenClaw 与 mini-SWE 数据，以及实际使用的 vLLM 0.19.1 容器源码。

**已证实：高并发时，原本相同的历史输入前缀未被复用，导致大量历史 token 再次进入 prefill，同时输出吞吐下降、排队增加。尚未证实：缓存/状态为什么未命中，以及 2048-token 调度预算对下降的贡献。当前结果能作为此配置和请求处理方式的探索性观测，尚不能作为已经确认的稳态容量基线。**

## 两类负载的共同现象

| 负载 / cc | 输出 tokens/s | 实际 prompt token 复用 | 本地 prefill 计算 tokens/s | 初始排队 p95（秒） |
|---|---:|---:|---:|---:|
| OpenClaw 16 | 410.9 | 93.8% | 4,254 | 0.45 |
| OpenClaw 32 | 289.2 | 81.1% | 8,371 | 35.53 |
| OpenClaw 64 | 184.9 | 21.8% | 14,971 | 104.08 |
| mini-SWE 32 | 715.9 | 95.8% | 4,434 | 0.22 |
| mini-SWE 64 | 342.4 | 45.6% | 17,475 | 55.15 |

数据来自 [OpenClaw 汇总](../2026-09-08-openclaw-qwen36-scale/summary.csv) 和 [mini-SWE 汇总](../2026-09-13-minisweagent-qwen36-scale/summary.csv)。本地 prefill 是输入 token 的计算记账量，并非直接测得的 kernel 时间；新增计数的采样跨度约 1799 秒，输出率使用完整 1800 秒。

## 请求级证据

| 项目 | OpenClaw cc32 | mini-SWE cc64 |
|---|---:|---:|
| trace | `676e5e31-a554-4acc-9286-b60d90a92d26` | `marshmallow-code__marshmallow-1343` |
| task / node | `00067 / llm-024` | `00039 / llm-065` |
| 前一次 prompt tokens | 168,331 | 40,872 |
| 本次 prompt tokens | 168,814 | 40,962 |
| 两次实际输入的相同 token 前缀长度 | 168,330 | 40,871 |
| 本次复用 tokens | 4,752 | 0 |
| 本次重新计算的输入 tokens | 164,062 | 40,962 |
| 调度后至首 token（秒） | 15.24 | 2.52 |
| 客户端 TTFT（秒） | 约 15.57 | 50.39 |

mini-SWE 同一 trace、task、node 在 cc32 的 TTFT 为 0.374 秒，初始排队接近零；cc64 初始排队为 47.70 秒。OpenClaw 对应下降片段中，多条既有请求完成，但新请求长时间未进入运行，KV 活跃块占用随之下降。不能把这个 KV gauge 下降解释为所有缓存都空了。

这些例子排除了“只是本次输入内容变了，所以自然没有相同前缀”的解释。它们仍不能区分 attention 缓存淘汰、Mamba/GDN 状态 checkpoint 丢失或匹配限制等机制。

请求级 cached tokens 的旧日志未直接保存。本次选取原始 Prometheus 相邻采样间只有一个请求产生首 token、且 prompt 计数增量精确等于该请求预检 token 数的区间，推导该请求的 cached/compute 计数，再按输出 token 数和完成时间匹配客户端报告。mini-SWE cc64 得到 760 个这种区间；它们是诊断子集，不用于估算总体 miss 率。具体例子和离线前缀核对见 [request_examples.json](request_examples.json) 与 [prefix_checks.json](prefix_checks.json)。

## 参数核对后的判断

- TP4、BF16、context 262144、max_num_seqs 64、memory fraction 0.90、APC 和 thinking 均按记录启用，七档一致。`max_model_len` 是请求长度上限；源码中的 full-ISL 准入检查使用请求实际长度，并非每条请求都预留整个 262144-token 上下文。
- **2048 是当前最值得做对照的参数。** 它是整轮调度预算。实际模型启用 Mamba `align`，block size 为 528；既有长 prefill 会连续消耗预算，剩余预算不足一个对齐块时，等待请求可以被推迟。实际容器中的 `schedule()`、Mamba 对齐与缓存管理逻辑已和前次检查的版本核对。不能从配置值本身推断某段等待实际走了哪条分支。
- vLLM 0.19.1 官方说明，小预算如 2048 偏向改善 token 间延迟，大预算更有利于 prefill/TTFT，并建议针对吞吐尝试大于 8192 的预算。因此 2048 是合法配置，但不是已经验证过的吞吐最优设置。[官方调优说明](https://docs.vllm.ai/en/v0.19.1/configuration/optimization/#chunked-prefill)
- KV usage 只计算非空闲块。空闲块仍可能保存会被后续分配淘汰的历史前缀；混合模型还要求状态 checkpoint 可复用。七档零 preemption 不排除这种历史缓存损失。详见 [缓存指标定义](../2026-09-08-openclaw-qwen36-scale/cache_interpretation.md)。
- **当前最符合证据的机制假设**是并发会话交错使历史前缀/状态的保留变差，增加长 prefill；小调度预算又延长这些 prefill 和等待。此假设能解释输入计算增多而输出减少，但尚不能当作唯一根因。两类负载在不同 cc 出现拐点，也不能单凭这些均值归因于某一个 trace 特征。

## 另一个需要明确的请求兼容性问题

已核对的 mini-SWE trace 的 assistant 消息使用 `reasoning_content` 保存历史思考。实际容器的 `vllm.entrypoints.chat_utils._parse_chat_message_content()` 读取 `message.get("reasoning")`，再把它转成模板中的 reasoning 字段；仅提供 `reasoning_content` 的历史消息没有经这条路径保留下来。[对应版本解析源码](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/entrypoints/chat_utils.py)

本地使用相同模型 tokenizer/template、相同 tool-call 参数规范化和实际 reasoning 处理后，两个例子的输入 token 数均与当时 `/tokenize` 预检及实际 prompt 计数一致。前缀核对因此针对服务端实际看到的输入。mini-SWE `llm-065` 如果保留录制思考，输入将从 40,962 增至 43,561 tokens；本次 OpenClaw 示例没有这一差异。具体见 `prefix_checks.json`，不把 mini-SWE 示例的影响范围泛化为所有 trace。

这不等于新生成的 thinking 没开启；**新输出生成 thinking** 和 **把历史录制 thinking 放回输入**是两件事。该处理方式在各 cc 一致，不能解释并发增大后的突然下降，但会改变原始 trace 与实际服务输入之间的关系。如果实验定义是完整保留录制思考，应做字段兼容映射，并重新检查每条 trace 的 context eligibility；如果有意不保留，就应把它明确写入实验定义。不能悄悄改变请求处理后继续合并旧、新结果。

## 成为多节点基线前要完成的验证

1. 先在当前请求处理方式下复现 OpenClaw cc32，对比 `max_num_batched_tokens=2048 / 8192`；根据结果补 16384 或缓存相关对照，再在 mini-SWE cc64 确认。复用既有 runner、冻结的 trace pool 和 seed，所有调用完成后自然排空。无需先重跑完整七档。
2. 对照中记录每请求 prompt/cached tokens、各缓存组匹配长度，以及等待原因：预算/对齐不足、full-ISL gate、块分配失败或 running 数上限。原请求日志已补 prompt/cached 两个字段，相关 11 项测试通过；缓存组与调度分支诊断仍待接入。增大 batch 预算可能改变内存 profiling 和可用 KV 块数，需要一起记录，避免误判机制。
3. 明确历史 thinking 的输入处理，冻结最终配置和 trace eligibility。正式容量统计需要更充分的 warmup/measurement 或固定同一批 trace 全部完成的 batch 试验。mini-SWE cc64 现有 30 分钟窗口的 15 个完成任务全来自 warmup，不能作为稳态任务完成率。
4. 多节点如果采用每节点一个 TP4 副本，应固定单副本配置和同一 trace 连续调用的路由规则，明确总 cc 与每副本 cc。否则节点数变化还会同时改变前缀缓存复用条件。

当前已准备四个参数对照配置，位于本地 `runs/scaling/baseline-validation-7608115-20260913/`。**尚未启动 GPU 对照**：新 allocation 7608115 的节点目前空闲，资源用途澄清待答复。本报告没有把未执行的对照写成已经验证的结果。

## 本地诊断输入与复现

输入根目录为 `runs/scaling/qwen36-expanded-7594555-20260908/`。本次补充脚本保存在其 `diagnostics/throughput-cause/`，包括 `audit_existing.py`、`check_prefix.py` 及实际容器源码副本；旧 OpenClaw 请求例子位于 `diagnostics/cache-drop/`。脚本只读取既有记录，未调用模型或工具 API。原始模型、trace 和大型事件日志仍保留在本地。
