# Qwen3.6-35B-A3B TP4 后端

新模型使用独立的启动入口、回放 profile 和 scaling 配置。原 Qwen3-32B 配置继续用于尚未结束的实验。

## 配置

| 项目 | 设置 |
|---|---|
| 权重 | `Qwen/Qwen3.6-35B-A3B`，BF16，无量化 |
| API model | `qwen/qwen3.6-35b-a3b` |
| 容器 | 现有 `vllm-openai-v0.19.1.sif` |
| 默认并行 | TP4 / DP1 |
| 上下文 / 序列上限 | 262144 / 64 |
| 批处理 token 预算 | 2048 |
| GPU memory utilization | 0.90 |
| 权重读取 | `--safetensors-load-strategy eager`，适合 Eagle/Lustre |
| Prefix caching | 开启；此混合注意力架构在 vLLM 0.19.1 下默认使用 Mamba `align` cache 模式 |
| Tool / reasoning parser | `qwen3_coder` / `qwen3` |
| 模态 / thinking | `--language-model-only`；默认 `enable_thinking=true` |

模型缓存位于 `/lus/eagle/projects/lc-mpi/ZhijingYe/Models/huggingface`，即 SIGMETRICS_2027 项目根目录的 `../Models/huggingface`。从 `trace_framework` 仓库根目录看，它是 `../../Models/huggingface`。编译缓存仍使用同一 Models 根目录下的 `vllm-cache`。

官方 [模型卡](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) 推荐 vLLM >=0.19.0，并说明了文本模式、工具解析器及关闭 thinking 的参数。本机 0.19.1 容器已确认包含该模型的 `Qwen3_5MoeForConditionalGeneration` 架构。

## 启动与基础检查

在已分配的推理节点、仓库根目录运行：

```bash
bash examples/scaling/vllm_qwen36_35b_a3b.sh serve
```

另一个终端检查固定长度流式输出：

```bash
VLLM_BASE_URL=http://<inference-node>:8000/v1 \
  bash examples/scaling/vllm_qwen36_35b_a3b.sh smoke
```

启动脚本沿用已有的端口、缓存目录、模型长度、批处理及显存比例环境变量覆盖方式。

## 回放与新的 trace pool

回放分别使用：

- `examples/openclaw_gaia/replay_profile_qwen36.json`
- `examples/minisweagent_swebench/replay_profile_qwen36.json`

两份 profile 使用新的 LLM model 名称和固定输出 token 长度策略。OpenClaw profile 的 `web_search_mode` 为 `recorded_delay`：只有 `web_search` 按记录中的 `tookMs` 异步等待并返回原结果，其他工具继续原生执行，mini-SWE 工具执行方式也保持原样。搜索节点不会调用 Brave，因此该回放模式不需要 Brave API key。请求中的 recorded messages、tools 和工具参数保持原样。

节点报告通过 `tool_replay` 标记模拟的搜索耗时及其来源。缓存命中记录中的 `tookMs` 是首次请求耗时，作为延迟近似保留并标注 `recorded_cached`；缺失或非法耗时会在 setup 阶段报错。这个系列研究固定搜索延迟下的推理后端表现，各 cc 档位应统一使用该模式。原 Qwen3-32B profile 继续使用 `native` 搜索。详见 [搜索回放语义](../examples/openclaw_gaia/README.md#web_search-recorded-delay)。

新模型的 tokenizer/chat template 会改变实际输入 token 数，需要重新生成 preflight，而不能把旧 Qwen3-32B 的 32K 筛选结果直接当成新模型结果。

```bash
.venv/bin/python -m agenttrace.experiments.prepare \
  --repo "$PWD" \
  --base-url http://<inference-node>:8000/v1 \
  --model qwen/qwen3.6-35b-a3b --max-model-len 262144 \
  --output runs/scaling/preflight-qwen36-NEW
```

两节点 scaling 配置为 `examples/scaling/single-backend-qwen36-tp4.json`，覆盖 cc 1/2/4/8/16/32/64，各档预热 120 秒、测量 1800 秒，之后自然排空。先将主机占位符替换成实际分配的回放、推理节点，再通过 `examples/scaling/run.sh run --config ... --pools ... --output ... --job-id ...` 使用新的 pool。该配置不会被旧模型的默认 PBS 补跑任务自动采用。

Qwen3.6 TP4 与原 Qwen3-32B TP4 使用相同数量的 GPU，但模型、上下文上限和可用 trace pool 不同，应保存为独立实验系列。

## 本次验证记录

2026-09-07，debug 作业 `7597636`，节点 `x3003c0s37b1n0`。初始 TP2 / 32768 验证记录位于 `runs/model-switch/qwen36-tp2-7597636/`；TP4 / 262144 验证记录位于 `runs/model-switch/qwen36-tp4-262144-7597636/`。推理与工具同处该 debug 节点，仅用于兼容性验证。

TP4 / 262144 已完成的检查：

- 启动日志确认 TP4、BF16、上下文 262144，CUDA Graph 与 `torch.compile` 开启。每卡模型加载约 16.3 GiB，可用缓存预算约 18.4 GiB。
- 7 项 API 检查通过：16 / 64 token 固定长度流式输出、自动工具调用，以及 4 个并发的 64 token 请求；chat API 的输入 token 用量与 `/tokenize` 一致。
- 合成 token ID 输入验证上下文边界：262128 输入 + 16 输出 = 262144 tokens，成功生成，见 `context-check.json`。该检查验证单请求长度可用性，不表示高并发容量或真实 trace 的吞吐。
- 全量 preflight 检查了 2216 次 LLM 调用：OpenClaw 24/24 条、mini-SWE 22/22 条完整 trace 均可纳入新 pool，最大输入加目标输出分别为 212461 / 107132 tokens。新 pool 在本次 TP4 记录的 `preflight/{openclaw,minisweagent}/traces.txt`。

原 TP2 验证还完成了 OpenClaw 和 mini-SWE 各一条完整回放，输出分别为 734 / 3226 tokens；mini-SWE 的 7 个原生工具错误与原 trace 的错误节点逐个一致，没有新增错误节点。TP4 的 API 与上下文检查单独记录，不将 TP2 的完整回放结果记作 TP4 结果。
