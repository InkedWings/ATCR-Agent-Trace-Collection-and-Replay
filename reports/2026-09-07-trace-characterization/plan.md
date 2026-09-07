# Trace 统计与展示方案

建议将这一部分组织为工作负载画像，回答四个问题：一个任务要交互多少轮、每轮处理多少 token、token 来自哪里、工具在做什么。性能回放中的延迟、GPU、排队、cache hit 放到另一部分，再用这里的工作负载特征解释。

## 样本与现有数据

主画像统计当前完整候选池：GAIA run2 的 24 条、mini-SWE dev-20260906T004724Z 的 22 条，不重复累计不同 cc 下的同一条 trace。32K 可回放子集（11/10 条）作为单独分组，展示筛选对调用数、长度和工具分布的影响。它们都是当前采集样本，不代表全部 GAIA 或 SWE-bench。

| 指标 | OpenClaw / GAIA | mini-SWE / SWE-bench |
|---|---:|---:|
| Trace 数 | 24 | 22 |
| LLM call 总数 | 309 | 1,907 |
| Tool call 总数 | 288 | 1,906 |
| 每 trace LLM calls：min / median / max | 3 / 9 / 55 | 30 / 70.5 / 250 |
| 每 trace tool calls：min / median / max | 2 / 8.5 / 54 | 30 / 70.5 / 250 |
| 记录的输出 token 总数 | 103,340 | 372,000 |
| 每 call 输出 token：min / median / max | 42 / 125 / 7,572 | 33 / 81 / 4,096 |
| 每 call 输入 token：min / median / max，Qwen 回放 tokenizer | 6,763 / 28,850 / 202,686 | 1,280 / 24,865 / 101,854 |

完整候选池保留终止状态：OpenClaw 22 条正常 stop、2 条 aborted；mini-SWE 21 条 Submitted、1 条 LimitsExceeded。Submitted 并不等于题目正确。不能把达到步数/输出长度限制的轨迹当作自然结束，不能为了画像好看删掉失败轨迹。

## 统计层级与展示

| 主题 | 统计单位与指标 | 建议图形 | 要回答的问题 |
|---|---|---|---|
| 交互长度 | 每 trace 的 LLM calls、tool calls、tool/LLM 比、不同工具数；median、IQR、范围 | ECDF；正文把 LLM/tool 放一组双面板 | mini-SWE 是否存在明显更长的交互链和长尾？ |
| 每轮 token | 每 LLM call 的 input、output；输出长度限制命中比例；input vs output 关联 | 输入/输出 ECDF，跨度大用 log-x；相关性散点作为辅助 | 是长输入、长输出，还是调用次数多造成负载？ |
| 每任务 token | 每 trace 累计 input、累计 output、最大 context、首轮/末轮 context | ECDF 或散点：调用数 vs 累计输入 | 哪些任务贡献最多处理量？长任务是否反复处理越来越长的历史？ |
| 上下文增长 | 按 trace 归一化步骤进度，统计 prompt 长度及组成 | 中位数曲线 + IQR；每 trace 在每进度区间最多贡献一个汇总值 | context 增长发生在何时，是否由工具结果主导？ |
| 输入构成 | system、工具 schema、user/task、assistant 历史、tool observations、模板开销 | 100% 堆叠柱；可附 early/middle/late 三阶段 | 固定开销、历史推理和工具结果各贡献多少？ |
| 输出构成 | reasoning、non-reasoning、unknown；更细的命令/正文分解另计 | 100% 堆叠柱 | 两种 agent 的 reasoning token 占比是否不同？ |
| 工具组成 | API 名称、语义用途、每类调用占比和任务覆盖率 | 横向条形图或堆叠柱；颜色跨图一致 | 两类 agent 实际在操作什么？ |
| 工具产生的上下文 | 每类工具结果送入 LLM 的 token 总量/长度分布 | 调用占比 vs observation token 占比并列条形图 | 哪类工具调用少却制造大量 context？ |

任务数只有 24/22 条，正文优先 median/IQR/范围和 ECDF；任务级 p95/p99 的样本不足，不宜当作稳定尾部分位数。调用级有 309/1907 个样本，但同一 trace 的调用不是独立重复；若做置信区间，应以 trace 为簇抽样。

## Token breakdown 的准确口径

### 输入：按来源拆分

推荐统一为：

1. **System / framework instructions**：系统提示、框架注入的规则和说明。
2. **Tool definitions**：API tools/function schema；与工具返回内容区分。
3. **User / task**：题目、补充用户消息和任务描述。
4. **Assistant history**：历史正文及工具调用参数；若实际模板保留 reasoning，可另分子类。
5. **Tool observations**：实际送入后续 LLM 的搜索、网页、代码、测试等结果。
6. **Template / serialization overhead**：角色标记、分隔符和其他无法归入上述内容的模板 token。

正式图应依据实际 chat template 渲染后的输入归因，确认模板是否保留 reasoning_content 和 tool schema。不要直接 tokenize 原始 JSON，也不要把每段单独 tokenize 后的和宣称为完整 prompt 的精确 token 数；边界与模板序列化可能改变结果。若采用内容单独 tokenize 的近似方案，必须标注近似，不与精确总输入混用。

同时输出两种汇总口径：全池 token 加权构成表示总体处理量；各 trace 比例的等权平均表示典型任务。正文选择一种明确标注，另一种可在附录或表中保留，防止一个超长任务决定整张图。

### 输出：先做 reasoning / non-reasoning

原始 trajectory 中有 usage，可先做可信的粗粒度分解：

| 输出部分 | OpenClaw | mini-SWE |
|---|---:|---:|
| 已知 reasoning tokens | 60,534 | 112,144 |
| 已知 non-reasoning tokens | 39,770 | 259,856 |
| 构成未知的输出 tokens | 3,036 | 0 |
| 合计 | 103,340 | 372,000 |

OpenClaw 的两次 aborted 调用分别记录 1,669 和 1,367 个输出 token，但原 trajectory 的 usage 被归零。这部分保留 unknown；不能当作 reasoning=0，也不能用不完整 usage 合计 100,304 代替完整 trace 的 103,340。

Non-reasoning 包括工具调用参数、正文和可能的格式开销，不能直接标为“最终答案”。若进一步拆为 reasoning / action arguments / visible text / formatting，应对原始返回内容做结构解析并用同一 tokenizer 计数；不能假定重算总数必然与 provider usage 一致。正文先采用 usage 支持的粗分，细分作为后续分析。

mini-SWE 统计 provider response 时不能只选 `role=assistant`：有一条响应存储在 user-role 事件的 `extra.response` 中。按 response 事件统计可以覆盖全部 1,907 次调用。

### 区分采集与回放 tokenizer

- 原始采集模型是 Nemotron-3-Ultra，原生 usage 的 input/output/reasoning 是采集口径。
- 现有 preflight 的 prompt_tokens 使用 Qwen3-32B serving tokenizer/template，对应回放输入负载。
- 回放固定生成的输出长度来自捕获的 output_tokens，是指定的生成长度目标；不是把原始 Nemotron 输出重新 tokenize 后得到的长度。
- 工作负载描述与回放资源需求都可以研究，但图注必须写明。不能把 Qwen 输入与 Nemotron usage 输出的比值表述为原采集模型的 input/output ratio。

### 区分上下文存量与反复处理量

每轮完整 prompt 求和表示请求层累计输入处理需求，会重复计入历史消息；这是有效的服务负载指标，不应称为“唯一信息量”。Prefix cache 是否避免了实际 prefill，属于回放性能部分。

工具结果可以统计两个互补量：每条结果首次送入模型的 token 数，以及它在后续历史中反复出现的累计 token 数。两者之比能说明历史重用带来的请求输入放大。遇到截断或上下文压缩，不能简单把相邻 prompt token 差当作新增 token。

## Tools 分布应分两层

原生 API 层现在可直接统计：OpenClaw 为 web_search 159 次（55.2%）、web_fetch 80 次（27.8%）、exec 48 次（16.7%）、read 1 次（0.3%）；mini-SWE 是 bash 1,906 次（100%）。

语义层应对 OpenClaw exec 与 mini-SWE bash 使用统一分类：

| 语义类别 | 例子 |
|---|---|
| Search / retrieval | 网页搜索、抓取远程资源 |
| Inspect / read | ls、find、rg/grep、cat、读取代码 |
| Modify / write | patch、编辑源文件、生成文件 |
| Test / execute | pytest、运行复现脚本、执行程序 |
| Environment / dependencies | pip/conda 安装、环境检查与构建准备 |
| Diff / submit | git diff、生成提交 patch、提交标记 |
| Mixed / uncertain | 一个调用包含多个实质操作，或无法仅凭请求确定用途 |

一次 bash API 调用可以包含多个命令。调用数仍按一个 tool node 计算，不能拆开命令后拿来和原 API calls 做同一个分母。初稿采用互斥主用途 + mixed/uncertain，之后按 trace 抽样复核；不应仅凭 Python 命令名就认定它是测试或计算。只有单独声明多标签统计时，各类占比才允许超过 100%。

除调用占比，再给每类工具的 trace 覆盖率（多少任务使用过）、每任务调用次数、错误率，以及 observation token 占比。原生记录中输出可能被截断/溢写，因此实际发给 LLM 的 tool 消息应作为 observation token 主口径，不能只量 `recorded_result` 中保留下来的片段。

原始工具 `tookMs` / `durationMs` 字段不齐全，不能将缺失耗时填零。需要耗时图时，以单独选定的真实回放基线统计，并清楚标注样本池和回放环境，放到性能分析部分。

## 正文建议布局

**一张概览表 + 四组图**足以形成逻辑完整的画像：

1. Calls per trace：两类负载的 LLM/tool 调用数 ECDF。
2. Token workload：每 call input/output 分布，加一张 context 随任务进度增长的曲线。
3. Token composition：input 来源构成与 output reasoning/non-reasoning 构成。
4. Tool composition：工具语义类别的调用份额与 observation token 份额对照。

完整池 vs 32K 子集的对照放在采样说明或附录，用于连接后续性能实验。无需为每个 cc 重画 trace 画像，因为同一个固定 trace 的结构不随回放并发改变。

当前 `preview.png` / `preview.pdf` 是第一版六面板预览，包含已可准确统计的调用数、token 分布、输出粗分、工具 API 分布；输入来源拆分、bash 语义分类和上下文增长尚未实施。`tasks.csv`、`llm_calls.csv`、`tool_calls.csv` 是已提取的基础分析表，记录完整池与回放子集标记，便于后续扩展。

在仓库根目录复现：

```bash
MPLCONFIGDIR=/tmp/agenttrace-matplotlib .venv/bin/python reports/2026-09-07-trace-characterization/characterize.py
```
