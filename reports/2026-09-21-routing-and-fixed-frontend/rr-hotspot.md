# RR 热点：请求数量均匀，未完成请求集中

2026-09-21 请求级补查。以下完整窗口统计解释了热点如何维持，但单靠它们不足以解释最初分化。后续 [热点形成与前缀核对报告](../2026-09-21-rr-hotspot-cause/README.md) 已在 warmup 转折附近定位到：同一任务相邻请求未换后端、公共前缀未变，实际复用仍大幅下降。该报告补充了缓存相关服务能力退化与 RR 放大过程，并区分已确认事实和未确认的底层缓存机制。

## 实际分发与请求成本

原始后端 middleware ledger 记录的 measurement 内到达请求：OpenClaw 四个副本均为 **1,956**；mini-SWE 为 **5,579 / 5,580 / 5,580 / 5,580**。这些调用全部完成，包含后来任务级 drain 被取消的 mini-SWE 任务在窗口内已经发出的调用。因此“热点收到更多请求”不能解释本窗口的巨大差异。

后端统计按 queued time 落入 measurement 的完成请求计算，允许其随后完成：

| Workload / 副本 | 平均 prompt tokens | 平均未复用输入 tokens | 后端排队均值 | 解码阶段经历时间均值 | 后端总滞留均值 |
|---|---:|---:|---:|---:|---:|
| OpenClaw r00 | 49,146 | 9,478 | 0.066 s | 2.54 s | 3.08 s |
| OpenClaw r01 | 45,498 | 9,940 | 0.050 s | 3.66 s | 4.23 s |
| OpenClaw r02 | 47,404 | 9,583 | 0.052 s | 2.86 s | 3.40 s |
| **OpenClaw r03** | **45,053** | **39,542** | **25.30 s** | **71.23 s** | **99.08 s** |
| mini-SWE r00 | 29,288 | 6,758 | 0.016 s | 2.03 s | 2.38 s |
| **mini-SWE r01** | **29,507** | **27,074** | **60.66 s** | **49.77 s** | **112.15 s** |
| mini-SWE r02 | 29,257 | 6,831 | 0.013 s | 2.02 s | 2.36 s |
| mini-SWE r03 | 29,354 | 6,936 | 0.015 s | 2.23 s | 2.59 s |

后端总滞留包含 queued→finished；解码阶段时间是首 token→末 token 的墙钟时间，包含连续批处理下的调度/资源共享影响，不能解释为该请求独占 GPU 的计算时间。热点的解码阶段经历时间也显著变长，因此不能只用首 token 延迟描述问题。

mini-SWE 每后端约 1.03 requests/s，到达量近乎相同；每请求滞留约 112 s 的副本会保有约百量级在途请求，而约 2.4 s 的副本只有几个。这与完整窗口采样中热点 running+waiting≈115、其他副本≈2.4–2.6 的差异相符。该乘积是解释数量级，有限窗口/非稳态下不要求精确相等。

两个 workload 的热点平均 prompt 并没有显著更长；mini-SWE 热点的平均输出长度 191.9 tokens，也低于 r03 的 193.1。四副本启动日志报告相同的 available KV memory 17.84 GiB、KV capacity 467,280 tokens，排除了这些已记录容量配置的差异；尚未排除节点运行时速度差异。

## 分化在 warmup 期间出现

mini-SWE 最初几分钟四副本的请求耗时接近。按 warmup 第 5、6、7、8 分钟到达的请求分组，r01 的平均总滞留约 **11.8、21.6、28.4、63.1 秒**，同时未复用输入从约 **4.1k→7.6k→8.6k→20.2k tokens** 增加。其他副本保持低得多的滞留。OpenClaw r03 也在 warmup 后段出现类似分化。这里是按到达时间划分的请求 cohort，延迟可能跨越该分钟；不能据此严格判定“先缓存下降”还是“先调度变慢”。

与数据相符的反馈机制：早期性能/缓存状态出现差异 → RR 持续等量送入请求 → 较慢副本累积在途请求、增加缓存与调度压力 → 有效复用和请求推进进一步恶化。闭环 agent 任务等待该副本时也减少后续请求产生，其他副本因此低并发运行。**这解释热点如何维持/放大，尚未闭合最初触发原因的因果证据。**

## 官方 cache-aware 同时包含负载反馈

已核对与安装版本相同的官方源码：

- [v0.1.15 round_robin.rs](https://github.com/vllm-project/router/blob/v0.1.15/src/policies/round_robin.rs)：原子递增计数器，对健康副本数取模，不读取 worker load 或 prompt。
- [v0.1.15 cache_aware.rs](https://github.com/vllm-project/router/blob/v0.1.15/src/policies/cache_aware.rs)：当最大/最小在途负载差超过绝对和相对阈值时，选择最小负载副本；其余情况下考虑前缀匹配，低匹配时也选择最小负载副本。树是请求历史的近似，不直接读取 GPU KV。

当前阈值为绝对差 8、相对比 1.5。实际 router Prometheus 边界增量：

| Workload | cache-aware 决策数 | 触发负载不均分支 | 比例 |
|---|---:|---:|---:|
| OpenClaw | 20,460 | 2,699 | 13.19% |
| mini-SWE | 69,993 | 7,829 | 11.19% |

此计数只覆盖“不均衡触发”的分支，不包含低前缀匹配时的最小负载选择。取 measurement 边界之后首个 Prometheus 样本，故决策数与严格事件窗口有几个请求的边界差异。细节见 [router_feedback.json](router_feedback.json)。

因此当前 RR/cache-aware 是**顺序均分**与**前缀亲和 + 负载反馈**的比较。吞吐提高 2.63×/3.11× 不能全部归为前缀亲和本身。最有价值的新增 routing baseline 是官方 [power_of_two](https://github.com/vllm-project/router/blob/v0.1.15/src/policies/power_of_two.rs)，用负载感知、无前缀亲和的策略帮助区分作用；本次补查没有提交实验或修改实验配置。

复现：`.venv/bin/python reports/2026-09-21-routing-and-fixed-frontend/rr_hotspot_requests.py`。脚本只读 RR 原始 request_finished 与 routing ledger，跳过 token 事件的 JSON 解码。输出 [逐请求 cohort 汇总](rr_hotspot_requests.json)、[简表](rr_hotspot_requests.csv)、[含 warmup 的分钟序列](rr_hotspot_request_timeseries.csv)。Backend 到达与 middleware 到达口径略有边界差异：mini-SWE r01 分别为 5,581 / 5,580；不能跨 cohort 直接相减当作缺失请求。
