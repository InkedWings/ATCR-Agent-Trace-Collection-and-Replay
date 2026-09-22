# 多节点实验数据检查（2026-09-14，约 16:59 UTC）

本轮查到 5 个正式提交：3 个退出，1 个运行，1 个排队。退出的 3 个 PBS Exit_status 均为 1，但数据可用性不同。

| Job | 工作负载 | 推理＋前端节点 | 检查结论 |
|---|---|---|---|
| 7616564 | OpenClaw | 2＋2 | 完成全部测量与 drain；主数据完整，单个前端采样间隔触发全局失败标记 |
| 7616566 | OpenClaw | 4＋4 | 排队，尚无运行数据 |
| 7616567 | OpenClaw | 8＋8 | 运行中；8 个后端均 ready，前端尚在 precache |
| 7616568 | OpenClaw | 16＋16 | 运行中发生 Eagle 项目磁盘配额错误，只有部分数据，需补跑完整点 |
| 7616569 | mini-SWE | 16＋16 | Docker Hub 匿名拉取限流，准备阶段失败，没有正式测量数据 |

本轮没有查到 OpenClaw N=1、mini-SWE N=1/2/4/8 的新提交或结果；旧 2048 运行和 smoke 不纳入本轮 8192 正式数据。

## OpenClaw 4 个物理节点：主记录齐全，校验标记需要区分指标

- 配置为 budget 8192、TP4、context 262144、每副本 cc16；measurement 完整 3600 秒。
- 两个 worker 都是 completed、measurement.valid=true，全部 939 次重放完成，覆盖 110 条不同 trace。
- 逐条检查 939 份 report：节点集合与原 trace 一致，events.jsonl、console.log、status.json 引用文件存在。
- 全程 13,490 次 LLM call、12,657 次 tool call、3,879,920 个输出 token；LLM call 与输出 token 总数均与两个后端汇总完全相等。所有 LLM 输出均满足目标 token 数。
- 测量窗口完成 767 个任务、输出 3,128,535 tokens，即 869.0375 tokens/s；实际前缀复用率 93.895%。
- 全局 summary 唯一 validation error 是 `x3209c0s37b0n0: missing/failed metric samples`。该前端有 3583 个有效窗口采样、0 error samples、0 JSON 解析错误；仅在 06:14:30.757634 UTC 后出现一次 5.594719 秒采样间隔，超过代码 5 秒阈值。其他前端和两个推理节点采样覆盖均通过。
- 因而任务吞吐、LLM token、时延及推理后端指标可以保留用于分析；这个前端的资源时间序列需要标注采样缺口。原始 valid=false / status=failed 保持不变，没有改写结果或重新生成全局汇总。
- 工具错误并非全部来自原始 trace：全程 895 次 native_error，其中 44 次为原成功→重放 error（web_fetch 38，exec 6）；测量窗口原始报告的 native_tool_errors 为 737。44 次转换的具体工具错误原因未逐一排查，工具行为一致性及工具时间分析需保留此限制。

逐条检查计数见 [openclaw-n2-record-check.json](openclaw-n2-record-check.json)。
原始结果目录：`runs/scaling/multinode/openclaw-balanced-n2-7616564.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/`。

## OpenClaw 32 个物理节点：配额导致测量中断

计划窗口为 14:27:42–15:27:42 UTC。16 个推理与 16 个前端采样尾部落在约 15:07:20–15:07:59，留下约 40 分钟记录，缺少剩余约 20 分钟以及完整 drain。
多个 frontend 的 `metrics.py` 写入报 `OSError: [Errno 122] Disk quota exceeded`，任务随后失败或取消；根 status.json 为 0 字节，没有全局 summary。16 个 worker 中只有 11 个 summary 路径存在，其中 1 个为空；其余已解析的 worker summary 均不是 completed。
这些原始记录可用于中断前的诊断，不能作为完整 60 分钟的正式 scaling 点。

## mini-SWE 32 个物理节点：没有进入测量

16 个推理服务均 ready，但前端在准备第 7 条 trace `django__django-10914` 时，Docker Hub 返回 `TOOMANYREQUESTS` / unauthenticated pull rate limit，重试 3 次后失败。没有 control/start.json、worker summary 或全局 summary。后续 sandbox_dir AttributeError 是初始化失败后的清理异常。

## 当前存储状态与下一步

约 16:59 UTC，Eagle 项目 20219（lc-mpi）占用 23,620,380,908 KiB，硬上限 23,622,320,128 KiB，只剩约 1.85 GiB；是项目字节配额，home 未超限。正在运行的 OpenClaw 16 节点任务和待启动任务存在再次写入失败的直接风险。

优先释放或增加项目空间，并让 mini-SWE 使用可复用的完整镜像缓存，之后再补跑两个 32 节点点。4 节点主数据先保留，并区分资源采样质量与主吞吐数据有效性。本次仅检查与记录，没有取消、重启、提交任务，没有修改实验代码或原始结果。
