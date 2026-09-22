# 2026-09-19：两个 small / fixed-frontend n16 作业检查

PBS 历史确认 7637016、7637017 均为 F，Exit_status=1；均申请 17 节点、3 小时，实际运行分别为 00:41:56 和 02:09:03。它们是 1 frontend + 16 个独立 TP4 backend 的 task-sticky/direct 实验，没有启用新准备的官方 router。

| Job | Workload / 全局 task cc | 首个失败位置 | 完成情况 | 正式结果 |
|---|---|---|---|---|
| 7637016 | OpenClaw / 256 | r08/task 00010，tool-003 web_fetch 调用本地 Gateway 时 httpx.ReadError | warmup 开始后 147 秒失败；全程完成 5 个 task，未进入正式窗口 | 不可用，需要修复后重跑 |
| 7637017 | mini-SWE / 512 | r07/task 00066，llm-040 调用 vLLM 时 httpx.ReadError | 已进入测量约 78.1/90 分钟；首次失败前测量窗口内完成 1491 个 task | 仅保留为部分窗口诊断，不计入完整正式点 |

## 已确认的触发链

两边都是单条 task 的 HTTP 读取错误，触发 benchmark 的 fail-fast，再取消同节点其他逻辑 worker 和剩余任务。前端最终的 cleanup failed 是外层退出报告，不是最初根因。两任务均未达到 3 小时 walltime，PBS 不是超时终止。

OpenClaw 的错误发生在工具 Gateway 路径，此 task 前三个 LLM call 都已成功。Gateway 在异常后接到清理用的 SIGTERM，日志没有显示它在错误前崩溃。

mini-SWE 的错误发生在等待 LLM HTTP 响应头阶段，约 3 毫秒即 ReadError。对应 r07 的推理日志没有 ERROR / Traceback / CUDA OOM，后端仍持续输出 metrics，随后由协调器关闭。当前证据只能确认连接层失败，尚不能唯一归因于网络、连接复用、服务端关闭连接或 CPU 压力。

## 已收集数据的诊断价值

- OpenClaw：失败前最后 60 秒 frontend CPU 平均 93.34%，load average 约 227；这是启动/warmup 阶段，不能作为稳态饱和结论。
- mini-SWE：首次失败前的正式测量窗口内 frontend CPU 样本均值 **97.44%**，约 **98.18%** 的样本不低于 90%。这段数据说明在 1:16、全局 cc512 下单前端持续接近 CPU 饱和，但尚不确定 sandbox 创建、工具运行等各自占比。
- 两个前端的可用内存和本地盘均有余量，没有存储或主机内存耗尽证据。mini-SWE 的 61 个 SIF 成功复制到本地，约 64.66 GiB，耗时 74.6 秒；此次失败位置不在 Docker Hub pull 或 SIF staging。
- 部分任务取消时 summary 保留 running 状态；这是失败退出的未完成记录，不代表作业现在仍在运行。

后续先排查 HTTP 连接读取错误及“一条请求失败取消整组”的处理，再重跑这两档。不要把部分测量窗口悄悄当成完整正式结果，也不要直接对可能已执行的工具调用进行无条件重试。

原始证据目录：

- `runs/scaling/multinode/openclaw-fixed_frontend-n16-7637016.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/`
- `runs/scaling/multinode/minisweagent-fixed_frontend-n16-7637017.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov/`

提取结果见 [evidence.json](evidence.json)。本次只核对调度记录、失败任务日志、服务日志和已保存资源采样；未修改运行代码、未连接计算节点、未重交任务。
