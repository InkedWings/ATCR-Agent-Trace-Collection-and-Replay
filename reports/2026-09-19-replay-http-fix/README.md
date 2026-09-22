# 2026-09-19：replay HTTP 失败隔离及 n16 重交

修复对象为 [两个 n16 small 作业的失败](../2026-09-19-fixed-frontend-small-check/README.md)：7637016（OpenClaw）和 7637017（mini-SWE）。两者均已结束，不需要 qdel。

## 实现

- LLM replay 和 OpenClaw Gateway 的 HTTP 客户端均关闭空闲连接复用，每次请求使用新连接，并发送 `Connection: close`。官方 router 的上游转发也保留该请求头。TTFT 仍包含请求建连时间。
- 不自动重发失败的 POST。即使尚未读到响应头，服务端也可能已经执行过推理或工具；重发会造成重复执行及计数歧义。
- 多节点负载驱动遇到 HTTP 传输失败（包括 ReadError、超时、流中断），或 HTTP 500/502/503/504 时，将当前 task 记为 failed，保留事件和错误类型，清理当前 task 后继续接纳池中的下一个 task。其他并发 task 和测量窗口继续运行。官方 router 会将部分上游连接错误转换成 HTTP 500，因此这些状态码也需要隔离。
- 非 HTTP 故障、400/401 等请求或认证错误、cleanup 失败、后端进程退出仍明确失败。普通单机 benchmark 默认保留原来的 fail-fast；多节点 frontend 显式开启上述失败隔离。
- 报告区分 `collection_complete` 和 `valid`。完整采集到窗口末尾不等于无错误；失败 task 不计入成功吞吐，另输出失败数、窗口内失败比例及 `failed-tasks.csv`。存在失败或计数不符时仍保留 `valid=false`，协调器在完成采集和保存报告后返回非零，不会把它作为无错误正式 scaling 点导出。
- 新 run 的 config 和 worker manifest 记录连接/失败策略。导出时不会把不同策略的结果自动混成同一组比较。

原始日志尚不能唯一证明 ReadError 的底层来源；关闭复用消除了空闲连接关闭竞争这一可能路径。没有改动模型、TP4、context length、8192 token budget、task cc、trace 池或测量时长。

## 排队任务

检查时以下 10 个作业均为 Q。已提交的 PBS 脚本在启动时调用当前仓库的 editable Python 代码，因此此次修复无需修改 PBS 资源申请或冻结配置，也无需撤销这些作业的排队位置。

| 作业 | 内容 | 处理 |
|---|---|---|
| 7637010、7637011 | 两类任务，1 frontend + 2 backends | 保留 |
| 7637012、7637013 | 两类任务，1 frontend + 4 backends | 保留 |
| 7637014、7637015 | 两类任务，1 frontend + 8 backends | 保留 |
| 7637645、7637646 | OpenClaw，官方 RR / cache-aware，4 backends | 保留 |
| 7637647、7637650 | mini-SWE，官方 RR / cache-aware，4 backends | 保留 |
| 7637016、7637017 | 两个已失败的 n16 | 重新提交下面两条命令 |

已有 capacity allocation 7636974 与此次重交无关，保留。

```bash
cd /lus/eagle/projects/lc-mpi/ZhijingYe/SIGMETRICS_2027/trace_framework
qsub runs/fixed-frontend-jobs-20260919/jobs/openclaw-fixed_frontend-n16.pbs
qsub runs/fixed-frontend-jobs-20260919/jobs/minisweagent-fixed_frontend-n16.pbs
```

两份脚本均申请 `prod`、17 个物理节点、3 小时。PBS 可将 prod 路由到 small；不要重新提交仍在排队的 n2/n4/n8 和 router 四组。

## 验证范围

本机真实 HTTP/1.1 socket 测试覆盖连接重置、响应中途断开、500/503、失败请求不重发、其他 task 继续、完整窗口与失败计数；另验证 400/401 和 cleanup 错误仍失败。使用实际安装的官方 vLLM Router 进程验证转发后的连接关闭与原始请求/流/路由行为。相关 replay、Gateway、frontend、全局统计回归测试和 12 份正式 PBS 脚本语法检查结果见 `validation.json`。

本次未连接计算节点、未运行 GPU 实验、未代提交或取消作业；本机验证不能替代 17 节点满负载复测。
