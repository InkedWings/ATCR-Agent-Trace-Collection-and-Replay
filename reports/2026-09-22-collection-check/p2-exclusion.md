# P2 暂不纳入性能对比

2026-09-22 数据分析补查，按用户要求暂不纳入合并主表、图表及性能结论。此前恢复的完整 measurement 窗口计数保留，但计数完整不代表预期的负载感知机制生效。

本地证据来自 `minisweagent-routing-n4-power_of_two-7643321.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov`：

- `frontends/x3003c0s13b1n0/router/config.json`：官方 vllm-router 0.1.15，policy 为 power_of_two。
- 同目录 `service.log`：全运行 41,170 条 `Power-of-two selection` 日志，两个候选的 load **全部为 0/0**。
- `backends/r02/service.log`：启动时注册 `/load`，实际收到的 `GET /get_load` 持续返回 404。

已核对对应版本官方源码：[P2 策略](https://github.com/vllm-project/router/blob/v0.1.15/src/policies/power_of_two.rs) 优先使用外部负载、缺失时使用本地计数，相等时选择随机抽取的第一个候选；[HTTP router](https://github.com/vllm-project/router/blob/v0.1.15/src/routers/http/router.rs) 查询 `/get_load`，而 typed request 路径仅对 cache-aware 增减本地 load。因此这轮 P2 没有得到可区分后端的负载信号，选择行为退化为随机，不能作为“有效负载均衡、无前缀亲和”的消融对照。

这是该版本和当前接入路径的兼容/计数问题证据，不说明 P2 算法本身无效。此前仅检查两个未结束流被分到不同后端的测试不足以排除此问题：随机选择也可能通过单次检查。后续若修复，需验证**运行时负载确实非零且能避开已知忙后端**，再决定是否纳入对比。本次没有修改 router、提交或重跑实验。
