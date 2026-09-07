# 实验分析报告

这里保存 2026-09-07 的分析快照、统计脚本、CSV 和图表。后续结果应注明选用的 run 和测量口径。

| 报告 | 内容 |
|---|---|
| [cc=1/2/4 数据复核](2026-09-07-cc124/report.md) | 六个有效点的计数复核、性能趋势与后续实验建议 |
| [Trace 统计与展示方案](2026-09-07-trace-characterization/plan.md) | 24 条 OpenClaw 与 22 条 mini-SWE 原始 trace 的调用、token、工具画像；包含预览图与逐调用统计 |
| [cc=1/2/4/8/32 scaling 汇总](2026-09-07-scale-cc1-32/README.md) | 十个有效点的完整指标表、窗口定义与趋势解释；cc=16 在此快照中仍缺失 |

## 输入与复现

原始 trace、任务工作区、模型和运行日志保留在本地，遵循仓库现有的数据忽略规则。CSV 中的原始路径用于追溯这次实验；只克隆代码仓库不会获得这些输入。统计脚本读取各次运行的 manifest/preflight 中记录的 trace 路径，因此重现时也需要这些路径可用。

在仓库根目录，使用 Python 3.11+ 环境运行。绘图依赖单独列在 [requirements.txt](requirements.txt)，可安装到现有环境：

```bash
uv pip install --python .venv/bin/python -r reports/requirements.txt
```

随后执行统计：

```bash
MPLCONFIGDIR=/tmp/agenttrace-matplotlib .venv/bin/python reports/2026-09-07-cc124/analyze.py
MPLCONFIGDIR=/tmp/agenttrace-matplotlib .venv/bin/python reports/2026-09-07-trace-characterization/characterize.py
.venv/bin/python reports/2026-09-07-scale-cc1-32/aggregate.py
```

脚本会更新所在报告目录的生成文件，不会执行推理、回放工具或提交计算任务。所需本地输入包括：

- `runs/scaling/single-backend-cc124-20260907-2/`
- `runs/scaling/preflight-20260906-1/`
- `runs/scaling/preemptable-cc8-7596639/experiment/`
- `runs/scaling/debug-cc8-7597420/experiment/`
- `runs/scaling/preemptable-cc32-7596641/experiment/`
- `runs/openclaw-gaia/fixed-first30-7576424/tasks/` 中的原生 trajectory，以及 preflight 指向的 GAIA 和 mini-SWE trace/trajectory。

Scaling 表使用 `summary.json.measurement` 的测量窗口。Trace 画像对每条原始 trace 只统计一次，完整候选池与 32K 可回放子集分开标记。两类报告的统计单位不同，具体定义见各报告。
