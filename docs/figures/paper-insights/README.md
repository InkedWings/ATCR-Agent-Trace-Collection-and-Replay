# Paper insight figures

Five figures accompany [Paper Insights](../../paper-insights.md). Each is exported as a 300-dpi PNG and a vector PDF; [paper-insights-figures.pdf](paper-insights-figures.pdf) collects the five pages. Text is embedded as TrueType fonts in the PDFs.

The style follows the existing routing/frontend report: DejaVu Sans, blue/green/orange with purple and grey accents, light horizontal grids, and no top/right spines. Layouts are redrawn at 7.2-inch paper width with 8–9-point labels, short panel titles, and captions outside the artwork. The local manuscript currently has no figure references from which to infer another style.

From the repository root:

```bash
.venv/bin/python docs/figures/paper-insights/plot.py
```

Dependencies are the existing reporting environment (`matplotlib`, `numpy`). The script reads committed aggregates only. It does not access raw runs, submit experiments, or modify earlier reports. [figure-data.json](figure-data.json) records the selected numeric values and source paths used by the plots. Each configuration has one run; no between-run confidence intervals are estimated.

| Figure | Evidence | Source |
| --- | --- | --- |
| 1: State retention | OpenClaw CC16/32/64 with budget8192; two executions of an identical prompt in a separate CC32 diagnostic | [Cache-state evidence](../../../reports/2026-09-16-kv-cache-cause/evidence.json) |
| 2: Weak scaling | Nine measured configurations; one frontend per inference replica | [Earlier measured aggregates](../../../reports/2026-09-15-weak-scaling/analysis.json), [updated measurements](../../../reports/2026-09-22-routing-and-frontend/analysis.json) |
| 3: Frontend capacity | One-frontend CPU demand; mini-SWE at eight backends with controlled frontend and creation-slot counts | [Frontend comparisons](../../../reports/2026-09-22-routing-and-frontend/analysis.json) |
| 4: RR imbalance | OpenClaw eight-backend replica metrics, arrival-cohort residence, and five-minute trajectories | [Replica metrics](../../../reports/2026-09-22-routing-and-frontend/replica_diagnostics.json), [arrivals](../../../reports/2026-09-22-routing-and-frontend/routing_arrivals.json), [window series](../../../reports/2026-09-22-routing-and-frontend/timeseries.csv), [replica series](../../../reports/2026-09-22-routing-and-frontend/replica_timeseries.csv) |
| 5: Cache-aware routing | Four-backend RR/CA comparisons through the same official router | [Window metrics](../../../reports/2026-09-22-routing-and-frontend/analysis.json), [replica means](../../../reports/2026-09-22-routing-and-frontend/replica_diagnostics.json) |

Important interpretation details:

- Figure 1 separates the one-hour CC sweep from the 30-minute request-level diagnostic. Hollow markers retain the nonsteady status of CC32/64. The two diagnostic requests share the same prompt and output length, but background load was not controlled; latency is scheduling-to-first-token wall time, not exclusive GPU computation. The diagnostic does not establish that every long-run cache miss is caused by recurrent state.
- Figure 2 discards both theoretical placeholders in the earlier report and adds the measured OpenClaw four-replica point. mini-SWE at four replicas remains missing, with a line gap. Sixteen replicas mean sixteen inference nodes plus sixteen frontend nodes. OpenClaw at two replicas has an acknowledged frontend sampling gap; its throughput counts were checked in the source report. Each workload keeps its own CC, trace pool, warmup, and measurement duration fixed across scales.
- Figure 3 uses total task CC256 for all mini-SWE controls and an aggregate creation limit of 4, 8, and 4, respectively. Timings follow complete measurement-admission cohorts through natural drain. Setup bars separate slot waiting from creation; the tool phase is total tool wall time per task. More frontend nodes also add memory bandwidth and local I/O, so CPU utilization does not isolate CPU as the sole cause. The OpenClaw CPU comparison uses its evaluated CC128 and replay tool policy.
- Figure 4 uses all 2,006 measurement arrivals per replica for residence, including completions after the window. Output and running/waiting/reuse metrics use the measurement window itself. Output trajectories divide five-minute token counts by their duration; queue trajectories average the existing minute means. The peer line averages the seven non-hotspot replicas. These are temporal averages, not confidence intervals. Two warmup timeouts and nonsteady behavior remain documented.
- Figure 5's queue metric is the maximum of four per-backend window means. The OpenClaw RR window is nonsteady; mini-SWE RR has a verified complete measurement window but an incomplete subsequent drain. Only window throughput/reuse/queue metrics are used, not truncated task-tail latency. CA includes both affinity and load feedback, so the comparison does not separate their contributions. P2 and missing eight-backend CA points are excluded.
