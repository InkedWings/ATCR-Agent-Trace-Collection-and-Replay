"""Publication-friendly static plots; input is the derived analysis, never raw runs."""
import csv
import json
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "atcr-report-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
import numpy as np

OUT = Path(__file__).resolve().parent
COLORS = {"routing_rr": "#CF6C36", "routing_cache": "#168575", "fixed_sticky": "#376AB3", "balanced": "#8C9299"}
NAMES = {"openclaw": "OpenClaw", "minisweagent": "mini-SWE-agent"}


def draw(report):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.labelsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .17, "axes.axisbelow": True,
                         "savefig.dpi": 150, "pdf.fonttype": 42})
    results = report["results"]
    with (OUT / "timeseries.csv").open() as f:
        series = list(csv.DictReader(f))
    with (OUT / "sandbox_setup.csv").open() as f:
        setups = list(csv.DictReader(f))
    figdir = OUT / "figures"
    figdir.mkdir(exist_ok=True)

    def point(workload, group, n=4):
        return next(r for r in results if r["workload"] == workload and r["group"] == group and r["backends"] == n)

    def rows(r):
        return [s for s in series if s["run"] == r["run"]]

    def values(r, key):
        return np.array([float(s[key]) if s.get(key) else np.nan for s in rows(r)])

    def finish(fig, name, pdf, note=""):
        if note:
            fig.text(.02, .015, note, fontsize=8.5, color="#555555", va="bottom")
        fig.tight_layout(rect=(0, .065 if note else 0, 1, .94))
        fig.savefig(figdir / f"{name}.png", bbox_inches="tight")
        fig.savefig(figdir / f"{name}.pdf", bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    with PdfPages(OUT / "results.pdf") as pdf:
        fig, axes = plt.subplots(2, 3, figsize=(12.4, 7))
        fig.suptitle("Cache-aware routing improves throughput at the same 1+4 placement", fontsize=15)
        for ri, workload in enumerate(NAMES):
            pair = [point(workload, g) for g in ("routing_rr", "routing_cache")]
            for ci, (key, label, scale) in enumerate([
                ("output_tokens_s", "Output tokens / second", 1),
                ("tasks_per_minute", "Completed tasks / minute", 1),
                ("prefix_reuse", "Actual cached / prompt tokens (%)", 100)]):
                ax = axes[ri, ci]
                ys = [r[key] * scale for r in pair]
                bars = ax.bar([0, 1], ys, width=.56, color=[COLORS[r["group"]] for r in pair])
                if workload == "minisweagent":
                    bars[0].set_hatch("//")
                    bars[0].set_edgecolor("#704528")
                ax.set_xticks([0, 1], ["RR †" if ri == 0 else "RR ‡", "Cache-aware"])
                ax.set_ylabel(label)
                ax.set_title(NAMES[workload] if ci == 0 else "")
                ax.set_ylim(0, max(ys) * 1.24)
                for bar, y in zip(bars, ys):
                    ax.text(bar.get_x() + bar.get_width()/2, y + max(ys)*.025,
                            f"{y:,.0f}" if key == "output_tokens_s" else f"{y:.2f}", ha="center", fontsize=11)
                if ci < 2:
                    ax.text(.98, .92, f"{ys[1]/ys[0]:.2f}×", transform=ax.transAxes,
                            ha="right", weight="bold", color=COLORS["routing_cache"])
        finish(fig, "01_routing_performance", pdf,
               "Full windows: OpenClaw 60 min, mini-SWE 90 min. † RR is nonsteady. ‡ Full-window counts recovered; drain timed out.\n"
               "Single run per point; error bars are unavailable. Mini-SWE RR task/LLM latency cohorts are censored and omitted.")

        fig, axes = plt.subplots(2, 3, figsize=(12.4, 7))
        fig.suptitle("Routing trajectories: reuse, throughput, and backend queues", fontsize=15)
        for ri, workload in enumerate(NAMES):
            for group, name in (("routing_rr", "RR"), ("routing_cache", "Cache-aware")):
                r = point(workload, group)
                for ci, (key, title, scale) in enumerate([
                    ("output_tokens_per_second", "Output tokens / second", 1),
                    ("actual_prefix_reuse", "Actual prefix reuse (%)", 100),
                    ("waiting_per_replica_mean", "Queued requests / backend", 1)]):
                    ax = axes[ri, ci]
                    ax.plot(values(r, "minute") + .5, values(r, key)*scale,
                            color=COLORS[group], label=name, lw=1.5)
                    ax.set_xlabel("Minutes into measurement")
                    ax.set_ylabel(title)
                    if ci == 0:
                        ax.set_title(NAMES[workload])
                axes[ri, 0].legend(frameon=False)
            for ax in axes[ri]:
                ax.relim()
                ax.autoscale_view()
                ax.set_ylim(bottom=0)
        finish(fig, "02_routing_timeseries", pdf,
               "60-second bins; queued requests use all backend samples. Curves cover measurement only, excluding warmup/drain.\n"
               "RR nonstationarity and mini-SWE drain cancellation limit steady-state capacity / tail-latency claims.")

        fig, axes = plt.subplots(2, 3, figsize=(12.4, 7))
        fig.suptitle("Routing changes locality and the amount of uncached prefill work", fontsize=15)
        for ri, workload in enumerate(NAMES):
            pair = [point(workload, g) for g in ("routing_rr", "routing_cache")]
            for ci, label in enumerate(["Adjacent-call backend switches (%)", "Uncached prompt tokens / output token"]):
                ys = [r["routing"]["adjacent_call_backend_switch_fraction"]*100 if ci == 0
                      else r["uncached_prompt_per_output"] for r in pair]
                ax = axes[ri, ci]
                ax.bar([0, 1], ys, color=[COLORS[r["group"]] for r in pair], width=.56)
                ax.set_xticks([0, 1], ["RR", "Cache-aware"])
                ax.set_ylabel(label)
                ax.set_ylim(0, max(ys)*1.2)
                for i, y in enumerate(ys):
                    ax.text(i, y+max(ys)*.025, f"{y:.1f}", ha="center")
                if ci == 0:
                    ax.set_title(NAMES[workload])
            ax = axes[ri, 2]
            for offset, r in zip([-.18, .18], pair):
                ax.bar(np.arange(4)+offset, [p["output_tokens_s"] for p in r["replicas"]],
                       width=.34, color=COLORS[r["group"]], label=f'{"RR" if r["group"] == "routing_rr" else "Cache-aware"}: CV {r["replica_output_cv"]*100:.1f}%')
            ax.set_xticks(range(4), ["r00", "r01", "r02", "r03"])
            ax.set_ylabel("Output tokens / second / backend")
            ax.set_ylim(0, 1.3 * max(p["output_tokens_s"] for r in pair for p in r["replicas"]))
            ax.legend(frameon=False, fontsize=8)
        finish(fig, "03_routing_locality", pdf,
               "Switch rate: adjacent observed completed calls of tasks with ≥8 window calls; weighted by adjacent-call pairs.\n"
               "Uncached input/output is a window-level work ratio, not a matched-request estimate. Cache-aware is not strict task stickiness.")

        fig, axes = plt.subplots(2, 3, figsize=(12.4, 7))
        fig.suptitle("One frontend scales differently for the two workloads", fontsize=15)
        for ri, workload in enumerate(NAMES):
            baseline = point(workload, "balanced", 1)
            fixed = [baseline] + sorted([r for r in results if r["workload"] == workload and r["group"] == "fixed_sticky"], key=lambda r:r["backends"])
            balanced = sorted([r for r in results if r["workload"] == workload and r["group"] == "balanced" and r["backends"] <= 8], key=lambda r:r["backends"])
            for ci, (key, title) in enumerate([("output_tokens_s", "Output tokens / second"), ("tasks_per_minute", "Completed tasks / minute")]):
                ax = axes[ri, ci]
                ax.plot([r["backends"] for r in balanced], [r[key] for r in balanced], "s--",
                        color=COLORS["balanced"], label="N frontends + N backends (historical)")
                ax.plot([r["backends"] for r in fixed], [r[key] for r in fixed], "o-",
                        color=COLORS["fixed_sticky"], label="1 frontend + N backends (sticky)")
                ax.set_ylabel(title)
                if ci == 0:
                    ax.set_title(NAMES[workload])
                ax.set_ylim(bottom=0)
            ax = axes[ri, 2]
            ax.plot([r["backends"] for r in fixed], [r["frontend_cpu_mean_percent"] for r in fixed], "o-", color=COLORS["fixed_sticky"])
            ax.axhline(90, ls=":", color="#9C3A32", label="90% node CPU")
            ax.set_ylabel("Frontend node CPU busy (%)")
            ax.set_ylim(0, 110)
            last = fixed[-1]
            ax.annotate(f'{last["frontend_cpu_mean_percent"]:.1f}%', (8, last["frontend_cpu_mean_percent"]),
                        xytext=(-5, 9), textcoords="offset points", ha="right")
            for ax in axes[ri]:
                ax.set_xlim(.6, 8.4)
                ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
                ax.set_xlabel("Backend nodes (TP4 on each)")
            if workload == "minisweagent":
                axes[ri, 0].text(.04, .94, "1+4 sticky: no completed result", transform=axes[ri, 0].transAxes, va="top", fontsize=8.5)
            axes[ri, 0].set_title(NAMES[workload])
        axes[0, 0].legend(frameon=False, fontsize=7.5)
        finish(fig, "04_fixed_frontend_scaling", pdf,
               "All markers are measured; segments merely connect available points. N=1 is the shared historical baseline.\n"
               "Historical balanced runs differ in HTTP/build policy: useful capacity references, not controlled frontend-count comparisons.")

        fig, axes = plt.subplots(2, 2, figsize=(12, 7.7))
        fig.suptitle("Frontend pressure and where task time is spent", fontsize=15)
        components = [("setup", "Environment setup", "#BD6573"), ("llm_only", "LLM", "#376AB3"),
                      ("tool_only", "Tools", "#DCAC45"), ("llm_tool_overlap", "LLM/tool overlap", "#8D74AF"),
                      ("other_and_cleanup", "Other / cleanup", "#92989F")]
        for ci, workload in enumerate(NAMES):
            ax = axes[0, ci]
            for n, color in ((2, "#8297AA"), (8, COLORS["fixed_sticky"])):
                r = point(workload, "fixed_sticky", n)
                ax.plot(values(r, "minute")+.5, values(r, "frontend_cpu_percent"), label=f"1+{n} sticky", color=color)
            ax.axhline(90, color="#9C3A32", ls=":", lw=1)
            ax.set_ylim(0, 105)
            ax.set_xlabel("Minutes into measurement")
            ax.set_ylabel("Frontend node CPU busy (%)")
            ax.set_title(NAMES[workload])
            ax.legend(frameon=False)
            ax = axes[1, ci]
            comparisons = [point(workload, "fixed_sticky", 2), point(workload, "balanced", 8), point(workload, "fixed_sticky", 8)]
            bottom = np.zeros(3)
            for key, label, color in components:
                ys = np.array([r["execution_seconds_mean"][key] for r in comparisons])
                ax.bar(range(3), ys, bottom=bottom, label=label, color=color, width=.58)
                bottom += ys
            for i, val in enumerate(bottom):
                ax.text(i, val + max(bottom)*.02, f"{val:.0f}s", ha="center")
            ax.set_ylim(0, max(bottom)*1.5)
            ax.set_xticks(range(3), ["1+2 sticky", "8+8 historical", "1+8 sticky"])
            ax.set_ylabel("Mean task lifecycle (seconds)")
        axes[1, 0].legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
        finish(fig, "05_frontend_bottleneck", pdf,
               "Top: full-window CPU samples aggregated into 60s means. Bottom: measurement-admission cohort, completed through drain.\n"
               "Time components are disjoint interval unions; mean bars do not imply identical task composition across runs.")

        fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9))
        fig.suptitle("mini-SWE environment setup: slot waiting and build slowdown", fontsize=15)
        compared = [point("minisweagent", "fixed_sticky", 2), point("minisweagent", "routing_rr"),
                    point("minisweagent", "routing_cache"), point("minisweagent", "fixed_sticky", 8)]
        labels = ["1+2\nsticky", "1+4\nRR", "1+4\ncache-aware", "1+8\nsticky"]
        for ax, field, title in zip(axes, ["slot_wait_seconds", "build_seconds"], ["Wait for a build slot (seconds)", "Environment build time (seconds)"]):
            arrays = [[float(row[field]) for row in setups if row["run"] == r["run"]] for r in compared]
            boxes = ax.boxplot(arrays, tick_labels=labels, showfliers=False, whis=(5, 95), patch_artist=True,
                               medianprops={"color": "#333333", "linewidth": 1.2})
            for patch, r in zip(boxes["boxes"], compared):
                patch.set_facecolor(COLORS[r["group"]]); patch.set_alpha(.7)
            means = [np.mean(a) for a in arrays]
            ax.scatter(range(1, 5), means, color="#222222", marker="D", s=25, zorder=3, label="Mean")
            ax.set_ylabel(title)
            ax.set_ylim(-.5, max(np.percentile(a, 95) for a in arrays)*1.17)
            ax.legend(frameon=False)
            for i, r in enumerate(compared):
                ax.text(i+1, .98, f'n={r["setup"]["records"]}', ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=8)
        finish(fig, "06_sandbox_setup", pdf,
               "All measurement admissions with setup records, including later-cancelled RR tasks; no missing records.\n"
               "Box: p25–p75, center: median, whiskers: p5–p95; outliers hidden. All four runs allow 4 concurrent builds per frontend.")

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
        fig.suptitle("Completed OpenClaw routing latency cohorts", fontsize=15)
        pair = [point("openclaw", g) for g in ("routing_rr", "routing_cache")]
        for ax, field, ylabel in zip(axes, ["ttft_seconds", "task_lifecycle_seconds"], ["Client TTFT (seconds)", "Task lifecycle (seconds)"]):
            for off, r in zip([-.18, .18], pair):
                ax.bar(np.arange(3)+off, [r[field][k] for k in ("mean", "p50", "p95")], width=.34,
                       color=COLORS[r["group"]], label="RR (nonsteady)" if r["group"] == "routing_rr" else "Cache-aware")
            ax.set_xticks(range(3), ["Mean", "p50", "p95"])
            ax.set_ylabel(ylabel)
            ax.legend(frameon=False)
        finish(fig, "07_openclaw_routing_latency", pdf,
               "OpenClaw: full completion through drain; task admissions / LLM starts within the measurement window.\n"
               "Mini-SWE RR latency omitted: 36 admission-cohort tasks were cancelled after the 30-minute drain limit.")
        diagnostics = OUT / "replica_diagnostics.json"
        if diagnostics.exists():
            data = json.loads(diagnostics.read_text())
            fig, axes = plt.subplots(2, 3, figsize=(12.4, 7))
            fig.suptitle("RR hides a hotspot behind similar per-backend output rates", fontsize=15)
            for ri, workload in enumerate(NAMES):
                for ci, (key, label, scale) in enumerate([
                    ("prefix_reuse", "Actual prefix reuse (%)", 100),
                    ("running_mean", "Mean running requests", 1),
                    ("waiting_mean", "Mean queued requests", 1)]):
                    ax = axes[ri, ci]
                    for offset, group, name in ((-.18, "routing_rr", "RR"), (.18, "routing_cache", "Cache-aware")):
                        selected = sorted([r for r in data if r["workload"] == workload and r["group"] == group], key=lambda r:r["replica"])
                        ax.bar(np.arange(4)+offset, [r[key]*scale for r in selected], width=.34, color=COLORS[group], label=name)
                    ax.set_xticks(range(4), ["r00", "r01", "r02", "r03"])
                    ax.set_ylabel(label)
                    ax.set_ylim(bottom=0)
                    if ci == 0:
                        ax.set_title(NAMES[workload])
                    if ci == 2:
                        ax.legend(frameon=False)
            finish(fig, "08_replica_hotspot", pdf,
                   "Full-window samples. RR hotspots: OpenClaw r03, mini-SWE r01. Closed-loop task progress can be limited by these replicas.\n"
                   "This identifies queue/cache concentration; the initiating cause needs repeats / request-level context and routing diagnostics.")
    print(f"Wrote {figdir} and {OUT / 'results.pdf'}", flush=True)
