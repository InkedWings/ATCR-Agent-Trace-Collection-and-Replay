"""Paper-sized evidence figures from committed aggregates; no raw-run access."""

import csv
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "atcr-paper-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
REPORT = "reports/2026-09-22-routing-and-frontend"
BLUE, GREEN, ORANGE = "#376AB3", "#168575", "#CF6C36"
PURPLE, GREY = "#8659A5", "#92989E"
WORKLOADS = (("openclaw", "OpenClaw", BLUE, "o"),
             ("minisweagent", "mini-SWE", ORANGE, "s"))


def read_json(path):
    return json.loads((ROOT / path).read_text())


def read_csv(path):
    with (ROOT / path).open() as stream:
        return list(csv.DictReader(stream))


def collect():
    sources = {
        "state": "reports/2026-09-16-kv-cache-cause/evidence.json",
        "weak_scaling": "reports/2026-09-15-weak-scaling/analysis.json",
        "merged_results": f"{REPORT}/analysis.json",
        "replicas": f"{REPORT}/replica_diagnostics.json",
        "arrivals": f"{REPORT}/routing_arrivals.json",
        "timeseries": f"{REPORT}/timeseries.csv",
        "replica_timeseries": f"{REPORT}/replica_timeseries.csv",
    }
    evidence = read_json(sources["state"])
    results = read_json(sources["merged_results"])["results"]
    replicas = read_json(sources["replicas"])

    def point(workload, group, backends):
        matches = [r for r in results if (r["workload"], r["group"], r["backends"])
                   == (workload, group, backends)]
        assert len(matches) == 1, (workload, group, backends)
        return matches[0]

    cc = [{k: r[k] for k in ("name", "source", "output_tokens_per_second",
                             "actual_prefix_reuse", "uncached_prompt_per_output",
                             "valid", "steady")}
          for r in evidence["long_runs"] if r["name"].startswith("openclaw-")]
    assert [r["name"] for r in cc] == ["openclaw-cc16", "openclaw-cc32", "openclaw-cc64"]
    pair = [evidence["paired_example"][i] for i in (0, 2)]
    assert pair[0]["prompt"] == pair[1]["prompt"] == 168814
    assert pair[0]["attention_hit"] == pair[1]["attention_hit"]
    assert pair[0]["output_tokens"] == pair[1]["output_tokens"]

    # Replace earlier placeholders with newer observations, never plot imputed points.
    weak = {}
    for r in read_json(sources["weak_scaling"])["rows"]:
        if r["source_kind"] != "measured":
            continue
        key = (r["workload"], r["inference_replicas"])
        weak[key] = dict(workload=key[0], backends=key[1],
                         source_run=r["source_run"], output_tokens_s=r["output_tokens_s"],
                         prefix_reuse=r["prompt_reuse_pct"] / 100,
                         quality_note=r["quality_note"])
    for r in results:
        if r["group"] == "balanced":
            weak[(r["workload"], r["backends"])] = dict(
                workload=r["workload"], backends=r["backends"], source_run=r["root"],
                output_tokens_s=r["output_tokens_s"], prefix_reuse=r["prefix_reuse"],
                quality_note=r["status"])
    assert len(weak) == 9 and ("minisweagent", 4) not in weak
    for (w, n), r in weak.items():
        base = weak[(w, 1)]["output_tokens_s"]
        r["token_speedup"] = r["output_tokens_s"] / base
        r["token_efficiency_percent"] = 100 * r["token_speedup"] / n

    cpu = [{k: point(w, "fixed_sticky", 8)[k] for k in
            ("workload", "run", "total_cc", "frontend_cpu_mean_percent")}
           for w, *_ in WORKLOADS]
    frontend = []
    for group, label, slots in (("fixed_sticky", "1 FE / 4 slots", 4),
                               ("frontend_f1_build8", "1 FE / 8 slots", 8),
                               ("frontend_f2_build2", "2 FE / 4 slots total", 4)):
        r = point("minisweagent", group, 8)
        frontend.append(dict(label=label, run=r["run"], frontends=r["frontends"],
                             total_slots=slots, total_cc=r["total_cc"],
                             output_tokens_s=r["output_tokens_s"],
                             cpu_per_frontend=r["frontend_cpu_mean_percent"],
                             slot_wait_s=r["setup"]["slot_wait_seconds"]["mean"],
                             environment_creation_s=r["setup"]["build_seconds"]["mean"],
                             tool_phase_s=r["execution_seconds_mean"]["tool_only"]))
    assert all(r["total_cc"] == 256 for r in frontend)

    rr = point("openclaw", "routing_rr", 8)
    arrivals = {r["replica"]: r for r in read_json(sources["arrivals"])
                if r["run"] == rr["run"]}
    hotspot = []
    for r in sorted((r for r in replicas if r["run"] == rr["run"]), key=lambda r: r["replica"]):
        a = arrivals[r["replica"]]
        hotspot.append(dict(**r, arrivals=a["arrivals"],
                            residence_s=a["residence_seconds"]["mean"]))
    assert len(hotspot) == 8 and {r["arrivals"] for r in hotspot} == {2006}
    ts = [r for r in read_csv(sources["timeseries"]) if r["run"] == rr["run"]]
    replica_ts = [r for r in read_csv(sources["replica_timeseries"]) if r["run"] == rr["run"]]
    assert len(ts) == 60 and len(replica_ts) == 60 * 8
    trajectory = []
    for start in range(0, 60, 5):
        rows = [r for r in ts if start <= float(r["minute"]) < start + 5]
        gauges = [r for r in replica_ts if start <= float(r["minute"]) < start + 5]
        trajectory.append(dict(
            minute=start + 2.5,
            output_tokens_s=sum(float(r["output_tokens"]) for r in rows) /
                            sum(float(r["duration_seconds"]) for r in rows),
            hotspot_waiting=float(np.mean([float(r["waiting_mean"]) for r in gauges
                                           if r["replica"] == "r07"])),
            other_waiting=float(np.mean([float(r["waiting_mean"]) for r in gauges
                                         if r["replica"] != "r07"]))))
    halves = [sum(float(r["output_tokens"]) for r in ts
                  if start <= float(r["minute"]) < start + 30) / 1800
              for start in (0, 30)]

    routing = []
    for w, *_ in WORKLOADS:
        for group in ("routing_rr", "routing_cache"):
            r = point(w, group, 4)
            rs = [x for x in replicas if x["run"] == r["run"]]
            assert len(rs) == 4
            routing.append(dict(workload=w, policy=group, run=r["run"],
                                status=r["status"], output_tokens_s=r["output_tokens_s"],
                                prefix_reuse=r["prefix_reuse"],
                                max_replica_mean_waiting=max(x["waiting_mean"] for x in rs)))
    return dict(sources=sources, insight1=dict(cc=cc, pair=pair),
                insight2=sorted(weak.values(), key=lambda r: (r["workload"], r["backends"])),
                insight3=dict(cpu=cpu, controls=frontend),
                insight4=dict(run=rr["run"], status=rr["status"], replicas=hotspot,
                              trajectory=trajectory, half_output_tokens_s=halves),
                insight5=routing)


def style():
    # Match the repository's existing reports; size and spacing target a two-column paper.
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.labelsize": 8.5, "axes.titlesize": 9, "axes.titlepad": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .7, "axes.axisbelow": True,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7.5, "legend.frameon": False,
        "lines.linewidth": 1.4, "lines.markersize": 4.5,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": .07,
        "figure.constrained_layout.w_pad": .05,
        "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "hatch.linewidth": .55,
    })


def axis(ax, title, ylabel, ylim=None):
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#CCCCCC", alpha=.6, linewidth=.5, linestyle=":")
    if ylim is not None:
        ax.set_ylim(*ylim)


def bar_labels(ax, bars, fmt="{:.1f}", size=8, padding=3):
    ax.bar_label(bars, labels=[fmt.format(b.get_height()) for b in bars],
                 padding=padding, fontsize=size)


def finish(fig, name, pdf):
    fig.savefig(OUT / f"{name}.png", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", facecolor="white")
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def state_retention(data, pdf):
    rows, pair = data["cc"], data["pair"]
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 4.4))
    for a, key, scale, title, label, limit in (
        (ax[0, 0], "output_tokens_per_second", 1, "(a) Concurrency scaling", "Output tokens/s", (0, 510)),
        (ax[0, 1], "actual_prefix_reuse", 100, "(b) Effective history reuse", "Reused prompt (%)", (0, 110)),
    ):
        y = [r[key] * scale for r in rows]
        a.plot(range(3), y, color=BLUE)
        for i, (r, value) in enumerate(zip(rows, y)):
            a.plot(i, value, "o", color=BLUE, markerfacecolor=BLUE if r["steady"] else "white")
            a.annotate(f"{value:.1f}", (i, value), xytext=(0, 7),
                       textcoords="offset points", ha="center", fontsize=8)
        a.set_xticks(range(3), [16, 32, 64])
        a.set_xlabel("Concurrent agent tasks")
        a.set_xlim(-.3, 2.3)
        axis(a, title, label, limit)
    x, width = np.arange(2), .32
    for offset, key, color, label in ((-.5, "attention_hit", BLUE, "Attention KV match"),
                                       (.5, "cached", GREEN, "Final reuse")):
        bars = ax[1, 0].bar(x + offset * width, [r[key] / 1000 for r in pair],
                            width, color=color, label=label)
        bar_labels(ax[1, 0], bars)
    ax[1, 0].set_xticks(x, ["Earlier call", "Later call"])
    ax[1, 0].legend(loc="upper left", ncol=2, fontsize=7.2,
                    handlelength=1.2, columnspacing=1.2)
    axis(ax[1, 0], "(c) Same prompt, different reuse", "Prompt tokens (×10³)", (0, 237))
    bars = ax[1, 1].bar(x, [r["scheduled_to_first_token_seconds"] for r in pair],
                         .5, color=[BLUE, ORANGE])
    bar_labels(ax[1, 1], bars, "{:.2f}")
    ax[1, 1].set_xticks(x, ["Earlier call", "Later call"])
    axis(ax[1, 1], "(d) Latency after scheduling", "Scheduled to first token (s)", (0, 11.5))
    finish(fig, "01_state_retention", pdf)


def weak_scaling(rows, pdf):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55))
    ns = np.array([1, 2, 4, 8, 16])
    for w, label, color, marker in WORKLOADS:
        by_n = {r["backends"]: r for r in rows if r["workload"] == w}
        for ax, key, scale in zip(axes, ("token_speedup", "token_efficiency_percent", "prefix_reuse"), (1, 1, 100)):
            y = [by_n[n][key] * scale if n in by_n else np.nan for n in ns]
            ax.plot(ns, y, color=color, marker=marker, label=label,
                    markerfacecolor="white" if w == "openclaw" else color)
    axes[0].plot(ns, ns, "--", color=GREY, linewidth=1, zorder=0, label="Ideal")
    axes[0].legend(loc="upper left")
    axes[1].axhline(100, color=GREY, linestyle="--", linewidth=1, zorder=0)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns, ns)
        ax.set_xlim(.87, 18.5)
        ax.set_xlabel("Inference replicas")
    axis(axes[0], "(a) Weak scaling", "Output throughput speedup (×)", (0, 18))
    axis(axes[1], "(b) Scaling efficiency", "Throughput efficiency (%)", (90, 105))
    axis(axes[2], "(c) Reuse across scales", "Reused prompt (%)", (90, 100))
    finish(fig, "02_weak_scaling", pdf)


def frontend_capacity(data, pdf):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.5))
    cpu, rows = data["cpu"], data["controls"]
    bars = axes[0, 0].bar([0, 1], [r["frontend_cpu_mean_percent"] for r in cpu],
                          .52, color=[BLUE, ORANGE])
    bar_labels(axes[0, 0], bars)
    axes[0, 0].set_xticks([0, 1], ["OpenClaw", "mini-SWE"])
    axes[0, 0].axhline(100, color=GREY, ls="--", lw=.8)
    axis(axes[0, 0], "(a) One frontend, eight backends", "Frontend CPU utilization (%)", (0, 115))
    labels = ["1 FE\n4 slots", "1 FE\n8 slots", "2 FE\n4 slots total"]
    colors = [BLUE, PURPLE, GREEN]
    bars = axes[0, 1].bar(range(3), [r["output_tokens_s"] / 1000 for r in rows], .55, color=colors)
    bar_labels(axes[0, 1], bars, "{:.2f}")
    axis(axes[0, 1], "(b) mini-SWE throughput", "Output tokens/s (×10³)", (0, 6.6))
    x, width = np.arange(3), .32
    for offset, key, color, label in ((-.5, "slot_wait_s", GREY, "Slot wait"),
                                     (.5, "environment_creation_s", ORANGE, "Creation")):
        bars = axes[1, 0].bar(x + offset * width, [r[key] for r in rows], width,
                              color=color, label=label)
        bar_labels(axes[1, 0], bars, size=7.5)
    axes[1, 0].legend(loc="upper right")
    axis(axes[1, 0], "(c) mini-SWE environment setup", "Mean time per task (s)", (0, 125))
    bars = axes[1, 1].bar(range(3), [r["tool_phase_s"] for r in rows], .55, color=colors)
    bar_labels(axes[1, 1], bars)
    axis(axes[1, 1], "(d) mini-SWE tool execution", "Mean tool phase per task (s)", (0, 315))
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.set_xticks(range(3), labels)
    finish(fig, "03_frontend_capacity", pdf)


def rr_imbalance(data, pdf):
    rows, trajectory = data["replicas"], data["trajectory"]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.65))
    x = np.arange(8)
    colors = [ORANGE if r["replica"] == "r07" else BLUE for r in rows]
    axes[0, 0].bar(x, [r["output_tokens_s"] for r in rows], .65, color=colors)
    axes[0, 0].text(.04, .91, "148–160 tokens/s", transform=axes[0, 0].transAxes, fontsize=7.5)
    axis(axes[0, 0], "(a) Similar output", "Output tokens/s", (0, 195))
    axes[0, 1].bar(x, [100 * r["prefix_reuse"] for r in rows], .65, color=colors)
    axes[0, 1].annotate(f"{100 * rows[-1]['prefix_reuse']:.1f}", (7, 100 * rows[-1]["prefix_reuse"]),
                         xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7.5)
    axis(axes[0, 1], "(b) Unequal reuse", "Reused prompt (%)", (0, 85))
    axes[0, 2].bar(x, [r["running_mean"] for r in rows], .65, color=BLUE, label="Running")
    axes[0, 2].bar(x, [r["waiting_mean"] for r in rows], .65, color=ORANGE,
                   bottom=[r["running_mean"] for r in rows], label="Waiting")
    axes[0, 2].legend(loc="upper left")
    axis(axes[0, 2], "(c) Concentrated backlog", "Mean outstanding requests", (0, 125))
    axes[1, 0].bar(x, [r["residence_s"] for r in rows], .65, color=colors)
    axes[1, 0].text(.04, .92, "Others: 4.5–5.2 s", transform=axes[1, 0].transAxes, fontsize=7.5)
    axes[1, 0].annotate(f"{rows[-1]['residence_s']:.1f}", (7, rows[-1]["residence_s"]),
                         xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7.5)
    axis(axes[1, 0], "(d) Unequal residence", "Mean request residence (s)", (0, 235))
    for ax in (*axes[0], axes[1, 0]):
        ax.set_xticks(x, [str(i) for i in x])
        ax.set_xlabel("Backend replica")
    t = [r["minute"] for r in trajectory]
    axes[1, 1].plot(t, [r["output_tokens_s"] / 1000 for r in trajectory], color=BLUE,
                    marker="o", markersize=3, label="5-min mean")
    for i, mean in enumerate(data["half_output_tokens_s"]):
        axes[1, 1].hlines(mean / 1000, i * 30, (i + 1) * 30, color=ORANGE,
                          linestyle="--", linewidth=1.1, label="Half-window mean" if i == 0 else None)
    axes[1, 1].legend(loc="lower left", fontsize=6.8)
    axis(axes[1, 1], "(e) System-wide slowdown", "Output tokens/s (×10³)", (0, 2))
    axes[1, 1].set_yticks([0, .5, 1, 1.5, 2])
    for key, color, label in (("hotspot_waiting", ORANGE, "Replica 7"),
                              ("other_waiting", BLUE, "Other replicas (mean)")):
        axes[1, 2].plot(t, [r[key] for r in trajectory], color=color, label=label)
    axes[1, 2].legend(loc="lower left", fontsize=6.8)
    axis(axes[1, 2], "(f) Persistent hotspot", "Mean waiting requests", (0, 105))
    for ax in axes[1, 1:]:
        ax.set_xlim(0, 60)
        ax.set_xticks([0, 20, 40, 60])
        ax.set_xlabel("Measurement time (min)")
    finish(fig, "04_rr_imbalance", pdf)


def cache_aware(rows, pdf):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65))
    x, width = np.arange(2), .31
    for policy, offset, color, label in (("routing_rr", -.5, ORANGE, "RR"),
                                        ("routing_cache", .5, GREEN, "Cache-aware")):
        subset = [next(r for r in rows if r["workload"] == w and r["policy"] == policy)
                  for w, *_ in WORKLOADS]
        for ax, key, scale, fmt in zip(axes, ("output_tokens_s", "prefix_reuse", "max_replica_mean_waiting"),
                                       (.001, 100, 1), ("{:.2f}", "{:.1f}", "{:.2f}")):
            bars = ax.bar(x + offset * width, [r[key] * scale for r in subset], width,
                          color=color, label=label, hatch="//" if policy == "routing_rr" else None,
                          edgecolor="white", linewidth=.5)
            bar_labels(ax, bars, fmt, size=7.5)
    axes[0].legend(loc="upper left", fontsize=7)
    for i, (w, *_) in enumerate(WORKLOADS):
        a, b = [next(r for r in rows if r["workload"] == w and r["policy"] == policy)
                for policy in ("routing_rr", "routing_cache")]
        axes[0].text(i, b["output_tokens_s"] / 1000 + .35,
                     f"{b['output_tokens_s'] / a['output_tokens_s']:.2f}×", ha="center", fontsize=8)
    for ax in axes:
        ax.set_xticks(x, ["OpenClaw", "mini-SWE"])
    axis(axes[0], "(a) Throughput gain", "Output tokens/s (×10³)", (0, 3.35))
    axis(axes[1], "(b) Prefix reuse", "Reused prompt (%)", (0, 112))
    axis(axes[2], "(c) Worst backend queue", "Max. mean waiting requests", (0, 78))
    finish(fig, "05_cache_aware", pdf)


def main():
    data = collect()
    style()
    with PdfPages(OUT / "paper-insights-figures.pdf") as pdf:
        pdf.infodict()["Title"] = "Evidence for agentic serving insights"
        state_retention(data["insight1"], pdf)
        weak_scaling(data["insight2"], pdf)
        frontend_capacity(data["insight3"], pdf)
        rr_imbalance(data["insight4"], pdf)
        cache_aware(data["insight5"], pdf)
    (OUT / "figure-data.json").write_text(json.dumps(data, indent=2) + "\n")
    print("Wrote 5 PNG/PDF figures, a combined PDF, and the selected plotting data.")


if __name__ == "__main__":
    main()
