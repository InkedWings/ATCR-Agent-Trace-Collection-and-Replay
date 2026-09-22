"""Merged measured comparisons. P2 and unmeasured points are never plotted."""
import csv
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "atcr-report-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

OUT = Path(__file__).resolve().parent
BLUE, GREEN, PURPLE, ORANGE, GREY = "#376AB3", "#168575", "#8659A5", "#CF6C36", "#92989E"


def read_csv(name):
    with (OUT / name).open() as stream:
        return list(csv.DictReader(stream))


def draw(report):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .17, "axes.axisbelow": True,
                         "savefig.dpi": 160, "pdf.fonttype": 42})
    results = report["results"]
    assert all("power_of_two" not in r["run"] for r in results)
    series, setups = read_csv("timeseries.csv"), read_csv("sandbox_setup.csv")
    replicas = json.loads((OUT / "replica_diagnostics.json").read_text())
    audits = {r["run"]: r for r in json.loads((OUT / "completed_call_audit.json").read_text())}
    figdir = OUT / "figures"
    figdir.mkdir(exist_ok=True)

    def point(w, g, n):
        return next((r for r in results if r["workload"] == w and r["group"] == g and r["backends"] == n), None)

    def values(r, key):
        return np.array([float(s[key]) if s.get(key) else np.nan for s in series if s["run"] == r["run"]])

    def smooth(r, key, scale=1):
        x, y = values(r, "minute") + .5, values(r, key) * scale
        if key == "actual_prefix_reuse":
            prompt, cached = values(r, "prompt_tokens"), values(r, "cached_prompt_tokens")
            aggregated = [scale*np.sum(cached[i:i+5])/np.sum(prompt[i:i+5]) for i in range(0,len(y),5)]
        else:
            aggregated = [np.nanmean(y[i:i+5]) for i in range(0,len(y),5)]
        return [np.mean(x[i:i+5]) for i in range(0,len(x),5)], aggregated

    def finish(fig, name, pdf, note):
        fig.text(.025, .014, note, fontsize=8.5, color="#555555", va="bottom")
        fig.tight_layout(rect=(0, .075, 1, .94))
        fig.savefig(figdir / (name + ".png"), bbox_inches="tight")
        fig.savefig(figdir / (name + ".pdf"), bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    fronts = [point("minisweagent", g, 8) for g in ("fixed_sticky", "frontend_f2_build2", "frontend_f1_build8")]
    labels = ["1+8\n4 slots", "2+8\n2 slots/FE", "1+8\n8 slots"]
    colors = [BLUE, GREEN, PURPLE]
    legend_labels = ["1+8, 4 build slots", "2+8, 2 build slots/FE", "1+8, 8 build slots"]

    def bar_values(ax, ys, ylabel, ylim=None, digits=1):
        bars = ax.bar(range(3), ys, width=.58, color=colors)
        ax.set_xticks(range(3), labels)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, ylim or max(ys)*1.24)
        for b, y in zip(bars, ys):
            ax.annotate(f"{y:,.{digits}f}", (b.get_x()+b.get_width()/2,y), xytext=(0,5),
                        textcoords="offset points", ha="center", fontsize=10)

    with PdfPages(OUT / "results.pdf") as pdf:
        fig, axes = plt.subplots(2,3,figsize=(12.5,7.5))
        fig.suptitle("mini-SWE: more frontend resources help; more build slots do not", fontsize=15)
        for ax, key, label in zip(axes[0], ("output_tokens_s", "tasks_per_minute", "frontend_cpu_mean_percent"),
                                  ("Output tokens / second", "Completed tasks / minute", "CPU busy per frontend (%)")):
            bar_values(ax, [r[key] for r in fronts], label, 112 if "cpu" in key else None)
        bar_values(axes[1,0], [r["setup"]["slot_wait_seconds"]["mean"] for r in fronts], "Build-slot wait: mean seconds")
        bar_values(axes[1,1], [r["setup"]["build_seconds"]["mean"] for r in fronts], "Actual build: mean seconds")
        bar_values(axes[1,2], [100*audits[r["run"]]["window_tools"].get("success_to_error",0)/
                                  audits[r["run"]]["window_tools"]["calls"] for r in fronts],
                   "Recorded success → replay tool error (%)", digits=2)
        finish(fig,"01_frontend_controls",pdf,
               "All measured: 8 TP4 backends, total cc256, sticky routing, 90-minute windows; one run per configuration.\n"
               "2+8 keeps 4 build slots in total. CPU is averaged per frontend, not summed; tool-error denominator is all window tool calls.")

        fig, axes = plt.subplots(1,3,figsize=(13,5.5), gridspec_kw={"width_ratios":[1.3,1,1]})
        fig.suptitle("Where frontend time goes: slot waiting, build work, and native tools", fontsize=15)
        bottom = np.zeros(3)
        components = [("setup","Setup",ORANGE),("llm_only","LLM",BLUE),("tool_only","Tools",GREEN),
                      ("other_and_cleanup","Cleanup / other",GREY)]
        for key,name,color in components:
            ys = [r["execution_seconds_mean"][key] for r in fronts]
            axes[0].bar(range(3),ys,bottom=bottom,label=name,color=color,width=.6)
            bottom += ys
        axes[0].set_xticks(range(3), labels)
        axes[0].set_ylabel("Mean task lifecycle (seconds)")
        axes[0].legend(fontsize=8,ncol=2,loc="upper left")
        axes[0].set_ylim(0,max(bottom)*1.3)
        for i,y in enumerate(bottom):
            axes[0].text(i,y+12,f"{y:.0f}",ha="center")
        for ax,key,title in zip(axes[1:],("slot_wait_seconds","build_seconds"),("Build-slot waiting","Actual environment build")):
            data = [[float(s[key]) for s in setups if s["run"] == r["run"]] for r in fronts]
            boxes = ax.boxplot(data,whis=(5,95),showfliers=False,showmeans=True,patch_artist=True,
                               meanprops={"marker":"D","markerfacecolor":"black","markeredgecolor":"black","markersize":4})
            for patch,c in zip(boxes["boxes"],colors):
                patch.set_facecolor(c);patch.set_alpha(.65)
            ax.set_xticks([1,2,3],labels)
            ax.set_ylabel("Seconds")
            ax.set_title(title)
            ax.set_ylim(bottom=0)
        finish(fig,"02_frontend_time_breakdown",pdf,
               "Complete measurement-admission cohorts, including natural drain. Boxes: p25–p75; whiskers: p5–p95; diamonds: means.\n"
               "LLM time is task wall time, not exclusive GPU compute. Different closed-loop runs need not have identical task mixtures.")

        fig, axes = plt.subplots(2,2,figsize=(12,7.2))
        fig.suptitle("Frontend controls: full-window trajectories",fontsize=15)
        for r,c,label in zip(fronts,colors,legend_labels):
            for ax,key in zip([axes[0,0],axes[0,1],axes[1,0]],
                              ["output_tokens_per_second","frontend_cpu_percent","running_per_replica_mean"]):
                ax.plot(*smooth(r,key),color=c,label=label)
            ss = [s for s in setups if s["run"] == r["run"]]
            x,y=[],[]
            for low in range(0,90,5):
                v=[float(s["slot_wait_seconds"]) for s in ss if low <= float(s["admission_minute"]) < low+5]
                if v: x.append(low+2.5);y.append(np.mean(v))
            axes[1,1].plot(x,y,color=c,label=label)
        for ax,label in zip(axes.flat,("Output tokens / second","CPU busy per frontend (%)",
                                       "Running requests per backend","Build-slot wait (mean seconds)")):
            ax.set_ylabel(label);ax.set_xlabel("Measurement minute");ax.set_ylim(bottom=0)
        axes[0,0].legend(fontsize=8,loc="lower left")
        axes[0,1].set_ylim(0,105)
        finish(fig,"03_frontend_timeseries",pdf,
               "Nonoverlapping 5-minute averages. Slot waiting is grouped by task admission time.\n"
               "Throughput steadiness does not imply that frontend waiting or task latency has stabilized.")

        fig, axes = plt.subplots(2,3,figsize=(13,7.4))
        fig.suptitle("Routing: measured 1+4 comparisons, with the new OpenClaw 1+8 RR window",fontsize=14)
        for row,w in enumerate(("openclaw","minisweagent")):
            pts=[point(w,g,n) for n,g in ((4,"routing_rr"),(4,"routing_cache"),(8,"routing_rr"),(8,"routing_cache"))]
            for ax,key,ylabel,scale in zip(axes[row],("output_tokens_s","tasks_per_minute","prefix_reuse"),
                         ("Output tokens / second","Completed tasks / minute","Actual cached / prompt tokens (%)"),(1,1,100)):
                valid=[r[key]*scale for r in pts if r]
                ax.set_ylim(0, max(valid)*1.26)
                for i,r in enumerate(pts):
                    if r:
                        y=r[key]*scale
                        b=ax.bar(i,y,color=ORANGE if r["group"] == "routing_rr" else GREEN,width=.58)[0]
                        if r["status"] != "complete_steady": b.set_hatch("//");b.set_edgecolor("#666666")
                        ax.annotate(f"{y:,.1f}",(i,y),xytext=(0,4),textcoords="offset points",ha="center",fontsize=9)
                    else:
                        ax.text(i,.1,"No data",rotation=90,transform=ax.get_xaxis_transform(),ha="center",color=GREY)
                ax.set_xticks(range(4),["RR\n1+4","CA\n1+4","RR\n1+8","CA\n1+8"])
                ax.set_xlim(-.6,3.6)
                ax.set_ylabel(ylabel)
            axes[row,0].set_title("OpenClaw" if row == 0 else "mini-SWE-agent")
        finish(fig,"04_routing_comparison",pdf,
               "Hatched: OpenClaw RR N4 nonsteady / N8 warmup failures; mini-SWE RR N4 recovered window with drain timeout.\n"
               "Blank positions are missing runs, not zero throughput. P2 is excluded. CA = cache-aware; single run per point.")

        fig, axes = plt.subplots(3,3,figsize=(13,9))
        fig.suptitle("OpenClaw: similar output per backend can hide a concentrated queue",fontsize=15)
        for row,(g,n,label) in enumerate((("routing_rr",4,"RR 1+4"),("routing_cache",4,"Cache-aware 1+4"),("routing_rr",8,"RR 1+8"))):
            r=point("openclaw",g,n)
            rs=sorted([x for x in replicas if x["run"] == r["run"]],key=lambda v:v["replica"])
            x=np.arange(n)
            axes[row,0].bar(x,[v["output_tokens_s"] for v in rs],color=BLUE)
            axes[row,1].bar(x,[100*v["prefix_reuse"] for v in rs],color=GREEN)
            axes[row,2].bar(x,[v["running_mean"] for v in rs],color=BLUE,label="Running")
            axes[row,2].bar(x,[v["waiting_mean"] for v in rs],bottom=[v["running_mean"] for v in rs],color=ORANGE,label="Waiting")
            for ax in axes[row]:
                ax.set_xticks(x,[v["replica"] for v in rs],fontsize=8)
            axes[row,0].set_ylabel(label+"\nOutput tokens / second")
            axes[row,1].set_ylabel("Actual prefix reuse (%)");axes[row,1].set_ylim(0,105)
            axes[row,2].set_ylabel("Mean requests");axes[row,2].set_ylim(0,125)
        axes[0,2].legend(fontsize=8)
        finish(fig,"05_openclaw_replica_hotspot",pdf,
               "Full-window per-backend means. Prefix reuse is token weighted. Queue and running counts are independent backend samples.\n"
               "RR N4 is nonsteady; RR N8 has two warmup failures and is not declared steady. These are window averages.")

        fig, axes = plt.subplots(2,3,figsize=(13,7))
        fig.suptitle("OpenClaw RR 4→8 backends: scaling and persistent imbalance",fontsize=15)
        for n,c in ((4,ORANGE),(8,BLUE)):
            r=point("openclaw","routing_rr",n)
            keys=["output_tokens_per_second","actual_prefix_reuse","waiting_per_replica_mean",
                  "running_per_replica_mean","tasks_per_minute","frontend_cpu_percent"]
            for ax,key,scale in zip(axes.flat,keys,[1/n,100,1,1,1/n,1]):
                ax.plot(*smooth(r,key,scale),label=f"1+{n} RR",color=c)
        names=["Output tokens / second / backend","Actual prefix reuse (%)","Waiting requests / backend",
               "Running requests / backend","Completed tasks / minute / backend","Frontend CPU busy (%)"]
        for ax,label in zip(axes.flat,names):
            ax.set_ylabel(label);ax.set_xlabel("Measurement minute");ax.set_ylim(bottom=0)
        axes[0,0].legend()
        finish(fig,"06_openclaw_rr_scaling",pdf,
               "Nonoverlapping 5-minute averages; 60-minute measurements, cc16 per logical worker.\n"
               "Normalized rates show scale-out efficiency, not recovery to cache-aware/sticky capacity. No steady-state capacity claim.")

        fig, axes = plt.subplots(2,2,figsize=(11.5,7.3))
        fig.suptitle("Frontend/backend scaling: updated measured comparisons",fontsize=15)
        for row,w in enumerate(("openclaw","minisweagent")):
            sticky=[point(w,"balanced",1)] + [point(w,"fixed_sticky",n) for n in (2,4,8)]
            historical=[r for r in results if r["workload"] == w and r["group"] == "balanced" and r["backends"] <=8]
            historical.sort(key=lambda r:r["backends"])
            for col,key in enumerate(("output_tokens_s","frontend_cpu_mean_percent")):
                ax=axes[row,col]
                # NaN deliberately breaks the missing mini-SWE 1+4 segment.
                ax.plot([1,2,4,8],[r[key] if r else np.nan for r in sticky],"o-",color=BLUE,label="1 frontend, sticky; N1 historical")
                ax.plot([r["backends"] for r in historical],[r[key] for r in historical],"s--",color=GREY,label="F=B historical policy")
                if w == "minisweagent":
                    for r,c,marker,label in zip(fronts[1:],colors[1:],["D","^"],legend_labels[1:]):
                        ax.scatter([8],[r[key]],color=c,marker=marker,s=65,label=label,zorder=4)
                ax.set_xticks([1,2,4,8]);ax.set_xlim(.6,8.5);ax.set_ylim(bottom=0)
                ax.set_xlabel("Inference backend nodes (TP4 each)")
                ax.set_ylabel(("OpenClaw" if row == 0 else "mini-SWE")+"\n"+("Output tokens / second" if col == 0 else "CPU busy per frontend (%)"))
            axes[row,0].legend(fontsize=8)
        finish(fig,"07_frontend_scaling_updated",pdf,
               "Historical F=B runs use an older HTTP/build policy: capacity references, not controlled frontend-only comparisons.\n"
               "Mini-SWE 1+4 sticky is missing; no interpolation. New mini-SWE points keep 8 backends and total cc256.")

    print("Wrote 7 PNG/PDF figures and results.pdf; P2 excluded.")
