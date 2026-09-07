"""Optional exploratory plots; all numeric results also remain in summary.csv."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
args = parser.parse_args()
with (args.run / "summary.csv").open() as handle:
    rows = [row for row in csv.DictReader(handle) if row["valid"] == "True"]
for key, label in (("task_per_s", "Completed tasks / s"),
                   ("backend_active_output_token_per_s", "Backend active output tokens / s"),
                   ("task_latency_p95_s", "Task lifecycle p95 (s)")):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for workload in ("openclaw", "minisweagent"):
        points = sorted((row for row in rows if row["point"].startswith(workload) and row[key]),
                        key=lambda row: int(row["task_cc"]))
        if points:
            ax.plot([int(row["task_cc"]) for row in points], [float(row[key]) for row in points],
                    marker="o", label=workload)
    ax.set(xlabel="Concurrent task lifecycles", ylabel=label, xticks=[1, 2, 4, 8],
           title="Single backend · exploratory, one repetition")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.run / f"{key}.png", dpi=160)
    plt.close(fig)
