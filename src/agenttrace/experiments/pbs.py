"""One task-concurrency point, selected workloads, inside a two-node PBS job."""

from __future__ import annotations

import argparse
import copy
import json
import os
import socket
from pathlib import Path

from agenttrace.cli import _run_interruptible
from agenttrace.experiments.scale import precache, run


def job_config(base: dict, replay_node: str, inference_node: str, concurrency: int) -> dict:
    if concurrency < 1:
        raise ValueError("task concurrency must be positive")
    if replay_node == inference_node:
        raise ValueError("replay and inference require separate nodes")
    config = copy.deepcopy(base)
    config.update(replay_node=replay_node, inference_node=inference_node,
                  concurrency=[concurrency])
    return config


async def run_job(args) -> None:
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    config = job_config(json.loads(args.config.read_text()),
                        socket.gethostname().split(".")[0],
                        args.inference_node, args.concurrency)
    if args.workloads:
        config["workloads"] = args.workloads
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"PBS job {args.job_id}: task cc={args.concurrency}, "
          f"replay={config['replay_node']}, inference={config['inference_node']}", flush=True)
    # New allocations have cold node-local OCI caches. Populate them serially
    # before measurement; this executes setup/close, not LLM or tool nodes.
    for workload in config["workloads"]:
        await precache(args.pools / workload / "traces.txt",
                       root / f"precache-{workload}", Path(config["profiles"][workload]))
    await run(argparse.Namespace(config=config_path, pools=args.pools,
                                output=root / "experiment", job_id=args.job_id,
                                smoke=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("examples/scaling/single-backend.json"))
    parser.add_argument("--pools", type=Path, default=Path("runs/scaling/preflight-20260906-1"))
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--workloads", nargs="+", choices=["openclaw", "minisweagent"])
    parser.add_argument("--inference-node", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-id", default=os.environ.get("PBS_JOBID"))
    args = parser.parse_args()
    if not args.job_id:
        parser.error("run inside PBS or supply --job-id")
    _run_interruptible(run_job(args))


if __name__ == "__main__":
    main()
