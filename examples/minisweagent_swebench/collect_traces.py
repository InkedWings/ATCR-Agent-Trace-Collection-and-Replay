#!/usr/bin/env python3
"""Collect AgentTrace records from mini-SWE-agent on SWE-bench instances."""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import re
import signal
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset
from minisweagent import __version__ as minisweagent_version
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import get_config_from_spec
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.run.benchmarks.swebench import get_sb_environment

from agenttrace.adapters.minisweagent import (
    write_minisweagent_trace,
)
from agenttrace.capture.openai import CaptureProxy


DEFAULT_UPSTREAM = (
    "https://inference-api.alcf.anl.gov/resource_server/minerva/api/v1"
)
DEFAULT_MODEL = "openai/nemotron-3-ultra"


class TaskDeadline(TimeoutError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_component(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    if not result:
        raise ValueError("empty path component")
    return result[:160]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def public_case(instance: dict[str, Any]) -> dict[str, Any]:
    return {
        "instance_id": instance["instance_id"],
        "repo": instance["repo"],
        "base_commit": instance["base_commit"],
        "problem_statement": instance["problem_statement"],
        "version": instance.get("version", ""),
        "created_at": instance.get("created_at", ""),
    }


def run_agent(
    *,
    instance: dict[str, Any],
    trajectory_path: Path,
    proxy_url: str,
    trace_id: str,
    step_limit: int,
    cost_limit: float,
    task_timeout: int,
    shell_timeout: int,
    max_output_tokens: int,
) -> dict[str, Any]:
    config = get_config_from_spec("swebench.yaml")
    agent_config = copy.deepcopy(config["agent"])
    agent_config.update(
        {
            "step_limit": step_limit,
            "cost_limit": cost_limit,
            "wall_time_limit_seconds": task_timeout,
            "output_path": trajectory_path,
        }
    )

    model_config = copy.deepcopy(config["model"])
    model_config.update(
        {
            "model_name": DEFAULT_MODEL,
            "cost_tracking": "ignore_errors",
            "model_kwargs": {
                "api_base": proxy_url,
                "drop_params": True,
                "extra_headers": {"X-Trace-Id": trace_id},
                "max_tokens": max_output_tokens,
                "num_retries": 0,
                "parallel_tool_calls": True,
                "temperature": 0.2,
                "top_p": 0.95,
                "extra_body": {
                    "chat_template_kwargs": {
                        "enable_thinking": True,
                        "force_nonempty_content": True,
                    }
                },
            },
        }
    )
    environment_config = copy.deepcopy(config["environment"])
    environment_config.update(
        {
            "environment_class": "singularity",
            "timeout": shell_timeout,
            "forward_env": [
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "http_proxy",
                "https_proxy",
                "NO_PROXY",
                "no_proxy",
            ],
        }
    )
    singularity_executable = os.environ.get("MSWEA_SINGULARITY_EXECUTABLE")
    if singularity_executable:
        environment_config["executable"] = singularity_executable
    config["environment"] = environment_config
    environment = get_sb_environment(config, instance)
    interrupted = False
    try:
        agent = DefaultAgent(LitellmModel(**model_config), environment, **agent_config)
        return agent.run(instance["problem_statement"])
    except KeyboardInterrupt:
        interrupted = True
        raise
    finally:
        # Removing a writable SWE-bench sandbox can take long enough to make
        # Ctrl-C appear ignored. Leave node-local scratch behind on an explicit
        # interrupt; the allocation cleanup will remove it.
        if not interrupted:
            cleanup = getattr(environment, "cleanup", None)
            if cleanup is not None:
                cleanup()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--run-root", type=Path, default=Path("runs/minisweagent-swebench"))
    parser.add_argument("--run-id")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-tasks", type=int, default=23)
    parser.add_argument("--step-limit", type=int, default=250)
    parser.add_argument("--cost-limit", type=float, default=0)
    parser.add_argument("--task-timeout", type=int, default=0)
    parser.add_argument("--run-timeout", type=int, default=0)
    parser.add_argument("--shell-timeout", type=int, default=60)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--llm-upstream", default=DEFAULT_UPSTREAM)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must contain the current ALCF access token")
    if args.start_index < 0 or args.max_tasks < 1:
        parser.error("start-index must be >= 0 and max-tasks must be >= 1")
    if args.step_limit < 1 or args.cost_limit < 0:
        parser.error("step-limit must be >= 1 and cost-limit must be >= 0")
    if args.task_timeout < 0 or args.run_timeout < 0:
        parser.error("task-timeout and run-timeout must be >= 0")

    dataset = load_dataset(args.dataset, split=args.split)
    instances = [dict(row) for row in dataset]
    selected = instances[args.start_index : args.start_index + args.max_tasks]
    if not selected:
        raise ValueError("task selection is empty")

    default_id = datetime.now(timezone.utc).strftime("dev-%Y%m%dT%H%M%SZ")
    run_id = safe_component(args.run_id or default_id)
    run_dir = args.run_root.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    tasks_dir = run_dir / "tasks"
    tasks_dir.mkdir()
    write_jsonl(run_dir / "manifest.jsonl", [public_case(row) for row in selected])
    write_json(
        run_dir / "metadata.json",
        {
            "run_id": run_id,
            "started_at": utc_now(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "framework": "mini-swe-agent",
            "framework_version": minisweagent_version,
            "workload": f"SWE-bench_Lite/{args.split}",
            "dataset": args.dataset,
            "split": args.split,
            "model": DEFAULT_MODEL,
            "environment": "SingularityEnvironment",
            "start_index": args.start_index,
            "max_tasks": args.max_tasks,
            "step_limit": args.step_limit,
            "cost_limit": args.cost_limit,
            "task_timeout_seconds": args.task_timeout,
            "run_timeout_seconds": args.run_timeout,
            "credentials_persisted": False,
        },
    )

    capture_path = run_dir / "llm_calls.jsonl"
    proxy = CaptureProxy(args.llm_upstream, capture_path)
    proxy.start()
    started = time.monotonic()
    statuses: list[dict[str, Any]] = []

    def deadline_handler(signum: int, frame: object) -> None:
        raise TaskDeadline(f"task exceeded {args.task_timeout} seconds")

    def interrupt_handler(signum: int, frame: object) -> None:
        print(
            "\nStopping collection; the current partial trajectory is preserved.",
            file=sys.stderr,
            flush=True,
        )
        raise KeyboardInterrupt

    previous_alarm_handler = signal.signal(signal.SIGALRM, deadline_handler)
    previous_interrupt_handler = signal.signal(signal.SIGINT, interrupt_handler)
    interrupted = False
    try:
        for offset, instance in enumerate(selected):
            if args.run_timeout and time.monotonic() - started >= args.run_timeout:
                break
            index = args.start_index + offset
            instance_id = str(instance["instance_id"])
            trace_id = safe_component(instance_id)
            task_dir = tasks_dir / f"{index + 1:03d}_{trace_id}"
            task_dir.mkdir()
            trajectory_path = task_dir / "trajectory.json"
            status: dict[str, Any] = {
                "index": index,
                "instance_id": instance_id,
                "started_at": utc_now(),
                "status": "running",
            }
            task_started = time.monotonic()
            print(f"[{index + 1}/{len(instances)}] {instance_id}", flush=True)
            try:
                signal.alarm(args.task_timeout)
                result = run_agent(
                    instance=instance,
                    trajectory_path=trajectory_path,
                    proxy_url=proxy.base_url,
                    trace_id=trace_id,
                    step_limit=args.step_limit,
                    cost_limit=args.cost_limit,
                    task_timeout=args.task_timeout,
                    shell_timeout=args.shell_timeout,
                    max_output_tokens=args.max_output_tokens,
                )
                status["agent_exit_status"] = result.get("exit_status", "")
                status["status"] = "agent_complete"
            except Exception as error:
                status["status"] = "agent_error"
                status["error"] = f"{type(error).__name__}: {error}"
            finally:
                signal.alarm(0)

            try:
                proxy.wait_idle(trace_id, timeout=30)
                if trajectory_path.is_file():
                    trace = write_minisweagent_trace(
                        task_dir / "trace.json",
                        trace_id=trace_id,
                        trajectory_path=trajectory_path,
                        capture_path=capture_path,
                        workload=f"SWE-bench_Lite/{args.split}",
                        framework_version=minisweagent_version,
                        captured_workspace_root="/testbed",
                        source_repository=str(instance["repo"]),
                        base_commit=str(instance["base_commit"]),
                        instance_id=instance_id,
                    )
                    status["llm_nodes"] = sum(
                        node["type"] == "llm" for node in trace["nodes"]
                    )
                    status["tool_nodes"] = sum(
                        node["type"] == "tool" for node in trace["nodes"]
                    )
                    status["status"] = "trace_complete"
            except Exception as error:
                status["trace_error"] = f"{type(error).__name__}: {error}"

            status["elapsed_seconds"] = round(time.monotonic() - task_started, 3)
            status["finished_at"] = utc_now()
            write_json(task_dir / "status.json", status)
            statuses.append(status)
            write_jsonl(run_dir / "statuses.jsonl", statuses)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_alarm_handler)
        signal.signal(signal.SIGINT, previous_interrupt_handler)
        proxy.close()

    complete_count = sum(row["status"] == "trace_complete" for row in statuses)
    write_json(
        run_dir / "summary.json",
        {
            "run_id": run_id,
            "updated_at": utc_now(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "tasks_selected": len(selected),
            "tasks_attempted": len(statuses),
            "traces_complete": complete_count,
            "complete": not interrupted and complete_count == len(selected),
            "interrupted": interrupted,
        },
    )
    print(f"run_dir={run_dir} traces_complete={complete_count}/{len(statuses)}")
    if interrupted:
        return 130
    return 0 if complete_count == len(statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
