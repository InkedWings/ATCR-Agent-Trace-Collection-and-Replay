"""Bounded finite-list or steady-load replay with isolated child processes."""

from __future__ import annotations

import asyncio
import json
import os
import random
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from agenttrace.metrics import monitor, summarize_metrics
from agenttrace.loader import load_profile


def distribution(values: list[float]) -> dict[str, Any]:
    values = sorted(values)
    def percentile(p: float) -> float | None:
        if not values:
            return None
        offset = (len(values) - 1) * p
        low = int(offset)
        return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (offset - low)
    return {"count": len(values), "mean": sum(values) / len(values) if values else None,
        "p50": percentile(.50), "p95": percentile(.95), "p99": percentile(.99)}


async def stop_process(process: asyncio.subprocess.Process) -> None:
    """Signal only the session created for this replay, never the allocation."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        if sig == signal.SIGTERM:
            await asyncio.sleep(.5)
    await process.wait()


def summarize(root: Path, tasks: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    reports = [json.loads((root / row["report"]).read_text())
               for row in tasks if row["status"] == "completed"]
    nodes = [node for report in reports for node in report["nodes"]]
    llm = [node for node in nodes if node["type"] == "llm"]
    tool = [node for node in nodes if node["type"] == "tool"]
    active = peak = 0
    timeline = [(row[key], change) for row in tasks
                for key, change in (("started_unix", 1), ("ended_unix", -1)) if key in row]
    for _, change in sorted(timeline):
        active += change
        peak = max(peak, active)
    return {"completed": len(reports), "elapsed_seconds": elapsed,
        "peak_in_flight_tasks": peak,
        "latency_summary_scope": "completed_tasks",
        "task_throughput_per_second": len(reports) / elapsed if elapsed else None,
        "llm_calls_per_second": len(llm) / elapsed if elapsed else None,
        "tool_calls_per_second": len(tool) / elapsed if elapsed else None,
        "output_tokens_per_second": sum(node["actual_output_tokens"] for node in llm) / elapsed if elapsed else None,
        "task_lifecycle_seconds": distribution([row["lifecycle_seconds"] for row in tasks if row["status"] == "completed"]),
        "task_replay_seconds": distribution([report["replay_makespan_seconds"] for report in reports]),
        "task_setup_seconds": distribution([report["setup_seconds"] for report in reports]),
        "llm_latency_seconds": distribution([node["elapsed_seconds"] for node in llm]),
        "tool_latency_seconds": distribution([node["elapsed_seconds"] for node in tool]),
        "llm_ttft_seconds": distribution([node["ttft_seconds"] for node in llm if node.get("ttft_seconds") is not None]),
        "llm_tpot_estimate_seconds": distribution([node["tpot_estimate_seconds"] for node in llm if node.get("tpot_estimate_seconds") is not None]),
        "target_output_tokens": sum(node["target_output_tokens"] for node in llm),
        "actual_output_tokens": sum(node["actual_output_tokens"] for node in llm),
        "native_tool_errors": sum(node["native_error"] for node in tool)}


async def benchmark(
    traces: list[Path], *, profile: Path | None, output: Path,
    concurrency: int = 1, repeat: int | None = None, dry_run: bool = False,
    sample_interval: float = 1, metrics_url: str | None = None,
    backend_events: Path | None = None,
    warmup_seconds: float = 0, duration_seconds: float | None = None, seed: int = 42,
) -> dict[str, Any]:
    if duration_seconds is not None and repeat is not None:
        raise ValueError("--repeat and --duration-seconds are mutually exclusive")
    if warmup_seconds < 0 or (duration_seconds is not None and duration_seconds <= 0):
        raise ValueError("warmup must be nonnegative and duration must be positive")
    if warmup_seconds and duration_seconds is None:
        raise ValueError("warmup requires --duration-seconds")
    if concurrency < 1 or (repeat is not None and repeat < 1) or sample_interval <= 0 or not traces:
        raise ValueError("traces must be nonempty; concurrency, repeat and sample interval must be positive")
    paths = [path.resolve(strict=True) for path in traces]
    llm_config = load_profile(profile).get("llm_executor", {}).get("config", {}) if profile else {}
    root = output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "tasks").mkdir()
    local_tmp_root = None
    if os.environ.get("TMPDIR"):
        Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
        local_tmp_root = Path(tempfile.mkdtemp(prefix=f"agenttrace-{root.name}-", dir=os.environ["TMPDIR"]))
    rows = [{"task_id": f"{index:05d}", "trace_path": str(path), "status": "pending"}
            for index, path in enumerate(paths * (repeat or 1), 1)] if duration_seconds is None else []
    stop = asyncio.Event()
    ready = asyncio.Event()
    sampler = asyncio.create_task(monitor(root / "metrics.jsonl", interval=sample_interval,
        metrics_url=metrics_url, stop=stop, ready=ready, label="replay"))
    ready_task = asyncio.create_task(ready.wait())
    try:
        await asyncio.wait([sampler, ready_task], return_when=asyncio.FIRST_COMPLETED)
    except BaseException:
        sampler.cancel()
        ready_task.cancel()
        await asyncio.gather(sampler, ready_task, return_exceptions=True)
        raise
    if sampler.done():
        ready_task.cancel()
        await asyncio.gather(ready_task, return_exceptions=True)
        await sampler
    started_unix = time.time()
    started = time.perf_counter()
    measurement_start = started_unix + warmup_seconds
    measurement_end = measurement_start + duration_seconds if duration_seconds is not None else None
    deadline = started + warmup_seconds + duration_seconds if duration_seconds is not None else None
    manifest = {"concurrency": concurrency, "repeat": repeat, "dry_run": dry_run,
        "mode": "steady" if deadline is not None else "finite", "seed": seed,
        "trace_pool": [str(path) for path in paths],
        "warmup_seconds": warmup_seconds, "duration_seconds": duration_seconds,
        "measurement_started_unix": measurement_start, "measurement_ended_unix": measurement_end,
        "profile_path": str(profile.resolve()) if profile else None,
        "llm_replay": {key: llm_config[key] for key in ("model_override", "ignore_eos", "max_tokens_field", "trust_env") if key in llm_config},
        "cache_policy": "preserve_existing_endpoint_state",
        "started_unix": started_unix, "sample_interval": sample_interval,
        "backend_events_path": str(backend_events.resolve()) if backend_events else None,
        "local_tmp_root": str(local_tmp_root) if local_tmp_root else None,
        "concurrency_scope": "task_lifecycle_including_setup_and_cleanup", "tasks": rows}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    for row in rows:
        queue.put_nowait(row)
    failed = asyncio.Event()
    rng = random.Random(seed)
    cycle: list[Path] = []
    cycle_number = 0
    admissions = (root / "admissions.jsonl").open("x", buffering=1)

    def next_row() -> dict[str, Any] | None:
        nonlocal cycle_number
        # No await between deadline check and admission: workers cannot oversubscribe.
        if failed.is_set():
            return None
        if deadline is None:
            return queue.get_nowait() if not queue.empty() else None
        if time.perf_counter() >= deadline:
            return None
        if not cycle:
            cycle.extend(paths)
            rng.shuffle(cycle)
            cycle_number += 1
        row = {"task_id": f"{len(rows) + 1:05d}", "trace_path": str(cycle.pop(0)),
            "status": "pending", "cycle": cycle_number}
        rows.append(row)
        return row

    async def phases() -> None:
        if deadline is None:
            return
        await asyncio.sleep(max(0, started + warmup_seconds - time.perf_counter()))
        print(f"Measurement started: {duration_seconds:g}s, task cc={concurrency}", flush=True)
        await asyncio.sleep(max(0, deadline - time.perf_counter()))
        print("Measurement ended; admissions stopped, draining active tasks naturally", flush=True)

    async def worker() -> None:
        while (row := next_row()) is not None:
            task_root = root / "tasks" / row["task_id"]
            task_root.mkdir()
            row.update(status="running", started_unix=time.time(),
                admission_wait_seconds=time.perf_counter() - started,
                report=f"tasks/{row['task_id']}/report.json")
            row["admission_phase"] = "warmup" if row["started_unix"] < measurement_start else "measurement"
            admissions.write(json.dumps(row) + "\n")
            task_started = time.perf_counter()
            command = [sys.executable, "-m", "agenttrace.cli", "replay", row["trace_path"],
                "--run-dir", str(task_root / "run"), "--output", str(task_root / "report.json"),
                "--events", str(task_root / "events.jsonl")]
            if profile:
                command += ["--profile", str(profile.resolve())]
            if dry_run:
                command += ["--dry-run"]
            task_tmp = task_root / "process-tmp"
            task_tmp.mkdir()
            # Native container scratch stays local if the launcher supplied TMPDIR.
            if local_tmp_root:
                task_tmp = local_tmp_root / row["task_id"]
                task_tmp.mkdir(parents=True, exist_ok=False)
            env = dict(os.environ, TMPDIR=str(task_tmp))
            trace_path = Path(row["trace_path"])
            case_name = trace_path.parent.name if trace_path.name == "trace.json" else trace_path.stem
            print(f"[{row['task_id']}] started {case_name}", flush=True)
            process = None
            try:
                with (task_root / "console.log").open("x") as log:
                    process = await asyncio.create_subprocess_exec(*command,
                        stdout=log, stderr=asyncio.subprocess.STDOUT, env=env, start_new_session=True)
                    code = await process.wait()
                row["returncode"] = code
                row["status"] = "completed" if code == 0 else "failed"
                if code:
                    failed.set()
                    raise RuntimeError(f"task {row['task_id']} failed; see {task_root / 'console.log'}")
            except BaseException:
                if process is not None:
                    await stop_process(process)
                if row["status"] == "running":
                    row["status"] = "cancelled"
                raise
            finally:
                row["lifecycle_seconds"] = time.perf_counter() - task_started
                row["ended_unix"] = time.time()
                (task_root / "status.json").write_text(json.dumps(row, indent=2) + "\n")
                print(f"[{row['task_id']}] {row['status']} {row['lifecycle_seconds']:.2f}s", flush=True)

    workers = [asyncio.create_task(worker()) for _ in range(concurrency if deadline is not None else min(concurrency, len(rows)))]
    phase_task = asyncio.create_task(phases())
    status = "completed"
    try:
        await asyncio.gather(*workers)
    except BaseException as error:
        status = "cancelled" if isinstance(error, asyncio.CancelledError) else "failed"
        for worker_task in workers:
            worker_task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        raise
    finally:
        ended = time.perf_counter()
        phase_task.cancel()
        await asyncio.gather(phase_task, return_exceptions=True)
        admissions.close()
        stop.set()
        try:
            await sampler
        finally:
            report = {"status": status, "dry_run": dry_run, "started_unix": started_unix,
                "ended_unix": time.time(), "concurrency": concurrency, "tasks": rows,
                **summarize(root, rows, ended - started)}
            report["metrics"] = summarize_metrics(root / "metrics.jsonl")
            report["backend_throughput"] = {"status": "unavailable",
                "reason": "no backend request timestamps configured"}
            if backend_events and status == "completed" and not dry_run:
                from agenttrace.vllm_backend import summarize_backend

                try:
                    backend = summarize_backend(backend_events, started_unix=started_unix,
                        ended_unix=report["ended_unix"])
                except (OSError, ValueError) as error:
                    backend = {"status": "unavailable", "reason": str(error)}
                if backend["status"] == "measured" and (
                        backend["requests"] != report["llm_latency_seconds"]["count"]
                        or backend["output_tokens"] != report["actual_output_tokens"]):
                    backend = {"status": "unavailable", "reason": "backend/client count mismatch",
                        "backend_requests": backend.get("requests"),
                        "backend_output_tokens": backend.get("output_tokens")}
                report["backend_throughput"] = backend
            if measurement_end is not None:
                from agenttrace.replay.window import summarize_window

                report["measurement"] = summarize_window(root, rows, measurement_start, measurement_end)
                report["measurement"]["valid"] = status == "completed" and ended >= deadline
                report["measurement"]["metrics"] = summarize_metrics(root / "metrics.jsonl",
                    started_unix=measurement_start, ended_unix=measurement_end)
                if backend_events and status == "completed" and not dry_run:
                    from agenttrace.vllm_backend import summarize_backend

                    try:
                        report["measurement"]["backend_throughput"] = summarize_backend(backend_events,
                            started_unix=measurement_start, ended_unix=measurement_end, window=True)
                    except (OSError, ValueError) as error:
                        report["measurement"]["backend_throughput"] = {"status": "unavailable", "reason": str(error)}
            (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
