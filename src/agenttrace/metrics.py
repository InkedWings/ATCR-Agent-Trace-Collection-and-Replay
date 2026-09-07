"""Periodic node-level hardware and raw vLLM Prometheus samples."""

from __future__ import annotations

import asyncio
import csv
import json
import math
import re
import shutil
import socket
import time
from pathlib import Path
from typing import Any

import httpx


def prometheus_values(text: str) -> dict[str, float]:
    """Keep labels in keys so independent engine series are never conflated."""
    values = {}
    for line in text.splitlines():
        match = re.fullmatch(r'(vllm:[a-zA-Z0-9_:]+(?:\{.*\})?)\s+(\S+)(?:\s+\S+)?', line)
        if match:
            value = float(match[2])
            if math.isfinite(value):
                values[match[1]] = value
    return values


def summarize_metrics(path: Path, *, started_unix: float | None = None,
                      ended_unix: float | None = None) -> dict[str, Any]:
    samples = 0
    errors = 0
    gpu_statuses: dict[str, int] = {}
    hardware: dict[str, list[float]] = {}
    first: dict[str, float] | None = None
    last: dict[str, float] = {}
    first_time = last_time = None
    invalid_series: set[str] = set()
    vllm_samples = 0
    gauges: dict[str, list[float]] = {}
    first_hw = last_hw = None
    handle = path.open(encoding="utf-8")
    for line in handle:
        sample = json.loads(line)
        if started_unix is not None and sample["timestamp_unix"] < started_unix:
            continue
        if ended_unix is not None and sample["timestamp_unix"] >= ended_unix:
            continue
        samples += 1
        errors += int("hardware_error" in sample or "vllm_error" in sample)
        metrics = sample.get("hardware", {})
        if metrics:
            if first_hw is None:
                first_hw = sample
            last_hw = sample
        gpu_status = metrics.get("gpu_status", "unknown")
        gpu_statuses[gpu_status] = gpu_statuses.get(gpu_status, 0) + 1
        for key in ("cpu_busy_percent", "cpu_iowait_percent", "memory_available_bytes"):
            if metrics.get(key) is not None:
                hardware.setdefault(key, []).append(metrics[key])
        for gpu in metrics.get("gpus", []):
            for key in ("gpu_busy_percent", "memory_busy_percent", "memory_used_mib", "power_watts"):
                if gpu.get(key) is not None:
                    hardware.setdefault(f"gpu.{gpu['index']}.{key}", []).append(gpu[key])
        if "vllm_prometheus" in sample:
            values = prometheus_values(sample["vllm_prometheus"])
            for key, value in values.items():
                if key.split("{", 1)[0] in ("vllm:num_requests_running", "vllm:num_requests_waiting",
                        "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"):
                    gauges.setdefault(key, []).append(value)
            if first is None:
                first, first_time = values, sample["timestamp_unix"]
            else:
                invalid_series.update(last.keys() ^ values.keys())
                invalid_series.update(key for key in last.keys() & values.keys() if values[key] < last[key])
            last, last_time = values, sample["timestamp_unix"]
            vllm_samples += 1
    handle.close()
    summary: dict[str, Any] = {"samples": samples, "error_samples": errors, "gpu_status_counts": gpu_statuses,
        "hardware_sample_statistics": {key: {"mean": sum(values) / len(values), "max": max(values)}
                                       for key, values in hardware.items()}}
    summary["vllm_gauge_sample_statistics"] = {key: {"mean": sum(v) / len(v), "max": max(v)}
        for key, v in gauges.items()}
    if first_hw is not None and last_hw is not None:
        seconds = last_hw["timestamp_unix"] - first_hw["timestamp_unix"]
        io = {}
        for kind, identity, counters in (("disks", "device", ("read_bytes", "write_bytes", "io_milliseconds")),
                                         ("network", "interface", ("rx_bytes", "tx_bytes"))):
            old = {item[identity]: item for item in first_hw["hardware"].get(kind, [])}
            for item in last_hw["hardware"].get(kind, []):
                if item[identity] in old:
                    for key in counters:
                        delta = item[key] - old[item[identity]][key]
                        io[f"{kind}.{item[identity]}.{key}"] = {
                            "delta": delta if delta >= 0 else None,
                            "per_second": delta / seconds if delta >= 0 and seconds > 0 else None}
        summary["node_io"] = {"sample_seconds": seconds, "counters": io,
            "scope": "per-device/interface node counters; block devices are not Lustre filesystem traffic"}
    if vllm_samples >= 2 and first is not None:
        counter_names = ("vllm:prompt_tokens_total", "vllm:generation_tokens_total",
            "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total")
        delta = {}
        resets = []
        for key in sorted(first.keys() | last.keys()):
            name = key.split("{", 1)[0]
            if name in counter_names or name.endswith(("_sum", "_count", "_bucket")):
                if key in invalid_series:
                    resets.append(key)
                else:
                    delta[key] = last[key] - first[key]
        queries = sum(value for key, value in delta.items() if key.split("{", 1)[0] == counter_names[2])
        hits = sum(value for key, value in delta.items() if key.split("{", 1)[0] == counter_names[3])
        cache_reset = any(key.split("{", 1)[0] in counter_names[2:] for key in resets)
        summary["vllm"] = {"scope": "endpoint_process_totals_over_sample_window",
            "first_sample_unix": first_time, "last_sample_unix": last_time,
            "counter_deltas": delta, "reset_or_missing_series": resets,
            "prefix_cache_hit_ratio": hits / queries if queries and not cache_reset else None}
    return summary


class HardwareSampler:
    def __init__(self, proc_root: Path = Path("/proc")) -> None:
        self.proc_root = proc_root
        self.previous: tuple[float, list[int]] | None = None
        self.gpu_executable = shutil.which("nvidia-smi")

    def cpu_memory_io(self) -> dict[str, Any]:
        now = time.monotonic()
        ticks = list(map(int, (self.proc_root / "stat").read_text().splitlines()[0].split()[1:9]))
        busy = iowait = None
        if self.previous:
            delta = [a - b for a, b in zip(ticks, self.previous[1])]
            total = sum(delta)
            if total > 0:
                busy = 100 * (total - delta[3] - delta[4]) / total
                iowait = 100 * delta[4] / total
        self.previous = (now, ticks)
        mem = {line.split()[0].rstrip(":"): int(line.split()[1]) * 1024
               for line in (self.proc_root / "meminfo").read_text().splitlines()}
        disks = []
        for line in (self.proc_root / "diskstats").read_text().splitlines():
            fields = line.split()
            if len(fields) >= 14:
                disks.append({"device": fields[2], "read_bytes": int(fields[5]) * 512,
                    "write_bytes": int(fields[9]) * 512, "io_milliseconds": int(fields[12])})
        network = []
        for line in (self.proc_root / "net/dev").read_text().splitlines()[2:]:
            name, values = line.split(":", 1)
            counters = values.split()
            network.append({"interface": name.strip(), "rx_bytes": int(counters[0]),
                "tx_bytes": int(counters[8])})
        return {"cpu_busy_percent": busy, "cpu_iowait_percent": iowait,
            "memory_total_bytes": mem["MemTotal"], "memory_available_bytes": mem["MemAvailable"],
            "disks": disks, "network": network}

    async def sample(self) -> dict[str, Any]:
        result = self.cpu_memory_io()
        result["gpus"] = []
        if not self.gpu_executable:
            result["gpu_status"] = "nvidia-smi unavailable"
            return result
        process = await asyncio.create_subprocess_exec(
            self.gpu_executable,
            "--query-gpu=index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), 5)
            if process.returncode:
                result["gpu_status"] = f"nvidia-smi exit {process.returncode}"
            else:
                for row in csv.reader(stdout.decode().splitlines()):
                    values = [value.strip() for value in row]
                    gpu: dict[str, Any] = {"index": int(values[0]), "uuid": values[1]}
                    for key, value in zip(("gpu_busy_percent", "memory_busy_percent",
                            "memory_used_mib", "memory_total_mib", "power_watts"), values[2:]):
                        try:
                            gpu[key] = float(value)
                        except ValueError:
                            gpu[key] = None
                    result["gpus"].append(gpu)
                result["gpu_status"] = "ok"
        except TimeoutError:
            result["gpu_status"] = "nvidia-smi timeout"
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        return result


async def monitor(
    output: str | Path, *, interval: float = 1, metrics_url: str | None = None,
    stop: asyncio.Event | None = None, duration: float | None = None,
    label: str = "node", ready: asyncio.Event | None = None,
) -> None:
    """Sample this node; remote inference hardware needs a sampler on that node."""
    if interval <= 0 or (duration is not None and duration <= 0):
        raise ValueError("sample interval and duration must be positive")
    stop = stop or asyncio.Event()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    sampler = HardwareSampler()
    started = time.monotonic()
    async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
        with path.open("x", encoding="utf-8", buffering=1) as handle:
            while True:
                sample_started = time.monotonic()
                sample: dict[str, Any] = {"timestamp_unix": time.time(),
                    "elapsed_seconds": sample_started - started,
                    "hostname": socket.gethostname(), "label": label, "scope": "whole_node"}
                try:
                    sample["hardware"] = await sampler.sample()
                except (OSError, ValueError, KeyError) as error:
                    sample["hardware_error"] = type(error).__name__
                if metrics_url:
                    try:
                        response = await client.get(metrics_url)
                        response.raise_for_status()
                        sample["vllm_prometheus"] = response.text
                    except httpx.HTTPError as error:
                        # Keep holes explicit rather than reporting zeros.
                        sample["vllm_error"] = type(error).__name__
                sample["collection_seconds"] = time.monotonic() - sample_started
                handle.write(json.dumps(sample) + "\n")
                if ready:
                    ready.set()
                if stop.is_set() or (duration and time.monotonic() - started >= duration):
                    break
                try:
                    await asyncio.wait_for(stop.wait(), max(0.001, interval - sample["collection_seconds"]))
                except TimeoutError:
                    pass
                # Capture a final boundary sample after stop is requested.
