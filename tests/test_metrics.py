import asyncio
import json

from agenttrace.metrics import HardwareSampler, monitor, prometheus_values, summarize_metrics


def test_cpu_accounting_and_unavailable_gpu(tmp_path, monkeypatch):
    (tmp_path / "stat").write_text("cpu 100 0 0 100 0 0 0 0\n")
    (tmp_path / "meminfo").write_text("MemTotal: 100 kB\nMemAvailable: 60 kB\n")
    (tmp_path / "diskstats").write_text("8 0 disk 1 0 2 0 1 0 3 0 0 9 0\n")
    (tmp_path / "net").mkdir()
    (tmp_path / "net/dev").write_text("header\nheader\n eth0: 10 0 0 0 0 0 0 0 20\n")
    sampler = HardwareSampler(tmp_path)
    sampler.gpu_executable = None
    assert sampler.cpu_memory_io()["cpu_busy_percent"] is None
    (tmp_path / "stat").write_text("cpu 130 0 0 160 10 0 0 0\n")
    sample = asyncio.run(sampler.sample())
    assert sample["cpu_busy_percent"] == 30
    assert sample["cpu_iowait_percent"] == 10
    assert sample["memory_total_bytes"] == 102400
    assert sample["gpus"] == [] and "unavailable" in sample["gpu_status"]


def test_interval_prefix_ratio_and_counter_reset(tmp_path):
    path = tmp_path / "metrics.jsonl"
    def row(t, hits, queries):
        return json.dumps({"timestamp_unix": t, "vllm_prometheus":
            f'vllm:prefix_cache_hits_total{{model_name="a"}} {hits}\n'
            f'vllm:prefix_cache_queries_total{{model_name="a"}} {queries}\n'}) + "\n"
    path.write_text(row(1, 900, 1000) + row(2, 910, 1100))
    assert summarize_metrics(path)["vllm"]["prefix_cache_hit_ratio"] == .1
    path.write_text(row(1, 900, 1000) + row(2, 0, 0) + row(3, 1000, 1200))
    assert summarize_metrics(path)["vllm"]["prefix_cache_hit_ratio"] is None
    assert len(summarize_metrics(path)["vllm"]["reset_or_missing_series"]) == 2
    assert prometheus_values('# HELP hi\nvllm:a{model_name="a b"} 4\n') == {'vllm:a{model_name="a b"}': 4}


def test_single_busy_core_is_distinct_from_node_saturation(tmp_path):
    (tmp_path / "stat").write_text("cpu 100 0 0 100 0 0 0 0\n"
                                 "cpu0 50 0 0 50 0 0 0 0\ncpu1 50 0 0 50 0 0 0 0\n")
    (tmp_path / "meminfo").write_text("MemTotal: 100 kB\nMemAvailable: 60 kB\n")
    (tmp_path / "diskstats").write_text("")
    (tmp_path / "net").mkdir()
    (tmp_path / "net/dev").write_text("header\nheader\n")
    (tmp_path / "loadavg").write_text("1.5 1.0 0.5 4/100 123\n")
    sampler = HardwareSampler(tmp_path)
    sampler.cpu_memory_io()
    (tmp_path / "stat").write_text("cpu 130 0 0 160 10 0 0 0\n"
                                 "cpu0 80 0 0 50 10 0 0 0\ncpu1 50 0 0 110 0 0 0 0\n"
                                 "procs_running 4\nprocs_blocked 2\n")
    sample = sampler.cpu_memory_io()
    assert sample["cpu_busy_percent"] == 30
    assert sample["cpu_busy_percent_by_logical_cpu"] == {"cpu0": 75, "cpu1": 0}
    assert sample["cpu_max_busy_percent"] == 75
    assert sample["cpu_busy_logical_cpus"] == .75
    assert sample["cpu_logical_count"] == 2
    assert sample["cpu_runnable_processes"] == 4 and sample["cpu_blocked_processes"] == 2
    assert sample["load_average_1m"] == 1.5
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps({"timestamp_unix": 1, "hardware": sample}) + "\n")
    summary = summarize_metrics(path)["hardware_sample_statistics"]
    assert summary["cpu_max_busy_percent"]["mean"] == 75
    assert summary["cpu_busy_percent"]["mean"] == 30


def test_scrape_failure_is_not_a_zero(tmp_path):
    path = tmp_path / "samples.jsonl"
    asyncio.run(monitor(path, duration=.02, interval=.01, metrics_url="http://127.0.0.1:1/metrics"))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows and "vllm_error" in rows[0] and "vllm_prometheus" not in rows[0]


def test_gpu_busy_vram_and_missing_power(monkeypatch):
    class Process:
        returncode = 0

        async def communicate(self):
            return b"0, GPU-uuid, 80, 45, 12000, 40000, [Not Supported]\n", b""

    async def spawn(*args, **kwargs):
        assert "power.draw" in args[1]
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    sampler = HardwareSampler()
    sampler.gpu_executable = "fake-nvidia-smi"
    monkeypatch.setattr(sampler, "cpu_memory_io", lambda: {})
    sample = asyncio.run(sampler.sample())
    gpu = sample["gpus"][0]
    assert gpu["gpu_busy_percent"] == 80
    assert gpu["memory_busy_percent"] == 45
    assert gpu["memory_used_mib"] == 12000
    assert gpu["power_watts"] is None
