import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from agenttrace.cli import main
from agenttrace.vllm_backend import install_request_logger, summarize_backend, union_seconds


def request(start, first, end, tokens, **overrides):
    return {"hostname": "gpu-node", "finish_reason": "length",
        "scheduled_monotonic": start, "first_token_monotonic": first,
        "last_token_monotonic": end, "scheduled_unix": 1000 + start,
        "finished_unix": 1000 + end, "output_tokens": tokens, **overrides}


def write_requests(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_union_excludes_idle_and_does_not_sum_concurrent_requests(tmp_path):
    path = tmp_path / "backend.jsonl"
    write_requests(path, [request(10, 11, 14, 101), request(12, 13, 16, 101),
        request(100, 101, 102, 51)])
    r = summarize_backend(path)
    assert r["requests"] == 3 and r["output_tokens"] == 253
    assert r["active_seconds"] == 8  # [10,16] U [100,102], not 10 or 92 seconds
    assert r["active_output_tokens_per_second"] == 253 / 8
    assert r["decode_seconds"] == 6  # [11,16] U [101,102]
    assert r["decode_tokens"] == 250 and r["decode_tokens_per_second"] == 250 / 6
    assert union_seconds([(10, 20), (12, 14), (20, 21)]) == 11
    assert union_seconds([]) == 0


def test_window_and_single_token_request(tmp_path):
    path = tmp_path / "backend.jsonl"
    write_requests(path, [request(10, 11, 14, 101), request(100, 101, 101, 1)])
    r = summarize_backend(path, started_unix=1090, ended_unix=1102)
    assert r["requests"] == 1 and r["active_output_tokens_per_second"] == 1
    assert r["decode_tokens_per_second"] is None
    assert summarize_backend(path, started_unix=1200)["status"] == "unavailable"
    with pytest.raises(ValueError, match="crosses benchmark start"):
        summarize_backend(path, started_unix=1012)


@pytest.mark.parametrize("overrides", [
    {"scheduled_monotonic": 0}, {"first_token_monotonic": 20},
    {"last_token_monotonic": float("nan")}, {"output_tokens": 0},
    {"finish_reason": "abort"}, {"hostname": "other-node"},
])
def test_invalid_backend_data_is_not_reported_as_throughput(tmp_path, overrides):
    path = tmp_path / "backend.jsonl"
    write_requests(path, [request(10, 11, 14, 101), request(12, 13, 16, 101, **overrides)])
    with pytest.raises(ValueError):
        summarize_backend(path)


def test_request_hook_preserves_native_call_without_recording_content(tmp_path, monkeypatch):
    class Stats:
        def update_from_output(self, output, timestamp, prefilling, prompt_len, req_stats, *args):
            req_stats.num_generation_tokens += len(output.new_token_ids)

        def update_from_finished_request(self, *args):
            self.args = args
            return "native-result"

    for name in ("vllm", "vllm.v1", "vllm.v1.metrics", "vllm.v1.metrics.stats"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["vllm"].__version__ = "0.19.1"
    sys.modules["vllm.v1.metrics.stats"].IterationStats = Stats
    path = tmp_path / "backend.jsonl"
    handle = install_request_logger(path)
    stats = Stats()
    req = SimpleNamespace(queued_ts=9, scheduled_ts=10, first_token_ts=11, last_token_ts=14,
        num_generation_tokens=101, response_body="PRIVATE")
    try:
        assert stats.update_from_finished_request("length", 20, 101, req, 0) == "native-result"
        assert stats.args == ("length", 20, 101, req, 0)
        row = json.loads(path.read_text())  # flushed at request finish
        assert row["output_tokens"] == 101
        assert row["last_token_monotonic"] - row["scheduled_monotonic"] == 4
        assert "PRIVATE" not in path.read_text()
    finally:
        handle.close()


def test_backend_cli(tmp_path, monkeypatch):
    path, out = tmp_path / "backend.jsonl", tmp_path / "summary.json"
    write_requests(path, [request(10, 11, 14, 101)])
    monkeypatch.setattr(sys, "argv", ["agenttrace", "backend-throughput", str(path), "--output", str(out)])
    assert main() == 0
    assert json.loads(out.read_text())["active_output_tokens_per_second"] == 101 / 4


def test_window_clips_crossing_requests_and_counts_only_in_window_tokens(tmp_path):
    path = tmp_path / "events.jsonl"
    rows = [request(10, 11, 14, 4, event="request_finished", request_id="a",
                    first_token_unix=1011, queued_monotonic=9),
            request(12, 13, 16, 4, event="request_finished", request_id="b",
                    first_token_unix=1013, queued_monotonic=11)]
    for key, times in (("a", [1011, 1012, 1013, 1014]), ("b", [1013, 1014, 1015, 1016])):
        for i, t in enumerate(times):
            rows.append({"event": "tokens", "request_id": key, "hostname": "gpu-node",
                "timestamp_unix": t, "output_tokens": 1, "decode_tokens": int(i > 0)})
    write_requests(path, rows)
    result = summarize_backend(path, started_unix=1012, ended_unix=1015, window=True)
    assert result["output_tokens"] == 5  # neither pre-window nor t==end tokens count
    assert result["decode_tokens"] == 4
    assert result["active_seconds"] == result["decode_seconds"] == 3
    assert result["queue_seconds"]["count"] == 1  # b scheduled in window
    rows.pop()
    write_requests(path, rows)
    with pytest.raises(ValueError, match="incomplete"):
        summarize_backend(path, started_unix=1012, ended_unix=1015, window=True)
