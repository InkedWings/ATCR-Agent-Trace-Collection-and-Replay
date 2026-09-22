"""Observer invariants without requiring vLLM/GPU in the development venv."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS

import pytest


@pytest.fixture
def scheduler(tmp_path, monkeypatch):
    class Native:
        def __init__(self):
            self.scheduler_config = NS(async_scheduling=True)
            self.connector = None
            self.max_num_scheduled_tokens = 2048
            self.max_num_running_reqs = 64
            self.scheduler_reserve_full_isl = True
            self.need_mamba_block_aligned_split = True
            self.cache_config = NS(block_size=528)
            self.running, self.waiting, self.skipped_waiting = [], [], []
            self.requests = {}
            self.native_calls = []
            self.block_result = object()
            self.output = NS(total_num_scheduled_tokens=1584, num_scheduled_tokens={})
            spec = NS(block_size=528)

            class Full:
                @classmethod
                def find_longest_cache_hit(cls, **kwargs):
                    self.native_calls.append("full")
                    return ([object()] * 10,)

            class Mamba:
                @classmethod
                def find_longest_cache_hit(cls, **kwargs):
                    self.native_calls.append("mamba")
                    return ([object()],)

            coordinator = NS(attention_groups=[(spec, [0], Full), (spec, [1], Mamba)],
                             get_num_blocks_to_allocate=lambda **kw: 70)

            def lookup(request):
                self.native_calls.append("lookup")
                for group_spec, ids, cls in coordinator.attention_groups:
                    cls.find_longest_cache_hit(max_length=6000, kv_cache_group_ids=ids,
                                              kv_cache_spec=group_spec)
                return self.block_result, 528

            def fits(request):
                return coordinator.get_num_blocks_to_allocate(request_id=request.request_id) <= 60

            self.kv_cache_manager = NS(coordinator=coordinator,
                block_pool=NS(num_gpu_blocks=100, get_num_free_blocks=lambda: 60),
                kv_cache_config=NS(kv_cache_groups=[NS(kv_cache_spec=spec, layer_names=["layer"])]),
                get_computed_blocks=lookup, can_fit_full_sequence=fits,
                allocate_slots=lambda *args: self.block_result)

        def _mamba_block_aligned_split(self, request, tokens, *a, **kw):
            self.native_calls.append("alignment")
            return tokens // 528 * 528

        def schedule(self):
            self.native_calls.append("schedule")
            return self.output

        def shutdown(self):
            self.native_calls.append("shutdown")

    for name in ("vllm", "vllm.v1", "vllm.v1.core", "vllm.v1.core.sched",
                 "vllm.v1.core.sched.async_scheduler"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["vllm"].__version__ = "0.19.1"
    sys.modules["vllm.v1.core.sched.async_scheduler"].AsyncScheduler = Native
    path = tmp_path / "scheduler.jsonl"
    monkeypatch.setenv("AGENTTRACE_SCHEDULER_EVENTS", str(path))
    spec = importlib.util.spec_from_file_location("diagnostics_under_test",
        Path(__file__).parents[1] / "src/agenttrace/vllm_diagnostics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    instance = module.DiagnosticAsyncScheduler()
    yield instance, path
    instance.shutdown()


def test_observations_preserve_cache_and_scheduler_results(scheduler):
    instance, path = scheduler
    request = NS(request_id="a", num_prompt_tokens=6001, status=NS(name="WAITING"))
    manager = instance.kv_cache_manager
    for _ in range(2):
        assert manager.get_computed_blocks(request)[0] is instance.block_result
    assert instance.native_calls == ["lookup", "full", "mamba"] * 2
    assert manager.allocate_slots(request, 1584) is instance.block_result
    instance.output.num_scheduled_tokens = {"a": 1584}
    assert instance.schedule() is instance.output
    assert instance.native_calls[-1] == "schedule"
    instance._flush()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    admitted = next(row for row in rows if row["event"] == "cache_at_admission")
    assert admitted["queries"] == 2
    assert admitted["last"]["cached_tokens"] == 528
    assert [g["hit_tokens"] for g in admitted["last"]["groups"]] == [5280, 528]
    assert not instance._diag_pending


def test_actual_gate_failure_is_distinguished_from_alignment(scheduler):
    instance, path = scheduler
    request = NS(request_id="a", num_prompt_tokens=6001, status=NS(name="WAITING"))
    assert instance._mamba_block_aligned_split(request, 464) == 0
    assert instance._mamba_block_aligned_split(request, 1584) == 1584
    assert instance.kv_cache_manager.can_fit_full_sequence(request) is False
    instance._flush()
    row = json.loads(path.read_text().splitlines()[-1])
    assert row["counts"] == {"alignment_zero_waiting": 1, "full_isl_rejections": 1}
    assert row["examples"]["full_isl_rejections"]["required_blocks"] == 70
    assert row["examples"]["full_isl_rejections"]["free_blocks"] == 60


def test_native_exception_propagates_and_observer_context_is_cleared(scheduler):
    instance, _ = scheduler
    request = NS(request_id="a", num_prompt_tokens=6001)
    error = ValueError("native failure")

    def fail(*args):
        raise error

    with pytest.raises(ValueError) as caught:
        instance._cache_lookup(fail, request)
    assert caught.value is error and instance._diag_groups is None
    with pytest.raises(ValueError) as caught:
        instance._can_fit(fail, request)
    assert caught.value is error and instance._diag_gate is None
