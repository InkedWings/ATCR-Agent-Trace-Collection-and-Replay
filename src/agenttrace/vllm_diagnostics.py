"""Opt-in observations of vLLM 0.19.1's native asynchronous scheduler.

Use only for the single-engine, text-only diagnostic runs. Native methods run
exactly once; cache lookups are observed, never repeated to probe the cache.
Counts describe scheduling iterations/calls, not fractions of GPU execution time.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path

import vllm
from vllm.v1.core.sched.async_scheduler import AsyncScheduler


class DiagnosticAsyncScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs):
        if vllm.__version__ != "0.19.1":
            raise RuntimeError("scheduler diagnostics require vLLM 0.19.1")
        super().__init__(*args, **kwargs)
        if not self.scheduler_config.async_scheduling:
            raise RuntimeError("diagnostic scheduler requires async_scheduling=True")
        if self.connector is not None:
            raise RuntimeError("scheduler diagnostics do not support a KV connector")
        path = Path(os.environ["AGENTTRACE_SCHEDULER_EVENTS"])
        path.parent.mkdir(parents=True, exist_ok=True)
        self._diag_handle = path.open("x", buffering=65536)
        self._diag_counts = Counter()
        self._diag_pending = {}
        self._diag_groups = None
        self._diag_examples = {}
        self._diag_last_flush = time.monotonic()
        self._diag_gate = None
        self._diag_required_blocks = None
        manager = self.kv_cache_manager
        coordinator = manager.coordinator
        self._diag_restores = []
        self._diag_original_groups = coordinator.attention_groups
        coordinator.attention_groups = [
            (spec, ids, self._observe_group(cls))
            for spec, ids, cls in coordinator.attention_groups
        ]
        self._replace(manager, "get_computed_blocks", self._cache_lookup)
        self._replace(manager, "can_fit_full_sequence", self._can_fit)
        self._replace(manager, "allocate_slots", self._allocate)
        self._replace(coordinator, "get_num_blocks_to_allocate", self._required_blocks)
        self._emit("scheduler_config", max_scheduled_tokens=self.max_num_scheduled_tokens,
                   max_num_seqs=self.max_num_running_reqs,
                   async_scheduling=self.scheduler_config.async_scheduling,
                   reserve_full_isl=self.scheduler_reserve_full_isl,
                   mamba_alignment=self.need_mamba_block_aligned_split,
                   block_size=self.cache_config.block_size,
                   num_gpu_blocks=manager.block_pool.num_gpu_blocks,
                   groups=[{"spec": type(group.kv_cache_spec).__name__,
                            "block_size": group.kv_cache_spec.block_size,
                            "layers": len(group.layer_names)}
                           for group in manager.kv_cache_config.kv_cache_groups])
        self._diag_handle.flush()

    def _replace(self, owner, name, observer):
        original = getattr(owner, name)
        # Restore the instance's previous attribute, including absent attributes.
        self._diag_restores.append((owner, name, name in vars(owner), vars(owner).get(name)))
        setattr(owner, name, lambda *a, **kw: observer(original, *a, **kw))

    def _observe_group(self, cls):
        original = cls.find_longest_cache_hit

        def observed(_cls, *args, **kwargs):
            result = original(*args, **kwargs)
            if self._diag_groups is not None:
                # HybridKVCacheCoordinator supplies these arguments by keyword.
                self._diag_groups.append({
                    "manager": cls.__name__, "group_ids": kwargs["kv_cache_group_ids"],
                    "candidate_tokens": kwargs["max_length"],
                    "hit_tokens": len(result[0]) * kwargs["kv_cache_spec"].block_size,
                })
            return result

        return type("Observed" + cls.__name__, (cls,),
                    {"find_longest_cache_hit": classmethod(observed)})

    def _cache_lookup(self, original, request, *args, **kwargs):
        self._diag_groups = []
        try:
            result = original(request, *args, **kwargs)
            snapshot = {"cached_tokens": result[1], "groups": self._diag_groups}
            pending = self._diag_pending.setdefault(request.request_id, {
                "first_query_unix": time.time(), "prompt_tokens": request.num_prompt_tokens,
                "queries": 0, "first": snapshot})
            pending["queries"] += 1
            pending["last"] = snapshot
            self._diag_counts["cache_lookup_calls"] += 1
            return result
        finally:
            self._diag_groups = None

    def _blocked(self, reason, request, **details):
        self._diag_counts[reason] += 1
        if reason not in self._diag_examples:
            self._diag_examples[reason] = {
                "request_id": request.request_id, "status": request.status.name,
                "prompt_tokens": request.num_prompt_tokens,
                "free_blocks": self.kv_cache_manager.block_pool.get_num_free_blocks(),
                **details,
            }

    def _required_blocks(self, original, *args, **kwargs):
        result = original(*args, **kwargs)
        if self._diag_gate is not None:
            self._diag_required_blocks = result
        return result

    def _can_fit(self, original, request, *args, **kwargs):
        self._diag_gate = "full_isl"
        self._diag_required_blocks = None
        try:
            result = original(request, *args, **kwargs)
            if not result:
                self._blocked("full_isl_rejections", request,
                              required_blocks=self._diag_required_blocks)
            return result
        finally:
            self._diag_gate = None

    def _allocate(self, original, request, *args, **kwargs):
        result = original(request, *args, **kwargs)
        if result is None:
            self._blocked("allocation_failures_" + request.status.name.lower(), request)
        return result

    def _mamba_block_aligned_split(self, request, num_new_tokens, *args, **kwargs):
        result = super()._mamba_block_aligned_split(request, num_new_tokens, *args, **kwargs)
        if num_new_tokens > 0 and result == 0:
            self._blocked("alignment_zero_" + request.status.name.lower(), request,
                          available_tokens=num_new_tokens)
        return result

    def _emit(self, event, **fields):
        self._diag_handle.write(json.dumps({"event": event, "timestamp_unix": time.time(),
                                           **fields}, separators=(",", ":")) + "\n")

    def schedule(self):
        output = super().schedule()
        self._diag_counts["steps"] += 1
        used = output.total_num_scheduled_tokens
        self._diag_counts["scheduled_tokens"] += used
        self._diag_counts["unused_budget_tokens"] += self.max_num_scheduled_tokens - used
        waiting = len(self.waiting) + len(self.skipped_waiting)
        if waiting:
            self._diag_counts["steps_ending_with_waiters"] += 1
            if used == self.max_num_scheduled_tokens:
                self._diag_counts["steps_ending_with_waiters_and_budget_exhausted"] += 1
            if len(self.running) == self.max_num_running_reqs:
                self._diag_counts["steps_ending_with_waiters_and_running_limit"] += 1
        for request_id in output.num_scheduled_tokens:
            pending = self._diag_pending.pop(request_id, None)
            if pending is not None:
                self._emit("cache_at_admission", request_id=request_id, **pending)
        if time.monotonic() - self._diag_last_flush >= 1:
            self._flush()
        return output

    def _flush(self):
        self._emit("scheduler_interval", interval_seconds=time.monotonic() - self._diag_last_flush,
                   counts=dict(self._diag_counts), examples=self._diag_examples,
                   running=len(self.running), waiting=len(self.waiting),
                   skipped_waiting=len(self.skipped_waiting),
                   free_blocks=self.kv_cache_manager.block_pool.get_num_free_blocks())
        self._diag_handle.flush()
        self._diag_counts.clear()
        self._diag_examples.clear()
        self._diag_last_flush = time.monotonic()
        # Aborted waiting requests must not accumulate in a long diagnostic run.
        for request_id in list(self._diag_pending):
            if request_id not in self.requests:
                del self._diag_pending[request_id]

    def shutdown(self):
        try:
            self._flush()
        finally:
            self._diag_handle.close()
            for owner, name, existed, value in reversed(self._diag_restores):
                if existed:
                    setattr(owner, name, value)
                else:
                    delattr(owner, name)
            self.kv_cache_manager.coordinator.attention_groups = self._diag_original_groups
            super().shutdown()
