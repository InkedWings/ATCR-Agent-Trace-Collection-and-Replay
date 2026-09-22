# Single-engine scheduler diagnostics

Set `VLLM_SCHEDULER_DIAGNOSTICS=1` in the existing serving environment to write
`backend-scheduler.jsonl` beside `backend.jsonl`. This explicitly selects a
subclass of vLLM 0.19.1 `AsyncScheduler` and retains its native scheduling and
cache methods. It is intended for the current text-only hybrid Qwen model,
without a KV connector. Other configurations are not covered.

The observer records native results without performing additional cache
lookups or changing admission decisions. Cache-manager wrappers belong to the
scheduler instance; they do not replace the global cache-manager classes.
Admissions use buffered writes; interval counters flush approximately once per
second while the scheduler is active. There is no per-token diagnostic log.
Instrumentation still adds CPU and I/O work: enable it for both sides of a
parameter comparison, and confirm the final performance setting without it.

| Event / field | Interpretation |
|---|---|
| `scheduler_config` | Actual block count, cache groups, block size and scheduling limits. Compare this between budgets because memory profiling can change KV capacity. |
| `cache_at_admission` | First and last real cache query before scheduling, number of queries, and group match lengths. The final cached-token count is the intersection, not the sum across groups. |
| Group `candidate_tokens` / `hit_tokens` | Candidate passed to that native group lookup and length it returned. A Mamba reduction after a longer attention match identifies a state-checkpoint matching limitation for that query; it does not by itself establish why the checkpoint was unavailable. |
| `alignment_zero_waiting` | Actual positive token allowance reduced to zero by Mamba alignment for a waiting request. The native waiting loop stops at this condition. |
| `alignment_zero_running` | Same condition for a running request; the native scheduler skips that request for this iteration. |
| `full_isl_rejections` | Native full-input-length admission check returned false. Examples include its actual required block count and currently free blocks. |
| `allocation_failures_*` | Native slot allocation returned `None`, separated by request status. |
| `steps_ending_with_waiters_and_*` | End-of-iteration budget-exhaustion/running-limit observations. These are not exclusive explanations of every request's wait. |

Interval counters count calls or iterations, not seconds of GPU work. A request
can be queried repeatedly while waiting, so lookup-level ratios must not be
presented as actual prompt reuse. Use prompt/cached token counters or the
completed-request log for that ratio, with a stated time/cohort boundary.
Snapshot gauges at an interval's end are not time-weighted averages. Idle gaps
can make the next interval longer than one second.

Tests in `tests/test_vllm_diagnostics.py` check that native methods run once,
return objects and exceptions survive observation, and different rejection
conditions remain distinguishable. GPU smoke and backend/client token-count
validation use the existing scale runner.
