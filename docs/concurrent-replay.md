# Concurrent replay and instrumentation

For continuous refill, fixed warmup/measurement windows and the managed two-node
experiment, see [Single-backend scaling](single-backend-scaling.md).
The finite-list mode below remains supported.

`agenttrace benchmark` replays a finite list of traces with a configurable
maximum number of tasks in flight. Each task runs in its own process, with its
own executor, workspace, and temporary files. The same trace can appear more
than once. The schema and the DAG inside each trace are unchanged.

## Concurrency and timing

`--concurrency N` defaults to 1. At most N task lifecycles run simultaneously:
process startup, executor/container setup, DAG replay, and cleanup all occupy
the slot. A completed task releases its slot to the next listed task.
`--repeat R` repeats the whole input list R times in the same order.

The task limit does not cap LLM requests. If two tasks each have two ready LLM
branches, task concurrency 2 can produce four simultaneous LLM requests. The
vLLM sequence limit is a separate server setting; choose it according to the
request concurrency the workload can produce, including within-task branches.

This is a bounded closed-loop workload, with a startup and a final drain phase.
The active count can be below N during those phases and the DAG execution count
can be below N while containers are being built. It is not an arrival-rate
generator or a synchronized steady-state benchmark.

Measurements distinguish:

| Field | Meaning |
| --- | --- |
| `admission_wait_seconds` | Time from benchmark start until this task gets a slot |
| `lifecycle_seconds` | Slot occupancy, including subprocess startup, setup and cleanup |
| `setup_seconds` | Workspace restoration and executor setup inside the task |
| `replay_makespan_seconds` | DAG execution after setup, excluding cleanup |
| node `elapsed_seconds` | Actual executor call duration, excluding dependency waits |
| node `ttft_seconds` | Client time to first nonempty content, reasoning or tool-argument delta |
| `output_stream_seconds` | Time between first and last nonempty output chunks |
| `tpot_estimate_seconds` | Output-stream time divided by completion tokens minus one; null for fewer than two chunks/tokens |

The last metric is an estimate because one SSE chunk can contain multiple
tokens. Client TTFT includes network, server queueing, processing and buffering;
it is not a measurement of GPU prefill time. LLM bodies are discarded after
timing. The recorded token target is still checked for each request.

Summary latency statistics (mean, p50, p95, p99) use completed tasks only.
Task throughput is completed tasks divided by the entire workload window,
including setup and the drain phase. Summing sibling node durations does not
give task wall time when a trace has parallel branches.

## Backend token throughput (excluding replay/tool idle gaps)

`output_tokens_per_second` is end-to-end replay throughput. For backend-only
throughput, the Qwen launcher now records vLLM 0.19.1 EngineCore request
timestamps in a separate `*-requests.jsonl` file. The in-process hook preserves
vLLM's original metrics update and writes incremental token-count events plus
one timing record per finished request; it does not save prompts or generated bodies. No container rebuild or
additional runtime dependency is needed. This integration targets one API
process and one engine on one serving host (tensor parallelism is supported).

Set the same **absolute shared-file path** in both shells, before starting a
new server and the benchmark:

```bash
export VLLM_BACKEND_EVENTS="$PWD/runs/vllm/experiment-requests.jsonl"
```

The server requires a new filename. The benchmark launcher passes it as
`--backend-events` and adds `backend_throughput` to `summary.json`:

- `active_output_tokens_per_second`: total output tokens divided by the union
  of `[first scheduled, last token]` intervals (prefill + decode).
- `decode_tokens_per_second`: total output tokens minus one per request,
  divided by the union of `[first token, last token]` intervals. The first token
  is excluded because it is produced during prefill in this metric definition.

Intervals are merged before division: two requests running for the same second
contribute one second, not two. Idle gaps for container setup and tool execution
are excluded, as is queueing before first scheduling. These are backend request
intervals, **not CUDA-kernel-only time**; vLLM's intervals include preemption and
in-engine stalls. Decode intervals can overlap prefill work for other requests.
See the [vLLM 0.19.1 timestamp definitions](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/v1/metrics/stats.py).

Use a dedicated idle endpoint at benchmark start. Only complete successful
requests in that benchmark's time window are used; request/token totals must
match the replay reports or the result is marked unavailable. Non-completed
benchmarks and runs without backend timestamps do not get invented values.
An entire server session can also be summarized with:

```bash
.venv/bin/agenttrace backend-throughput "$VLLM_BACKEND_EVENTS"
```

Existing servers must be relaunched through the updated launcher to enable this
hook. Old 1-second Prometheus samples cannot recover exact active intervals;
the four-trace run `real-four-c2-20260906-1` cannot be backfilled precisely.
The vLLM log's periodic throughput also uses a wall-clock window, so it can
include idle gaps and is not a substitute for this busy-time metric.

## Two-node Polaris commands

Use the repository `.venv` prepared by the mini-SWE bootstrap script. These
commands do not request or terminate PBS allocations.

On the GPU node, configure a server sequence limit suitable for the expected
LLM request concurrency. The previous smoke configuration defaults to
one sequence, which limits server-side batching even if many clients connect.
Keep this server limit fixed during a concurrency sweep unless that limit is
itself an experimental variable:

```bash
VLLM_MAX_NUM_SEQS=8 VLLM_PREFIX_CACHING=1 \
  ./examples/minisweagent_swebench/scripts/vllm_qwen3_32b.sh serve
```

In another shell **on the GPU node**, start node hardware monitoring. The local
sampler is needed even though the replay coordinator can scrape remote vLLM
metrics. Choose a duration that covers the run, or omit it and stop with Ctrl-C:

```bash
.venv/bin/agenttrace monitor \
  --label inference --interval 1 --duration 1800 \
  --output runs/metrics/inference-c4.jsonl
```

On the tool/replay node, run real mini-SWE traces:

```bash
MINISWE_REPLAY_RUN_ID=c4 \
./examples/minisweagent_swebench/scripts/benchmark_qwen3_32b.sh \
  http://<gpu-node>:8000/v1 \
  runs/minisweagent-swebench/<capture-run>/tasks/*/trace.json \
  --concurrency 4
```

The launcher selects the native Apptainer executable and prepares node-local
build scratch, proxy settings, and the local Qwen replay profile. The coordinator
samples the replay node automatically and scrapes the remote `/metrics` endpoint.
The benchmark runs in the foreground. Task start/completion messages appear in
the terminal; each task's console output and live node events are saved below.

For a generic adapter or a controlled repeated-trace test:

```bash
AGENTTRACE_LLM_BASE_URL=http://<gpu-node>:8000/v1 \
.venv/bin/agenttrace benchmark path/to/trace.json \
  --profile examples/minisweagent_swebench/replay_profile_local_vllm.json \
  --concurrency 2 --repeat 4 \
  --vllm-metrics-url http://<gpu-node>:8000/metrics \
  --output-dir runs/benchmarks/repeated-c2
```

`--trace-list paths.txt` accepts one trace path per line, relative to the list
file. Blank lines and `#` comments are ignored. Existing output directories are
not overwritten. Use a new run ID/directory for each experiment.

## Outputs and metric sources

```text
runs/benchmarks/<run-id>/
├── manifest.json            # ordered task list, concurrency and measurement policy
├── summary.json             # timing distributions, throughput, sample summaries
├── metrics.jsonl            # replay-node hardware + raw vLLM Prometheus snapshots
└── tasks/<unique-task-id>/
    ├── console.log
    ├── events.jsonl         # live setup, node start/end/failure and cleanup events
    ├── status.json
    ├── report.json          # completed task's per-node performance report
    └── run/                 # private workspace and tmp
```

The explicit task IDs disambiguate repeated trace IDs. `events.jsonl` records
monotonic relative time and Unix timestamps; hardware snapshots record Unix
timestamps and hostname. Match time windows across nodes using their clocks;
durations inside each process use a monotonic clock. The inference hardware file
is separate and is not automatically merged into the coordinator summary.

The hardware sampler reads `/proc` for whole-node CPU busy/iowait, available
memory, per-device disk I/O counters, and per-interface network counters.
Compute byte rates using successive counter deltas divided by sample time.
Do not sum disk partitions and their parent disks; these overlap. Local block
device statistics do not measure Lustre client file operations or attribute
traffic to the frontend process. That deeper I/O attribution needs additional
profiling.

On NVIDIA nodes, `nvidia-smi` adds each GPU's busy percentage, memory-controller
busy percentage, used/total VRAM, and power. GPU busy percentage describes time
with executing kernels, not FLOP utilization; memory busy percentage is distinct
from allocated VRAM. See [NVIDIA's query definitions](https://docs.nvidia.com/deploy/nvidia-smi/index.html).

Whole-node measurements include other jobs/processes on that node. Use dedicated
nodes for formal results. Hardware means in `summary.json` are sample means,
not per-task attribution or time-weighted integrals. Missing GPUs, unavailable
fields, scrape failures and counter resets are represented as missing/error
states rather than invented zero measurements.

The raw vLLM snapshots preserve queue length, running requests, KV occupancy,
token counters and the server's request-latency histograms, including queue,
prefill, decode and TTFT. They are endpoint-wide aggregates and are not joined
to individual task nodes. Available metrics follow the deployed version; see
[vLLM 0.19.1 metrics](https://docs.vllm.ai/en/v0.19.1/usage/metrics/).

The summary subtracts the first successful counter snapshot from the last.
Prefix hit rate uses `delta(hits) / delta(queries)`, rather than a server-lifetime
ratio. Counter decreases or disappearing series invalidate the affected delta.
Raw histograms remain available for interval-specific backend percentiles.
Snapshots may lag request completion according to the backend reporting period,
so their counts need not match the client report exactly. The sample window
brackets workload launch and cleanup; see its timestamps in the summary.

Sampling defaults to one second and records its own collection duration. There
is no Prometheus server, Grafana installation or OpenTelemetry collector required.

## Cache policy, failures and validation

The runner preserves existing server cache state. Repeating a trace or rerunning
an experiment can reuse prefixes from earlier requests. `VLLM_PREFIX_CACHING=0`
passes the explicit `--no-enable-prefix-caching` flag on server startup. For a
cold-cache experiment, restart the serving instance before the measurement;
for warm-cache experiments, keep a consistent warm-up procedure. Record the
server configuration and cache policy with the experiment results.

Tool success or failure never changes the recorded replay path. Native `isError`
results and OpenClaw Gateway `error.type: "tool_error"` responses (including HTTP
400/403/500) are recorded as `native_error: true`, with their execution latency.
Subsequent nodes and tasks still run; LLM inputs remain the recorded payloads.
These calls count toward tool latency and `native_tool_errors`, without retry.
Node `status: "completed"` means the invocation finished, not that the tool succeeded.
Runtime setup, authentication, transport, malformed Gateway responses or
LLM token-count failures still stop admissions and cancel other active tasks;
these are not native tool results. Ctrl-C or SIGTERM stops this benchmark's child
process groups and writes the partial summary. The surrounding interactive PBS
shell and unrelated collection/server processes are not signalled. Cancelled
native sandboxes may remain in the run's scratch directory; no broad cleanup is
performed.

No-GPU scheduling check:

```bash
.venv/bin/agenttrace benchmark examples/synthetic/trace.json \
  --dry-run --concurrency 2 --repeat 4 --output-dir runs/benchmarks/dry-c2
.venv/bin/pytest -q
```

Dry-run reports are explicitly marked `dry_run`; their token counts are synthetic
and must not be used as real performance results. Integration tests exercise
bounded overlapping HTTP requests, independent tool workspaces, failure without
retry, streaming timing, metric counter deltas and Ctrl-C descendant cleanup.
Before a formal sweep, verify concurrency 1 and 2 on the actual allocated nodes,
confirm that inference-node GPU samples exist, and inspect actual running-request
counts in vLLM. Choose traces that fit the served model's context window.
