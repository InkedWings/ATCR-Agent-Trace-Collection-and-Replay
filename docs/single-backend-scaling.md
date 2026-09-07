# Single-backend exploratory scaling

This experiment runs real fixed-path DAG replay against one dedicated vLLM
backend. OpenClaw and mini-SWE-agent are separate workloads, not a mixed pool.
New LLM outputs are discarded; recorded tool calls execute in native runtimes.

## Protocol

`examples/scaling/single-backend.json` records the configuration:

- Qwen3-32B BF16, vLLM 0.19.1, TP=4 on four A100s.
- Context 32768, GPU memory fraction 0.90, `max_num_seqs=32`,
  `max_num_batched_tokens=2048`, prefix caching enabled.
- Task concurrency 1, 2, 4, 8; one repetition per point.
- 120 seconds of warmup, then a 600-second measurement window.
- Immediately refill free task slots. At the deadline, stop admissions and
  let all admitted tasks finish naturally. Do not truncate long requests/tasks.
- Restart the experiment-owned backend before each point, so the initial
  prefix cache is empty. Model loading/startup is outside the warmup window.
- Shuffle the frozen trace pool each cycle with one RNG seeded to 42; reset
  that RNG for every point. Record every actual admission and pool cycle.

Concurrency counts complete task lifecycles, including setup and cleanup.
It does **not** cap LLM requests: one DAG may have several ready LLM branches.
The fixed server sequence/token budgets are separate experimental controls.
These settings are an initial exploration, not a claim that they maximize
throughput or that cc=8 is maximum sustainable concurrency.

## Prepare the pools once

Run on the replay node with an existing **Qwen3-32B** endpoint. `/tokenize`
does not run inference or touch the prefix cache. It uses the serving
tokenizer/template and the recorded messages, tools, and template overrides.

```bash
cd /lus/eagle/projects/lc-mpi/ZhijingYe/SIGMETRICS_2027/trace_framework
source examples/scaling/environment.sh
.venv/bin/python -m agenttrace.experiments.prepare \
  --base-url http://x3005c0s1b1n0:8000/v1 \
  --output runs/scaling/preflight-YYYYMMDD
```

The default candidates are GAIA `../datasets/GAIA_Trace/run2` (24 traces) and
`runs/minisweagent-swebench/dev-20260906T004724Z` (22 traces). Validation checks
schema/DAG, supported protocols/tools, workspace seeds and artifact existence.
If **any** LLM request's prompt tokens + recorded output target exceed 32768,
exclude the entire trace. Do not shorten inputs/outputs or filter by answers.
`preflight.json` records every check's counts and exclusion reason;
`traces.txt` freezes the eligible paths. Original datasets remain unchanged.

The 2026-09-06 preflight in `runs/scaling/preflight-20260906-1` found
**11 eligible GAIA traces and 10 eligible mini-SWE traces**. All other candidate
exclusions were context overflow. Results apply to these filtered pools;
long-context traces are underrepresented and this must be reported.

Precache each pool's native runtime/images before measuring:

```bash
bash examples/scaling/run.sh precache \
  --pool runs/scaling/preflight-20260906-1/openclaw/traces.txt \
  --profile examples/openclaw_gaia/replay_profile_local_vllm.json \
  --output runs/scaling/precache-openclaw-NEW
bash examples/scaling/run.sh precache \
  --pool runs/scaling/preflight-20260906-1/minisweagent/traces.txt \
  --profile examples/minisweagent_swebench/replay_profile_local_vllm.json \
  --output runs/scaling/precache-miniswe-NEW
```

This only sets up/closes tools; it does not invoke trace nodes. Subsequent
tasks still create independent workspaces, Gateways and writable container
sandboxes. Their setup latency remains part of task lifecycle latency.
Each OpenClaw task seeds its private managed-npm directory from the installed
Brave plugin project. A configured load path alone is insufficient: OpenClaw's
startup checks may otherwise reinstall Brave in every fresh state directory.
Sessions and credentials are not copied. The replay profile allows 180 seconds
for Gateway startup; this is a readiness limit, not an added sleep or tool retry.
Traces without `web_search` explicitly disable search so an inherited Brave API
key cannot trigger an unnecessary plugin installation.
Brave credentials are loaded from the existing user credential file; no ALCF
access token is needed for local Qwen replay. No credential values are printed.

## Start the experiment

Edit the two hostnames in the JSON config when changing allocation. The master
runs **on the replay node**, uses SSH to the inference node, and shares the
repository/output paths over Eagle. Both hosts need synchronized system clocks
for window selection; backend interval durations come from EngineCore's clock.
The repository Python environment and existing vLLM SIF are reused.

Port 8000 must be free. The script refuses an occupied port; it never kills an
unrelated or pre-existing service. Stop only the known server you own before
using the managed launch flow.

First run a real acceptance pass: one complete short trace per framework,
followed by a short cc=2 window and natural drain for each framework.

```bash
bash examples/scaling/run.sh run --smoke \
  --job-id 7589784 \
  --pools runs/scaling/preflight-20260906-1 \
  --output runs/scaling/smoke-NEW
```

After it validates, run the eight exploratory points:

```bash
bash examples/scaling/run.sh run \
  --job-id 7589784 \
  --pools runs/scaling/preflight-20260906-1 \
  --output runs/scaling/single-backend-NEW
```

Both commands run in the foreground. Task start/completion and phase changes
are displayed; detailed task output is in `tasks/<id>/console.log`. Server
startup logs are in each point's `service.log`. Ctrl-C cancels the coordinator's
own replay process groups and closes its SSH control pipe; the remote worker
stops its own server and sampler. It never runs `qdel` or signals the PBS shell.
No automatic retry is performed. Native tool failures do not stop a trace or
the sweep: `isError` results and OpenClaw Gateway `error.type: "tool_error"`
responses (including HTTP 400/403/500) are recorded as `native_error: true`,
with their latency. Subsequent recorded calls still execute, and LLM inputs
remain unchanged. Runtime setup/auth/transport and LLM execution failures
stop the sweep and preserve partial results; these are not native tool results.

The script checks remaining allocation time before starting and before every
point. The fixed warmup/measurement windows alone total 96 minutes. Server
restarts and natural drain add workload-dependent time; the script budgets an
additional ten minutes per point, which is an estimate, not an upper bound.

## Split cc=8, 16, 32 into preemptable jobs

Submit **three independent two-node jobs**, one task-concurrency value per job.
Each job runs OpenClaw followed by mini-SWE-agent, with a fresh backend for each
workload. Each allocation has a two-hour limit; it exits as soon as both points
finish. Existing cc=1/2/4 results are not touched.

```bash
cd /lus/eagle/projects/lc-mpi/ZhijingYe/SIGMETRICS_2027/trace_framework
mkdir -p runs/pbs-logs
qsub -N at-scale-cc8  -v SCALE_CC=8  examples/scaling/pbs/single_backend_preemptable.pbs
qsub -N at-scale-cc16 -v SCALE_CC=16 examples/scaling/pbs/single_backend_preemptable.pbs
qsub -N at-scale-cc32 -v SCALE_CC=32 examples/scaling/pbs/single_backend_preemptable.pbs
```

The batch script selects the replay/inference hosts from `PBS_NODEFILE`; no
hostname edits are necessary. It reuses the frozen `preflight-20260906-1` pools
and the same backend settings, seed and 120/600-second windows. New node-local
tool/image caches are populated sequentially before starting the experiment.
Credentials are loaded at runtime by `environment.sh`, not passed through
`qsub -V`. Do not submit these commands again to inspect an existing job.

For a job numbered `JOBID`, live logs are
`runs/scaling/preemptable-ccCC-JOBID.log`; results are in
`runs/scaling/preemptable-ccCC-JOBID/experiment/`, including per-workload
directories and `summary.csv` / `summary.md`. Configuration and precache output
are in the parent directory. `qstat -u "$USER"` shows queue/running state.

[Polaris preemptable jobs](https://docs.alcf.anl.gov/polaris/running-jobs/)
may be killed without warning. `#PBS -r n` disables automatic reruns. Persisted
partial results are retained, but a point without a completed, valid measurement
summary is not a successful experiment. Separate jobs have separate nodes,
workspaces and results; if scheduled together, they still share external web
services and Eagle storage, which can affect tool latencies and error rates.

To rerun **only mini-SWE-agent cc=8 in debug** and **both cc=16 workloads in
preemptable**, reuse the batch script with PBS overrides:

```bash
qsub -q debug -l walltime=01:00:00 -N at-ms-cc8 \
  -v SCALE_CC=8,SCALE_WORKLOADS=minisweagent \
  examples/scaling/pbs/single_backend_preemptable.pbs
qsub -q preemptable -l walltime=02:00:00 -N at-scale-cc16 \
  -v SCALE_CC=16 examples/scaling/pbs/single_backend_preemptable.pbs
```

The first job uses `debug-cc8-JOBID` in log/result paths and does not prepare
or execute OpenClaw. New job IDs produce separate output directories; successful
previous points remain unchanged. All measurement/model settings stay the same.

Gateway ports now carry per-user, node-local cross-process leases for their full
lifetime; task-private temporary directories cannot coordinate these leases.
Readiness verifies bearer authentication with an empty `/tools/invoke` request
that is rejected before any tool executes; public `/health` alone is insufficient.
The vLLM launcher execs Apptainer with a private PID namespace and its init shim.
Shutdown waits for the owned process group, not only the launcher, before another
point may start. No tool/LLM retry is added by either fix.

## Measurements and cohorts

The primary results are under `summary.json.measurement`. Top-level totals
cover warmup through drain and are useful for count cross-checks, **not** the
steady-window results.

| Metric | Population / denominator |
| --- | --- |
| Task throughput | Tasks completed inside `[T0,T1)` / 600 seconds, including warmup admissions |
| Task lifecycle/setup/replay latency | Tasks admitted inside the window; include their full latency after drain |
| Client LLM/tool latency, TTFT | Calls started inside the window; include their full latency after drain |
| Backend active output tokens/s | Tokens generated inside the window / union of scheduled-to-last-token intervals clipped to the window |
| Backend decode tokens/s | Decode tokens generated inside the window / union of first-to-last-token intervals clipped to the window |
| Backend queue/prefill/decode latency | Requests first scheduled inside the window; complete intervals through drain |
| Hardware | One-second samples within the window, separately on both nodes |
| Prefix-cache hit ratio | Hit/query counter deltas between the first/last successful samples inside the window |

The vLLM hook records request IDs, incremental token counts, EngineCore event
timestamps and request completion timings, but no prompt/output text. Counts
from requests straddling a boundary are assigned by **token event timestamp**,
not by request completion time. The first token is excluded from decode-token
counts. Overlapping intervals count once in the denominator. These are backend
request-active intervals, not CUDA-kernel-only execution: preemptions/stalls
are included, and another request's prefill may overlap a decode interval.
Writes are buffered with at most a one-second flush delay while generating,
and flushed at request completion.

Hardware includes CPU busy/iowait, memory availability, per-GPU utilization,
VRAM and power, plus per-device disk and network counters. Disk counters are
**not a measurement of Lustre filesystem traffic**. Raw Prometheus samples
retain running/waiting requests, KV-cache occupancy, token/cache counters and
latency histograms. Client task/LLM concurrency is reconstructed from exact
start/end events into `concurrency.json`; server concurrency is sampled.

After drain, backend finished-request/token totals are checked against client
reports and vLLM generation-counter deltas. Missing backend events or count
mismatches invalidate the point, rather than substituting end-to-end throughput.
Sampling holes are explicit. Preserve hardware warnings with the run; a corrected
hardware warning alone is not proof of invalid data. Native external tool errors
(for example search-service rate limits) are not vLLM failures.

Every latency statistic includes its sample count, and trace coverage/admission
order is retained. Pool cycles may not finish during a window. One exploratory
repetition does not establish confidence intervals or an SLO capacity limit.

## Output

Each point contains `service.json`, `service.log`, `backend.jsonl`,
`inference-metrics.jsonl`, and `benchmark/` with its manifest, admissions,
concurrency timeline, replay-node metrics, task logs/events/reports, and summary.
The sweep produces `summary.csv` and `summary.md` after each completed point.

Optional plots (not an additional core runtime dependency):

```bash
uv pip install --python .venv/bin/python -e '.[analysis]'
.venv/bin/python examples/scaling/plot.py runs/scaling/single-backend-NEW
```

## Generic steady replay without the two-node orchestration

```bash
.venv/bin/agenttrace benchmark --trace-list POOL/traces.txt \
  --profile PROFILE.json --output-dir runs/benchmarks/NEW \
  --concurrency 2 --warmup-seconds 120 --duration-seconds 600 --seed 42 \
  --vllm-metrics-url http://GPU_NODE:8000/metrics \
  --backend-events /ABSOLUTE/SHARED/backend.jsonl
```

This generic command preserves an existing endpoint's cache. It does not
restart vLLM or create a remote hardware sampler. `--repeat` remains the finite
list mode and cannot be combined with `--duration-seconds`.
