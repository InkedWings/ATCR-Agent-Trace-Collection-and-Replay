# Real mini-SWE-agent Replay on Polaris

For multiple tasks and hardware/latency measurements, see
[Concurrent replay and instrumentation](concurrent-replay.md).

This workflow replays one complete mini-SWE-agent trace using two allocated
Polaris compute nodes:

```text
GPU node:  Qwen3-32B served by vLLM on four A100 GPUs
                         ^
                         | OpenAI-compatible requests
                         |
Tool node: AgentTrace DAG + one persistent mini-SWE Apptainer sandbox
```

The replay follows the recorded DAG. LLM request payloads are resent to vLLM;
the replay profile selects the served model, fixes the output length, enables
streaming usage, and requests `ignore_eos`. Generated text is consumed but not
saved and does not change the recorded path. Recorded bash commands execute in
order inside the native mini-SWE `SingularityEnvironment`.

## Prerequisites

- Both nodes belong to active allocations and can reach each other by hostname.
- The tool-node allocation supports `singularity_fakeroot=true`.
- Run `examples/minisweagent_swebench/scripts/bootstrap_python.sh` once from
  the repository to create `.venv`.
- A trace exists under `runs/minisweagent-swebench/.../trace.json` or another
  readable path.

No inference API token is needed for the local vLLM endpoint. A Hugging Face
token is only needed if the Qwen weights are not already cached.

## 1. Start vLLM on the GPU node

SSH to the node that will dedicate all four GPUs to inference:

```bash
cd /lus/eagle/projects/lc-mpi/ZhijingYe/SIGMETRICS_2027/trace_framework
./examples/minisweagent_swebench/scripts/vllm_qwen3_32b.sh serve
```

The server runs in the foreground and writes the same output to
`runs/vllm/qwen3-32b-<host>-<timestamp>.log`. `Ctrl-C` stops it. The default
configuration is:

- model: `Qwen/Qwen3-32B`;
- served model name: `qwen/qwen3-32b`;
- tensor parallel size: 4;
- context length: 32,768;
- maximum active sequences: 1;
- GPU memory utilization: 0.90.

Weights and vLLM caches are stored under
`/lus/eagle/projects/lc-mpi/ZhijingYe/Models`. The vLLM 0.19.1 Apptainer image
is reused from `/lus/eagle/projects/lc-mpi/ZhijingYe/Agentic/containers`.

## 2. Check the endpoint from the tool node

Replace `<gpu-node>` with the serving node hostname:

```bash
cd /lus/eagle/projects/lc-mpi/ZhijingYe/SIGMETRICS_2027/trace_framework
VLLM_BASE_URL=http://<gpu-node>:8000/v1 \
  ./examples/minisweagent_swebench/scripts/vllm_qwen3_32b.sh smoke
```

The smoke request checks `/models`, requests exactly 16 output tokens with
streaming usage enabled, and verifies that vLLM returns 16 completion tokens.

## 3. Replay one complete trace on the tool node

```bash
./examples/minisweagent_swebench/scripts/replay_qwen3_32b.sh \
  runs/minisweagent-swebench/<capture-run>/tasks/<task>/trace.json \
  http://<gpu-node>:8000/v1
```

The script runs in the foreground. It does not launch background workers, and
`Ctrl-C` interrupts the replay. Each invocation creates:

```text
runs/minisweagent-replay/
├── qwen3-32b-<task>-<timestamp>/       # replay workspace and private tmp
├── qwen3-32b-<task>-<timestamp>.json   # performance report
└── qwen3-32b-<task>-<timestamp>.log    # console log
```

Apptainer build files use node-local `/local/scratch`. The cache is shared by
replays on that node, while each replay gets its own temporary directory. The
private directory is bind-mounted at `/tmp` inside the container, so recorded
absolute `/tmp` paths preserve their original semantics without sharing state
between replays.

At completion the script prints:

- total nodes and replay timing;
- LLM call count and target/actual output-token totals;
- tool call count, native error count, and error-flag differences from capture.

An LLM token mismatch, endpoint/transport failure, container setup failure, or
executor exception terminates the replay. A normal tool result with a nonzero
exit code is recorded as `native_error=true` and replay continues, matching the
fixed-path performance semantics.

## Configuration overrides

The common overrides are:

```bash
MINISWE_REPLAY_RUN_ID=my-run \
APPTAINER_CACHEDIR=/local/scratch/$USER/existing-apptainer-cache \
MINISWE_REPLAY_LOCAL_SCRATCH=/local/scratch/$USER/my-replay \
./examples/minisweagent_swebench/scripts/replay_qwen3_32b.sh \
  path/to/trace.json http://<gpu-node>:8000/v1
```

The replay profile is
`examples/minisweagent_swebench/replay_profile_local_vllm.json`. Its
`model_override` maps the model captured in the trace to the served
`qwen/qwen3-32b` name. It also enables vLLM `ignore_eos`, allowing every replay
call to produce exactly the recorded output-token count.

## Verified run

The workflow was exercised end to end with `sqlfluff__sqlfluff-2419`:

- 60 of 60 nodes completed;
- 30 LLM calls produced 3,226 target and 3,226 actual output tokens;
- 30 bash calls executed through one persistent sandbox;
- all seven recorded tool error flags were reproduced with no extra mismatch.

Generated replay reports and logs remain under `runs/` and are not committed.
