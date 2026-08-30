# OpenClaw + GAIA on Polaris

This example collects exact OpenClaw execution into AgentTrace schema v2 and
replays it through an OpenAI-compatible inference endpoint plus OpenClaw's
official `POST /tools/invoke` API. GAIA answers, credentials, OpenClaw state,
raw runs, and generated datasets stay outside Git.

Run commands below from the AgentTrace repository root.

## 1. Install

```bash
./examples/openclaw_gaia/scripts/bootstrap_python.sh
./examples/openclaw_gaia/scripts/install_openclaw.sh
```

The pinned versions are OpenClaw `2026.7.1-2`, its Brave plugin, and Node
`24.19.0`. The bootstrap installs AgentTrace editable into `.venv` and adds the
small GAIA preparation dependencies.

## 2. Credentials

Never paste a key into source, profiles, or traces. Brave Search is loaded from
`~/.config/sigmetrics-2027/openclaw-gaia.env`; the short-lived ALCF token is
obtained from the Globus-backed `alcf-ai` cache.

```bash
./examples/openclaw_gaia/scripts/set_brave_key.sh
./examples/openclaw_gaia/scripts/login_alcf.sh
./examples/openclaw_gaia/scripts/auth_status.sh
./examples/openclaw_gaia/scripts/preflight_services.sh
```

The scripts only report credential presence and expiry. They never print or
persist token values.

## 3. Prepare GAIA

The GAIA validation split is gated on Hugging Face. Authenticate the local `hf`
CLI once, then prepare either the deterministic pilot or the first 30 source
rows:

```bash
./examples/openclaw_gaia/scripts/prepare_gaia_pilot.sh
./examples/openclaw_gaia/scripts/prepare_gaia_first30.sh
```

Files are placed under the ignored `data/gaia/` tree. The generated manifests
contain task IDs, questions, levels, attachment paths, and source revision, but
no benchmark answer field.

## 4. Collect

Submit the pilot to the Polaris debug queue:

```bash
mkdir -p runs/pbs-logs
qsub examples/openclaw_gaia/pbs/collect_pilot_debug.pbs
```

Override the slice at submission when needed:

```bash
qsub -v GAIA_START_INDEX=0,GAIA_MAX_TASKS=2 \
  examples/openclaw_gaia/pbs/collect_pilot_debug.pbs
```

From an already allocated compute node, collect the first 30 rows directly:

```bash
GAIA_RUN_ID=first30 ./examples/openclaw_gaia/scripts/run_gaia_first30_direct.sh
```

Collection is serial. Every case receives an independent OpenClaw session and
workspace. Before OpenClaw starts, the collector copies `workspace_seed/`.
After execution it retains the native trajectory and immediately copies any
available `details.fullOutputPath` spill file. The provider proxy stores the
actual request payload plus response ID and completion-token count, but no
authorization headers or response text.

Each task directory includes:

```text
workspace_seed/          initial state used by replay
workspace/               framework execution workspace
trajectory.jsonl         native OpenClaw trajectory
artifacts/               copied tool spill files, when present
trace.json               unified schema-v2 trace
openclaw.stdout.json
openclaw.stderr.log
status.json
```

Polaris compute nodes must use the site proxy for both ALCF and public HTTPS.
The supplied launchers deliberately set `NO_PROXY=localhost,127.0.0.1`; do not
inherit a login-node `*.alcf.anl.gov` bypass.

## 5. Build GAIA_Trace

The builder retains unavailable index rows, reconstructs only the initial
workspace, copies all referenced GAIA attachments, and rebuilds traces from the
native trajectory plus provider capture:

```bash
uv run python examples/openclaw_gaia/build_gaia_trace_dataset.py \
  --manifest data/gaia/pilots/validation_first30.jsonl \
  --gaia-dir data/gaia \
  --source-run runs/openclaw-gaia/<first-run> \
  --source-run runs/openclaw-gaia/<continuation-run> \
  --output datasets/GAIA_Trace
```

The output path must not already exist. Historical spill files that vanished
with a compute node are not fabricated; the persisted native result still
contains its recorded path, and a live replay regenerates the new path.

Validate the complete dataset:

```bash
for trace in datasets/GAIA_Trace/traces/*.json; do
  uv run agenttrace validate "$trace"
done
(cd datasets/GAIA_Trace && sha256sum -c checksums.sha256)
```

## 6. Replay one trace

Replay uses one long-lived OpenClaw Gateway per trace. It binds only to
loopback, generates an ephemeral bearer token, uses one session key, points at
the isolated replay workspace, and enables the six observed tools:
`web_search`, `web_fetch`, `exec`, `read`, `write`, and `edit`.

OpenClaw 2026.7 exposes the web tools directly through `/tools/invoke`, but its
Gateway resolver omits coding tools used during normal agent turns. AgentTrace
therefore creates a temporary replay-only Gateway plugin: it imports the public
OpenClaw `agent-sessions` SDK, registers aliases for the native
`read`/`write`/`edit`/`bash` implementations, and maps recorded `exec` to native
`bash`. All six tools are still invoked with `POST /tools/invoke`; no filesystem
or shell tool is reimplemented. The plugin, Gateway state, and ephemeral token
live only in the replay run directory, and the Gateway is closed after replay.

```bash
source examples/openclaw_gaia/scripts/load_credentials.sh
export OPENCLAW_BIN="$PWD/.tools/openclaw/bin/openclaw"
export NO_PROXY=localhost,127.0.0.1
export no_proxy="$NO_PROXY"

uv run agenttrace replay \
  datasets/GAIA_Trace/traces/001_c61d22de-5f6c-4958-a7f6-5e9707bd3466.json \
  --profile examples/openclaw_gaia/replay_profile.json \
  --run-dir replay-runs/gaia-001 \
  --output replay-runs/gaia-001.json
```

For the dataset already built beside this repository in the SIGMETRICS
workspace, replace `datasets/GAIA_Trace` with `../datasets/GAIA_Trace`.

Useful real acceptance cases are:

- trace 001 for `web_search`/`web_fetch`;
- trace 010 for `write`/`edit`/`exec` side effects;
- trace 006 for an XLSX attachment plus `exec`.

Gateway startup is reported as setup latency, not as tool-node time. The replay
report contains node timing, native error flags, and LLM target/actual token
counts. It does not contain generated model text. A native tool result with
`isError: true` completes normally; HTTP, authentication, executor, or LLM
transport failure stops the trace without retry.
