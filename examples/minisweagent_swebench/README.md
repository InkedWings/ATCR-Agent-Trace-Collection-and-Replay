# mini-SWE-agent + SWE-bench Lite

This example collects framework-native mini-SWE-agent trajectories and
AgentTrace schema-v2 replay graphs for the 23 cases in the SWE-bench Lite `dev`
split. Cases run sequentially (`workers=1`).

Each case uses mini-SWE-agent's native `SingularityEnvironment` and the official
per-instance SWE-bench image from `docker.io/swebench`. On Polaris the launch
script loads Apptainer 1.4.1, places its temporary files and image cache under
node-local `/local/scratch`, and invokes the environment with its standard
`--fakeroot --writable` execution mode.

## Setup

From the AgentTrace repository:

```bash
./examples/minisweagent_swebench/scripts/bootstrap_python.sh
./examples/openclaw_gaia/scripts/login_alcf.sh
```

The ALCF access token stays in the process environment and is not written to a
trajectory or trace. The collector stores no SWE-bench gold patch or test
patch.

## Collect the dev split

On an allocated compute node:

```bash
./examples/minisweagent_swebench/scripts/run_dev_direct.sh
```

The launch defaults follow mini-SWE-agent 2.4.5's built-in `swebench.yaml`:
250 agent steps, a 60-second shell-command timeout, and no agent wall-time
limit. The cost limit is set to zero because the ALCF endpoint does not report
a meaningful API price. Tasks and traces are collected sequentially.

For a new PBS allocation, ALCF recommends requesting
`singularity_fakeroot=true`. An existing allocation can be checked with:

```bash
module use /soft/modulefiles
module load spack-pe-base apptainer
apptainer version
```

Useful sequential resume controls are:

```bash
MINISWE_START_INDEX=5 MINISWE_MAX_TASKS=18 \
MINISWE_RUN_ID=dev-from-6 \
./examples/minisweagent_swebench/scripts/run_dev_direct.sh
```

Each completed case contains:

```text
runs/minisweagent-swebench/<run-id>/tasks/<index>_<instance-id>/
├── trajectory.json
├── trace.json
└── status.json
```

The native trajectory is retained. The writable container sandbox is temporary;
replay rebuilds it from the recorded per-instance image URI.

## Dependency semantics

mini-SWE-agent calls the model, then executes the returned bash actions with a
sequential list comprehension. The adapter therefore records:

```text
LLM -> bash-1 -> bash-2 -> next LLM
```

This is framework control flow, not a dependency inferred from timestamps.

## Replay one trace

Load the current ALCF token, then replay:

```bash
module use /soft/modulefiles
module load spack-pe-base apptainer
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy="$http_proxy" HTTP_PROXY="$http_proxy" HTTPS_PROXY="$http_proxy"
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1
export MSWEA_SINGULARITY_EXECUTABLE="$(command -v apptainer)"
export TMPDIR=/local/scratch/$USER/agenttrace-miniswe-replay/tmp
export APPTAINER_TMPDIR=/local/scratch/$USER/agenttrace-miniswe-replay/apptainer-tmp
export APPTAINER_CACHEDIR=/local/scratch/$USER/agenttrace-miniswe-replay/apptainer-cache
mkdir -p "$TMPDIR" "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
source examples/minisweagent_swebench/scripts/load_credentials.sh
agenttrace replay path/to/trace.json \
  --profile examples/minisweagent_swebench/replay_profile.json \
  --run-dir replay-runs/minisweagent-one \
  --output replay-runs/minisweagent-one.json
```

Container construction is setup work and is excluded from replay makespan.
Bash commands execute through one persistent mini-SWE-agent
`SingularityEnvironment` sandbox.
