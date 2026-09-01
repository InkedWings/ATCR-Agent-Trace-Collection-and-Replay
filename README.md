# AgentTrace

AgentTrace records framework-native agent execution as one portable DAG and
replays that recorded path for performance experiments. The core does not know
how OpenClaw, a coding agent, or a scientific workflow implements tools. Each
framework supplies a collection adapter and one long-lived native tool
executor.

Replay is deliberately fixed-path:

- LLM nodes resend the exact provider request captured at the model boundary.
- The recorded output-token count becomes the new generation-length target.
- Replay consumes the stream, checks the actual token count, and discards text.
- Tool nodes execute the recorded name and arguments through the framework's
  native runtime; their new outputs never alter the recorded DAG.
- Sibling tool calls run concurrently and dependent nodes join on all parents.

The first implementation includes OpenAI-compatible LLM capture/replay and the
OpenClaw `/tools/invoke` adapter. OpenClaw's web tools are directly visible on
that endpoint. For coding tools, the executor creates a replay-only plugin that
aliases the public OpenClaw SDK implementations of `read`, `write`, `edit`, and
`bash` onto the same endpoint (`exec` maps to `bash`); AgentTrace contains no
duplicate tool implementation. Multi-trace load generation, mini-SWE-agent,
ChemGraph, retries, and distributed execution are intentionally outside v0.1.

## Install and test

AgentTrace requires Python 3.11 or newer. `uv` is convenient but not required.

```bash
uv sync --extra test
uv run pytest -q
uv run agenttrace validate examples/synthetic/trace.json
uv run agenttrace replay examples/synthetic/trace.json --dry-run \
  --run-dir replay-runs/synthetic
```

The synthetic dry-run needs no service or credential. An editable pip install
also works:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e '.[test]'
```

## Repository layout

```text
src/agenttrace/
├── schema.py                 # schema-v2 validation
├── loader.py                 # module:factory profile loading
├── capture/openai.py         # request/usage capture proxy
├── replay/engine.py          # single-trace DAG scheduler
├── replay/llm.py             # OpenAI-compatible LLM executor
├── replay/bindings.py        # workspace/tmp/resource remapping
└── adapters/openclaw.py      # collection conversion + native tools
examples/openclaw_gaia/       # complete Polaris collection/replay example
docs/trace-format-comparison.md # STS, OTel/OpenInference, and AgentTrace
tests/                        # unit and fake-service integration tests
```

Generated runs, framework state, gated input data, datasets, and credentials
are ignored by this Git repository.

## Trace schema v2

Each trace is one JSON object. LLM response text is absent; native persisted
tool results are retained.

```json
{
  "schema_version": 2,
  "trace_id": "task-id",
  "source": {
    "framework": "openclaw",
    "workload": "GAIA",
    "version": "OpenClaw 2026.7.1-2"
  },
  "context": {
    "captured_workspace_root": "/captured/workspace",
    "captured_tmp_root": "/tmp",
    "workspace_seed": "../workspace_seeds/task-id"
  },
  "artifacts": [],
  "nodes": [
    {
      "id": "llm-001",
      "type": "llm",
      "depends_on": [],
      "request": {
        "protocol": "openai-chat-completions",
        "endpoint": "/chat/completions",
        "payload": {"model": "model-id", "messages": []}
      },
      "output_tokens": 128
    },
    {
      "id": "tool-001",
      "type": "tool",
      "depends_on": ["llm-001"],
      "request": {
        "protocol": "openclaw-tools-invoke",
        "name": "web_fetch",
        "arguments": {"url": "https://example.org"}
      },
      "recorded_result": {
        "toolCallId": "call-id",
        "toolName": "web_fetch",
        "content": [],
        "details": {},
        "isError": false
      }
    }
  ]
}
```

`agenttrace validate` rejects duplicate nodes, unknown dependencies, cycles,
incomplete LLM/tool records, and every schema version other than v2.

## Replay profiles

Executors are selected with ordinary Python `module:factory` references. There
is no registry or entry-point layer.

```json
{
  "llm_executor": {
    "factory": "agenttrace.replay.llm:create_openai_executor",
    "config": {
      "base_url": "https://inference.example/v1",
      "api_key_env": "INFERENCE_TOKEN",
      "ignore_eos": true
    }
  },
  "tool_executor": {
    "factory": "my_adapter:create_tool_executor",
    "config": {}
  }
}
```

```bash
agenttrace replay path/to/trace.json --profile profile.json \
  --run-dir replay-runs/one-trace --output replay-runs/one-trace.json
```

The report contains setup latency, replay makespan, node start time and
duration, node status, LLM target/actual tokens, and native tool error state.
It never contains newly generated LLM text.

During replay AgentTrace restores the initial workspace seed and referenced
artifacts into `workspace/`, creates a private `tmp/`, and rewrites strings in
tool arguments. It maps the captured workspace and `/tmp` roots immediately.
When a native result returns a new `details.fullOutputPath`, session ID, or
resource ID, the binding is learned for later recorded nodes, including paths
embedded in shell command strings.

## Adding another framework

Add four framework-specific pieces without changing the schema or replay core:

1. a `CollectionAdapter` that runs one case and preserves its native log;
2. a converter that emits schema-v2 nodes and initial workspace metadata;
3. a long-lived `ToolExecutor` that calls the framework's own tool runtime;
4. a replay profile pointing at the executor factory.

The extension protocols live in `agenttrace.interfaces`. See
`examples/openclaw_gaia/` for a complete implementation.
