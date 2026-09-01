# Trace Representations for Agent Replay

This document compares Hugging Face Session Trace Simple Format (STS),
OpenTelemetry with OpenInference, and AgentTrace using one agent execution.
The comparison separates three questions that are easy to conflate:

1. What can the representation store?
2. What did the collector actually observe?
3. What execution semantics did an adapter add when converting the data?

A format does not discover dependencies. A collector observes framework events,
and an adapter converts those events into a representation. An explicit edge is
only as reliable as the framework event or declared policy that produced it.

## Running example

Assume the application itself defines this logical execution:

```python
async def run_agent():
    response_1 = await llm_call(request_1)

    search_result, fetch_result = await asyncio.gather(
        web_search("A"),
        web_fetch("B"),
    )

    response_2 = await llm_call(request_2)
```

The program tells us, independently of any trace format, that the two tools
have no dependency on each other and that the second LLM call waits for both:

```text
             +--> web_search("A") --+
             |                       |
LLM-1 -------+                       +--> LLM-2
             |                       |
             +--> web_fetch("B")  --+
```

The logical replay dependencies are therefore:

```text
web_search depends on LLM-1
web_fetch  depends on LLM-1
LLM-2      depends on web_search and web_fetch
```

This example starts with known application semantics. If a collector sees only
messages and results, it cannot recover the use of `asyncio.gather` merely from
their order in a file.

## Hugging Face STS: a session representation

Hugging Face STS is a JSONL session header followed by message records:

```jsonl
{"type":"session","harness":"openclaw","id":"task-1"}
{"type":"message","message":{"role":"user","content":"Find information about A and fetch B."}}
{"type":"message","message":{"role":"assistant","content":"","toolCalls":[{"id":"call-1","function":{"name":"web_search","arguments":"{\"query\":\"A\"}"}},{"id":"call-2","function":{"name":"web_fetch","arguments":"{\"url\":\"B\"}"}}]}}
{"type":"message","message":{"role":"tool","toolCallId":"call-1","content":"search result"}}
{"type":"message","message":{"role":"tool","toolCallId":"call-2","content":"fetch result"}}
{"type":"message","message":{"role":"assistant","content":"final answer"}}
```

STS explicitly represents:

- message order;
- assistant tool requests;
- tool-call/result pairing through `toolCalls[].id` and `toolCallId`;
- message content;
- optional per-message timestamps and model names.

STS does not define a dependency field or a replay scheduler contract. From the
file alone, two calls in one assistant message could be interpreted as a batch,
but STS does not state whether the harness executed that batch concurrently or
sequentially.

This does not prevent an adapter from converting STS into a logical DAG. For
example, an adapter may declare that all tool calls in one assistant message
are independent and generate the same sibling nodes as AgentTrace. That edge
structure then comes from the adapter policy, not from additional information
present in STS.

STS also does not require the following replay-specific data:

- the exact provider request payload;
- a recorded generation-length target;
- a native tool-executor protocol;
- workspace restoration;
- artifact and dynamic-path bindings.

STS is therefore a useful session interchange and visualization format. It may
also be an input to a replay converter, but it is not by itself a complete
fixed-path replay bundle.

## OpenTelemetry and OpenInference: operation telemetry

OpenTelemetry represents operations as spans. A span can contain a parent,
start and end timestamps, attributes, events, and links to other span
contexts. Instrumentation decides which operations become spans and how their
contexts are propagated.

OpenInference adds AI-specific semantic conventions on top of OpenTelemetry,
including LLM, agent, chain, and tool operation kinds and attributes for
messages, model parameters, tool calls, results, and token counts. It does not
choose an agent framework's span boundaries or define replay scheduling.

### Span hierarchy depends on instrumentation

One minimal instrumentation may keep only an agent-run span active around the
whole execution:

```text
invoke_agent
|-- LLM-1
|-- web_search
|-- web_fetch
`-- LLM-2
```

Another instrumentation may add turn or graph-node spans:

```text
invoke_agent
|-- turn-1
|   |-- LLM-1
|   |-- web_search
|   `-- web_fetch
`-- turn-2
    `-- LLM-2
```

Neither hierarchy is mandated by OpenTelemetry or OpenInference. A parent-child
relationship describes the nesting or propagated causal context chosen by the
instrumentation. It should not automatically be interpreted as "the parent
must finish before the child can start" in a replay scheduler.

### Links can carry non-tree relationships

Each span has at most one parent, but OpenTelemetry supports links to multiple
other spans. Instrumentation can therefore encode the running example as:

```json
[
  {
    "span_id": "llm-1",
    "name": "chat",
    "links": []
  },
  {
    "span_id": "search",
    "name": "execute_tool web_search",
    "links": [
      {"span_id": "llm-1", "attributes": {"agenttrace.relation": "depends_on"}}
    ]
  },
  {
    "span_id": "fetch",
    "name": "execute_tool web_fetch",
    "links": [
      {"span_id": "llm-1", "attributes": {"agenttrace.relation": "depends_on"}}
    ]
  },
  {
    "span_id": "llm-2",
    "name": "chat",
    "links": [
      {"span_id": "search", "attributes": {"agenttrace.relation": "depends_on"}},
      {"span_id": "fetch", "attributes": {"agenttrace.relation": "depends_on"}}
    ]
  }
]
```

The custom link attribute is necessary because OpenTelemetry links imply an
association or causal relationship but do not standardize a fixed-path replay
meaning. With an agreed replay profile, the links can be converted directly to
`depends_on`; no timestamp inference is needed for those declared edges.

### Timestamps describe an observed schedule

Suppose operation spans contain these intervals:

```text
LLM-1:      [0, 100]
web_search:          [110, 210]
web_fetch:           [110,       300]
LLM-2:                              [310, 400]
```

The overlap shows that `web_search` and `web_fetch` ran concurrently in this
execution. It also shows that `LLM-2` started after both tools completed.
Timestamps alone do not prove that `LLM-2` had a required data dependency on
both tools. An unrelated operation or a single-worker executor can produce the
same observed ordering without a logical dependency.

Timing analysis is useful when the goal is to reproduce the observed schedule,
or as a fallback when framework causality is unavailable. It should not be
presented as proof of a logical dependency.

For workload-DAG replay, the preferred dependency sources are explicit
framework dispatch/await events, tool-call and result identifiers, graph edges,
or replay-typed span links. Timestamps remain valuable performance data but do
not have to determine the DAG.

## AgentTrace: an executable replay representation

AgentTrace stores a normalized replay graph. In the running example, its core
shape is:

```json
{
  "schema_version": 2,
  "trace_id": "task-1",
  "source": {
    "framework": "openclaw",
    "workload": "example",
    "version": "2026.7"
  },
  "context": {
    "captured_workspace_root": "/captured/workspace",
    "captured_tmp_root": "/tmp",
    "workspace_seed": null
  },
  "artifacts": [],
  "nodes": [
    {
      "id": "llm-1",
      "type": "llm",
      "depends_on": [],
      "request": {
        "protocol": "openai-chat-completions",
        "endpoint": "/chat/completions",
        "payload": {
          "model": "nemotron-3-ultra",
          "messages": [
            {"role": "user", "content": "Find information about A and fetch B."}
          ]
        }
      },
      "output_tokens": 24
    },
    {
      "id": "search",
      "type": "tool",
      "depends_on": ["llm-1"],
      "request": {
        "protocol": "openclaw-tools-invoke",
        "name": "web_search",
        "arguments": {"query": "A"}
      },
      "recorded_result": {
        "toolCallId": "call-1",
        "toolName": "web_search",
        "content": [],
        "details": {},
        "isError": false
      }
    },
    {
      "id": "fetch",
      "type": "tool",
      "depends_on": ["llm-1"],
      "request": {
        "protocol": "openclaw-tools-invoke",
        "name": "web_fetch",
        "arguments": {"url": "B"}
      },
      "recorded_result": {
        "toolCallId": "call-2",
        "toolName": "web_fetch",
        "content": [],
        "details": {},
        "isError": false
      }
    },
    {
      "id": "llm-2",
      "type": "llm",
      "depends_on": ["search", "fetch"],
      "request": {
        "protocol": "openai-chat-completions",
        "endpoint": "/chat/completions",
        "payload": {
          "model": "nemotron-3-ultra",
          "messages": [
            {"role": "tool", "tool_call_id": "call-1", "content": "search result"},
            {"role": "tool", "tool_call_id": "call-2", "content": "fetch result"}
          ]
        }
      },
      "output_tokens": 32
    }
  ]
}
```

The file gives the replay engine an unambiguous scheduling contract:

- `search` and `fetch` become ready after `llm-1`;
- both may run concurrently because neither depends on the other;
- `llm-2` becomes ready after both complete.

AgentTrace does not discover these edges. A collection adapter must supply
them from framework semantics or a declared conversion policy.

### Dependency provenance in the current OpenClaw adapter

The current OpenClaw adapter constructs turn-level dependencies as follows:

1. Pair captured provider calls with assistant events using `responseId`, with
   sequence order as a fallback.
2. Pair tool calls and results using `toolCallId`.
3. Make every tool call in an assistant turn depend on that turn's LLM call.
4. Make the next LLM call depend on all tools from the preceding turn.
5. Treat multiple tools in one assistant turn as independent siblings.

The first four rules recover the agent's turn structure. The fifth rule is the
declared replay policy implemented by the adapter; it is not evidence that the
original OpenClaw runtime executed those tools concurrently. If tools have a
required ordering through shared state, artifacts, or framework control flow,
the adapter must add the corresponding edge.

AgentTrace additionally requires replay data that STS and generic
OpenTelemetry/OpenInference instrumentation do not require, including the
actual provider request payload, the recorded output-token target, the native
tool protocol, workspace restoration, and artifact bindings. Those fields are
the reason to keep AgentTrace as a compact replay format or internal replay IR.

## Correct comparison

| Question | Hugging Face STS | OpenTelemetry + OpenInference | AgentTrace |
|---|---|---|---|
| Primary abstraction | Ordered session messages | Instrumented operation spans | Executable replay nodes |
| Who chooses captured structure? | Harness | Instrumentation | Collection adapter |
| Tool/result pairing | `toolCalls[].id` and `toolCallId` | GenAI/OpenInference attributes when instrumented | Required tool request and recorded result |
| Explicit logical dependency | No standard field | Possible with links, but replay meaning is not standardized | Required `depends_on` field |
| Observed concurrency | Not generally available; message timestamps are optional points | Span intervals show observed overlap | Not represented by the current schema |
| Multi-input join | Can be derived by an adapter policy | Can be associated through multiple links | Explicit dependency list |
| Exact provider request | Not required | Instrumentation-dependent | Required for LLM nodes |
| Workspace and artifact replay | Not defined | Not defined by generic tracing conventions | Part of the replay bundle |
| Main use | Session interchange and viewing | Runtime observability and semantic telemetry | Fixed-path workload-DAG replay |

The most accurate summary is:

```text
STS represents what appeared in a session.
OpenTelemetry/OpenInference represents instrumented operations, timing, and
declared associations.
AgentTrace materializes adapter-supplied dependencies and replay state as an
executable workload graph.
```

The representations can be complementary. STS can provide a session view;
OpenTelemetry/OpenInference can be a collection, interchange, or replay-output
layer; and AgentTrace can remain the compact replay IR. Converters between
them must document which relationships were observed and which were introduced
by policy.

## References

- [Hugging Face Session Traces Format](https://huggingface.co/docs/hub/session-traces-format)
- [OpenTelemetry trace and span concepts](https://opentelemetry.io/docs/concepts/signals/traces/)
- [OpenTelemetry Trace API](https://opentelemetry.io/docs/specs/otel/trace/api/)
- [OpenInference specification](https://arize-ai.github.io/openinference/spec/)
- [AgentTrace schema implementation](../src/agenttrace/schema.py)
