# Trace Format Comparison: Hugging Face STS, OpenTelemetry, and AgentTrace

This document compares three trace representations using the same agent
execution. They serve different purposes:

- Hugging Face Session Trace Simple Format (STS) is a session transcript for
  storage, sharing, and visualization.
- OpenTelemetry is a runtime observability model based on timed spans.
- AgentTrace is an executable dependency graph for fixed-path performance
  replay.

The comparison uses Hugging Face's generic STS format, not the native session
formats that the Hugging Face Hub also recognizes for specific agents.

## One execution, three representations

Assume an agent performs the following operation:

```python
async def run_agent():
    response_1 = await llm_call(request_1)

    search_result, fetch_result = await asyncio.gather(
        web_search("A"),
        web_fetch("B"),
    )

    response_2 = await llm_call(request_2)
```

The first LLM call produces two tool calls. The tools run concurrently, and
the second LLM call starts only after both tools finish. Its execution DAG is:

```text
             +--> web_search("A") --+
             |                       |
LLM-1 -------+                       +--> LLM-2
             |                       |
             +--> web_fetch("B")  --+
```

Equivalently:

```text
web_search depends on LLM-1
web_fetch  depends on LLM-1
LLM-2      depends on both web_search and web_fetch
```

## Hugging Face STS: message order

Hugging Face STS stores one session header followed by message records in a
JSONL file:

```jsonl
{"type":"session","harness":"openclaw","id":"task-1"}
{"type":"message","message":{"role":"user","content":"Find information about A and fetch B."}}
{"type":"message","message":{"role":"assistant","content":"","toolCalls":[{"id":"call-1","function":{"name":"web_search","arguments":"{\"query\":\"A\"}"}},{"id":"call-2","function":{"name":"web_fetch","arguments":"{\"url\":\"B\"}"}}]}}
{"type":"message","message":{"role":"tool","toolCallId":"call-1","content":"search result"}}
{"type":"message","message":{"role":"tool","toolCallId":"call-2","content":"fetch result"}}
{"type":"message","message":{"role":"assistant","content":"final answer"}}
```

This representation explicitly records:

- the conversational order;
- the two tool requests made by the assistant;
- the association between each tool call and result through `toolCallId`;
- the assistant output text.

STS does not define an explicit dependency field. A converter can reasonably
interpret two tool calls in one assistant message as one batch, but STS itself
does not specify whether they must run concurrently or sequentially. It also
does not express the join before the final assistant message as:

```text
LLM-2 depends_on [web_search, web_fetch]
```

STS is therefore a good session transcript, but it is not by itself a replay
scheduler specification. Exact provider payloads, generation-length targets,
workspace restoration, tool runtime protocols, and dynamic artifact bindings
are also outside its required schema.

## OpenTelemetry: operation nesting and time

An instrumented version of the same execution might look conceptually like
this:

```python
with tracer.start_as_current_span("invoke_agent openclaw"):
    response_1 = await instrumented_llm_call(request_1)

    search_result, fetch_result = await asyncio.gather(
        instrumented_web_search("A"),
        instrumented_web_fetch("B"),
    )

    response_2 = await instrumented_llm_call(request_2)
```

OpenTelemetry assigns a new span's parent from the active span context. The
outer `invoke_agent` span remains active for the entire run. In contrast, the
first inference span ends when the first model response has been fully
received. When tool execution begins, the active enclosing span is once again
`invoke_agent`, not the completed inference span. The same is true when the
second inference begins.

The resulting span hierarchy is therefore commonly:

```text
invoke_agent openclaw
|-- LLM-1
|-- web_search
|-- web_fetch
`-- LLM-2
```

A simplified logical representation is:

```json
[
  {
    "span_id": "agent",
    "parent_span_id": null,
    "name": "invoke_agent openclaw",
    "start": 0,
    "end": 400
  },
  {
    "span_id": "llm-1",
    "parent_span_id": "agent",
    "name": "chat nemotron",
    "start": 0,
    "end": 100
  },
  {
    "span_id": "search",
    "parent_span_id": "agent",
    "name": "execute_tool web_search",
    "start": 110,
    "end": 210
  },
  {
    "span_id": "fetch",
    "parent_span_id": "agent",
    "name": "execute_tool web_fetch",
    "start": 110,
    "end": 300
  },
  {
    "span_id": "llm-2",
    "parent_span_id": "agent",
    "name": "chat nemotron",
    "start": 310,
    "end": 400
  }
]
```

The example is intentionally a simplified view of the span data model rather
than a complete OTLP export envelope.

### Why the four operation spans are siblings

A parent span means that an operation is part of an active enclosing
operation. It does not mean that the parent is the immediate execution
predecessor. Sibling spans may execute sequentially, concurrently, or in a
mixture of both.

Making the tool spans children of `LLM-1` would also distort the inference
span boundary. `LLM-1` is intended to measure the model request through receipt
of its response. Keeping it open while tools execute would mix model latency
with tool latency.

The join creates another problem for a pure parent tree: `LLM-2` depends on two
tools, while a span has only one `parent_span_id`. OpenTelemetry span links can
associate a span with multiple other spans, and custom attributes can carry an
explicit dependency list, but the standard GenAI span model does not define
those links as replay-scheduler dependencies.

The timestamps reveal the observed schedule:

```text
LLM-1:      [0, 100]
web_search:          [110, 210]
web_fetch:           [110,       300]
LLM-2:                              [310, 400]
```

From this schedule, a system such as XPerf can infer that the tools overlapped
and that `LLM-2` followed both. That is a graph reconstructed from timing and
span hierarchy, not a dependency graph directly represented by
`parent_span_id`.

## AgentTrace: explicit replay dependencies

AgentTrace stores the execution relationship directly in `depends_on`. The
same run is represented as a schema-v2 trace like this:

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
            {"role": "user", "content": "Find information about A and fetch B."},
            {
              "role": "assistant",
              "content": "",
              "tool_calls": [
                {
                  "id": "call-1",
                  "type": "function",
                  "function": {
                    "name": "web_search",
                    "arguments": "{\"query\":\"A\"}"
                  }
                },
                {
                  "id": "call-2",
                  "type": "function",
                  "function": {
                    "name": "web_fetch",
                    "arguments": "{\"url\":\"B\"}"
                  }
                }
              ]
            },
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

The scheduler semantics are unambiguous:

- `search` and `fetch` become ready after `llm-1` and can run concurrently.
- `llm-2` becomes ready only after both tools complete.
- No timestamps are needed to reconstruct this dependency graph.

AgentTrace additionally records the actual LLM provider request, the recorded
output-token target, the native tool request protocol, the initial workspace,
and artifact metadata needed by fixed-path replay. LLM response text is not
stored because newly generated text does not alter the recorded path.

## Summary

| Question | Hugging Face STS | OpenTelemetry | AgentTrace |
|---|---|---|---|
| Primary abstraction | Ordered messages | Timed spans | Executable DAG nodes |
| Tool/result pairing | `toolCalls[].id` and `toolCallId` | Optional GenAI attributes | Tool request and recorded result |
| Concurrency | Not explicitly specified | Observed through overlapping time intervals | Explicit sibling dependencies |
| Multi-parent join | Not explicitly specified | Not represented by one parent ID | Explicit dependency list |
| Exact provider request | Not required | Optional/custom instrumentation | Required for LLM nodes |
| Workspace and artifacts | Not defined | Not defined for replay | Part of replay context |
| Main use | Session viewing and sharing | Runtime observability | Fixed-path performance replay |

In short:

```text
Hugging Face STS records what appeared in the session.
OpenTelemetry records where operations ran and how long they took.
AgentTrace records what must complete before each replay operation can run.
```

The formats can be complementary. AgentTrace can remain the authoritative
replay representation, an HF exporter can provide a human-oriented session
view, and replay execution can emit OpenTelemetry spans for performance
analysis.

## References

- [Hugging Face Session Traces Format](https://huggingface.co/docs/hub/session-traces-format)
- [OpenTelemetry trace and span concepts](https://opentelemetry.io/docs/concepts/signals/traces/)
- [OpenTelemetry GenAI agent spans](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-agent-spans.md)
- [OpenTelemetry GenAI inference and tool spans](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md)
- [AgentTrace schema implementation](../src/agenttrace/schema.py)
