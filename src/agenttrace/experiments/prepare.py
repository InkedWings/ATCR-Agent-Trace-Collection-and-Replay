"""Freeze pools after schema, path, protocol and exact serving-tokenizer checks."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx

from agenttrace.schema import load_trace


async def prepare(paths: list[Path], output: Path, base_url: str,
                  model: str = "qwen/qwen3-32b", max_model_len: int = 32768) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    async with httpx.AsyncClient(base_url=base_url.removesuffix("/v1").rstrip("/"),
                                 trust_env=False, timeout=120) as client:
        for path in sorted(paths):
            path = path.resolve()
            trace = load_trace(path)  # malformed data is an error, not context filtering
            framework = trace["source"]["framework"]
            seed = trace["context"].get("workspace_seed")
            if seed and not (path.parent / seed).is_dir():
                raise FileNotFoundError(f"missing workspace seed: {path}")
            for artifact in trace["artifacts"]:
                if not (path.parent / artifact["path"]).is_file():
                    raise FileNotFoundError(f"missing artifact {artifact['id']}: {path}")
            calls, overflow = [], []
            for node in trace["nodes"]:
                request = node["request"]
                if node["type"] == "tool":
                    supported = ({"web_search", "web_fetch", "exec", "read", "write", "edit"}
                        if framework == "openclaw" else {"bash"})
                    protocol = "openclaw-tools-invoke" if framework == "openclaw" else "minisweagent-singularity"
                    if request["protocol"] != protocol or request["name"] not in supported:
                        raise ValueError(f"unsupported tool in {path}: {node['id']}")
                    continue
                if request["protocol"] != "openai-chat-completions" or request["endpoint"] != "/chat/completions":
                    raise ValueError(f"unsupported LLM request in {path}: {node['id']}")
                payload = request["payload"]
                # Same messages/tools and template overrides as replay. Sampling
                # parameters do not change tokenization; server supplies its template.
                body = {key: payload[key] for key in ("messages", "tools", "tool_choice",
                    "chat_template", "chat_template_kwargs", "add_generation_prompt",
                    "continue_final_message") if key in payload}
                body["model"] = model
                response = await client.post("/tokenize", json=body)
                response.raise_for_status()
                count = response.json()["count"]
                call = {"node_id": node["id"], "prompt_tokens": count,
                    "target_output_tokens": node["output_tokens"],
                    "total_tokens": count + node["output_tokens"]}
                calls.append(call)
                if call["total_tokens"] > max_model_len:
                    overflow.append(call)
            row = {"trace_path": str(path), "trace_id": trace["trace_id"], "eligible": not overflow,
                "llm_calls": calls, "exclusion_reason": "context_overflow" if overflow else None,
                "overflow_nodes": overflow}
            rows.append(row)
            print(f"Preflight {path.parent.name if path.name == 'trace.json' else path.stem}: "
                  f"{'excluded' if overflow else 'eligible'}; max prompt+output={max(c['total_tokens'] for c in calls)}", flush=True)
    eligible = [row["trace_path"] for row in rows if row["eligible"]]
    report = {"model": model, "max_model_len": max_model_len, "tokenizer_endpoint": base_url,
        "candidate_count": len(rows), "eligible_count": len(eligible),
        "policy": "exclude whole trace on any context overflow; no truncation or answer grading", "traces": rows}
    (output / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "traces.txt").write_text("".join(path + "\n" for path in eligible))
    if not eligible:
        raise ValueError("no eligible traces")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    for workload, paths in (
        ("openclaw", list((repo.parent / "datasets/GAIA_Trace/run2/traces").glob("*.json"))),
        ("minisweagent", list((repo / "runs/minisweagent-swebench/dev-20260906T004724Z/tasks").glob("*/trace.json"))),
    ):
        asyncio.run(prepare(paths, args.output / workload, args.base_url))


if __name__ == "__main__":
    main()
