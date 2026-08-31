#!/usr/bin/env python3
"""Build the answer-free GAIA_Trace schema-v2 dataset from raw OpenClaw runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agenttrace.adapters.openclaw import build_openclaw_trace
from agenttrace.schema import write_trace

INITIAL_WORKSPACE_FILES = (
    "AGENTS.md",
    "SOUL.md",
    "TOOLS.md",
    "IDENTITY.md",
    "USER.md",
    "HEARTBEAT.md",
    "BOOTSTRAP.md",
    "prompt.txt",
)
BOOTSTRAP_TEXT = "This benchmark workspace is already initialized; do not run onboarding.\n"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        # JSON strings may contain Unicode line separators; JSONL uses only
        # the ASCII newline byte as its record delimiter.
        for line in path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def source_version(run: Path) -> str:
    value = read_json(run / "metadata.json").get("openclaw")
    return str(value or "unknown")


def build_workspace_seed(task_dir: Path, destination: Path) -> None:
    saved_seed = task_dir / "workspace_seed"
    source = saved_seed if saved_seed.is_dir() else task_dir / "workspace"
    destination.mkdir(parents=True)
    for name in INITIAL_WORKSPACE_FILES:
        candidate = source / name
        target = destination / name
        if candidate.is_file():
            shutil.copy2(candidate, target)
        elif name == "BOOTSTRAP.md":
            target.write_text(BOOTSTRAP_TEXT, encoding="utf-8")
        else:
            raise FileNotFoundError(f"initial workspace file is missing: {candidate}")


def copy_input_attachment(
    gaia_dir: Path, attachment: str, output: Path, copied: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    source = (gaia_dir / attachment).resolve()
    if not source.is_file() or not source.is_relative_to(gaia_dir.resolve()):
        raise FileNotFoundError(f"GAIA attachment is missing: {source}")
    destination = output / "artifacts" / "gaia" / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
    metadata = {
        "path": destination.relative_to(output).as_posix(),
        "sha256": sha256(destination),
    }
    copied[attachment] = metadata
    return metadata


def existing_tool_artifacts(
    task_dir: Path, output: Path, number: int
) -> list[dict[str, Any]]:
    source_trace = read_json(task_dir / "trace.json")
    if source_trace.get("schema_version") != 2:
        return []
    copied: list[dict[str, Any]] = []
    for artifact in source_trace.get("artifacts", []):
        if artifact.get("kind") != "tool_output":
            continue
        source = (task_dir / artifact["path"]).resolve()
        if not source.is_file():
            continue
        destination = output / "artifacts" / "tool_outputs" / f"{number:03d}" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(
            {
                **artifact,
                "path": Path("..", destination.relative_to(output)).as_posix(),
                "sha256": sha256(destination),
            }
        )
    return copied


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gaia-dir", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    gaia_dir = args.gaia_dir.resolve()
    source_runs = [path.resolve() for path in args.source_run]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"dataset output already exists: {output}")
    tasks = read_jsonl(manifest_path)[: args.limit]
    if len(tasks) != args.limit:
        raise ValueError(f"expected {args.limit} manifest rows, found {len(tasks)}")

    output.mkdir(parents=True)
    (output / "traces").mkdir()
    (output / "workspace_seeds").mkdir()
    index_rows: list[dict[str, Any]] = []
    copied_attachments: dict[str, dict[str, Any]] = {}
    total_llm = 0
    total_tools = 0
    total_tool_results = 0
    total_output_tokens = 0

    for zero_index, task in enumerate(tasks):
        number = zero_index + 1
        task_id = str(task["task_id"])
        candidates: list[tuple[Path, Path]] = []
        for run in source_runs:
            candidates.extend(
                (run, task_dir)
                for task_dir in (run / "tasks").glob(f"{number:03d}_{task_id}")
            )
        traced = [
            item
            for item in candidates
            if (item[1] / "trajectory.jsonl").is_file()
            and (item[1] / "trace.json").is_file()
            and (item[0] / "llm_calls.jsonl").is_file()
        ]
        if len(traced) > 1:
            hashes = {sha256(task_dir / "trace.json") for _, task_dir in traced}
            if len(hashes) != 1:
                raise ValueError(f"conflicting traces for task {number}: {task_id}")
        selected = traced[0] if traced else None
        observed = selected or (candidates[-1] if candidates else None)
        status = read_json(observed[1] / "status.json") if observed else {}
        attachment = task.get("attachment")
        attachment_metadata = None
        if attachment:
            attachment_metadata = copy_input_attachment(
                gaia_dir, str(attachment), output, copied_attachments
            )

        row: dict[str, Any] = {
            "index": number,
            "task_id": task_id,
            "split": "validation",
            "level": task["level"],
            "attachment": attachment_metadata,
            "available": selected is not None,
            "trace_path": None,
            "workspace_seed_path": None,
            "source_run": observed[0].name if observed else None,
            "agent_status": status.get("agent_status", status.get("status")),
        }
        if selected is None:
            row["missing_reason"] = status.get("replay_trace_error") or "trace absent"
            index_rows.append(row)
            continue

        run, task_dir = selected
        relative_name = f"{number:03d}_{task_id}"
        seed_relative = Path("workspace_seeds") / relative_name
        build_workspace_seed(task_dir, output / seed_relative)
        artifacts = existing_tool_artifacts(task_dir, output, number)
        if attachment and attachment_metadata:
            captured_attachment = task_dir / "workspace" / "attachments" / Path(str(attachment)).name
            artifacts.insert(
                0,
                {
                    "id": "gaia-attachment",
                    "kind": "input",
                    "captured_path": str(captured_attachment),
                    "path": Path("..", attachment_metadata["path"]).as_posix(),
                    "workspace_path": f"attachments/{Path(str(attachment)).name}",
                    "sha256": attachment_metadata["sha256"],
                },
            )

        trace = build_openclaw_trace(
            trace_id=task_id,
            trajectory_path=task_dir / "trajectory.jsonl",
            capture_path=run / "llm_calls.jsonl",
            workload="GAIA",
            framework_version=source_version(run),
            captured_workspace_root=str(task_dir / "workspace"),
            workspace_seed=Path("..", seed_relative).as_posix(),
            artifacts=artifacts,
        )
        trace_relative = Path("traces") / f"{relative_name}.json"
        write_trace(output / trace_relative, trace)
        llm_nodes = [node for node in trace["nodes"] if node["type"] == "llm"]
        tool_nodes = [node for node in trace["nodes"] if node["type"] == "tool"]
        output_tokens = sum(node["output_tokens"] for node in llm_nodes)
        total_llm += len(llm_nodes)
        total_tools += len(tool_nodes)
        total_tool_results += sum("recorded_result" in node for node in tool_nodes)
        total_output_tokens += output_tokens
        row.update(
            {
                "trace_path": trace_relative.as_posix(),
                "workspace_seed_path": seed_relative.as_posix(),
                "source_run": run.name,
                "sha256": sha256(output / trace_relative),
                "llm_nodes": len(llm_nodes),
                "tool_nodes": len(tool_nodes),
                "persisted_tool_results": len(tool_nodes),
                "output_tokens": output_tokens,
            }
        )
        index_rows.append(row)

    index_path = output / "index.jsonl"
    index_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in index_rows),
        encoding="utf-8",
    )
    missing = [row["index"] for row in index_rows if not row["available"]]
    metadata = {
        "name": "GAIA_Trace",
        "version": 2,
        "trace_schema_version": 2,
        "benchmark": "GAIA 2023",
        "split": "validation",
        "selection": f"first {args.limit} rows in source order",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "records_planned": len(index_rows),
        "records_available": len(index_rows) - len(missing),
        "records_missing": len(missing),
        "missing_indices": missing,
        "llm_nodes": total_llm,
        "tool_nodes": total_tools,
        "persisted_tool_results": total_tool_results,
        "output_tokens": total_output_tokens,
        "attachments": len(copied_attachments),
        "manifest_sha256": sha256(manifest_path),
        "source_runs": [run.name for run in source_runs],
        "ground_truth_answers_included": False,
        "llm_response_text_included": False,
        "credentials_included": False,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        """# GAIA_Trace

Framework-independent performance traces for the first 30 GAIA 2023
validation rows, collected with OpenClaw and Nemotron-3-Ultra. The dataset uses
AgentTrace schema v2. It stores exact LLM requests and output-token counts,
native tool requests and persisted tool results, initial workspace seeds, and
the five referenced GAIA attachments. It stores neither ground-truth answer
labels, credentials, nor LLM response text. Persisted tool results remain part
of the trace because they are required for performance-path reconstruction.

`index.jsonl` retains all 30 positions. Unavailable records have a null
`trace_path`; no synthetic trace is created. Historical `/tmp` spill files that
were gone before v2 migration are not fabricated: their recorded paths remain
in the native tool result and live replay regenerates them.

```bash
agenttrace validate traces/<trace>.json
agenttrace replay traces/<trace>.json --dry-run
```
""",
        encoding="utf-8",
    )
    checksum_files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )
    (output / "checksums.sha256").write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(output).as_posix()}\n"
            for path in checksum_files
        ),
        encoding="utf-8",
    )
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
