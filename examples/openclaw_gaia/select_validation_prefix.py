#!/usr/bin/env python3
"""Create an answer-free manifest from the first GAIA validation rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


DEFAULT_SOURCE_REVISION = "682dd723ee1e1697e00360edccf2366dc8418dd9"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaia-dir", type=Path, default=Path("data/gaia"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/gaia/pilots/validation_first30.jsonl")
    )
    parser.add_argument("--count", type=int, default=30)
    args = parser.parse_args()

    if args.count < 1:
        parser.error("count must be at least 1")

    metadata_path = args.gaia_dir / "2023/validation/metadata.parquet"
    rows = pq.read_table(metadata_path).to_pylist()
    selected = rows[: args.count]
    if len(selected) != args.count:
        raise RuntimeError(
            f"requested {args.count} rows, but validation contains only {len(rows)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            record = {
                "task_id": row["task_id"],
                "question": row["Question"],
                "level": int(row["Level"]),
                "attachment": row.get("file_path") or None,
                "source_revision": DEFAULT_SOURCE_REVISION,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "ok": True,
        "selection": "validation row order prefix",
        "manifest": str(args.output),
        "tasks": len(selected),
        "levels": {
            str(level): sum(int(row["Level"]) == level for row in selected)
            for level in (1, 2, 3)
        },
        "attachments": [
            {
                "task_id": row["task_id"],
                "type": Path(row["file_name"]).suffix.lower(),
            }
            for row in selected
            if row.get("file_name")
        ],
        "answers_in_manifest": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
