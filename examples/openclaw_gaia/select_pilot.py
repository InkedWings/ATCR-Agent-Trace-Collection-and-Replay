#!/usr/bin/env python3
"""Create a deterministic, answer-free GAIA validation pilot manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


DEFAULT_SOURCE_REVISION = "682dd723ee1e1697e00360edccf2366dc8418dd9"
TOOL_READABLE_EXTENSIONS = {
    ".csv",
    ".docx",
    ".jsonld",
    ".pdb",
    ".pdf",
    ".pptx",
    ".py",
    ".txt",
    ".xlsx",
    ".zip",
}


def stable_score(seed: str, level: int, bucket: str, task_id: str) -> str:
    material = f"{seed}:{level}:{bucket}:{task_id}".encode()
    return hashlib.sha256(material).hexdigest()


def select_rows(
    rows: list[dict[str, Any]], count_per_level: int, attached_per_level: int, seed: str
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    plain_per_level = count_per_level - attached_per_level
    if count_per_level < 1 or not 0 <= attached_per_level <= count_per_level:
        raise ValueError("invalid per-level sample counts")

    for level in (1, 2, 3):
        level_rows = [row for row in rows if int(row["Level"]) == level]
        plain = [row for row in level_rows if not row.get("file_name")]
        attached = [
            row
            for row in level_rows
            if row.get("file_name")
            and Path(row["file_name"]).suffix.lower() in TOOL_READABLE_EXTENSIONS
        ]
        plain.sort(
            key=lambda row: stable_score(seed, level, "plain", row["task_id"])
        )
        attached.sort(
            key=lambda row: stable_score(seed, level, "attached", row["task_id"])
        )
        if len(plain) < plain_per_level or len(attached) < attached_per_level:
            raise RuntimeError(f"not enough eligible Level {level} tasks")
        selected.extend(plain[:plain_per_level])
        selected.extend(attached[:attached_per_level])

    return sorted(selected, key=lambda row: (int(row["Level"]), row["task_id"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaia-dir", type=Path, default=Path("data/gaia"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/gaia/pilots/pilot12.jsonl")
    )
    parser.add_argument("--count-per-level", type=int, default=4)
    parser.add_argument("--attached-per-level", type=int, default=1)
    parser.add_argument("--seed", default="sigmetrics-2027-pilot-v1")
    args = parser.parse_args()

    metadata_path = args.gaia_dir / "2023/validation/metadata.parquet"
    rows = pq.read_table(metadata_path).to_pylist()
    selected = select_rows(
        rows, args.count_per_level, args.attached_per_level, args.seed
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
