"""Reuse the existing call/token and recorded-tool-error audit on five completed points."""
import importlib.util
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
spec = importlib.util.spec_from_file_location("weak_audit", OUT.parent / "2026-09-15-weak-scaling/analyze.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main():
    metadata, results = {}, []
    roots = []
    for job in ("7642785", "7642786", "7642787", "7642788"):
        roots.extend(sorted((REPO / "runs/scaling/multinode").glob(f"*-{job}.*")))
    for root in roots:
        if not (root / "summary.json").exists():
            continue
        config = module.read(root / "config.json")
        paths = tuple(config["trace_paths"])
        if paths not in metadata:
            metadata[paths] = module.tool_metadata(paths)
        result = module.audit_calls(root, module.read(root / "summary.json"), metadata[paths])
        results.append({"run": root.name, **result})
        print(json.dumps({"run": root.name, "checks_passed": all(result["checks"].values()),
                          "window_tools": result["window_tools"]}), flush=True)
    assert len(results) == 5
    (OUT / "completed_call_audit.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
