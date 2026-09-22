"""Prepare, manually submit, run and summarize the multi-node matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agenttrace.cli import _run_interruptible
from .multi.config import GROUPS, check, matrix, read_config, render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("check", "matrix", "render"):
        command = sub.add_parser(action)
        command.add_argument("--config", type=Path, default=Path("examples/scaling/multinode.json"))
        command.add_argument("--smoke", action="store_true")
        command.add_argument("--group", choices=GROUPS, default="all")
        if action == "render":
            command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("run", help="inside a rendered PBS job only")
    command.add_argument("--config", type=Path, required=True)
    command = sub.add_parser("frontend", help=argparse.SUPPRESS)
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--config", type=Path, required=True)
    command = sub.add_parser("summarize", help="aggregate completed run directories and draw figures")
    command.add_argument("--runs", type=Path, nargs="+", required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.action in ("check", "matrix", "render"):
        config = read_config(args.config)
        if args.action == "render":
            result = render(config, args.output, smoke=args.smoke, group=args.group)
        else:
            result = (check if args.action == "check" else matrix)(config, smoke=args.smoke, group=args.group)
    elif args.action == "run":
        from .multi.runtime import run
        result = _run_interruptible(run(args.config.resolve()))
    elif args.action == "frontend":
        from .multi.frontend import serve_frontend
        result = _run_interruptible(serve_frontend(args.root, json.loads(args.config.read_text())))
    else:
        from .multi.plots import export
        result = export(args.runs, args.output, plots=not args.no_plots)
    if result is not None:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
