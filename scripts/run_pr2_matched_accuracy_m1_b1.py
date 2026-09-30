#!/usr/bin/env python3
"""Freeze or execute the result-prior PR-2 M1-B1 compile map."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from trotterlib.pr2_matched_accuracy_m1_b1_contract import (
    M1_A_RESULT_RELATIVE,
    canonical_json,
    file_sha256,
    load_json,
)
from trotterlib.pr2_matched_accuracy_m1_b1_execution import (
    FAILURE_STATUS,
    RESULT_SCHEMA_VERSION,
    build_execution_plan,
    run_m1_b1,
)


DEFAULT_SOURCE_FILES = (
    "src/trotterlib/pr2_matched_accuracy_m1_b1_contract.py",
    "src/trotterlib/pr2_matched_accuracy_m1_b1_execution.py",
    "scripts/run_pr2_matched_accuracy_m1_b1.py",
    "tests/test_pr2_matched_accuracy_m1_b1_execution.py",
    "artifacts/pr2_matched_accuracy_m1_b1_contract/2026-09-30/"
    "pr2_matched_accuracy_m1_b1_result_schema_v2.json",
)


def _head(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(payload) + b"\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="create the source-bound zero-compute plan")
    plan.add_argument("--project-root", type=Path, required=True)
    plan.add_argument("--source-commit", required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--source-file", action="append", dest="source_files")
    run = subparsers.add_parser("run", help="execute the separately authorized compile map")
    run.add_argument("--project-root", type=Path, required=True)
    run.add_argument("--authorization", type=Path, required=True)
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--workers", type=int, default=6)
    run.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    root = args.project_root.resolve()
    if args.command == "plan":
        if _head(root) != args.source_commit:
            raise SystemExit("HEAD must equal --source-commit when freezing the plan")
        source_files = tuple(args.source_files or DEFAULT_SOURCE_FILES)
        source_hashes = {relative: file_sha256(root / relative) for relative in source_files}
        m1_path = root / M1_A_RESULT_RELATIVE
        plan = build_execution_plan(
            m1_a_result=load_json(m1_path),
            m1_a_result_sha256=file_sha256(m1_path),
            source_commit=args.source_commit,
            source_hashes=source_hashes,
        )
        output = args.output if args.output.is_absolute() else root / args.output
        if output.exists():
            raise SystemExit(f"refusing to overwrite plan: {output}")
        _write(output, plan)
        print(json.dumps({"output": str(output), "plan_fingerprint": plan["plan_fingerprint"]}, sort_keys=True))
        return 0

    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    authorization = args.authorization if args.authorization.is_absolute() else root / args.authorization
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    try:
        result = run_m1_b1(
            root,
            authorization_path=authorization,
            plan_path=plan_path,
            output_dir=output,
            workers=args.workers,
            resume=args.resume,
        )
    except Exception as exc:
        failure = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": FAILURE_STATUS,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "research_decision": None,
            "automatic_next_stage": None,
        }
        if output.exists() and output.is_dir():
            _write(output / "M1_B1_FAILURE.json", failure)
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "result_fingerprint": result["result_fingerprint"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
