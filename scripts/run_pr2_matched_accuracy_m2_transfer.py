#!/usr/bin/env python3
"""Plan M2 without data access, or run ONLY a later separately authorized transfer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from trotterlib import pr2_matched_accuracy_m2_transfer_contract as contract
from trotterlib import pr2_matched_accuracy_m2_transfer_execution as execution


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="zero-science source-bound execution plan; no authorization")
    plan.add_argument("--project-root", required=True, type=Path)
    plan.add_argument("--source-commit", required=True)
    plan.add_argument("--output", required=True, type=Path)
    run = commands.add_parser("run", help="requires committed result-prior authorization and final review")
    run.add_argument("--project-root", required=True, type=Path)
    run.add_argument("--plan", required=True, type=Path)
    run.add_argument("--authorization", required=True, type=Path)
    run.add_argument("--output-relative", required=True)
    run.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    root = args.project_root.absolute()
    if args.command == "plan":
        hashes = execution.committed_source_hashes(root, args.source_commit)
        path = root / execution.CONTRACT_PLAN_PATH
        if contract.file_sha256(path) != execution.CONTRACT_PLAN_SHA256:
            raise ValueError("frozen contract plan hash mismatch")
        output = args.output if args.output.is_absolute() else root / args.output
        # Prevent a zero-compute command from statting the held-out path as output.
        if output.suffix != ".json" or "held_out" in str(output):
            raise ValueError("zero-compute output must be a non-held-out JSON path")
        value = execution.build_execution_plan(
            contract.load_json(path), source_commit=args.source_commit,
            source_hashes=hashes, environment=execution.environment_identity(),
        )
        execution.validate_execution_plan(value, contract.load_json(path))
        contract.write_json_artifact(value, output)
        print(json.dumps({"status": value["status"], "plan_fingerprint": value["plan_fingerprint"],
                          "held_out_access": 0, "science_wrappers": 0, "execution_authorized": False}))
        return 0
    try:
        value = execution.run_transfer(
            root, plan_path=root / args.plan, authorization_path=root / args.authorization,
            output_relative=args.output_relative, workers=args.workers,
        )
    except Exception as exc:
        print(json.dumps({"status": execution.FAILURE, "exception_type": type(exc).__name__,
                          "exception_message": str(exc), "next_stage_authorized": False,
                          "mandatory_stop_reached": True}), file=sys.stderr)
        return 1
    print(json.dumps({"status": value["status"], "result_fingerprint": value["result_fingerprint"],
                      "next_stage_authorized": False, "mandatory_stop_reached": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
