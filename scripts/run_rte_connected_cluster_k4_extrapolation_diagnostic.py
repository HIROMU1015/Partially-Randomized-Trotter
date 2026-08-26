#!/usr/bin/env python3
"""Fit an L=4 residual as a K4 term and transfer it to L=6."""

from __future__ import annotations

import argparse
import hashlib
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import qiskit

from trotterlib.rte_connected_cluster_cost_validation import (
    diagnose_connected_cluster_k4_extrapolation,
    validate_connected_cluster_transfer_payload,
    write_connected_cluster_k4_extrapolation_diagnostic,
)


def _git_head() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _git_status() -> list[str]:
    completed = subprocess.run(
        ["git", "status", "--short"], check=False, capture_output=True, text=True
    )
    return (
        completed.stdout.splitlines()
        if completed.returncode == 0
        else ["git_status_unavailable"]
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("transfer", type=Path)
    parser.add_argument("--fit-length", type=int, default=4)
    parser.add_argument("--test-length", type=int, default=6)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    import json

    transfer = json.loads(args.transfer.read_text(encoding="utf-8"))
    validate_connected_cluster_transfer_payload(transfer)
    output = args.output or args.transfer.with_name(
        f"{args.transfer.stem}_k4_l{args.fit_length}_to_l{args.test_length}_v1.json"
    )
    source_paths = (
        Path("src/trotterlib/rte_connected_cluster_cost_validation.py"),
        Path("scripts/run_rte_connected_cluster_k4_extrapolation_diagnostic.py"),
    )
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_head(),
        "git_worktree_status_before_generation": _git_status(),
        "command": shlex.join(
            [
                ".venv311/bin/python",
                "scripts/run_rte_connected_cluster_k4_extrapolation_diagnostic.py",
                *sys.argv[1:],
            ]
        ),
        "python_version": platform.python_version(),
        "qiskit_version": qiskit.__version__,
        "platform": platform.platform(),
        "evidence_status": "local_worktree_validation_not_immutable_ci",
        "transfer_path": str(args.transfer),
        "transfer_sha256": _file_sha256(args.transfer),
        "source_sha256": {str(path): _file_sha256(path) for path in source_paths},
    }
    payload = diagnose_connected_cluster_k4_extrapolation(
        transfer,
        fit_length=args.fit_length,
        test_length=args.test_length,
        provenance=provenance,
    )
    write_connected_cluster_k4_extrapolation_diagnostic(payload, output)
    print(output)
    print(payload["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
