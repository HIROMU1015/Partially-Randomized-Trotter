#!/usr/bin/env python3
"""Run the preregistered finite-RTE phase/radius FR-1 validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

from trotterlib.finite_rte_phase_amplitude import (
    run_finite_rte_phase_amplitude_fr1,
    write_finite_rte_phase_amplitude_payload,
)


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_status() -> list[str]:
    output = subprocess.run(
        ["git", "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return output.splitlines()


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/finite_rte_phase_amplitude/2026-09-26/"
            "finite_rte_phase_amplitude_fr1_v1.json"
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    module_path = Path("src/trotterlib/finite_rte_phase_amplitude.py")
    runner_path = Path("scripts/run_finite_rte_phase_amplitude.py")
    contract_path = Path("docs/research/finite_rte_phase_amplitude_contract.md")
    preregistration_path = Path(
        "docs/research/finite_rte_phase_amplitude_fr1_preregistration.md"
    )
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_head(),
        "git_worktree_status_before_generation": _git_status(),
        "evidence_status": "local_dirty_worktree_validation_not_immutable_ci",
        "command": shlex.join([".venv311/bin/python", *sys.argv]),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "platform": platform.platform(),
        "source_sha256": {
            str(module_path): _file_sha256(module_path),
            str(runner_path): _file_sha256(runner_path),
            str(contract_path): _file_sha256(contract_path),
            str(preregistration_path): _file_sha256(preregistration_path),
        },
    }
    payload = run_finite_rte_phase_amplitude_fr1(provenance=provenance)
    write_finite_rte_phase_amplitude_payload(payload, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "validation_fingerprint": payload["validation_fingerprint"],
                "gates": payload["gates"],
                "summary": payload["summary"],
                "performance": payload["performance"],
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0 if payload["summary"]["execution_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
