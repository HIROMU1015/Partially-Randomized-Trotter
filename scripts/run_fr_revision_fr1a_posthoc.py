#!/usr/bin/env python3
"""Run the frozen FR-R1a positive-scalar posthoc reanalysis."""

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

from trotterlib.fr_revision_fr1a_posthoc import (
    build_fr1a_expected,
    run_fr_revision_fr1a_posthoc,
    write_fr1a_expected,
    write_fr1a_payload,
)


DEFAULT_DIRECTORY = Path("artifacts/fr_revision_fr1a_posthoc/2026-09-26")
SOURCE_RESULT = Path(
    "artifacts/finite_rte_phase_amplitude/2026-09-26/"
    "finite_rte_phase_amplitude_fr1_v1.json"
)
FROZEN_PREREGISTRATION = Path(
    "artifacts/finite_rte_phase_amplitude/2026-09-26/"
    "fr1_preregistration_frozen.md"
)
POSTHOC_PLAN = Path("docs/research/fr_revision_fr1a_posthoc_plan.md")


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_status() -> list[str]:
    return subprocess.run(
        ["git", "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected",
        type=Path,
        default=DEFAULT_DIRECTORY / "fr_revision_fr1a_expected_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DIRECTORY / "fr_revision_fr1a_posthoc_v1.json",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    module_path = Path("src/trotterlib/fr_revision_fr1a_posthoc.py")
    runner_path = Path("scripts/run_fr_revision_fr1a_posthoc.py")
    source = json.loads(SOURCE_RESULT.read_text(encoding="utf-8"))
    expected = build_fr1a_expected(source)
    write_fr1a_expected(expected, args.expected)
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_head(),
        "git_worktree_status_before_generation": _git_status(),
        "evidence_status": "local_dirty_worktree_posthoc_validation_not_immutable_ci",
        "command": shlex.join([".venv311/bin/python", *sys.argv]),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "platform": platform.platform(),
        "source_sha256": {
            str(module_path): _sha256(module_path),
            str(runner_path): _sha256(runner_path),
            str(SOURCE_RESULT): _sha256(SOURCE_RESULT),
            str(FROZEN_PREREGISTRATION): _sha256(FROZEN_PREREGISTRATION),
            str(POSTHOC_PLAN): _sha256(POSTHOC_PLAN),
            str(args.expected): _sha256(args.expected),
        },
    }
    payload = run_fr_revision_fr1a_posthoc(
        source,
        source_result_path=SOURCE_RESULT,
        frozen_preregistration_path=FROZEN_PREREGISTRATION,
        posthoc_plan_path=POSTHOC_PLAN,
        provenance=provenance,
    )
    write_fr1a_payload(payload, args.output)
    print(
        json.dumps(
            {
                "expected": str(args.expected),
                "expected_fingerprint": expected["expected_fingerprint"],
                "output": str(args.output),
                "validation_fingerprint": payload["validation_fingerprint"],
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
