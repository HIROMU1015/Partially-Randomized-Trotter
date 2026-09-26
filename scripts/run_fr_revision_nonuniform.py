#!/usr/bin/env python3
"""Freeze or run the preregistered FR-R1b nonuniform 4x4 validation."""

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

from trotterlib.fr_revision_nonuniform import (
    PREREGISTRATION_SHA256,
    build_fr_r1b_expected,
    run_fr_revision_nonuniform,
    validate_fr_r1b_expected,
    write_fr_r1b_expected,
    write_fr_r1b_payload,
)


DEFAULT_DIRECTORY = Path("artifacts/fr_revision_nonuniform/2026-09-27")
PREREGISTRATION = Path("docs/research/fr_revision_nonuniform_preregistration.md")
PARENT_CONTRACT = Path("docs/research/fr_revision_scalar_structure_contract.md")
FR1A_RESULT = Path(
    "artifacts/fr_revision_fr1a_posthoc/2026-09-26/"
    "fr_revision_fr1a_posthoc_v1.json"
)
MODULE = Path("src/trotterlib/fr_revision_nonuniform.py")
SHARED_SCALAR_MODULE = Path("src/trotterlib/fr_revision_fr1a_posthoc.py")
RUNNER = Path("scripts/run_fr_revision_nonuniform.py")
TEST = Path("tests/test_fr_revision_nonuniform.py")


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _git_status() -> list[str]:
    return subprocess.run(
        ["git", "status", "--short"], check=True, capture_output=True, text=True
    ).stdout.splitlines()


def _source_hashes() -> dict[str, str]:
    paths = (
        MODULE,
        SHARED_SCALAR_MODULE,
        RUNNER,
        TEST,
        PREREGISTRATION,
        PARENT_CONTRACT,
        FR1A_RESULT,
    )
    return {str(path): _sha256(path) for path in paths}


def _provenance(evidence_status: str) -> dict[str, object]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_head(),
        "git_worktree_status_before_generation": _git_status(),
        "evidence_status": evidence_status,
        "command": shlex.join([".venv311/bin/python", *sys.argv]),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "platform": platform.platform(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected",
        type=Path,
        default=DEFAULT_DIRECTORY / "fr_revision_nonuniform_expected_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DIRECTORY / "fr_revision_nonuniform_r1b_v1.json",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--expected-only",
        action="store_true",
        help="Freeze the expected specification without evaluating any matrix condition.",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Run only after the expected specification already exists and validates.",
    )
    return parser


def _validate_preregistration() -> None:
    if _sha256(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise ValueError("FR-R1b preregistration SHA-256 mismatch.")
    final_text = PREREGISTRATION.read_text(encoding="utf-8").rstrip()
    if not final_text.endswith("full compiled総costへ進まない。"):
        raise ValueError("FR-R1b preregistration completeness sentinel mismatch.")


def _load_expected(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_fr_r1b_expected(payload)
    current = _source_hashes()
    frozen = payload.get("source_sha256", {})
    mismatches = {
        key: {"expected": frozen.get(key), "actual": value}
        for key, value in current.items()
        if frozen.get(key) != value
    }
    if mismatches:
        raise ValueError(f"FR-R1b frozen source hash mismatch: {mismatches}")
    return payload


def main() -> int:
    args = _parser().parse_args()
    _validate_preregistration()
    if args.expected_only:
        expected = build_fr_r1b_expected(
            provenance=_provenance(
                "local_dirty_worktree_preregistered_expected_not_immutable_ci"
            ),
            source_sha256=_source_hashes(),
        )
        write_fr_r1b_expected(expected, args.expected)
        print(
            json.dumps(
                {
                    "mode": "expected_only",
                    "expected": str(args.expected),
                    "expected_fingerprint": expected["expected_fingerprint"],
                    "configuration_fingerprint": expected[
                        "configuration_fingerprint"
                    ],
                    "matrix_condition_count": len(expected["condition_ids"]),
                    "state_row_count": len(expected["state_ids"]),
                    "semantic_control_count": len(expected["semantic_control_ids"]),
                    "result_computation_performed": False,
                },
                sort_keys=True,
                indent=2,
            )
        )
        return 0
    if not args.expected.exists():
        raise FileNotFoundError(
            "FR-R1b expected specification does not exist; run --expected-only first."
        )
    expected = _load_expected(args.expected)
    provenance = _provenance(
        "local_dirty_worktree_preregistered_validation_not_immutable_ci"
    )
    provenance["expected_file_sha256"] = _sha256(args.expected)
    provenance["source_sha256"] = _source_hashes()
    payload = run_fr_revision_nonuniform(expected, provenance=provenance)
    write_fr_r1b_payload(payload, args.output)
    print(
        json.dumps(
            {
                "mode": "run",
                "expected": str(args.expected),
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
