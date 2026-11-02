#!/usr/bin/env python3
"""Formalize P-A v1 and audit the mechanism exercised by completed holdouts."""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from trotterlib.parallel_validation_executor import file_sha256
from trotterlib.research_direction_joint_synthesis_formalization import (
    evaluate_joint_synthesis_formalization,
    finalize_joint_synthesis_formalization_artifact,
    write_joint_synthesis_formalization_artifact,
)


DEFAULT_PILOT = Path(
    "artifacts/research_direction_joint_synthesis_pilot/2026-09-25/"
    "pa_h4_interval_union_joint_synthesis_v1.json"
)
DEFAULT_BLIND = Path(
    "artifacts/research_direction_joint_synthesis_blind_validation/2026-09-25/"
    "pa_v1_h5_physical_h4_opt2_blind_v1.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/research_direction_joint_synthesis_formalization/2026-09-25/"
    "pa_v1_formalization_and_mechanism_audit_v1.json"
)


def _git(command: list[str]) -> str | list[str] | None:
    result = subprocess.run(
        ["git", *command], check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    lines = result.stdout.splitlines()
    return lines[0] if len(lines) == 1 else lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--blind", type=Path, default=DEFAULT_BLIND)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    inputs = {
        "pilot": json.loads(args.pilot.read_text(encoding="utf-8")),
        "blind": json.loads(args.blind.read_text(encoding="utf-8")),
    }
    body = evaluate_joint_synthesis_formalization(**inputs)
    body["source_evidence"] = {
        "pilot": {
            "path": str(args.pilot),
            "sha256": file_sha256(args.pilot),
            "content_fingerprint": inputs["pilot"]["content_fingerprint"],
        },
        "blind": {
            "path": str(args.blind),
            "sha256": file_sha256(args.blind),
            "content_fingerprint": inputs["blind"]["content_fingerprint"],
        },
    }
    sources = (
        Path(
            "src/trotterlib/"
            "research_direction_joint_synthesis_formalization.py"
        ),
        Path("scripts/run_research_direction_joint_synthesis_formalization.py"),
    )
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_worktree_status_before_generation": _git(["status", "--short"]),
        "evidence_status": "local_dirty_worktree_not_externally_reproduced",
        "command": shlex.join([sys.executable, *sys.argv]),
        "python_version": platform.python_version(),
        "source_sha256": {str(path): file_sha256(path) for path in sources},
    }
    artifact = finalize_joint_synthesis_formalization_artifact(
        body, provenance=provenance
    )
    write_joint_synthesis_formalization_artifact(artifact, args.output)
    print(args.output)
    print(artifact["content_fingerprint"])
    print(artifact["decision"]["status"])


if __name__ == "__main__":
    main()
