#!/usr/bin/env python3
"""Run the preregistered PR-2 and PR-3 minimal pilots, then stop."""

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

from trotterlib.pr2_pr3_minimal_pilot import (
    run_pr2_pr3_minimal_pilots,
    write_pr2_pr3_payload,
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
            "artifacts/pr2_pr3_minimal_pilot/2026-09-27/"
            "pr2_pr3_minimal_pilot_v1.json"
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    source_paths = (
        Path("src/trotterlib/pr2_pr3_minimal_pilot.py"),
        Path("scripts/run_pr2_pr3_minimal_pilot.py"),
        Path("docs/research/pr2_pr3_minimal_pilot_preregistration.md"),
    )
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_head(),
        "git_worktree_status_before_generation": _git_status(),
        "evidence_status": "local_dirty_worktree_validation_not_immutable_ci",
        "command": shlex.join([".venv311/bin/python", *sys.argv]),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "source_sha256": {
            str(path): _file_sha256(path) for path in source_paths
        },
    }
    payload = run_pr2_pr3_minimal_pilots(provenance=provenance)
    write_pr2_pr3_payload(payload, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "result_fingerprint": payload["result_fingerprint"],
                "pr2_decision": payload["pr2"]["decision"],
                "pr3_decision": payload["pr3"]["decision"],
                "theme_selection": payload["theme_selection"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
