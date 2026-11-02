#!/usr/bin/env python3
"""Run only the frozen PR-2 V4/S1-prime correctness stage."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trotterlib.pr2_v4_s2_development_validation import (  # noqa: E402
    AUTHORIZATION_COMMIT,
    V4_PASS_STATUS,
    file_sha256,
    run_v4,
    validate_v1_v3_artifact,
    validate_v4_payload,
    write_json_artifact,
)


DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "pr2_v4_s2_development"
    / "2026-09-28"
    / "pr2_v4_correctness_result_v1.json"
)
EXECUTABLE_SOURCE_PATHS = (
    "src/trotterlib/pr2_v4_s2_development_validation.py",
    "scripts/run_pr2_v4_correctness.py",
    "scripts/run_pr2_s2_development.py",
    "tests/test_pr2_v4_s2_development_validation.py",
)
SOURCE_PATHS = EXECUTABLE_SOURCE_PATHS + (
    "docs/research/pr2_v4_s2_development_authorization_v5.md",
    "artifacts/pr2_v4_s2_development/2026-09-28/"
    "pr2_v4_s2_authorization_v1.json",
)


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _assert_source_frozen() -> str:
    head = _git("rev-parse", "HEAD")
    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", AUTHORIZATION_COMMIT, head),
        cwd=ROOT,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("The authorization commit is not an ancestor of HEAD.")
    for path in EXECUTABLE_SOURCE_PATHS:
        subprocess.run(
            ("git", "ls-files", "--error-unmatch", path),
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    for mode in ((), ("--cached",)):
        changed = subprocess.run(
            (
                "git",
                "diff",
                *mode,
                "--quiet",
                "HEAD",
                "--",
                *EXECUTABLE_SOURCE_PATHS,
            ),
            cwd=ROOT,
            check=False,
        )
        if changed.returncode != 0:
            raise RuntimeError("Executable PR-2 source differs from HEAD.")
    return head


def _junit_summary(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    summary = {
        key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    if summary["tests"] <= 0 or summary["failures"] or summary["errors"]:
        raise RuntimeError("Dedicated V4/S2 test log is not passing.")
    return summary


def _provenance(test_log: Path) -> dict[str, Any]:
    head = _assert_source_frozen()
    resolved = test_log.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Dedicated test log not found: {resolved}")
    status = _git("status", "--porcelain=v1")
    return {
        "stage": "V4",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": head,
        "git_branch": _git("branch", "--show-current"),
        "authorization_commit": AUTHORIZATION_COMMIT,
        "source_hashes": {
            relative: file_sha256(ROOT / relative) for relative in SOURCE_PATHS
        },
        "worktree_dirty": bool(status),
        "worktree_status_porcelain_v1": status.splitlines(),
        "command": [sys.executable, *sys.argv],
        "cwd": str(ROOT),
        "dedicated_test_log": {
            "path": str(resolved),
            "sha256": file_sha256(resolved),
            "junit": _junit_summary(resolved),
            "evidence_class": "local_validation",
            "independent_external_reproduction": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frozen PR-2 V4 correctness.")
    parser.add_argument("--test-log", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    if artifact.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {artifact}")
    validate_v1_v3_artifact(ROOT)
    payload = run_v4(ROOT, provenance=_provenance(args.test_log))
    write_json_artifact(payload, artifact, validator=validate_v4_payload)
    print(
        json.dumps(
            {
                "stage": "V4",
                "status": payload["status"],
                "artifact": str(artifact),
                "result_fingerprint": payload["result_fingerprint"],
                "S2_development_authorized": payload[
                    "S2_development_authorized"
                ],
                "S3_authorized": False,
                "automatic_next_stage": None,
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == V4_PASS_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
