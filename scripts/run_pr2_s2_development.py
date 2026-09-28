#!/usr/bin/env python3
"""Run the one authorized PR-2 development resource comparison and stop."""

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
    run_s2_development,
    validate_s2_payload,
    validate_v4_payload,
    write_json_artifact,
)


DEFAULT_V4 = (
    ROOT
    / "artifacts"
    / "pr2_v4_s2_development"
    / "2026-09-28"
    / "pr2_v4_correctness_result_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "pr2_v4_s2_development"
    / "2026-09-28"
    / "pr2_s2_development_resource_result_v1.json"
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
    if subprocess.run(
        ("git", "merge-base", "--is-ancestor", AUTHORIZATION_COMMIT, head),
        cwd=ROOT,
        check=False,
    ).returncode:
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
        if subprocess.run(
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
        ).returncode:
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


def _provenance(test_log: Path, v4_path: Path) -> dict[str, Any]:
    head = _assert_source_frozen()
    resolved = test_log.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Dedicated test log not found: {resolved}")
    status = _git("status", "--porcelain=v1")
    return {
        "stage": "S2-development",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": head,
        "git_branch": _git("branch", "--show-current"),
        "authorization_commit": AUTHORIZATION_COMMIT,
        "source_hashes": {
            relative: file_sha256(ROOT / relative) for relative in SOURCE_PATHS
        },
        "V4_artifact": {
            "path": str(v4_path.resolve()),
            "sha256": file_sha256(v4_path),
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
    parser = argparse.ArgumentParser(
        description="Run the frozen PR-2 S2 development comparison."
    )
    parser.add_argument("--test-log", type=Path, required=True)
    parser.add_argument("--v4-artifact", type=Path, default=DEFAULT_V4)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    if artifact.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {artifact}")
    v4_path = args.v4_artifact.resolve()
    v4_payload = json.loads(v4_path.read_text(encoding="utf-8"))
    validate_v4_payload(v4_payload)
    if v4_payload["status"] != V4_PASS_STATUS or v4_payload["deviations"]:
        raise RuntimeError("S2 requires a deviation-free V4 PASS artifact.")
    payload = run_s2_development(
        ROOT,
        v4_payload,
        provenance=_provenance(args.test_log, v4_path),
    )
    write_json_artifact(payload, artifact, validator=validate_s2_payload)
    print(
        json.dumps(
            {
                "stage": "S2-development",
                "status": payload["status"],
                "artifact": str(artifact),
                "result_fingerprint": payload["result_fingerprint"],
                "selected_B2": payload["decision"]["selected_B2"],
                "S3_authorized": False,
                "automatic_next_stage": None,
                "mandatory_stop_reached": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
