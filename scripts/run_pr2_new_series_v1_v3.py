#!/usr/bin/env python3
"""Run the frozen PR-2 snapshot-rebased V1--V3 package once."""

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

from trotterlib.pr2_new_series_validation import (  # noqa: E402
    PASS_STATUS,
    SPECIFICATION_COMMIT,
    file_sha256,
    run_v1_v3,
    validate_result_payload,
    write_json_artifact,
)


DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "pr2_new_series_validation"
    / "2026-09-28"
    / "pr2_v1_v3_result_v1.json"
)
EXECUTABLE_SOURCE_PATHS = (
    "src/trotterlib/df_hamiltonian.py",
    "src/trotterlib/df_partial_randomized_pf.py",
    "src/trotterlib/df_partial_s2.py",
    "src/trotterlib/df_rte_tail.py",
    "src/trotterlib/pr2_s0_s1_validation.py",
    "src/trotterlib/pr2_new_series_validation.py",
    "scripts/run_pr2_new_series_v1_v3.py",
    "tests/test_pr2_new_series_validation.py",
)
SOURCE_PATHS = EXECUTABLE_SOURCE_PATHS + (
    "docs/research/pr2_s0_review_and_research_reset_bf5d3b2_20260928.md",
    "docs/research/pr2_codex_restart_workpackage_bf5d3b2_20260928.md",
    "docs/research/pr2_v0_input_recovery_audit_bf5d3b2_20260928.md",
    "docs/research/pr2_new_series_amendment_v4.md",
    "artifacts/pr2_new_series_validation/2026-09-28/"
    "pr2_v1_v3_authorization_v1.json",
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
        ("git", "merge-base", "--is-ancestor", SPECIFICATION_COMMIT, head),
        cwd=ROOT,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("The result-prior specification is not an ancestor of HEAD.")
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
            raise RuntimeError(
                "Executable source or dedicated tests differ from HEAD; "
                "commit them before V1--V3 execution."
            )
    return head


def _junit_summary(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    summary = {
        key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    if summary["tests"] <= 0:
        raise RuntimeError("Dedicated V3 test log contains no tests.")
    if summary["failures"] or summary["errors"]:
        raise RuntimeError("Dedicated V3 test log is not passing.")
    return summary


def _provenance(test_log: Path) -> dict[str, Any]:
    head = _assert_source_frozen()
    resolved_log = test_log.resolve()
    if not resolved_log.is_file():
        raise FileNotFoundError(f"Dedicated V3 test log not found: {resolved_log}")
    status = _git("status", "--porcelain=v1")
    return {
        "stage": "V1-V3",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": head,
        "git_branch": _git("branch", "--show-current"),
        "specification_commit": SPECIFICATION_COMMIT,
        "source_hashes": {
            relative: file_sha256(ROOT / relative) for relative in SOURCE_PATHS
        },
        "worktree_dirty": bool(status),
        "worktree_status_porcelain_v1": status.splitlines(),
        "command": [sys.executable, *sys.argv],
        "cwd": str(ROOT),
        "dedicated_test_log": {
            "path": str(resolved_log),
            "sha256": file_sha256(resolved_log),
            "junit": _junit_summary(resolved_log),
            "independent_external_reproduction": False,
            "evidence_class": "local_validation",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run only the frozen PR-2 new-series V1--V3 validation."
    )
    parser.add_argument("--test-log", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    artifact = args.artifact.resolve()
    if artifact.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {artifact}")
    payload = run_v1_v3(ROOT, provenance=_provenance(args.test_log))
    validate_result_payload(payload)
    write_json_artifact(payload, artifact)
    print(
        json.dumps(
            {
                "stage": "V1-V3",
                "status": payload["status"],
                "artifact": str(artifact),
                "result_fingerprint": payload["result_fingerprint"],
                "V4_authorized": False,
                "automatic_next_stage": None,
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == PASS_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())

