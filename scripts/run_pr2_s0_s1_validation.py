#!/usr/bin/env python3
"""Run only the frozen PR-2 S0 or S1 correctness stage.

The two stages are intentionally separate commands.  There is no command that
automatically advances from S0 to S1, and this runner has no S2/S3 entry point.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trotterlib.pr2_s0_s1_validation import (  # noqa: E402
    AUTHORIZATION_COMMIT,
    file_sha256,
    run_s0,
    run_s1,
    s1_markdown_summary,
    validate_s0_payload,
    validate_s1_payload,
    write_json_artifact,
)


DEFAULT_OUTPUT_DIRECTORY = (
    ROOT / "artifacts" / "pr2_s0_s1_validation" / "2026-09-28"
)
DEFAULT_S0_ARTIFACT = DEFAULT_OUTPUT_DIRECTORY / "pr2_s0_validation_v1.json"
DEFAULT_S1_ARTIFACT = DEFAULT_OUTPUT_DIRECTORY / "pr2_s1_correctness_v1.json"
DEFAULT_S1_MARKDOWN = DEFAULT_OUTPUT_DIRECTORY / "pr2_s1_correctness_v1.md"

EXECUTABLE_SOURCE_PATHS = (
    "src/trotterlib/df_hamiltonian.py",
    "src/trotterlib/df_partial_randomized_pf.py",
    "src/trotterlib/df_partial_s2.py",
    "src/trotterlib/df_partial_s2_repeated.py",
    "src/trotterlib/df_rte_tail.py",
    "src/trotterlib/rte.py",
    "src/trotterlib/rpe_hadamard_interrogation.py",
    "src/trotterlib/rpe_hadamard_compiled_cost_benchmark.py",
    "src/trotterlib/pr2_s0_s1_validation.py",
    "scripts/run_pr2_s0_s1_validation.py",
    "tests/test_pr2_s0_s1_validation.py",
)
SOURCE_PATHS = EXECUTABLE_SOURCE_PATHS + (
    "docs/research/pr2_s0_s1_execution_amendment_v3.md",
    "artifacts/pr2_s1_s3_preregistration/2026-09-28/"
    "pr2_s0_s1_authorization_manifest_v3.json",
    "docs/research/pr2_s1_s3_preregistration_amendment_v2.md",
    "docs/research/pr2_s1_s3_preregistration.md",
    "docs/research/pr2_codex_validation_policy_d3e1723.md",
    "docs/research/pr2_s0_s1_external_review_d3e1723.md",
    "docs/research/pr2_primary_research_contract.md",
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
        raise RuntimeError("The frozen authorization commit is not an ancestor of HEAD.")
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
                "Runner, validation module, or dedicated test differs from HEAD; "
                "commit the implementation before numerical execution."
            )
    return head


def _provenance(stage: str, test_log: Path | None) -> dict[str, Any]:
    head = _assert_source_frozen()
    sources = {
        relative: file_sha256(ROOT / relative)
        for relative in SOURCE_PATHS
    }
    status = _git("status", "--porcelain=v1")
    record: dict[str, Any] = {
        "stage": stage,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": head,
        "git_branch": _git("branch", "--show-current"),
        "authorization_commit": AUTHORIZATION_COMMIT,
        "source_hashes": sources,
        "worktree_dirty": bool(status),
        "worktree_status_porcelain_v1": status.splitlines(),
        "command": [sys.executable, *sys.argv],
        "cwd": str(ROOT),
    }
    if test_log is not None:
        resolved = test_log.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Dedicated test log not found: {resolved}")
        record["dedicated_test_log"] = {
            "path": str(resolved),
            "sha256": file_sha256(resolved),
        }
    return record


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_text_non_overwrite(text: str, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _run_s0(args: argparse.Namespace) -> int:
    output_directory = args.output_directory.resolve()
    artifact = args.artifact.resolve()
    if artifact.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {artifact}")
    provenance = _provenance("S0", args.test_log)
    payload = run_s0(output_directory, provenance=provenance)
    write_json_artifact(payload, artifact, validator=validate_s0_payload)
    print(
        json.dumps(
            {
                "stage": "S0",
                "status": payload["status"],
                "artifact": str(artifact),
                "result_fingerprint": payload["result_fingerprint"],
                "automatic_next_stage": None,
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "S0_PASS_S1_AUTHORIZED" else 2


def _run_s1(args: argparse.Namespace) -> int:
    s0_artifact = args.s0_artifact.resolve()
    json_artifact = args.artifact.resolve()
    markdown_artifact = args.markdown.resolve()
    for target in (json_artifact, markdown_artifact):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite artifact: {target}")
    s0_payload = _read_json(s0_artifact)
    validate_s0_payload(s0_payload)
    provenance = _provenance("S1", args.test_log)
    provenance["S0_artifact"] = {
        "path": str(s0_artifact),
        "sha256": file_sha256(s0_artifact),
        "result_fingerprint": s0_payload["result_fingerprint"],
    }
    payload = run_s1(s0_payload, provenance=provenance)
    write_json_artifact(payload, json_artifact, validator=validate_s1_payload)
    _write_text_non_overwrite(s1_markdown_summary(payload), markdown_artifact)
    print(
        json.dumps(
            {
                "stage": "S1",
                "status": payload["status"],
                "json_artifact": str(json_artifact),
                "markdown_artifact": str(markdown_artifact),
                "result_fingerprint": payload["result_fingerprint"],
                "automatic_next_stage": None,
                "S2_authorized": False,
            },
            sort_keys=True,
        )
    )
    return (
        0
        if payload["status"]
        == "S1_CORRECTNESS_PASS_AWAITING_EXTERNAL_REVIEW"
        else 2
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen PR-2 S0/S1 correctness runner (no S2/S3 command)."
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)

    s0 = subparsers.add_parser("s0", help="Freeze inputs and execute S0 gates only.")
    s0.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    s0.add_argument("--artifact", type=Path, default=DEFAULT_S0_ARTIFACT)
    s0.add_argument("--test-log", type=Path, required=True)
    s0.set_defaults(handler=_run_s0)

    s1 = subparsers.add_parser(
        "s1", help="Execute S1 correctness only, requiring a passing S0 artifact."
    )
    s1.add_argument("--s0-artifact", type=Path, default=DEFAULT_S0_ARTIFACT)
    s1.add_argument("--artifact", type=Path, default=DEFAULT_S1_ARTIFACT)
    s1.add_argument("--markdown", type=Path, default=DEFAULT_S1_MARKDOWN)
    s1.add_argument("--test-log", type=Path, required=True)
    s1.set_defaults(handler=_run_s1)
    return parser


def main() -> int:
    os.chdir(ROOT)
    arguments = _parser().parse_args()
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
