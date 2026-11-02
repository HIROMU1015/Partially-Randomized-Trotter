#!/usr/bin/env python3
"""Generate the PR-2 M2 held-out transfer zero-compute plan."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/trotterlib/pr2_matched_accuracy_m2_transfer_contract.py"
SPEC = importlib.util.spec_from_file_location("pr2_m2_transfer_contract", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("unable to load M2 transfer contract source")
CONTRACT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTRACT)

DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/pr2_matched_accuracy_m2_transfer_contract/2026-10-04/"
    "pr2_matched_accuracy_m2_transfer_zero_compute_plan_v2.json"
)
REQUIRED_SOURCE_PATHS = (
    "src/trotterlib/pr2_matched_accuracy_m2_transfer_contract.py",
    "scripts/run_pr2_matched_accuracy_m2_transfer_contract.py",
    "tests/test_pr2_matched_accuracy_m2_transfer_contract.py",
    "docs/research/pr2_matched_accuracy_m2_held_out_transfer_contract_v1.md",
    "docs/research/pr2_matched_accuracy_m2_transfer_contract_amendment_v2.md",
    "artifacts/pr2_matched_accuracy_m2_transfer_contract/2026-10-04/"
    "pr2_matched_accuracy_m2_transfer_plan_schema_v2.json",
    "artifacts/pr2_matched_accuracy_m2_transfer_contract/2026-10-04/"
    "pr2_matched_accuracy_m2_transfer_result_schema_v2.json",
)


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    ).returncode == 0


def _require_committed_sources(source_commit: str, source_hashes: dict[str, str]) -> None:
    for relative, digest in source_hashes.items():
        blob = subprocess.run(
            ["git", "show", f"{source_commit}:{relative}"],
            cwd=ROOT, check=True, capture_output=True,
        ).stdout
        if hashlib.sha256(blob).hexdigest() != digest:
            raise ValueError(f"source differs from committed blob: {relative}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze five M2 transfer candidates and rules without held-out access."
    )
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--draft", action="store_true",
        help="Write an explicitly uncommitted draft; cannot be used for execution authorization.",
    )
    args = parser.parse_args()

    if _head() != args.source_commit:
        raise ValueError("HEAD differs from --source-commit")
    if not _is_ancestor(CONTRACT.M1_B1_EVIDENCE_COMMIT, args.source_commit):
        raise ValueError("source commit is not a descendant of M1-B1 evidence")
    source_hashes = {
        relative: CONTRACT.file_sha256(ROOT / relative)
        for relative in REQUIRED_SOURCE_PATHS
    }
    if not args.draft:
        _require_committed_sources(args.source_commit, source_hashes)
    result_path = ROOT / CONTRACT.M1_B1_RESULT_RELATIVE
    validation_path = ROOT / CONTRACT.M1_B1_VALIDATION_RELATIVE
    result = CONTRACT.load_json(result_path)
    validation = CONTRACT.load_json(validation_path)
    plan = CONTRACT.build_plan(
        result=result,
        validation=validation,
        result_sha256=CONTRACT.file_sha256(result_path),
        validation_sha256=CONTRACT.file_sha256(validation_path),
        source_commit=args.source_commit,
        source_hashes=source_hashes,
        source_committed=not args.draft,
    )
    output = args.artifact
    if args.draft and output == DEFAULT_OUTPUT:
        output = DEFAULT_OUTPUT.with_name(DEFAULT_OUTPUT.stem + "_draft.json")
    artifact = output.resolve()
    CONTRACT.write_json_artifact(plan, artifact)
    print(
        json.dumps(
            {
                "status": plan["status"],
                "artifact": str(artifact),
                "plan_fingerprint": plan["plan_fingerprint"],
                "source_binding_status": plan["source_binding_status"],
                "candidate_count": plan["resource_caps"]["candidate_count"],
                "future_full_wrapper_cap": plan["resource_caps"]["total_full_wrappers"],
                "held_out_access_authorized": False,
                "transfer_execution_authorized": False,
                "mandatory_stop_reached": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
