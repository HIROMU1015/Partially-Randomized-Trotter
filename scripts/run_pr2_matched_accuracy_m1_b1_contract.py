#!/usr/bin/env python3
"""Generate the zero-compute PR-2 M1-B1 bounded-compile plan."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/trotterlib/pr2_matched_accuracy_m1_b1_contract.py"
_SPEC = importlib.util.spec_from_file_location("pr2_m1_b1_contract_standalone", SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Unable to load the standalone M1-B1 contract source.")
_CONTRACT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CONTRACT)

AUTHORIZATION_SCHEMA_VERSION = _CONTRACT.AUTHORIZATION_SCHEMA_VERSION
M1_A_RESULT_RELATIVE = _CONTRACT.M1_A_RESULT_RELATIVE
STATUS = _CONTRACT.STATUS
build_plan = _CONTRACT.build_plan
file_sha256 = _CONTRACT.file_sha256
load_json = _CONTRACT.load_json
write_json_artifact = _CONTRACT.write_json_artifact

DEFAULT_AUTHORIZATION = (
    ROOT
    / "artifacts/pr2_matched_accuracy_m1_b1_contract/2026-09-30/"
    "pr2_matched_accuracy_m1_b1_implementation_authorization_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/pr2_matched_accuracy_m1_b1_contract/2026-09-30/"
    "pr2_matched_accuracy_m1_b1_zero_compute_plan_v1.json"
)
PLANNING_BASE_COMMIT = "6661bb5ad2e09ab54d4487e34b7a07c22bb6e88e"


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _validate_authorization(
    payload: Mapping[str, Any], *, source_commit: str
) -> dict[str, str]:
    if payload.get("schema_version") != AUTHORIZATION_SCHEMA_VERSION:
        raise ValueError("unexpected M1-B1 implementation authorization schema")
    if payload.get("status") != STATUS:
        raise ValueError("unexpected M1-B1 implementation authorization status")
    if payload.get("planning_base_commit") != PLANNING_BASE_COMMIT:
        raise ValueError("planning base commit differs from the frozen value")
    if payload.get("source_commit_policy") != "full_commit_descendant_of_planning_base":
        raise ValueError("unexpected source commit policy")
    if not _is_ancestor(PLANNING_BASE_COMMIT, source_commit):
        raise ValueError("source commit is not a descendant of the planning base")
    permissions = payload.get("permissions", {})
    if permissions.get("zero_compute_plan_generation_authorized") is not True:
        raise ValueError("zero-compute plan generation is not authorized")
    forbidden = (
        "m1_b1_scientific_execution_authorized",
        "development_npz_load_authorized",
        "signal_reevaluation_authorized",
        "trajectory_sampling_authorized",
        "circuit_build_authorized",
        "direct_compile_authorized",
        "held_out_access_authorized",
        "additional_96_trajectories_authorized",
        "transfer_authorized",
        "s3_authorized",
    )
    if any(permissions.get(name) is not False for name in forbidden):
        raise ValueError("authorization improperly enables scientific work")
    if permissions.get("automatic_next_stage") is not None:
        raise ValueError("authorization defines an automatic next stage")
    hashes = payload.get("required_source_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("authorization has no required source hashes")
    for relative, expected in hashes.items():
        path = ROOT / str(relative)
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"source hash differs from authorization: {relative}")
    return {str(key): str(value) for key, value in hashes.items()}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the 194+16 M1-B1 cell ledger and 12,448 wrapper cache "
            "identities without performing scientific computation."
        )
    )
    parser.add_argument("--authorization", type=Path, default=DEFAULT_AUTHORIZATION)
    parser.add_argument("--m1-a-result", type=Path, default=ROOT / M1_A_RESULT_RELATIVE)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    actual_head = _head()
    if actual_head != args.source_commit:
        raise ValueError(
            f"HEAD {actual_head} differs from --source-commit {args.source_commit}"
        )
    authorization_path = args.authorization.resolve()
    authorization = load_json(authorization_path)
    source_hashes = _validate_authorization(
        authorization, source_commit=args.source_commit
    )
    result_path = args.m1_a_result.resolve()
    result = load_json(result_path)
    plan = build_plan(
        m1_a_result=result,
        m1_a_result_sha256=file_sha256(result_path),
        authorization_sha256=file_sha256(authorization_path),
        source_hashes=source_hashes,
        source_commit=args.source_commit,
    )
    artifact = args.artifact.resolve()
    write_json_artifact(plan, artifact)
    print(
        json.dumps(
            {
                "status": plan["status"],
                "artifact": str(artifact),
                "plan_fingerprint": plan["plan_fingerprint"],
                "random_cells": plan["resource_caps"]["random_cells"],
                "baseline_cells": plan["resource_caps"][
                    "deterministic_or_discard_cells"
                ],
                "total_full_wrappers": plan["resource_caps"][
                    "total_full_wrappers"
                ],
                "m1_b1_scientific_execution_authorized": False,
                "mandatory_stop_reached": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
