#!/usr/bin/env python3
"""Generate the zero-compute PR-2 M1 implementation-contract dry run."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_SOURCE = ROOT / "src" / "trotterlib" / "pr2_matched_accuracy_m1_contract.py"
_SPEC = importlib.util.spec_from_file_location(
    "pr2_matched_accuracy_m1_contract_standalone",
    CONTRACT_SOURCE,
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Unable to load the standalone M1 contract source.")
_CONTRACT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CONTRACT)

AUTHORIZATION_SCHEMA_VERSION = _CONTRACT.AUTHORIZATION_SCHEMA_VERSION
BASE_COMMIT = _CONTRACT.BASE_COMMIT
PRIOR_ART_GATE_SHA256 = _CONTRACT.PRIOR_ART_GATE_SHA256
RESOURCE_CONTRACT_SHA256 = _CONTRACT.RESOURCE_CONTRACT_SHA256
STATUS = _CONTRACT.STATUS
build_dry_run = _CONTRACT.build_dry_run
file_sha256 = _CONTRACT.file_sha256
write_json_artifact = _CONTRACT.write_json_artifact


DEFAULT_AUTHORIZATION = (
    ROOT
    / "artifacts"
    / "pr2_matched_accuracy_m1_contract"
    / "2026-09-29"
    / "pr2_matched_accuracy_m1_implementation_authorization_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "pr2_matched_accuracy_m1_contract"
    / "2026-09-29"
    / "pr2_matched_accuracy_m1_dry_run_v1.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _validate_authorization(
    authorization: Mapping[str, Any],
) -> dict[str, str]:
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA_VERSION:
        raise ValueError("unexpected M1 implementation authorization schema")
    if authorization.get("status") != STATUS:
        raise ValueError("unexpected M1 implementation authorization status")
    if authorization.get("implementation_base_commit") != BASE_COMMIT:
        raise ValueError("implementation base commit differs from the frozen value")
    contract = authorization["contract_identity"]
    if contract["resource_contract_sha256"] != RESOURCE_CONTRACT_SHA256:
        raise ValueError("resource contract SHA-256 differs from the frozen value")
    if contract["prior_art_gate_sha256"] != PRIOR_ART_GATE_SHA256:
        raise ValueError("prior-art gate SHA-256 differs from the frozen value")
    permissions = authorization["permissions"]
    if not permissions["implementation_contract_and_dry_run_authorized"]:
        raise ValueError("implementation-contract dry run is not authorized")
    forbidden = (
        "m1_scientific_execution_authorized",
        "development_npz_load_authorized",
        "held_out_access_authorized",
        "signal_evaluation_authorized",
        "trajectory_sampling_authorized",
        "circuit_build_authorized",
        "direct_compile_authorized",
        "s3_authorized",
    )
    if any(permissions[name] for name in forbidden):
        raise ValueError("authorization improperly enables scientific work")
    if permissions["automatic_next_stage"] is not None:
        raise ValueError("authorization defines an automatic next stage")
    source_hashes = authorization["required_source_hashes"]
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("authorization has no required source hashes")
    for relative, expected in source_hashes.items():
        path = ROOT / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"source hash differs from authorization: {relative}")
    return {str(key): str(value) for key, value in source_hashes.items()}


def _read_s2_ledger(authorization: Mapping[str, Any]) -> dict[str, Any]:
    frozen = authorization["m0_read_only_input"]
    relative = str(frozen["path"])
    path = ROOT / relative
    observed_sha = file_sha256(path)
    if observed_sha != frozen["sha256"]:
        raise ValueError("known S2 artifact SHA-256 differs from authorization")
    payload = _load_json(path)
    if payload.get("status") != frozen["status"]:
        raise ValueError("known S2 status differs from authorization")
    if payload.get("result_fingerprint") != frozen["result_fingerprint"]:
        raise ValueError("known S2 result fingerprint differs from authorization")
    if payload.get("held_out_npz_loaded") is not False:
        raise ValueError("known S2 artifact does not preserve held-out closure")
    return {
        "path": relative,
        "sha256": observed_sha,
        "schema_version": payload["schema_version"],
        "status": payload["status"],
        "result_fingerprint": payload["result_fingerprint"],
        "held_out_npz_loaded": False,
        "read_purpose": "known_result_identity_only",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate candidate identities and a synthetic selector dry run; "
            "perform no M1 scientific computation."
        )
    )
    parser.add_argument(
        "--authorization", type=Path, default=DEFAULT_AUTHORIZATION
    )
    parser.add_argument("--artifact", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    authorization_path = args.authorization.resolve()
    authorization = _load_json(authorization_path)
    source_hashes = _validate_authorization(authorization)
    s2_ledger = _read_s2_ledger(authorization)
    payload = build_dry_run(
        authorization_sha256=file_sha256(authorization_path),
        source_hashes=source_hashes,
        s2_ledger=s2_ledger,
    )
    artifact = args.artifact.resolve()
    write_json_artifact(payload, artifact)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "artifact": str(artifact),
                "artifact_fingerprint": payload["artifact_fingerprint"],
                "base_candidate_count": payload["candidate_counts"]["base_total"],
                "maximum_signal_candidate_count": payload["candidate_counts"][
                    "maximum_signal_candidates"
                ],
                "selected_synthetic_compile_cells": payload[
                    "selector_dry_run"
                ]["selection"]["selected_count"],
                "m1_scientific_execution_authorized": False,
                "mandatory_stop_reached": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
