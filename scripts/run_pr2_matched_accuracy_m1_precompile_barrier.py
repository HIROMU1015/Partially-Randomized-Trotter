#!/usr/bin/env python3
"""Run the zero-compute M1-A/M1-B precompile-barrier dry run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = (
    ROOT / "artifacts/pr2_matched_accuracy_m1_contract/2026-09-29"
)
DEFAULT_AUTHORIZATION = (
    ARTIFACT_ROOT
    / "pr2_matched_accuracy_m1_preexecution_amendment_authorization_v2.json"
)
DEFAULT_OUTPUT = (
    ARTIFACT_ROOT
    / "pr2_matched_accuracy_m1_precompile_barrier_dry_run_v2.json"
)
V1_SOURCE = ROOT / "src/trotterlib/pr2_matched_accuracy_m1_contract.py"
BARRIER_SOURCE = (
    ROOT / "src/trotterlib/pr2_matched_accuracy_m1_precompile_barrier.py"
)


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _fingerprint(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _validate_authorization(
    authorization: Mapping[str, Any],
) -> dict[str, str]:
    expected_schema = (
        "pr2_matched_accuracy_m1_preexecution_amendment_authorization_v2"
    )
    expected_status = (
        "M1_PREEXECUTION_AMENDMENT_V2_FROZEN_SCIENCE_NOT_AUTHORIZED"
    )
    if authorization.get("schema_version") != expected_schema:
        raise ValueError("unexpected preexecution amendment authorization schema")
    if authorization.get("status") != expected_status:
        raise ValueError("unexpected preexecution amendment authorization status")
    if authorization.get("implementation_base_commit") != (
        "0ee4d641c10f4e248d2b03ebc59703c56f10e581"
    ):
        raise ValueError("implementation base commit differs from the frozen value")

    permissions = authorization.get("permissions")
    if not isinstance(permissions, Mapping):
        raise ValueError("authorization permissions are missing")
    if not permissions.get("amendment_dry_run_authorized"):
        raise ValueError("amendment dry run is not authorized")
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
    if any(permissions.get(name) is not False for name in forbidden):
        raise ValueError("authorization improperly enables scientific work")
    if permissions.get("automatic_next_stage") is not None:
        raise ValueError("authorization defines an automatic next stage")

    source_hashes = authorization.get("required_source_hashes")
    if not isinstance(source_hashes, Mapping) or len(source_hashes) < 7:
        raise ValueError("authorization has an incomplete source-hash set")
    verified: dict[str, str] = {}
    for relative, expected in source_hashes.items():
        path = ROOT / str(relative)
        if not path.is_file() or _file_sha256(path) != expected:
            raise ValueError(f"source hash differs from authorization: {relative}")
        verified[str(relative)] = str(expected)
    return verified


def _clear_control(selection: Mapping[str, Any]) -> dict[str, Any]:
    clear = json.loads(json.dumps(selection))
    clear["unselected_proxy_frontier_count"] = 0
    clear["unselected_proxy_frontier_fingerprints"] = []
    clear["unselected_boundary_fingerprints"] = []
    clear["unselected_tail_challenger_fingerprints"] = []
    clear["selection_limited"] = False
    clear["selection_limited_reasons"] = []
    return clear


def _build_payload(
    authorization_sha256: str,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    contract = _load_module("pr2_m1_contract_v1_for_barrier", V1_SOURCE)
    barrier = _load_module("pr2_m1_precompile_barrier_v2", BARRIER_SOURCE)

    candidates = contract.enumerate_base_candidates()
    fixture = contract.selector_fixture(candidates)
    limited_selection = fixture["selection"]
    limited_barrier = barrier.evaluate_precompile_barrier(limited_selection)
    if not limited_barrier["selection_limited"]:
        raise ValueError("frozen limited fixture did not exercise the stop branch")
    try:
        barrier.build_m1_b_compile_plan(
            limited_selection,
            deterministic_or_discard_fingerprints=[],
        )
    except barrier.SelectionLimitedStop:
        compile_plan_created = False
    else:
        raise ValueError("limited fixture improperly created a compile plan")

    clear_selection = _clear_control(limited_selection)
    clear_barrier = barrier.evaluate_precompile_barrier(clear_selection)
    deterministic = [
        item["candidate_fingerprint"]
        for item in candidates
        if item["method"] in {"B0", "B1"}
    ]
    clear_plan = barrier.build_m1_b_compile_plan(
        clear_selection,
        deterministic_or_discard_fingerprints=deterministic,
    )

    zero_names = (
        "development_npz_loads",
        "held_out_npz_loads",
        "molecular_calculations",
        "signal_evaluations",
        "random_trajectories_sampled",
        "circuits_built",
        "circuit_compilations",
        "full_wrappers_compiled",
        "quantum_shots_executed",
    )
    payload: dict[str, Any] = {
        "schema_version": (
            "pr2_matched_accuracy_m1_precompile_barrier_dry_run_v2"
        ),
        "series_id": "pr2-rebaseline-de7a5492-v1",
        "status": (
            "M1_PREEXECUTION_AMENDMENT_V2_FROZEN_SCIENCE_NOT_AUTHORIZED"
        ),
        "authorization_sha256": authorization_sha256,
        "source_hashes": dict(sorted(source_hashes.items())),
        "prior_art_gate": {
            "decision": "PROCEED_RESOURCE_STUDY",
            "papers_added": [
                "arXiv:2603.13495",
                "arXiv:2603.22778",
            ],
            "claim_scope": (
                "fixed_df_prefix_matched_accuracy_resource_study_only"
            ),
        },
        "limited_fixture": {
            "fixture_only": True,
            "scientific_values": False,
            "selector_fixture_sha256": _fingerprint(fixture),
            "barrier": limited_barrier,
            "compile_plan_created": compile_plan_created,
            "compile_jobs_created": 0,
        },
        "clear_control": {
            "fixture_only": True,
            "scientific_values": False,
            "barrier": clear_barrier,
            "compile_plan": clear_plan,
        },
        "zero_compute_counters": {name: 0 for name in zero_names},
        "permissions": {
            "m1_scientific_execution_authorized": False,
            "direct_compile_authorized": False,
            "held_out_access_authorized": False,
            "s3_authorized": False,
            "automatic_next_stage": None,
        },
        "mandatory_stop_reached": True,
    }
    payload["artifact_fingerprint"] = _fingerprint(payload)
    return payload


def _write_non_overwrite(payload: Mapping[str, Any], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Exercise the M1 precompile hard barrier without science."
    )
    parser.add_argument("--authorization", type=Path, default=DEFAULT_AUTHORIZATION)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    authorization_path = args.authorization.resolve()
    authorization = _load_json(authorization_path)
    source_hashes = _validate_authorization(authorization)
    payload = _build_payload(
        _file_sha256(authorization_path),
        source_hashes,
    )
    output = args.artifact.resolve()
    _write_non_overwrite(payload, output)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "artifact": str(output),
                "artifact_fingerprint": payload["artifact_fingerprint"],
                "limited_fixture_status": payload["limited_fixture"]["barrier"][
                    "status"
                ],
                "limited_compile_jobs_created": 0,
                "m1_scientific_execution_authorized": False,
                "mandatory_stop_reached": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
