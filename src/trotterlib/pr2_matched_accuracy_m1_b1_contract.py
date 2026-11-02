"""Zero-compute contract for the bounded PR-2 M1-B1 compile expansion.

This module is intentionally standard-library-only.  It reads the frozen M1-A
result, fixes the eligible random cells and all deterministic/discard baseline
cells, and materializes identities for future compile tasks.  It never loads a
molecular snapshot, samples a trajectory, builds a circuit, or invokes a
compiler.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "pr2_matched_accuracy_m1_b1_zero_compute_plan_v1"
AUTHORIZATION_SCHEMA_VERSION = (
    "pr2_matched_accuracy_m1_b1_implementation_authorization_v1"
)
STATUS = "M1_B1_BOUNDED_COMPILE_CONTRACT_FROZEN_EXECUTION_NOT_AUTHORIZED"
SERIES_ID = "pr2-rebaseline-de7a5492-v1"

M1_A_RESULT_RELATIVE = (
    "artifacts/pr2_matched_accuracy_m1_execution/2026-09-30/"
    "pr2_matched_accuracy_m1_a_result_v1.json"
)
M1_A_RESULT_COMMIT = "3c1831e326c27c5f679b3820997f27916d26ed9f"
M1_A_RESULT_SHA256 = (
    "1f960d7a33296e2dcb74d497e360572b26409dc9aeae01522335e7b91ed81086"
)
M1_A_RESULT_FINGERPRINT = (
    "422f898bba1e3849d0f45830082b76d4f42da436e2b49796e562cd79fc716c9e"
)

RANDOM_METHODS = ("B2", "B3")
BASELINE_METHODS = ("B0", "B1")
AXES = ("cosine", "sine")
TRAJECTORIES_PER_RANDOM_CELL = 32
RANDOM_CELL_COUNT = 194
BASELINE_CELL_COUNT = 16
RANDOM_TRAJECTORY_COUNT = RANDOM_CELL_COUNT * TRAJECTORIES_PER_RANDOM_CELL
RANDOM_WRAPPER_COUNT = RANDOM_TRAJECTORY_COUNT * len(AXES)
BASELINE_WRAPPER_COUNT = BASELINE_CELL_COUNT * len(AXES)
TOTAL_WRAPPER_COUNT = RANDOM_WRAPPER_COUNT + BASELINE_WRAPPER_COUNT
MAXIMUM_WORKERS = 6
MASTER_SEED = 2026093001

COMPILER_IDENTITY = {
    "qiskit_version": "1.3.0",
    "basis_gates": ["rz", "sx", "x", "cx"],
    "optimization_level": 1,
    "transpiler_seed": 17,
    "coupling_map": None,
    "backend_name": None,
    "layout_method": None,
    "routing_method": None,
}

POST_B1_DECISIONS = (
    "CONTINUE_RESOURCE_STUDY",
    "NARROW_TO_TECHNICAL_NOTE",
    "STOP_DUPLICATIVE",
    "COMPILE_RESULT_INCONCLUSIVE",
)

ZERO_COMPUTE_COUNTER_NAMES = (
    "development_npz_loads",
    "held_out_path_stats",
    "held_out_hash_reads",
    "held_out_npz_loads",
    "signal_evaluations",
    "trajectory_samples",
    "occurrence_samples",
    "circuits_built",
    "compiler_invocations",
    "full_wrappers_compiled",
    "quantum_shots_executed",
    "molecular_calculations",
    "gpu_queries",
    "gpu_allocations",
    "gpu_kernels",
)


def canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def fingerprint(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def validate_m1_a_result(payload: Mapping[str, Any], *, sha256: str) -> None:
    if sha256 != M1_A_RESULT_SHA256:
        raise ValueError("M1-A result SHA-256 differs from the frozen value")
    if payload.get("schema_version") != "pr2_matched_accuracy_m1_a_result_v2":
        raise ValueError("unexpected M1-A result schema")
    if payload.get("series_id") != SERIES_ID:
        raise ValueError("unexpected M1-A series")
    if payload.get("status") != "SELECTION_LIMITED":
        raise ValueError("M1-A result is not the frozen SELECTION_LIMITED result")
    if payload.get("result_fingerprint") != M1_A_RESULT_FINGERPRINT:
        raise ValueError("M1-A result fingerprint differs from the frozen value")
    selection = payload.get("compile_selection", {})
    if selection.get("compile_cap") != 16 or selection.get("selection_limited") is not True:
        raise ValueError("M1-A selector audit is not the frozen limited result")
    if selection.get("proxy_frontier_count") != 64:
        raise ValueError("M1-A proxy frontier count differs from the frozen value")
    if selection.get("unselected_proxy_frontier_count") != 52:
        raise ValueError("M1-A unselected frontier count differs from the frozen value")
    if payload.get("compile_records") != []:
        raise ValueError("M1-A unexpectedly contains compile records")
    counters = payload.get("counters", {})
    for name in (
        "random_trajectories_sampled",
        "random_trajectories_compiled",
        "circuits_built",
        "circuit_compilations",
        "full_wrappers_compiled",
        "quantum_shots_executed",
    ):
        if counters.get(name) != 0:
            raise ValueError(f"M1-A counter must remain zero: {name}")


def _validate_candidate(candidate: Mapping[str, Any]) -> None:
    identity = {
        key: value
        for key, value in candidate.items()
        if key not in {"candidate_id", "candidate_fingerprint"}
    }
    if fingerprint(identity) != candidate.get("candidate_fingerprint"):
        raise ValueError("candidate fingerprint does not match candidate identity")
    if candidate.get("compiler_identity") != {
        key: COMPILER_IDENTITY[key]
        for key in (
            "qiskit_version",
            "basis_gates",
            "optimization_level",
            "transpiler_seed",
            "coupling_map",
        )
    }:
        raise ValueError("candidate compiler identity differs from M1-B1 compiler")


def frozen_cells(payload: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = payload.get("signal_records")
    if not isinstance(records, list):
        raise ValueError("M1-A signal records are missing")
    random_cells: list[dict[str, Any]] = []
    baseline_cells: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("candidate"), dict):
            raise ValueError("invalid M1-A signal record")
        candidate = dict(record["candidate"])
        _validate_candidate(candidate)
        method = str(candidate["method"])
        cell = {
            "candidate_id": str(candidate["candidate_id"]),
            "candidate_fingerprint": str(candidate["candidate_fingerprint"]),
            "method": method,
            "rank": int(candidate["rank"]),
            "q": int(candidate["q"]),
            "r": int(candidate["r"]),
            "K": int(candidate["K"]),
            "accuracy_eligible": bool(record["accuracy_eligible"]),
            "signal_record_fingerprint": fingerprint(record),
        }
        if method in RANDOM_METHODS:
            if record.get("accuracy_eligible") is True:
                random_cells.append(cell)
        elif method in BASELINE_METHODS:
            baseline_cells.append(cell)
        else:
            raise ValueError(f"unsupported method in M1-A result: {method}")
    random_cells.sort(key=lambda item: item["candidate_id"])
    baseline_cells.sort(key=lambda item: item["candidate_id"])
    random_counts = {
        method: sum(item["method"] == method for item in random_cells)
        for method in RANDOM_METHODS
    }
    baseline_counts = {
        method: sum(item["method"] == method for item in baseline_cells)
        for method in BASELINE_METHODS
    }
    if random_counts != {"B2": 145, "B3": 49}:
        raise ValueError("frozen random method counts differ from M1-A")
    if baseline_counts != {"B0": 12, "B1": 4}:
        raise ValueError("frozen baseline method counts differ from M1-A")
    if len(random_cells) != RANDOM_CELL_COUNT:
        raise ValueError("frozen random cell count differs from 194")
    if len(baseline_cells) != BASELINE_CELL_COUNT:
        raise ValueError("frozen baseline cell count differs from 16")
    fingerprints = [item["candidate_fingerprint"] for item in random_cells + baseline_cells]
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("frozen candidate fingerprints are not unique")
    return random_cells, baseline_cells


def trajectory_seed(candidate_fingerprint: str, trajectory_index: int) -> int:
    if trajectory_index < 0 or trajectory_index >= TRAJECTORIES_PER_RANDOM_CELL:
        raise ValueError("trajectory index is outside the frozen B1 range")
    payload = {
        "policy": "pr2_m1_b1_candidate_trajectory_sha256_v1",
        "master_seed": MASTER_SEED,
        "candidate_fingerprint": candidate_fingerprint,
        "trajectory_index": trajectory_index,
    }
    return int.from_bytes(hashlib.sha256(canonical_json(payload)).digest()[:8], "big")


def occurrence_seed(
    candidate_fingerprint: str,
    trajectory_index: int,
    *,
    outer_step: int,
    tail_occurrence: int,
    rte_step: int,
) -> int:
    coordinates = (outer_step, tail_occurrence, rte_step)
    if any(value < 0 for value in coordinates):
        raise ValueError("occurrence coordinates must be non-negative")
    payload = {
        "policy": "pr2_m1_b1_occurrence_sha256_v1",
        "trajectory_seed": trajectory_seed(candidate_fingerprint, trajectory_index),
        "candidate_fingerprint": candidate_fingerprint,
        "trajectory_index": trajectory_index,
        "outer_step": outer_step,
        "tail_occurrence": tail_occurrence,
        "rte_step": rte_step,
    }
    return int.from_bytes(hashlib.sha256(canonical_json(payload)).digest()[:8], "big")


def wrapper_identity(
    *,
    source_commit: str,
    candidate_fingerprint: str,
    axis: str,
    trajectory_index: int | None,
) -> dict[str, Any]:
    if len(source_commit) != 40 or any(ch not in "0123456789abcdef" for ch in source_commit):
        raise ValueError("source commit must be a full lowercase Git object name")
    if axis not in AXES:
        raise ValueError("unsupported wrapper axis")
    seed = None
    if trajectory_index is not None:
        seed = trajectory_seed(candidate_fingerprint, trajectory_index)
    return {
        "schema_version": "pr2_matched_accuracy_m1_b1_wrapper_identity_v1",
        "source_commit": source_commit,
        "m1_a_result_sha256": M1_A_RESULT_SHA256,
        "compiler_identity": COMPILER_IDENTITY,
        "compiler_fingerprint": fingerprint(COMPILER_IDENTITY),
        "candidate_fingerprint": candidate_fingerprint,
        "axis": axis,
        "trajectory_index": trajectory_index,
        "trajectory_seed": seed,
        "wrapper_semantics": "full_measured_hadamard_wrapper_without_state_preparation",
    }


def _cell_plan(cell: Mapping[str, Any], *, source_commit: str, random: bool) -> dict[str, Any]:
    indices: Sequence[int | None]
    indices = range(TRAJECTORIES_PER_RANDOM_CELL) if random else (None,)
    keys: dict[str, list[str]] = {}
    for axis in AXES:
        keys[axis] = [
            fingerprint(
                wrapper_identity(
                    source_commit=source_commit,
                    candidate_fingerprint=str(cell["candidate_fingerprint"]),
                    axis=axis,
                    trajectory_index=index,
                )
            )
            for index in indices
        ]
    flattened = [key for axis in AXES for key in keys[axis]]
    return {
        **dict(cell),
        "trajectory_count": TRAJECTORIES_PER_RANDOM_CELL if random else 0,
        "wrapper_count": len(flattened),
        "wrapper_cache_keys": keys,
        "wrapper_cache_key_set_fingerprint": fingerprint(flattened),
    }


def build_plan(
    *,
    m1_a_result: Mapping[str, Any],
    m1_a_result_sha256: str,
    authorization_sha256: str,
    source_hashes: Mapping[str, str],
    source_commit: str,
) -> dict[str, Any]:
    validate_m1_a_result(m1_a_result, sha256=m1_a_result_sha256)
    random_cells, baseline_cells = frozen_cells(m1_a_result)
    random_plan = [
        _cell_plan(cell, source_commit=source_commit, random=True)
        for cell in random_cells
    ]
    baseline_plan = [
        _cell_plan(cell, source_commit=source_commit, random=False)
        for cell in baseline_cells
    ]
    ordered_keys = [
        key
        for cell in random_plan + baseline_plan
        for axis in AXES
        for key in cell["wrapper_cache_keys"][axis]
    ]
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "series_id": SERIES_ID,
        "status": STATUS,
        "source_commit": source_commit,
        "authorization_sha256": authorization_sha256,
        "source_hashes": dict(sorted(source_hashes.items())),
        "m1_a_result": {
            "path": M1_A_RESULT_RELATIVE,
            "commit": M1_A_RESULT_COMMIT,
            "sha256": m1_a_result_sha256,
            "result_fingerprint": M1_A_RESULT_FINGERPRINT,
            "status": "SELECTION_LIMITED",
            "legacy_selector_audit_preserved": True,
            "legacy_compile_cap": 16,
            "legacy_proxy_frontier_count": 64,
            "legacy_unselected_proxy_frontier_count": 52,
        },
        "decision": {
            "review_decision": "PROCEED_BOUNDED_COMPILE_EXPANSION",
            "old_selector_reinterpreted_as_failure": False,
            "bounded_direct_compile_selected": True,
            "science_execution_authorized": False,
            "automatic_next_stage": None,
        },
        "compiler_identity": COMPILER_IDENTITY,
        "compiler_fingerprint": fingerprint(COMPILER_IDENTITY),
        "seed_policy": {
            "trajectory": "pr2_m1_b1_candidate_trajectory_sha256_v1",
            "occurrence": "pr2_m1_b1_occurrence_sha256_v1",
            "master_seed": MASTER_SEED,
            "axis_excluded_from_random_trajectory": True,
            "same_random_trajectory_shared_by_cosine_and_sine": True,
            "occurrence_coordinates": [
                "candidate_fingerprint",
                "trajectory_index",
                "outer_step",
                "tail_occurrence",
                "rte_step",
            ],
        },
        "cache_identity_policy": {
            "required_coordinates": [
                "source_commit",
                "compiler_fingerprint",
                "candidate_fingerprint",
                "axis",
                "trajectory_index",
                "trajectory_seed",
            ],
            "reuse_requires_exact_identity": True,
            "cross_candidate_reuse_permitted": False,
            "partial_cache_cross_cell_reuse_permitted": False,
        },
        "resource_caps": {
            "random_cells": RANDOM_CELL_COUNT,
            "trajectories_per_random_cell": TRAJECTORIES_PER_RANDOM_CELL,
            "random_trajectories": RANDOM_TRAJECTORY_COUNT,
            "random_full_wrappers": RANDOM_WRAPPER_COUNT,
            "deterministic_or_discard_cells": BASELINE_CELL_COUNT,
            "deterministic_or_discard_full_wrappers": BASELINE_WRAPPER_COUNT,
            "total_full_wrappers": TOTAL_WRAPPER_COUNT,
            "axes_per_cell_or_trajectory": 2,
            "maximum_process_workers": MAXIMUM_WORKERS,
            "blas_threads_per_worker": 1,
            "extension_trajectories": 0,
            "maximum_trajectories_per_random_cell": TRAJECTORIES_PER_RANDOM_CELL,
        },
        "random_cells": random_plan,
        "deterministic_or_discard_cells": baseline_plan,
        "ordered_wrapper_cache_keys_sha256": fingerprint(ordered_keys),
        "post_b1_contract": {
            "allowed_decisions": list(POST_B1_DECISIONS),
            "stop_after_actual_compiled_resource_map": True,
            "additional_96_trajectories_authorized": False,
            "held_out_candidate_freeze_authorized": False,
            "held_out_access_authorized": False,
            "transfer_authorized": False,
            "winner_precision_extension_authorized": False,
            "s3_authorized": False,
        },
        "zero_compute_counters": {name: 0 for name in ZERO_COMPUTE_COUNTER_NAMES},
    }
    body["plan_fingerprint"] = fingerprint(body)
    validate_plan(body)
    return body


def _all_cache_keys(payload: Mapping[str, Any]) -> list[str]:
    return [
        str(key)
        for cell in list(payload["random_cells"])
        + list(payload["deterministic_or_discard_cells"])
        for axis in AXES
        for key in cell["wrapper_cache_keys"][axis]
    ]


def validate_plan(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("status") != STATUS:
        raise ValueError("unexpected M1-B1 zero-compute plan identity")
    body = {key: value for key, value in payload.items() if key != "plan_fingerprint"}
    if fingerprint(body) != payload.get("plan_fingerprint"):
        raise ValueError("M1-B1 plan fingerprint mismatch")
    if payload["decision"]["science_execution_authorized"] is not False:
        raise ValueError("zero-compute plan improperly authorizes science")
    random_cells = list(payload["random_cells"])
    baseline_cells = list(payload["deterministic_or_discard_cells"])
    if len(random_cells) != RANDOM_CELL_COUNT or len(baseline_cells) != BASELINE_CELL_COUNT:
        raise ValueError("M1-B1 cell counts differ from the frozen contract")
    if any(cell["wrapper_count"] != 64 for cell in random_cells):
        raise ValueError("random cell does not contain 32 paired-axis trajectories")
    if any(cell["wrapper_count"] != 2 for cell in baseline_cells):
        raise ValueError("baseline cell does not contain two axes")
    if any(cell["method"] not in RANDOM_METHODS for cell in random_cells):
        raise ValueError("non-random method in random B1 plan")
    if any(cell["method"] not in BASELINE_METHODS for cell in baseline_cells):
        raise ValueError("non-baseline method in baseline B1 plan")
    if any(cell["accuracy_eligible"] is not True for cell in random_cells):
        raise ValueError("ineligible random cell in B1 plan")
    keys = _all_cache_keys(payload)
    if len(keys) != TOTAL_WRAPPER_COUNT or len(keys) != len(set(keys)):
        raise ValueError("M1-B1 wrapper cache keys are missing or duplicated")
    if fingerprint(keys) != payload.get("ordered_wrapper_cache_keys_sha256"):
        raise ValueError("M1-B1 ordered wrapper key digest mismatch")
    source_commit = str(payload["source_commit"])
    for cell in random_cells:
        for index in range(TRAJECTORIES_PER_RANDOM_CELL):
            cosine = fingerprint(
                wrapper_identity(
                    source_commit=source_commit,
                    candidate_fingerprint=cell["candidate_fingerprint"],
                    axis="cosine",
                    trajectory_index=index,
                )
            )
            sine = fingerprint(
                wrapper_identity(
                    source_commit=source_commit,
                    candidate_fingerprint=cell["candidate_fingerprint"],
                    axis="sine",
                    trajectory_index=index,
                )
            )
            if cell["wrapper_cache_keys"]["cosine"][index] != cosine:
                raise ValueError("cosine wrapper identity mismatch")
            if cell["wrapper_cache_keys"]["sine"][index] != sine:
                raise ValueError("sine wrapper identity mismatch")
            identity_c = wrapper_identity(
                source_commit=source_commit,
                candidate_fingerprint=cell["candidate_fingerprint"],
                axis="cosine",
                trajectory_index=index,
            )
            identity_s = wrapper_identity(
                source_commit=source_commit,
                candidate_fingerprint=cell["candidate_fingerprint"],
                axis="sine",
                trajectory_index=index,
            )
            if identity_c["trajectory_seed"] != identity_s["trajectory_seed"]:
                raise ValueError("paired axes do not share the frozen trajectory")
    caps = payload["resource_caps"]
    expected_caps = {
        "random_cells": RANDOM_CELL_COUNT,
        "trajectories_per_random_cell": TRAJECTORIES_PER_RANDOM_CELL,
        "random_trajectories": RANDOM_TRAJECTORY_COUNT,
        "random_full_wrappers": RANDOM_WRAPPER_COUNT,
        "deterministic_or_discard_cells": BASELINE_CELL_COUNT,
        "deterministic_or_discard_full_wrappers": BASELINE_WRAPPER_COUNT,
        "total_full_wrappers": TOTAL_WRAPPER_COUNT,
    }
    if any(caps.get(key) != value for key, value in expected_caps.items()):
        raise ValueError("M1-B1 resource cap mismatch")
    if any(value != 0 for value in payload["zero_compute_counters"].values()):
        raise ValueError("M1-B1 zero-compute counter is nonzero")
    post = payload["post_b1_contract"]
    if tuple(post["allowed_decisions"]) != POST_B1_DECISIONS:
        raise ValueError("M1-B1 post-run decisions differ from the contract")
    forbidden = (
        "additional_96_trajectories_authorized",
        "held_out_candidate_freeze_authorized",
        "held_out_access_authorized",
        "transfer_authorized",
        "winner_precision_extension_authorized",
        "s3_authorized",
    )
    if any(post[name] for name in forbidden):
        raise ValueError("M1-B1 plan improperly authorizes a later stage")


def write_json_artifact(payload: Mapping[str, Any], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
