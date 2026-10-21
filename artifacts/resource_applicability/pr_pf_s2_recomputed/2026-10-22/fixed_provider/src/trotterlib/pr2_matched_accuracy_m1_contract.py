"""Zero-compute contract utilities for the PR-2 matched-accuracy M1 study.

This module is intentionally standard-library only.  It enumerates the frozen
candidate space, creates immutable candidate identities, derives occurrence
seeds, and exercises the result-prior compile selector on a synthetic fixture.
It does not load an NPZ, evaluate a signal, build a circuit, or compile one.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "pr2_matched_accuracy_m1_contract_dry_run_v1"
AUTHORIZATION_SCHEMA_VERSION = (
    "pr2_matched_accuracy_m1_implementation_authorization_v1"
)
STATUS = "M1_IMPLEMENTATION_CONTRACT_FROZEN_SCIENCE_NOT_AUTHORIZED"
SERIES_ID = "pr2-rebaseline-de7a5492-v1"

BASE_COMMIT = "61bbaadfa852f569726b7a392f231606790b2aac"
RESOURCE_CONTRACT_SHA256 = (
    "313843ef55e6740d1ab37e1d0fac0e2c68243445e9235bbc45cf92b481305416"
)
PRIOR_ART_GATE_SHA256 = (
    "39faf5aeaa25d391c3f537f4a89da7862aad5ad1dccc3531c717bcaa0b59cf2d"
)

DEVELOPMENT_SNAPSHOT_SHA256 = (
    "3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a"
)
DEVELOPMENT_HAMILTONIAN_HASH = (
    "de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424"
)
DEVELOPMENT_STATE_HASH = (
    "31e63b0104126c85136ee173f1dce7642aee2d272924e70e8b120ac340ab45bd"
)
DEVELOPMENT_STATE_VECTOR_HASH = (
    "c9aca811b5c023772d148d0331c82958bac6824b593a367cb65a5f89937f4f63"
)

TOTAL_TIME = 0.8
Q_VALUES = (1, 2, 4, 8)
R_VALUES = (1, 2, 4, 8, 16, 32)
K_VALUES = (2, 4)
DISCARD_RANKS = (3, 6, 9)
PARTIAL_RANKS = (3, 6, 9)
RANDOM_SPLITS = (0, 3, 6, 9)
BOUNDARY_R = 64
MAXIMUM_BOUNDARY_CANDIDATES = 4
MAXIMUM_RANDOM_COMPILE_CELLS = 16
INITIAL_TRAJECTORIES = 32
EXTENSION_TRAJECTORIES = 96
MAXIMUM_TRAJECTORIES_PER_CELL = 128
MASTER_SEED_INITIAL = 20260929101
MASTER_SEED_EXTENSION = 20260929102

ZERO_COMPUTE_COUNTER_NAMES = (
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

_METHODS = {
    "B0": {
        "mode": "discard",
        "random": False,
        "semantics": "df_prefix_discard_residual_deterministic_s2",
    },
    "B1": {
        "mode": "deterministic",
        "random": False,
        "semantics": "rank12_deterministic_s2",
    },
    "B2": {
        "mode": "partial",
        "random": True,
        "semantics": "df_prefix_deterministic_backbone_finite_rte_tail",
    },
    "B3": {
        "mode": "random_dominant",
        "random": True,
        "semantics": "rank0_two_body_random_one_body_deterministic",
    },
}


def canonical_json(payload: Any) -> bytes:
    """Return the canonical JSON representation used by all fingerprints."""

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


def _candidate_identity(
    method: str,
    rank: int,
    q: int,
    r: int,
    cutoff: int,
    *,
    boundary_parent_fingerprint: str | None = None,
) -> dict[str, Any]:
    if method not in _METHODS:
        raise ValueError(f"Unknown method: {method}")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError("rank must be a non-negative integer")
    if q not in Q_VALUES:
        raise ValueError(f"q must be one of {Q_VALUES}")
    random = bool(_METHODS[method]["random"])
    if random:
        if r not in (*R_VALUES, BOUNDARY_R) or cutoff not in K_VALUES:
            raise ValueError("random candidates require a frozen r/K value")
    elif r != 0 or cutoff != 0:
        raise ValueError("deterministic candidates require r=K=0")
    if method == "B0" and rank not in DISCARD_RANKS:
        raise ValueError("B0 rank is outside the frozen discard set")
    if method == "B1" and rank != 12:
        raise ValueError("B1 must use rank 12")
    if method == "B2" and rank not in PARTIAL_RANKS:
        raise ValueError("B2 rank is outside the frozen partial set")
    if method == "B3" and rank != 0:
        raise ValueError("B3 must use rank 0")
    if r == BOUNDARY_R and boundary_parent_fingerprint is None:
        raise ValueError("r=64 candidates require their frozen r=32 parent")
    if r != BOUNDARY_R and boundary_parent_fingerprint is not None:
        raise ValueError("only r=64 candidates may carry a boundary parent")

    delta = TOTAL_TIME / q
    return {
        "series_id": SERIES_ID,
        "snapshot_sha256": DEVELOPMENT_SNAPSHOT_SHA256,
        "hamiltonian_hash": DEVELOPMENT_HAMILTONIAN_HASH,
        "state_hash": DEVELOPMENT_STATE_HASH,
        "state_vector_hash": DEVELOPMENT_STATE_VECTOR_HASH,
        "method": method,
        "mode": _METHODS[method]["mode"],
        "method_semantics": _METHODS[method]["semantics"],
        "rank": rank,
        "T": TOTAL_TIME,
        "T_hex": TOTAL_TIME.hex(),
        "q": q,
        "delta": delta,
        "delta_hex": delta.hex(),
        "r": r,
        "K": cutoff,
        "boundary_parent_fingerprint": boundary_parent_fingerprint,
        "identity_policy": "extract_identity_phase",
        "coefficient_atol": 0.0,
        "outer_formula": "symmetric_second_order_product_formula",
        "wrapper_semantics": (
            "full_measured_hadamard_wrapper_without_state_preparation"
        ),
        "compiler_identity": {
            "qiskit_version": "1.3.0",
            "basis_gates": ["rz", "sx", "x", "cx"],
            "optimization_level": 1,
            "transpiler_seed": 17,
            "coupling_map": None,
        },
        "seed_policy": (
            "sha256_v1_candidate_axis_trajectory_outer_step_"
            "tail_occurrence_rte_step"
        ),
    }


def make_candidate(
    method: str,
    rank: int,
    q: int,
    r: int = 0,
    cutoff: int = 0,
    *,
    boundary_parent_fingerprint: str | None = None,
) -> dict[str, Any]:
    identity = _candidate_identity(
        method,
        rank,
        q,
        r,
        cutoff,
        boundary_parent_fingerprint=boundary_parent_fingerprint,
    )
    candidate_fingerprint = fingerprint(identity)
    return {
        "candidate_id": f"{method}-rank{rank}-q{q}-r{r}-K{cutoff}",
        "candidate_fingerprint": candidate_fingerprint,
        **identity,
    }


def enumerate_base_candidates() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for rank in DISCARD_RANKS:
        for q in Q_VALUES:
            candidates.append(make_candidate("B0", rank, q))
    for q in Q_VALUES:
        candidates.append(make_candidate("B1", 12, q))
    for rank in PARTIAL_RANKS:
        for q in Q_VALUES:
            for r in R_VALUES:
                for cutoff in K_VALUES:
                    candidates.append(make_candidate("B2", rank, q, r, cutoff))
    for q in Q_VALUES:
        for r in R_VALUES:
            for cutoff in K_VALUES:
                candidates.append(make_candidate("B3", 0, q, r, cutoff))
    validate_candidate_ledger(candidates)
    return candidates


def candidate_counts(candidates: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {method: 0 for method in _METHODS}
    for candidate in candidates:
        counts[str(candidate["method"])] += 1
    return {
        **counts,
        "deterministic_or_discard": counts["B0"] + counts["B1"],
        "random_base": counts["B2"] + counts["B3"],
        "base_total": len(candidates),
        "maximum_boundary_additions": MAXIMUM_BOUNDARY_CANDIDATES,
        "maximum_signal_candidates": len(candidates)
        + MAXIMUM_BOUNDARY_CANDIDATES,
        "maximum_random_direct_compile_cells": MAXIMUM_RANDOM_COMPILE_CELLS,
    }


def validate_candidate_ledger(candidates: Sequence[Mapping[str, Any]]) -> None:
    fingerprints = [str(item["candidate_fingerprint"]) for item in candidates]
    identifiers = [str(item["candidate_id"]) for item in candidates]
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("candidate fingerprints are not unique")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate identifiers are not unique")
    for candidate in candidates:
        q = int(candidate["q"])
        delta = float(candidate["delta"])
        if not math.isclose(q * delta, TOTAL_TIME, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("candidate violates q*delta=T")
        identity = {
            key: value
            for key, value in candidate.items()
            if key not in {"candidate_id", "candidate_fingerprint"}
        }
        if fingerprint(identity) != candidate["candidate_fingerprint"]:
            raise ValueError("candidate fingerprint does not match its identity")


def occurrence_seed(
    candidate_fingerprint: str,
    *,
    axis: str,
    trajectory: int,
    outer_step: int,
    tail_occurrence: int,
    rte_step: int,
    master_seed: int,
) -> int:
    if axis not in {"cosine", "sine"}:
        raise ValueError("axis must be cosine or sine")
    coordinates = (trajectory, outer_step, tail_occurrence, rte_step)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in coordinates):
        raise ValueError("seed coordinates must be non-negative integers")
    payload = {
        "policy": "pr2_m1_occurrence_seed_sha256_v1",
        "master_seed": int(master_seed),
        "candidate_fingerprint": candidate_fingerprint,
        "axis": axis,
        "trajectory": trajectory,
        "outer_step": outer_step,
        "tail_occurrence": tail_occurrence,
        "rte_step": rte_step,
    }
    return int.from_bytes(hashlib.sha256(canonical_json(payload)).digest()[:8], "big")


def _split(candidate: Mapping[str, Any]) -> int:
    return int(candidate["rank"])


def _fixture_proxy(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Create a non-scientific selector fixture from candidate metadata only."""

    if candidate["method"] not in {"B2", "B3"}:
        raise ValueError("selector fixture accepts random candidates only")
    rank = int(candidate["rank"])
    q = int(candidate["q"])
    r = int(candidate["r"])
    cutoff = int(candidate["K"])
    split_penalty = {0: 900, 3: 620, 6: 440, 9: 280}[rank]
    total_shots = int(math.ceil(520_000 / (r * cutoff))) + split_penalty + 17 * q
    n_det = q * (rank + 1)
    n_rand = q * r * (cutoff + 1) * (13 - rank)
    n_fixed = 2 * q + 3
    action = n_det + n_rand + n_fixed
    return {
        "candidate": dict(candidate),
        "candidate_fingerprint": candidate["candidate_fingerprint"],
        "accuracy_eligible": True,
        "total_shots": total_shots,
        "n_det": n_det,
        "n_rand": n_rand,
        "n_fixed": n_fixed,
        "W_action": total_shots * action,
        "W_tail": total_shots * n_rand,
        "fixture_only": True,
    }


def _proxy_key(record: Mapping[str, Any], metric: str) -> tuple[Any, ...]:
    candidate = record["candidate"]
    return (
        int(record[metric]),
        int(record["total_shots"]),
        int(record["n_det"]),
        int(record["n_rand"]),
        str(candidate["method"]),
        int(candidate["rank"]),
        int(candidate["q"]),
        int(candidate["r"]),
        int(candidate["K"]),
        str(record["candidate_fingerprint"]),
    )


def _nondominated(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    metrics = ("total_shots", "n_det", "n_rand")
    frontier = []
    for candidate in records:
        dominated = False
        for challenger in records:
            if challenger is candidate:
                continue
            no_worse = all(
                int(challenger[name]) <= int(candidate[name]) for name in metrics
            )
            strictly_better = any(
                int(challenger[name]) < int(candidate[name]) for name in metrics
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda item: _proxy_key(item, "W_action"))


def _preboundary_selection(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    selected: dict[str, str] = {}
    for split in RANDOM_SPLITS:
        options = [item for item in records if _split(item["candidate"]) == split]
        if options:
            best = min(options, key=lambda item: _proxy_key(item, "W_action"))
            selected[str(best["candidate_fingerprint"])] = "split_anchor"
    for q in Q_VALUES:
        options = [
            item
            for item in records
            if int(item["candidate"]["q"]) == q
            and item["candidate_fingerprint"] not in selected
        ]
        if options:
            best = min(options, key=lambda item: _proxy_key(item, "W_action"))
            selected[str(best["candidate_fingerprint"])] = "q_anchor"
    return selected


def boundary_requests(
    base_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    eligible = [item for item in base_records if bool(item["accuracy_eligible"])]
    selected = _preboundary_selection(eligible)
    frontier = {
        str(item["candidate_fingerprint"]) for item in _nondominated(eligible)
    }
    requests = []
    for split in RANDOM_SPLITS:
        options = [
            item
            for item in eligible
            if _split(item["candidate"]) == split
            and int(item["candidate"]["r"]) == max(R_VALUES)
            and (
                item["candidate_fingerprint"] in selected
                or item["candidate_fingerprint"] in frontier
            )
        ]
        if not options:
            continue
        parent = min(options, key=lambda item: _proxy_key(item, "W_action"))
        base = parent["candidate"]
        boundary = make_candidate(
            str(base["method"]),
            int(base["rank"]),
            int(base["q"]),
            BOUNDARY_R,
            int(base["K"]),
            boundary_parent_fingerprint=str(parent["candidate_fingerprint"]),
        )
        requests.append(
            {
                "split": split,
                "parent_candidate_fingerprint": parent["candidate_fingerprint"],
                "candidate": boundary,
            }
        )
    if len(requests) > MAXIMUM_BOUNDARY_CANDIDATES:
        raise ValueError("boundary request count exceeds its frozen cap")
    return requests


def _validate_proxy_records(records: Sequence[Mapping[str, Any]]) -> None:
    fingerprints = set()
    for record in records:
        if not record.get("accuracy_eligible", False):
            continue
        candidate = record["candidate"]
        if candidate["method"] not in {"B2", "B3"}:
            raise ValueError("random selector received a deterministic method")
        candidate_fingerprint = str(record["candidate_fingerprint"])
        if candidate_fingerprint != candidate["candidate_fingerprint"]:
            raise ValueError("proxy and candidate fingerprints differ")
        if candidate_fingerprint in fingerprints:
            raise ValueError("duplicate proxy candidate fingerprint")
        fingerprints.add(candidate_fingerprint)
        for name in ("total_shots", "n_det", "n_rand", "n_fixed"):
            value = record[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        expected_action = int(record["total_shots"]) * (
            int(record["n_det"])
            + int(record["n_rand"])
            + int(record["n_fixed"])
        )
        expected_tail = int(record["total_shots"]) * int(record["n_rand"])
        if record["W_action"] != expected_action or record["W_tail"] != expected_tail:
            raise ValueError("proxy work values do not match their frozen formulas")


def select_random_compile_cells(
    records: Sequence[Mapping[str, Any]],
    *,
    boundary_fingerprints: Iterable[str],
) -> dict[str, Any]:
    _validate_proxy_records(records)
    eligible = [item for item in records if bool(item["accuracy_eligible"])]
    boundary = set(boundary_fingerprints)
    selected: dict[str, dict[str, Any]] = {}

    def add(record: Mapping[str, Any], tier: str) -> None:
        key = str(record["candidate_fingerprint"])
        if key in selected or len(selected) >= MAXIMUM_RANDOM_COMPILE_CELLS:
            return
        selected[key] = {
            "candidate_id": record["candidate"]["candidate_id"],
            "candidate_fingerprint": key,
            "selection_tier": tier,
            "total_shots": record["total_shots"],
            "n_det": record["n_det"],
            "n_rand": record["n_rand"],
            "n_fixed": record["n_fixed"],
            "W_action": record["W_action"],
            "W_tail": record["W_tail"],
        }

    for split in RANDOM_SPLITS:
        options = [item for item in eligible if _split(item["candidate"]) == split]
        if options:
            add(min(options, key=lambda item: _proxy_key(item, "W_action")), "split_anchor")
    for q in Q_VALUES:
        options = [
            item
            for item in eligible
            if int(item["candidate"]["q"]) == q
            and item["candidate_fingerprint"] not in selected
        ]
        if options:
            add(min(options, key=lambda item: _proxy_key(item, "W_action")), "q_anchor")
    for split in RANDOM_SPLITS:
        options = [
            item
            for item in eligible
            if _split(item["candidate"]) == split
            and item["candidate_fingerprint"] in boundary
        ]
        if options:
            add(min(options, key=lambda item: _proxy_key(item, "W_action")), "boundary_check")
    for split in RANDOM_SPLITS:
        options = [
            item
            for item in eligible
            if _split(item["candidate"]) == split
            and item["candidate_fingerprint"] not in selected
        ]
        if options:
            add(min(options, key=lambda item: _proxy_key(item, "W_tail")), "tail_challenger")

    frontier = _nondominated(eligible)
    buckets = {
        split: [
            item
            for item in frontier
            if _split(item["candidate"]) == split
            and item["candidate_fingerprint"] not in selected
        ]
        for split in RANDOM_SPLITS
    }
    while len(selected) < MAXIMUM_RANDOM_COMPILE_CELLS and any(buckets.values()):
        for split in RANDOM_SPLITS:
            if len(selected) >= MAXIMUM_RANDOM_COMPILE_CELLS:
                break
            if buckets[split]:
                add(buckets[split].pop(0), "proxy_frontier_round_robin")
    if len(selected) < MAXIMUM_RANDOM_COMPILE_CELLS:
        for record in sorted(eligible, key=lambda item: _proxy_key(item, "W_action")):
            add(record, "W_action_fill")
            if len(selected) >= MAXIMUM_RANDOM_COMPILE_CELLS:
                break

    selected_keys = set(selected)
    unselected_frontier = [
        str(item["candidate_fingerprint"])
        for item in frontier
        if item["candidate_fingerprint"] not in selected_keys
    ]
    unselected_boundary = sorted(boundary - selected_keys)
    expected_tail = set()
    for split in RANDOM_SPLITS:
        options = [item for item in eligible if _split(item["candidate"]) == split]
        if options:
            expected_tail.add(
                str(min(options, key=lambda item: _proxy_key(item, "W_tail"))["candidate_fingerprint"])
            )
    unselected_tail = sorted(expected_tail - selected_keys)
    reasons = []
    if unselected_frontier:
        reasons.append("unselected_proxy_nondominated_candidates")
    if unselected_boundary:
        reasons.append("boundary_check_not_selected")
    if unselected_tail:
        reasons.append("W_tail_challenger_not_selected")
    return {
        "compile_cap": MAXIMUM_RANDOM_COMPILE_CELLS,
        "eligible_record_count": len(eligible),
        "selected_count": len(selected),
        "selected": [
            {"selection_ordinal": index, **item}
            for index, item in enumerate(selected.values())
        ],
        "tier_counts": {
            tier: sum(
                item["selection_tier"] == tier for item in selected.values()
            )
            for tier in (
                "split_anchor",
                "q_anchor",
                "boundary_check",
                "tail_challenger",
                "proxy_frontier_round_robin",
                "W_action_fill",
            )
        },
        "proxy_frontier_count": len(frontier),
        "unselected_proxy_frontier_count": len(unselected_frontier),
        "unselected_proxy_frontier_fingerprints": unselected_frontier,
        "unselected_boundary_fingerprints": unselected_boundary,
        "unselected_tail_challenger_fingerprints": unselected_tail,
        "selection_limited": bool(reasons),
        "selection_limited_reasons": reasons,
    }


def selector_fixture(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    base_random = [
        candidate
        for candidate in candidates
        if candidate["method"] in {"B2", "B3"}
    ]
    base_records = [_fixture_proxy(candidate) for candidate in base_random]
    requests = boundary_requests(base_records)
    boundary_records = [_fixture_proxy(item["candidate"]) for item in requests]
    all_records = [*base_records, *boundary_records]
    boundary_fingerprints = [
        item["candidate"]["candidate_fingerprint"] for item in requests
    ]
    selection = select_random_compile_cells(
        all_records,
        boundary_fingerprints=boundary_fingerprints,
    )
    return {
        "fixture_only": True,
        "scientific_values": False,
        "fixture_formula": "metadata_only_selector_fixture_v1",
        "base_proxy_record_count": len(base_records),
        "boundary_request_count": len(requests),
        "boundary_requests": requests,
        "all_proxy_record_count": len(all_records),
        "proxy_records_sha256": fingerprint(all_records),
        "selection": selection,
    }


def build_dry_run(
    *,
    authorization_sha256: str,
    source_hashes: Mapping[str, str],
    s2_ledger: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = enumerate_base_candidates()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "series_id": SERIES_ID,
        "status": STATUS,
        "implementation_base_commit": BASE_COMMIT,
        "authorization_sha256": authorization_sha256,
        "source_hashes": dict(sorted(source_hashes.items())),
        "contract_identity": {
            "resource_contract_sha256": RESOURCE_CONTRACT_SHA256,
            "prior_art_gate_sha256": PRIOR_ART_GATE_SHA256,
        },
        "m0_read_only_ledger": dict(s2_ledger),
        "candidate_counts": candidate_counts(candidates),
        "candidate_ledger": candidates,
        "selector_dry_run": selector_fixture(candidates),
        "seed_policy": {
            "name": "pr2_m1_occurrence_seed_sha256_v1",
            "initial_master_seed": MASTER_SEED_INITIAL,
            "extension_master_seed": MASTER_SEED_EXTENSION,
            "coordinates": [
                "candidate_fingerprint",
                "axis",
                "trajectory",
                "outer_step",
                "tail_occurrence",
                "rte_step",
            ],
            "fresh_iid_required_per_coordinate_tuple": True,
        },
        "resource_caps": {
            "random_direct_compile_cells": MAXIMUM_RANDOM_COMPILE_CELLS,
            "initial_trajectories_per_selected_cell": INITIAL_TRAJECTORIES,
            "extension_trajectories_per_triggered_cell": EXTENSION_TRAJECTORIES,
            "maximum_trajectories_per_selected_cell": MAXIMUM_TRAJECTORIES_PER_CELL,
            "maximum_random_trajectories": (
                MAXIMUM_RANDOM_COMPILE_CELLS * MAXIMUM_TRAJECTORIES_PER_CELL
            ),
            "maximum_random_full_wrappers": (
                2 * MAXIMUM_RANDOM_COMPILE_CELLS * MAXIMUM_TRAJECTORIES_PER_CELL
            ),
            "deterministic_or_discard_compile_cells": 16,
            "maximum_total_full_wrappers": (
                2 * MAXIMUM_RANDOM_COMPILE_CELLS * MAXIMUM_TRAJECTORIES_PER_CELL
                + 2 * 16
            ),
        },
        "zero_compute_counters": {
            name: 0 for name in ZERO_COMPUTE_COUNTER_NAMES
        },
        "access_audit": {
            "authorization_json_reads": 1,
            "known_s2_json_reads": 1,
            "development_snapshot_reads": 0,
            "held_out_path_stat_calls": 0,
            "held_out_hash_reads": 0,
            "held_out_npz_loads": 0,
        },
        "authorization": {
            "implementation_contract_authorized": True,
            "synthetic_selector_dry_run_authorized": True,
            "m1_scientific_execution_authorized": False,
            "held_out_access_authorized": False,
            "s3_authorized": False,
            "automatic_next_stage": None,
        },
        "mandatory_stop_reached": True,
        "next_review": "independent_review_before_m1_execution_authorization",
    }
    payload["artifact_fingerprint"] = fingerprint(payload)
    validate_dry_run(payload)
    return payload


def validate_dry_run(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected M1 contract dry-run schema")
    if payload.get("status") != STATUS:
        raise ValueError("unexpected M1 contract dry-run status")
    unsigned = dict(payload)
    observed = unsigned.pop("artifact_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("dry-run artifact fingerprint mismatch")
    candidates = payload["candidate_ledger"]
    validate_candidate_ledger(candidates)
    expected_candidates = enumerate_base_candidates()
    if candidates != expected_candidates:
        raise ValueError("dry-run candidate ledger differs from frozen enumeration")
    expected_counts = candidate_counts(expected_candidates)
    if payload["candidate_counts"] != expected_counts:
        raise ValueError("dry-run candidate counts differ from frozen counts")
    expected_selector = selector_fixture(expected_candidates)
    if payload["selector_dry_run"] != expected_selector:
        raise ValueError("selector dry-run differs from the frozen fixture")
    selection = payload["selector_dry_run"]["selection"]
    if selection["selected_count"] > MAXIMUM_RANDOM_COMPILE_CELLS:
        raise ValueError("selector dry-run exceeded the 16-cell cap")
    if payload["selector_dry_run"]["boundary_request_count"] > MAXIMUM_BOUNDARY_CANDIDATES:
        raise ValueError("selector dry-run exceeded the boundary cap")
    if any(int(value) != 0 for value in payload["zero_compute_counters"].values()):
        raise ValueError("dry-run contains a non-zero scientific counter")
    access = payload["access_audit"]
    if any(
        int(access[name]) != 0
        for name in (
            "development_snapshot_reads",
            "held_out_path_stat_calls",
            "held_out_hash_reads",
            "held_out_npz_loads",
        )
    ):
        raise ValueError("dry-run accessed a frozen scientific snapshot")
    authorization = payload["authorization"]
    if authorization["m1_scientific_execution_authorized"]:
        raise ValueError("dry-run must not authorize M1 scientific execution")
    if authorization["held_out_access_authorized"] or authorization["s3_authorized"]:
        raise ValueError("dry-run must keep held-out and S3 unauthorized")
    if authorization["automatic_next_stage"] is not None:
        raise ValueError("dry-run must not define an automatic next stage")


def write_json_artifact(payload: Mapping[str, Any], path: Path) -> None:
    validate_dry_run(payload)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
