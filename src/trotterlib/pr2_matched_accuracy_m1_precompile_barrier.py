"""Hard pre-compile barrier for the staged PR-2 matched-accuracy M1 study.

This module is standard-library only.  It consumes the result-prior selector
output produced after M1-A signal evaluation.  A selection-limited result is a
terminal stop: no deterministic, random, trajectory, circuit, or wrapper
compile job may be materialized.  It performs no scientific computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "pr2_matched_accuracy_m1_precompile_barrier_v2"
STATUS = "M1_PREEXECUTION_AMENDMENT_V2_FROZEN_SCIENCE_NOT_AUTHORIZED"
MAXIMUM_RANDOM_COMPILE_CELLS = 16
MAXIMUM_DETERMINISTIC_OR_DISCARD_COMPILE_CELLS = 16

LIMITED_STATUS = "SELECTION_LIMITED"
CLEAR_STATUS = "M1_A_COMPLETE_M1_B_ELIGIBLE"

_LIMIT_REASON_BY_FIELD = {
    "unselected_proxy_frontier_fingerprints": (
        "unselected_proxy_nondominated_candidates"
    ),
    "unselected_boundary_fingerprints": "boundary_check_not_selected",
    "unselected_tail_challenger_fingerprints": (
        "W_tail_challenger_not_selected"
    ),
}


class SelectionLimitedStop(RuntimeError):
    """Raised before compile-plan creation when the selector is limited."""


@dataclass(frozen=True)
class ValidatedSelection:
    selected_fingerprints: tuple[str, ...]
    selection_limited: bool
    reasons: tuple[str, ...]


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _fingerprint_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or len(item) != 64:
            raise ValueError(f"{name} must contain 64-character fingerprints")
        try:
            int(item, 16)
        except ValueError as exc:
            raise ValueError(f"{name} contains a non-hex fingerprint") from exc
        result.append(item)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicate fingerprints")
    return result


def validate_compile_selection(
    selection: Mapping[str, Any],
) -> ValidatedSelection:
    """Validate a v1 selector result before the M1-A/M1-B branch."""

    compile_cap = _non_negative_int(selection.get("compile_cap"), "compile_cap")
    if compile_cap != MAXIMUM_RANDOM_COMPILE_CELLS:
        raise ValueError("random compile cap differs from the frozen value 16")

    selected = selection.get("selected")
    if not isinstance(selected, list):
        raise ValueError("selected must be a list")
    selected_count = _non_negative_int(
        selection.get("selected_count"), "selected_count"
    )
    if selected_count != len(selected) or selected_count > compile_cap:
        raise ValueError("selected_count is inconsistent with selected or cap")

    fingerprints: list[str] = []
    ordinals: list[int] = []
    for record in selected:
        if not isinstance(record, Mapping):
            raise ValueError("selected records must be mappings")
        fingerprints.extend(
            _fingerprint_list(
                [record.get("candidate_fingerprint")],
                "selected candidate fingerprints",
            )
        )
        ordinals.append(
            _non_negative_int(record.get("selection_ordinal"), "selection_ordinal")
        )
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("selected candidate fingerprints are not unique")
    if ordinals != list(range(selected_count)):
        raise ValueError("selection ordinals are not canonical")

    selection_limited = selection.get("selection_limited")
    if not isinstance(selection_limited, bool):
        raise ValueError("selection_limited must be boolean")
    reasons = selection.get("selection_limited_reasons")
    if not isinstance(reasons, list) or any(
        not isinstance(reason, str) or not reason for reason in reasons
    ):
        raise ValueError("selection_limited_reasons must be a list of strings")
    if len(reasons) != len(set(reasons)):
        raise ValueError("selection_limited_reasons contains duplicates")

    expected_reasons = []
    for field, reason in _LIMIT_REASON_BY_FIELD.items():
        values = _fingerprint_list(selection.get(field), field)
        if values:
            expected_reasons.append(reason)
    if set(reasons) != set(expected_reasons):
        raise ValueError("selection_limited reasons do not match unresolved sets")
    if selection_limited != bool(expected_reasons):
        raise ValueError("selection_limited does not match unresolved sets")

    return ValidatedSelection(
        selected_fingerprints=tuple(fingerprints),
        selection_limited=selection_limited,
        reasons=tuple(reasons),
    )


def evaluate_precompile_barrier(selection: Mapping[str, Any]) -> dict[str, Any]:
    """Return the mandatory branch immediately after M1-A selection."""

    validated = validate_compile_selection(selection)
    limited = validated.selection_limited
    return {
        "schema_version": SCHEMA_VERSION,
        "status": LIMITED_STATUS if limited else CLEAR_STATUS,
        "selection_limited": limited,
        "selection_limited_reasons": list(validated.reasons),
        "selected_random_cell_count": len(validated.selected_fingerprints),
        "selected_random_candidate_fingerprints": list(
            validated.selected_fingerprints
        ),
        "m1_b_direct_compile_eligible": not limited,
        "m1_b_direct_compile_authorized_by_current_amendment": False,
        "compile_jobs_materialized_at_barrier": 0,
        "all_direct_compile_counters_required_zero_at_m1_a": True,
        "winner_claim_permitted": False,
        "held_out_candidate_selection_permitted": False,
        "mandatory_stop_reached": limited,
        "next_action": (
            "STOP_AND_REVIEW_COMPILE_BUDGET_OR_TECHNICAL_NOTE_SCOPE"
            if limited
            else "M1_B_ONLY_UNDER_RESULT_PRIOR_EXECUTION_AUTHORIZATION"
        ),
    }


def build_m1_b_compile_plan(
    selection: Mapping[str, Any],
    *,
    deterministic_or_discard_fingerprints: Sequence[str],
) -> dict[str, Any]:
    """Build cell identities only after a clear barrier.

    This is a pure planning function.  It never builds or compiles a circuit.
    A future scientific runner must call it before creating any compile task.
    """

    barrier = evaluate_precompile_barrier(selection)
    if barrier["selection_limited"]:
        raise SelectionLimitedStop(
            "SELECTION_LIMITED is a terminal pre-compile stop; no jobs created"
        )

    deterministic = _fingerprint_list(
        list(deterministic_or_discard_fingerprints),
        "deterministic_or_discard_fingerprints",
    )
    if len(deterministic) > MAXIMUM_DETERMINISTIC_OR_DISCARD_COMPILE_CELLS:
        raise ValueError("deterministic/discard compile-cell cap exceeded")

    random = list(barrier["selected_random_candidate_fingerprints"])
    return {
        "barrier_status": CLEAR_STATUS,
        "deterministic_or_discard_compile_cells": deterministic,
        "random_compile_cells": random,
        "deterministic_or_discard_compile_cell_count": len(deterministic),
        "random_compile_cell_count": len(random),
        "compile_cell_count": len(deterministic) + len(random),
        "circuits_built": 0,
        "circuit_compilations": 0,
        "full_wrappers_compiled": 0,
        "plan_only": True,
    }
