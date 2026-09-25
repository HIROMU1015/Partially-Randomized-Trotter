from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from trotterlib.research_direction_joint_synthesis_formalization import (
    IntervalObjective,
    evaluate_joint_synthesis_formalization,
    finalize_joint_synthesis_formalization_artifact,
    solve_lexicographic_interval_dp,
    validate_joint_synthesis_formalization_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
PILOT = (
    ROOT
    / "artifacts/research_direction_joint_synthesis_pilot/2026-09-25/"
    "pa_h4_interval_union_joint_synthesis_v1.json"
)
BLIND = (
    ROOT
    / "artifacts/research_direction_joint_synthesis_blind_validation/2026-09-25/"
    "pa_v1_h5_physical_h4_opt2_blind_v1.json"
)
ARTIFACT = (
    ROOT
    / "artifacts/research_direction_joint_synthesis_formalization/2026-09-25/"
    "pa_v1_formalization_and_mechanism_audit_v1.json"
)


def _inputs() -> dict[str, dict]:
    return {
        "pilot": json.loads(PILOT.read_text(encoding="utf-8")),
        "blind": json.loads(BLIND.read_text(encoding="utf-8")),
    }


def _brute_force(
    length: int,
    options: dict[tuple[int, int, str], tuple[int, int]],
) -> tuple[IntervalObjective, tuple[tuple[int, int, str], ...]]:
    candidates = []

    def visit(
        start: int,
        objective: IntervalObjective,
        segments: tuple[tuple[int, int, str], ...],
    ) -> None:
        if start == length:
            candidates.append((objective, segments))
            return
        for stop in range(start + 1, length + 1):
            for mode in ("full", "support_union"):
                operation_count, support_size = options[(start, stop, mode)]
                visit(
                    stop,
                    objective.plus(
                        basis_operation_count=operation_count,
                        support_union_size=support_size,
                        mode=mode,
                    ),
                    (*segments, (start, stop, mode)),
                )

    visit(0, IntervalObjective(0, 0, 0, 0), ())
    return min(candidates, key=lambda item: (item[0], item[1]))


def test_formal_interval_dp_matches_exhaustive_search_and_can_split() -> None:
    length = 4
    options = {}
    for start in range(length):
        for stop in range(start + 1, length + 1):
            interval_length = stop - start
            options[(start, stop, "full")] = (10, interval_length)
            options[(start, stop, "support_union")] = (
                1 if interval_length == 1 else 10 * interval_length,
                interval_length,
            )

    dynamic = solve_lexicographic_interval_dp(length, options)
    exhaustive = _brute_force(length, options)

    assert dynamic == exhaustive
    assert dynamic[0].segment_count == 4
    assert all(segment[2] == "support_union" for segment in dynamic[1])


def test_completed_evidence_narrows_the_empirical_mechanism() -> None:
    body = evaluate_joint_synthesis_formalization(**_inputs())
    audit = body["empirical_mechanism_audit"]

    assert body["overall_pass"]
    assert audit["holdout_record_count"] == 48
    assert audit["operator_probe_count"] == 6
    assert audit["records_with_within_run_split"] == 0
    assert audit["holdout_nonzero_taylor_order_event_count"] == 0
    assert audit["operator_probe_implied_extra_applications"] == 0
    assert audit["one_segment_per_source_run_matches_selected_plan_on_all_records"]
    assert not audit["incremental_interval_partition_compiled_benefit_identified"]
    assert body["decision"]["primary_theme_status"] == (
        "conditional_candidate_pending_nondegenerate_mechanism_validation"
    )


def test_formalization_artifact_is_tamper_evident_and_scope_limited() -> None:
    artifact = finalize_joint_synthesis_formalization_artifact(
        evaluate_joint_synthesis_formalization(**_inputs()),
        provenance={"test": True},
    )
    validate_joint_synthesis_formalization_artifact(artifact)

    tampered = deepcopy(artifact)
    tampered["scope"]["interval_partition_advantage_claimed"] = True
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_joint_synthesis_formalization_artifact(tampered)


def test_completed_formalization_artifact_validates() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    validate_joint_synthesis_formalization_artifact(payload)
    assert payload["decision"]["status"] == (
        "pa_v1_formalized_but_interval_mechanism_not_empirically_distinguished"
    )
