from __future__ import annotations

import copy

import pytest

from trotterlib import research_direction_pd_fair_comparison as s1


def _deterministic_stub() -> dict[str, object]:
    return {
        "deterministic_task_id": "nested_2nd_d0.1_m8",
        "construction": "nested",
        "formula_label": "2nd",
        "delta": 0.1,
        "outer_step_count": 8,
        "inner_hd_substeps": 8,
        "tail_coefficients": [1.0],
        "gamma_r": 1.0,
        "tail_occurrence_count": 1,
        "outer_stage_count_total": 24,
        "deterministic_occurrence_count_per_outer_step": 2,
        "inner_stages_per_hd_occurrence": 56,
        "deterministic_component_actions_total": 896,
        "outer_phase_error_rad": 1.0e-8,
        "inner_phase_error_rad": 2.0e-8,
        "deterministic_phase_bound_rad": 3.0e-8,
        "deterministic_actual_phase_error_rad": 1.5e-8,
        "deterministic_signal_radius": 1.0,
        "deterministic_signal_phase_rad": 0.0,
        "deterministic_feasible": True,
    }


def test_expected_manifest_freezes_common_time_and_scope() -> None:
    body = s1.expected_task_manifest_body()
    artifact = s1.finalize_expected_task_manifest(
        body,
        provenance={"test": True},
    )
    s1.validate_expected_task_manifest(artifact)
    assert body["configuration"]["total_physical_time_au"] == 0.8
    assert body["configuration"]["outer_step_counts"] == [8, 4, 2]
    assert body["primary_finite_task_count"] == 300
    assert body["mandatory_stop_after_s1"] is True
    assert body["scope"]["h12_evaluated"] is False


def test_expected_manifest_detects_post_freeze_change() -> None:
    artifact = s1.finalize_expected_task_manifest(
        s1.expected_task_manifest_body(),
        provenance={"test": True},
    )
    changed = copy.deepcopy(artifact)
    changed["configuration"]["total_physical_time_au"] = 1.0
    with pytest.raises(ValueError):
        s1.validate_expected_task_manifest(changed)


def test_finite_k4_reduces_bound_without_free_cost() -> None:
    deterministic = _deterministic_stub()
    k2 = s1._finite_candidate(
        deterministic,
        lambda_r=0.5,
        total_rte_steps=16,
        finite_order=2,
    )
    k4 = s1._finite_candidate(
        deterministic,
        lambda_r=0.5,
        total_rte_steps=16,
        finite_order=4,
    )
    assert k4["finite_signal_error_bound"] < k2["finite_signal_error_bound"]
    assert (
        k4["b4_c1shot_expected_component_actions"]
        >= k2["b4_c1shot_expected_component_actions"]
    )
    assert k2["b2_c1shot_proxy"] == k4["b2_c1shot_proxy"]


def test_result_contract_requires_stop_and_no_scope_overclaim() -> None:
    body = {
        "schema_version": s1.RESULT_SCHEMA,
        "method": s1.METHOD,
        "configuration": dict(s1.CONFIGURATION),
        "primary_k2_candidates": [{"candidate_id": "one"}],
        "classification": {
            "primary_case": "A",
            "mandatory_stop_before_s2": True,
        },
        "scope": dict(s1.expected_task_manifest_body()["scope"]),
    }
    artifact = s1.finalize_result(
        body,
        provenance={"test": True},
        source_evidence=[],
    )
    s1.validate_result(artifact)
    changed = copy.deepcopy(body)
    changed["scope"]["h12_evaluated"] = True
    with pytest.raises(ValueError):
        s1.finalize_result(
            changed,
            provenance={"test": True},
            source_evidence=[],
        )


def test_too_few_short_steps_is_recorded_as_infeasible() -> None:
    deterministic = _deterministic_stub()
    deterministic["tail_coefficients"] = [0.1] * 17
    deterministic["gamma_r"] = 1.7
    row = s1._finite_candidate(
        deterministic,
        lambda_r=0.5,
        total_rte_steps=16,
        finite_order=2,
    )
    assert row["allocation_feasible"] is False
    assert row["b4_feasible"] is False
    assert row["b4_objective"] is None
    assert row["b2_objective"] > 0.0

