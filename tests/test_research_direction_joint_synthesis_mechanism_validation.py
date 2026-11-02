from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from trotterlib.research_direction_joint_synthesis_mechanism_validation import (
    BLIND_FRAGMENTS,
    FINITE_TAYLOR_ORDER,
    SUPPORT_PROFILES,
    TRAINING_FRAGMENTS,
    validate_expected_task_manifest,
    validate_mechanism_validation_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = (
    ROOT
    / "artifacts"
    / "research_direction_joint_synthesis_mechanism_validation"
    / "2026-09-25"
)
EXPECTED = ARTIFACT_ROOT / "pa_forced_support_order2_expected_tasks_v1.json"
FINAL = ARTIFACT_ROOT / "pa_forced_support_order2_mechanism_validation_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_expected_tasks_are_frozen_disjoint_order2_streams() -> None:
    payload = _load(EXPECTED)
    validate_expected_task_manifest(payload)
    assert payload["content_fingerprint"] == (
        "e8b064e9821fae5c2e7a44d0c98d3f9a0ed973a4cd87945fb051151e446a96fc"
    )
    assert payload["task_count"] == 30
    assert set(TRAINING_FRAGMENTS).isdisjoint(BLIND_FRAGMENTS)
    assert {task["partition"] for task in payload["tasks"]} == {
        "training_diagnostic",
        "blind_holdout",
    }
    assert {
        task["profile_id"] for task in payload["tasks"]
    } == set(SUPPORT_PROFILES)
    assert all(
        task["event_orders"] == [FINITE_TAYLOR_ORDER] * task["event_count"]
        and task["source_run_count"] == 1
        for task in payload["tasks"]
    )


def test_final_artifact_preserves_preregistered_stop_decision() -> None:
    payload = _load(FINAL)
    validate_mechanism_validation_artifact(payload)
    assert payload["content_fingerprint"] == (
        "fdc89974e89a4a6809cecd2c5608a36d684d40d76d9b3055fbbe6ec9276abbaf"
    )
    assert payload["expected_task_fingerprint"] == (
        "e8b064e9821fae5c2e7a44d0c98d3f9a0ed973a4cd87945fb051151e446a96fc"
    )
    assert payload["overall_pass"] is False
    assert payload["decision"]["status"] == (
        "stop_pa_interval_dp_as_primary_and_return_to_pc"
    )
    assert payload["decision"]["primary_theme"] == (
        "P-C_geometry_energy_difference"
    )
    assert payload["decision"]["thresholds_changed_after_results"] is False


def test_no_split_or_incremental_compiled_benefit_was_observed() -> None:
    payload = _load(FINAL)
    for partition in ("training_diagnostic", "blind_holdout"):
        summary = payload[partition]
        assert summary["split_row_count"] == 0
        assert summary["changed_row_count"] == 0
        assert summary["improved_rz_row_count"] == 0
        for metric in summary["metrics"].values():
            assert metric["candidate_relative_to_baseline"] == 0.0
    assert all(row["within_run_split_count"] == 0 for row in payload["rows"])
    assert all(
        order == FINITE_TAYLOR_ORDER
        for row in payload["rows"]
        for order in row["event_orders"]
    )
    assert payload["gates"]["operator_equivalence_and_relative_phase_pass"]
    assert not payload["gates"][
        "within_run_split_transfers_to_at_least_2_blind_bases"
    ]
    assert not payload["gates"]["pooled_blind_rz_improves_by_at_least_2pct"]


def test_mechanism_artifacts_reject_tampering_and_scope_overstatement() -> None:
    expected = _load(EXPECTED)
    expected["tasks"][0]["event_orders"][0] = 0
    with pytest.raises(ValueError, match="fingerprint"):
        validate_expected_task_manifest(expected)

    final = _load(FINAL)
    tampered = copy.deepcopy(final)
    tampered["decision"]["status"] = (
        "advance_pa_after_nondegenerate_mechanism_validation"
    )
    with pytest.raises(ValueError, match="fingerprint"):
        validate_mechanism_validation_artifact(tampered)

    overstated = copy.deepcopy(final)
    overstated.pop("content_fingerprint")
    overstated["scope"]["scientific_superiority_claimed"] = True
    from trotterlib.research_direction_full_scope import fingerprint

    overstated["content_fingerprint"] = fingerprint(overstated)
    with pytest.raises(ValueError, match="scope overstated"):
        validate_mechanism_validation_artifact(overstated)
