from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from trotterlib.research_direction_full_scope import fingerprint
from trotterlib.research_direction_geometry_tracking_breakdown import (
    BLIND_GEOMETRIES,
    GEOMETRIES,
    POLICIES,
    TRAINING_GEOMETRIES,
    validate_expected_task_manifest,
    validate_geometry_tracking_breakdown_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = (
    ROOT
    / "artifacts"
    / "research_direction_geometry_tracking_breakdown"
    / "2026-09-25"
)
EXPECTED = ARTIFACT_ROOT / "pc_tracking_breakdown_expected_tasks_v1.json"
FINAL = ARTIFACT_ROOT / "pc_tracking_breakdown_validation_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_expected_tasks_freeze_disjoint_geometry_holdouts_and_hashes() -> None:
    payload = _load(EXPECTED)
    validate_expected_task_manifest(payload)
    assert payload["content_fingerprint"] == (
        "cbe260750d081316070d3684a2a24194d91d029ad700a54cf4f61152f2ed4a4e"
    )
    assert payload["task_count"] == len(GEOMETRIES) * len(POLICIES) == 16
    assert set(TRAINING_GEOMETRIES).isdisjoint(BLIND_GEOMETRIES)
    assert {task["policy"] for task in payload["tasks"]} == set(POLICIES)
    assert {
        task["geometry_angstrom"]
        for task in payload["tasks"]
        if task["role"] == "blind_holdout"
    } == set(BLIND_GEOMETRIES)
    assert payload["provenance"]["source_sha256"] == {
        "scripts/run_research_direction_geometry_tracking_breakdown.py": (
            "53410e399e4a4b8bc31991d0b6dcef1e9d746522b9bb4179906727fff4968912"
        ),
        "src/trotterlib/research_direction_geometry_tracking_breakdown.py": (
            "a77c427f569b36218baa0a6343ed77db96951767a12a0f9a373bef8f2da3195b"
        ),
    }


def test_final_artifact_preserves_preregistered_stop_decision() -> None:
    payload = _load(FINAL)
    validate_geometry_tracking_breakdown_artifact(payload)
    assert payload["content_fingerprint"] == (
        "26845effe8efda56390aabdf9e40d61fa3a033e3ac7e6ff6911e2e124156f07a"
    )
    assert payload["expected_task_fingerprint"] == (
        "cbe260750d081316070d3684a2a24194d91d029ad700a54cf4f61152f2ed4a4e"
    )
    assert payload["overall_pass"] is False
    assert payload["decision"]["status"] == (
        "stop_pc_current_h4_family_as_primary"
    )
    assert payload["decision"]["thresholds_changed_after_results"] is False
    assert payload["decision"]["current_primary_theme"] == "none_confirmed"


def test_tracking_is_identical_and_stretch_prediction_breaks_down() -> None:
    payload = _load(FINAL)
    assert payload["summary"]["changed_blind_prefix_count"] == 0
    assert payload["summary"]["tracking_median_error_reduction"] == 0.0
    assert payload["summary"]["tracked_blind_coefficient_pass_count"] == 3
    assert payload["summary"]["tracked_pair_prediction_pass_count"] == 1
    assert payload["summary"]["actual_breakdown_count"] == 2
    assert payload["summary"]["diagnostic_accuracy"] == pytest.approx(0.5)
    assert all(
        indices == [0, 1, 2]
        for indices in payload["tracked_original_fragment_indices"].values()
    )
    blind = {
        row["geometry_angstrom"]: row
        for row in payload["blind_coefficient_predictions"]["tracked"]
    }
    assert blind[1.4]["relative_error"] == pytest.approx(0.33653423584407843)
    assert blind[1.6]["relative_error"] == pytest.approx(1.2324543634081246)
    assert payload["gates"]["representation_integrity_pass"]
    assert payload["gates"]["delta_holdout_pass"]
    assert payload["gates"]["nontrivial_cancellation_pass"]
    assert not payload["gates"]["blind_coefficient_prediction_pass"]
    assert not payload["gates"]["pair_prediction_pass"]
    assert not payload["gates"]["diagnostic_transfer_pass"]
    assert not payload["gates"]["mechanism_discrimination_pass"]


def test_tracking_artifacts_reject_tampering_and_scope_overstatement() -> None:
    expected = _load(EXPECTED)
    expected["tasks"][0]["geometry_angstrom"] = 0.71
    with pytest.raises(ValueError, match="fingerprint"):
        validate_expected_task_manifest(expected)

    final = _load(FINAL)
    tampered = copy.deepcopy(final)
    tampered["decision"]["status"] = "advance_pc_tracking_and_breakdown_design"
    with pytest.raises(ValueError, match="fingerprint"):
        validate_geometry_tracking_breakdown_artifact(tampered)

    overstated = copy.deepcopy(final)
    overstated.pop("content_fingerprint")
    overstated["scope"]["scientific_superiority_claimed"] = True
    overstated["content_fingerprint"] = fingerprint(overstated)
    with pytest.raises(ValueError, match="overstates scope"):
        validate_geometry_tracking_breakdown_artifact(overstated)
