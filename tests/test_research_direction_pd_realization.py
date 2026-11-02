from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from trotterlib.research_direction_energy_tail_pareto import fingerprint
from trotterlib.research_direction_pd_realization import (
    D1_OPERATOR_ATOL,
    FRESH_HOLDOUT_LD,
    LD_ROLES,
    evaluate_d1,
    expected_task_manifest_body,
    signed_time_tasks,
    validate_expected_task_manifest,
    validate_result,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = (
    ROOT
    / "artifacts"
    / "research_direction_pd_realization"
    / "2026-09-25"
)
EXPECTED = ARTIFACT_ROOT / "pd_realization_expected_tasks_v2.json"
FINAL = ARTIFACT_ROOT / "pd_realization_go_no_go_v2.json"
EXPECTED_FINGERPRINT = "8924d637e52b03900f32e2f167e63593cf9729b4b78d4dee3af87fca661183f4"
RESULT_FINGERPRINT = "805a17f95497a4d61286748a126c01b1235fbe0d987528be86ea3938700b9ede"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_expected_body_has_only_negative_signed_time_tasks_and_fresh_ld5() -> None:
    body = expected_task_manifest_body()
    tasks = signed_time_tasks()
    assert tasks
    assert all(task["tail_coefficient"] < 0.0 for task in tasks)
    assert all(task["signed_short_time"] < 0.0 for task in tasks)
    assert [row[0] for row in LD_ROLES] == [3, 4, FRESH_HOLDOUT_LD]
    assert body["exploration_disclosure"][
        "ld5_outer_exact_or_internal_hd_seen_before_freeze"
    ] is False
    assert body["task_count"] == len(body["d1_tasks"]) + len(
        body["d2_d3_tasks"]
    )


def test_d1_signed_time_and_controlled_oracle_passes() -> None:
    result = evaluate_d1(expected_task_manifest_body())
    assert result["task_count"] == len(signed_time_tasks())
    assert result["failed_tasks"] == 0
    assert result["overall_pass"]
    assert max(
        row["ordinary_oracle_residual_spectral_norm"] for row in result["rows"]
    ) <= D1_OPERATOR_ATOL
    assert max(
        row["signed_adjoint_residual_spectral_norm"] for row in result["rows"]
    ) <= D1_OPERATOR_ATOL


def test_frozen_expected_and_result_artifacts() -> None:
    expected = _load(EXPECTED)
    validate_expected_task_manifest(expected)
    assert expected["content_fingerprint"] == EXPECTED_FINGERPRINT
    assert expected["content_fingerprint"] == fingerprint(
        {key: value for key, value in expected.items() if key != "content_fingerprint"}
    )

    final = _load(FINAL)
    validate_result(final)
    assert final["content_fingerprint"] == RESULT_FINGERPRINT
    assert final["expected_task_fingerprint"] == expected["content_fingerprint"]
    assert [row["ld"] for row in final["d2_d3_internal_hd_splits"]] == [3, 4, 5]
    assert all(final["gates"].values())
    assert final["decision"]["status"] == (
        "advance_pd_to_formal_primary_candidate_then_stop_for_research_redesign"
    )
    assert final["decision"]["stop_after_this_validation_for_research_redesign"]


def test_artifacts_reject_tampering_and_scope_overstatement() -> None:
    expected = _load(EXPECTED)
    tampered_expected = copy.deepcopy(expected)
    tampered_expected["d2_d3_tasks"][0]["ld"] = 99
    with pytest.raises(ValueError, match="fingerprint"):
        validate_expected_task_manifest(tampered_expected)

    final = _load(FINAL)
    tampered = copy.deepcopy(final)
    tampered["decision"]["status"] = "arbitrary"
    with pytest.raises(ValueError, match="fingerprint"):
        validate_result(tampered)

    overstated = copy.deepcopy(final)
    overstated.pop("content_fingerprint")
    overstated["scope"]["h12_evaluated"] = True
    overstated["content_fingerprint"] = fingerprint(overstated)
    with pytest.raises(ValueError, match="overstates scope"):
        validate_result(overstated)

