from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from trotterlib.parallel_validation_executor import file_sha256
from trotterlib.research_direction_joint_synthesis_blind_validation import (
    FROZEN_CORE_HASHES,
    FROZEN_PILOT_CONTENT_FINGERPRINT,
    FROZEN_PILOT_FILE_SHA256,
    H4_COMPILER_TRANSFER,
    H5_PHYSICAL_TRANSFER,
    evaluate_preregistered_gates,
    finalize_blind_validation_artifact,
    finalize_checkpoint,
    validate_blind_validation_artifact,
    validate_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]
PILOT = (
    ROOT
    / "artifacts"
    / "research_direction_joint_synthesis_pilot"
    / "2026-09-25"
    / "pa_h4_interval_union_joint_synthesis_v1.json"
)

BLIND_ARTIFACT = (
    ROOT
    / "artifacts"
    / "research_direction_joint_synthesis_blind_validation"
    / "2026-09-25"
    / "pa_v1_h5_physical_h4_opt2_blind_v1.json"
)

def _row(candidate_rz: int = 90) -> dict[str, object]:
    metrics = {
        "rz_count": candidate_rz,
        "rz_depth": candidate_rz,
        "cx_count": 10,
        "cx_depth": 10,
        "total_depth": candidate_rz,
        "circuit_size": candidate_rz + 10,
    }
    return {
        "task_key": "test__L3__sample0",
        "costs": {
            "full_basis_shared": {**metrics, "rz_count": 120},
            "event_support_restricted": {**metrics, "rz_count": 115},
            "support_run_le_1": {**metrics, "rz_count": 100},
            "interval_union_dp": metrics,
        },
        "interval_choice_differs_from_current": True,
        "interval_metadata": {
            "selected_multi_application_interval_count": 1,
            "selected_support_union_interval_count": 1,
        },
    }


def _probe() -> dict[str, object]:
    return {
        "operator_max_abs_difference": 0.0,
        "relative_ancilla_phase_matches": True,
    }


def test_preregistered_gates_pass_and_fail_without_threshold_changes() -> None:
    gates, summary = evaluate_preregistered_gates([_row()], [_probe()])
    assert all(gates.values())
    assert summary["all_preregistered_gates_pass"]

    failed, failed_summary = evaluate_preregistered_gates(
        [_row(candidate_rz=101)], [_probe()]
    )
    assert not failed[
        "pooled_rz_improvement_over_current_at_least_2pct"
    ]
    assert not failed_summary["all_preregistered_gates_pass"]


def test_frozen_v1_and_pilot_inputs_match_preregistration() -> None:
    actual = {
        path: file_sha256(ROOT / path) for path in FROZEN_CORE_HASHES
    }
    assert actual == FROZEN_CORE_HASHES
    assert file_sha256(PILOT) == FROZEN_PILOT_FILE_SHA256
    payload = json.loads(PILOT.read_text(encoding="utf-8"))
    assert payload["content_fingerprint"] == FROZEN_PILOT_CONTENT_FINGERPRINT


def test_blind_artifact_and_checkpoint_are_tamper_evident() -> None:
    strata = {
        spec.stratum_id: {
            "execution_valid": True,
            "summary": {"all_preregistered_gates_pass": True},
        }
        for spec in (H5_PHYSICAL_TRANSFER, H4_COMPILER_TRANSFER)
    }
    artifact = finalize_blind_validation_artifact(
        strata,
        source_evidence={
            "frozen_core_hashes_match": True,
            "frozen_pilot_artifact_matches": True,
        },
        provenance={"test": True},
    )
    validate_blind_validation_artifact(artifact)
    assert artifact["decision"]["blind_validation_passed"]

    tampered = deepcopy(artifact)
    tampered["decision"]["blind_validation_passed"] = False
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_blind_validation_artifact(tampered)

    checkpoint = finalize_checkpoint("holdout", {"task_key": "test", "x": 1})
    validate_checkpoint(checkpoint)
    broken = deepcopy(checkpoint)
    broken["row"]["x"] = 2
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_checkpoint(broken)


def test_completed_blind_artifact_validates_and_both_strata_pass() -> None:
    payload = json.loads(BLIND_ARTIFACT.read_text(encoding="utf-8"))
    validate_blind_validation_artifact(payload)
    assert payload["content_fingerprint"] == (
        "78af3474898dbf989780ea5f2881cb5b61595609dd2698164b9846c1ce1c5919"
    )
    assert payload["overall_pass"]
    assert payload["decision"]["blind_validation_passed"]
    assert payload["decision"]["status"] == (
        "advance_pa_v1_to_formal_primary_theme_candidate"
    )
    assert set(payload["strata"]) == {
        H5_PHYSICAL_TRANSFER.stratum_id,
        H4_COMPILER_TRANSFER.stratum_id,
    }
    for stratum in payload["strata"].values():
        assert stratum["execution_valid"]
        assert all(stratum["preregistered_gates"].values())
        assert len(stratum["holdout_rows"]) == 24
        assert len(stratum["operator_equivalence_probes"]) == 3
