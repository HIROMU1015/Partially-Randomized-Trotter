from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from trotterlib.fr_revision_nonuniform import (
    EXPECTED_MATRIX_CONDITIONS,
    EXPECTED_SEMANTIC_CONTROLS,
    EXPECTED_STATE_ROWS,
    METHOD_ORDER,
    PREREGISTRATION_SHA256,
    validate_fr_r1b_expected,
    validate_fr_r1b_payload,
    write_fr_r1b_payload,
)


DIRECTORY = Path("artifacts/fr_revision_nonuniform/2026-09-27")
EXPECTED = DIRECTORY / "fr_revision_nonuniform_expected_v1.json"
RESULT = DIRECTORY / "fr_revision_nonuniform_r1b_v1.json"


@pytest.fixture(scope="module")
def expected() -> dict:
    return json.loads(EXPECTED.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_expected_specification_was_frozen_before_result(expected: dict) -> None:
    validate_fr_r1b_expected(expected)
    assert expected["preregistered_not_result_adaptive"] is True
    assert expected["preregistration_sha256"] == PREREGISTRATION_SHA256
    assert len(expected["condition_ids"]) == EXPECTED_MATRIX_CONDITIONS
    assert len(expected["state_ids"]) == EXPECTED_STATE_ROWS
    assert len(expected["semantic_control_ids"]) == EXPECTED_SEMANTIC_CONTROLS
    assert expected["method_order"] == list(METHOD_ORDER)
    assert expected["phase_budgets_rad"] == [1e-2, 1e-3, 1e-4]
    assert expected["mandatory_stop_after_result"] is True


def test_result_completeness_soundness_and_mandatory_stop(result: dict) -> None:
    validate_fr_r1b_payload(result)
    summary = result["summary"]
    assert summary["matrix_condition_count"] == EXPECTED_MATRIX_CONDITIONS
    assert summary["state_record_count"] == EXPECTED_STATE_ROWS
    assert summary["semantic_control_count"] == EXPECTED_SEMANTIC_CONTROLS
    assert summary["method_record_count"] == EXPECTED_STATE_ROWS * len(METHOD_ORDER)
    assert summary["soundness_failure_count"] == 0
    assert summary["execution_valid"]
    assert summary["mandatory_stop_observed"] is True
    assert summary["fr_r2_started"] is False
    assert result["fr_r2_started"] is False
    assert result["h4_or_h12_evaluation_performed"] is False
    assert result["final_cost_evaluation_performed"] is False


def test_fixed_states_are_reused_and_information_layers_are_separated(result: dict) -> None:
    fingerprints: dict[tuple[float, str, int, str], set[str]] = {}
    rows = []
    for condition in result["conditions"]:
        spec = condition["condition"]
        for state in condition["state_records"]:
            rows.append(state)
            if state["state_label"] in {
                "fixed_supplied_superposition",
                "fixed_q8_reference_eigenstate",
                "fixed_total_ground_state",
            }:
                key = (
                    float(spec["nu"]),
                    str(spec["deterministic_block"]),
                    int(spec["sigma"]),
                    state["state_label"],
                )
                fingerprints.setdefault(key, set()).add(state["state_fingerprint"])
    assert all(len(values) == 1 for values in fingerprints.values())
    supplied = [row for row in rows if row["state_label"] == "fixed_supplied_superposition"]
    assert all(row["available_rho"] == 0.8 for row in supplied)
    assert all(not row["methods"]["OPT-SCALAR-NORM-I1"]["oracle"] for row in supplied)
    assert all(row["methods"]["DENSE-ORACLE-I2"]["oracle"] for row in rows)
    stress = [row for row in rows if row["state_label"] == "signal_near_zero_stress_state"]
    assert len(stress) == 1
    assert stress[0]["available_rho"] is None
    assert not stress[0]["methods"]["OPT-SCALAR-NORM-I1"]["applicable"]


def test_gates_and_decision_follow_the_frozen_priority(result: dict) -> None:
    gates = result["gates"]
    assert list(gates) == [
        "R0_completeness_and_semantics",
        "R1_input_certificate",
        "R2_soundness",
        "R3_nonuniform_mechanism",
        "R4_same_information_common_scalar_gain",
        "R5_decision_relevance",
        "R6_oracle_independence",
        "R7_control_transfer",
    ]
    scalar_only = result["summary"]["scalar_norm_explains_all_optimized_differences"]
    if not (gates["R0_completeness_and_semantics"] and gates["R1_input_certificate"] and gates["R2_soundness"] and gates["R7_control_transfer"]):
        expected_decision = "STOP_FR_R_INPUT_OR_SOUNDNESS"
    elif not gates["R3_nonuniform_mechanism"] or not gates["R4_same_information_common_scalar_gain"] or scalar_only:
        expected_decision = "STOP_FR_R_INVOLUTION_OR_SCALAR_ONLY"
    elif not gates["R5_decision_relevance"] or not gates["R6_oracle_independence"]:
        expected_decision = "MECHANISM_ONLY_NO_PRACTICAL_GO"
    else:
        expected_decision = "GO_FR_R2_CANDIDATE_CONDITIONAL_ON_SUPPLIED_STATE"
    assert result["summary"]["decision"] == expected_decision


def test_result_is_tamper_evident_and_non_overwriting(result: dict, tmp_path: Path) -> None:
    target = tmp_path / "result.json"
    write_fr_r1b_payload(result, target)
    tampered = deepcopy(result)
    tampered["gates"]["R5_decision_relevance"] = not tampered["gates"][
        "R5_decision_relevance"
    ]
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_fr_r1b_payload(tampered)
    target.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_fr_r1b_payload(result, target)
