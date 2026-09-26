from __future__ import annotations

from copy import deepcopy

import pytest

from trotterlib.finite_rte_phase_amplitude import (
    run_finite_rte_phase_amplitude_fr1,
    validate_finite_rte_phase_amplitude_payload,
    write_finite_rte_phase_amplitude_payload,
)


@pytest.fixture(scope="module")
def payload() -> dict:
    return run_finite_rte_phase_amplitude_fr1(provenance={"test": True})


def test_fr1_fixed_grid_is_sound_and_stops_after_classification(payload: dict) -> None:
    validate_finite_rte_phase_amplitude_payload(payload)

    assert payload["summary"]["condition_count"] == 33
    assert payload["summary"]["state_record_count"] == 99
    assert payload["summary"]["bound_violation_count"] == 0
    assert payload["summary"]["execution_valid"]
    assert payload["summary"]["fr2_started"] is False
    assert payload["gates"]["G0_semantic_consistency"]
    assert payload["gates"]["G1_all_applicable_bounds_sound"]
    assert payload["gates"]["G3_conditioning_and_rejection"]
    assert payload["gates"]["G4_sign_cutoff_and_asymmetry_sound"]

    g2 = payload["gates"]["G2_available_noncommuting_utility"]
    oracle = payload["summary"]["reference_or_oracle_utility_observed"]
    if g2:
        expected = "GO_FR2_AVAILABLE"
    elif oracle:
        expected = "GO_FR2_MECHANISM_ONLY"
    else:
        expected = "STOP_FR1_NO_NONCOMMUTING_GAIN"
    assert payload["summary"]["decision"] == expected


def test_identity_control_semantics_and_available_rho_contract(payload: dict) -> None:
    semantic = payload["semantic_checks"]
    assert semantic["all_pass"]
    assert len(semantic["records"]) == 4
    assert {item["event_count"] for item in semantic["records"]} == {3, 30}
    assert semantic["maximum_ordinary_operator_residual"] <= 1e-12
    assert semantic["maximum_controlled_relative_phase_residual"] <= 1e-12

    mixture_records = [
        state
        for condition in payload["conditions"]
        for state in condition["state_records"]
        if state["state_label"] == "analytic_mixture_state"
    ]
    assert mixture_records
    assert all(state["available_rho"] == 0.8 for state in mixture_records)
    assert all(state["available_rho_valid"] for state in mixture_records)
    assert all("PROPOSED_AVAILABLE" in state["methods"] for state in mixture_records)

    physical_records = [
        state
        for condition in payload["conditions"]
        for state in condition["state_records"]
        if state["state_label"] == "physical_ground_state"
    ]
    assert physical_records
    assert all(state["available_rho"] is None for state in physical_records)
    assert all("PROPOSED_AVAILABLE" not in state["methods"] for state in physical_records)


def test_scalar_order_diagnostic_and_strong_norm_are_not_grid_maxima(payload: dict) -> None:
    assert payload["analytic_strong_norm"]["finite_grid_maximum_used"] is False
    diagnostics = {
        item["cutoff"]: item for item in payload["local_order_diagnostic"]
    }
    assert set(diagnostics) == {0, 2, 4}
    for cutoff, record in diagnostics.items():
        assert record["radial_loglog_slope"] == pytest.approx(
            cutoff + 2, abs=0.15
        )
        assert record["tangential_loglog_slope"] == pytest.approx(
            cutoff + 3, abs=0.15
        )
    assert (
        payload["summary"]["maximum_matrix_vs_scalar_strong_abs_difference"]
        <= 2e-15
    )


def test_fr1_payload_is_tamper_evident_and_serializable(
    payload: dict,
    tmp_path,
) -> None:
    target = tmp_path / "fr1.json"
    write_finite_rte_phase_amplitude_payload(payload, target)
    assert target.exists()

    tampered = deepcopy(payload)
    tampered["gates"]["G2_available_noncommuting_utility"] = not tampered[
        "gates"
    ]["G2_available_noncommuting_utility"]
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_finite_rte_phase_amplitude_payload(tampered)

    invalid_scope = deepcopy(payload)
    invalid_scope.pop("validation_fingerprint")
    invalid_scope["h4_or_h12_evaluation_performed"] = True
    from trotterlib.finite_rte_phase_amplitude import _fingerprint

    invalid_scope["validation_fingerprint"] = _fingerprint(invalid_scope)
    with pytest.raises(ValueError, match="H4 or H12"):
        validate_finite_rte_phase_amplitude_payload(invalid_scope)
