from __future__ import annotations

import copy
import math

import pytest

from trotterlib.rpe_delta_round_schedule_validation import (
    METHOD,
    SCHEMA_VERSION,
    _fingerprint,
    screen_delta_candidate,
    validate_rpe_delta_round_schedule_payload,
)


def test_pf_delta_screen_uses_target_dependent_power_of_two_horizon() -> None:
    result = screen_delta_candidate(
        delta_time=0.02,
        target_energy_precision=1.5936001019904e-4,
        beta_rpe=0.4,
        beta_pf_budget=0.02,
        pf_coefficient=0.01342567,
    )
    assert result["maximum_round_index_M"] == 17
    assert result["q_max"] == 131072
    assert math.isclose(
        result["empirical_pf_phase_proxy_at_q_max"],
        0.014077835345920003,
    )
    assert result["empirical_pf_screen_pass"] is True


def _minimal_payload() -> dict:
    screen = screen_delta_candidate(
        delta_time=0.02,
        target_energy_precision=1.5936001019904e-4,
        beta_rpe=0.4,
        beta_pf_budget=0.02,
        pf_coefficient=0.01342567,
    )
    screen["source_role"] = "pf_surrogate_calibration_grid"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "scope": {
            "q_greater_than_8_circuit_compilation_performed": False,
            "compiled_cost_proxy_used_for_selection": False,
            "final_total_cost_evaluation_performed": False,
        },
        "configuration": {
            "target_energy_precision_ha": 1.5936001019904e-4,
            "beta_rpe": 0.4,
            "beta_pf_budget": 0.02,
            "pf_coefficient": 0.01342567,
            "delta_candidates": [0.02],
        },
        "delta_pf_screen": [screen],
        "summary": {"checks": {"example": True}, "overall_pass": True},
    }
    payload["content_fingerprint"] = _fingerprint(payload)
    return payload


def test_payload_validator_accepts_consistent_payload() -> None:
    validate_rpe_delta_round_schedule_payload(_minimal_payload())


def test_payload_validator_rejects_tampering() -> None:
    payload = copy.deepcopy(_minimal_payload())
    payload["delta_pf_screen"][0]["q_max"] = 8
    with pytest.raises(ValueError, match="fingerprint"):
        validate_rpe_delta_round_schedule_payload(payload)
