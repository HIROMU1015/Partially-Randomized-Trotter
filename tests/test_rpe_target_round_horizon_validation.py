from __future__ import annotations

from copy import deepcopy

import pytest

from trotterlib.rpe_target_round_horizon_validation import (
    _fingerprint,
    required_rpe_round_horizon,
    validate_rpe_target_round_horizon_payload,
)


def test_required_round_horizon_reproduces_q8_example() -> None:
    result = required_rpe_round_horizon(
        target_energy_precision=0.50,
        beta_rpe=0.40,
        delta_time=0.10,
    )
    assert result["maximum_round_index_M"] == 3
    assert result["round_count_M_plus_one"] == 4
    assert result["q_max"] == 8
    assert result["minimal_power_of_two_horizon"]


def test_required_round_horizon_for_ca_and_ca_over_ten() -> None:
    ca = 1.59360010199040e-3
    ca_result = required_rpe_round_horizon(
        target_energy_precision=ca, beta_rpe=0.40, delta_time=0.10
    )
    target_result = required_rpe_round_horizon(
        target_energy_precision=ca / 10.0,
        beta_rpe=0.40,
        delta_time=0.10,
    )
    assert (ca_result["maximum_round_index_M"], ca_result["q_max"]) == (
        12,
        4096,
    )
    assert (
        target_result["maximum_round_index_M"],
        target_result["q_max"],
    ) == (15, 32768)


def test_target_round_payload_is_tamper_evident() -> None:
    base = {
        "schema_version": "rpe_target_round_horizon_validation_v1",
        "method": "target_precision_horizon_and_fixed_schedule_matrix_diagnostic_v1",
        "scope": {
            "q_greater_than_8_circuit_compilation_performed": False,
            "final_total_cost_evaluation_performed": False,
        },
        "configuration": {"beta_rpe": 0.4, "delta_time": 0.1},
        "round_horizons": [],
        "summary": {"checks": {"a": True}, "overall_pass": True},
    }
    for label, epsilon in (("example", 0.5), ("ca", 0.001), ("target", 0.0001)):
        base["round_horizons"].append(
            {
                "label": label,
                **required_rpe_round_horizon(
                    target_energy_precision=epsilon,
                    beta_rpe=0.4,
                    delta_time=0.1,
                ),
            }
        )
    base["content_fingerprint"] = _fingerprint(base)
    validate_rpe_target_round_horizon_payload(base)
    tampered = deepcopy(base)
    tampered["summary"]["overall_pass"] = False
    with pytest.raises(ValueError, match="content_fingerprint"):
        validate_rpe_target_round_horizon_payload(tampered)
