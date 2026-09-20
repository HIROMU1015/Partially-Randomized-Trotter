from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from trotterlib.rpe_four_round_phase_validation import (
    _vectorized_reconstruction,
    reconstruct_rpe_phase,
    validate_rpe_four_round_phase_payload,
)


def test_sequential_branch_reconstruction_crosses_principal_branch() -> None:
    reference = 0.72
    q_values = (1, 2, 4, 8)
    round_phases = [
        float(np.angle(np.exp(1j * q_m * reference))) for q_m in q_values
    ]
    result = reconstruct_rpe_phase(round_phases, q_values)
    assert result["final_phase_estimate"] == pytest.approx(reference)
    assert [item["q_m"] for item in result["rounds"]] == [1, 2, 4, 8]


def test_vectorized_reconstruction_matches_scalar() -> None:
    q_values = (1, 2, 4, 8)
    references = np.asarray([0.22, 0.72, -0.61])
    phases = [np.angle(np.exp(1j * q_m * references)) for q_m in q_values]
    vectorized, _branches, ambiguous = _vectorized_reconstruction(phases, q_values)
    assert not np.any(ambiguous)
    for index, reference in enumerate(references):
        scalar = reconstruct_rpe_phase(
            [float(item[index]) for item in phases], q_values
        )
        assert vectorized[index] == pytest.approx(
            scalar["final_phase_estimate"]
        )
        assert vectorized[index] == pytest.approx(reference)


def test_phase_payload_is_tamper_evident() -> None:
    payload = {
        "schema_version": "rpe_four_round_phase_validation_v1",
        "method": "physical_q8_fresh_iid_and_sequential_branch_reconstruction_v1",
        "scope": {"final_total_cost_evaluation_performed": False},
        "physical_signals_and_exact_probabilities": {"checks": {"a": True}},
        "explicit_fresh_iid_trajectory_batch": {"checks": {"b": True}},
        "summary": {
            "all_finite_rte_signal_checks_pass": True,
            "all_exact_probability_checks_pass": True,
            "all_explicit_fresh_iid_checks_pass": True,
            "all_branch_reconstruction_checks_pass": True,
            "reconstruction_checks": {"c": True},
            "overall_pass": True,
        },
    }
    from trotterlib.rpe_four_round_phase_validation import _fingerprint

    payload["content_fingerprint"] = _fingerprint(payload)
    validate_rpe_four_round_phase_payload(payload)
    tampered = deepcopy(payload)
    tampered["summary"]["overall_pass"] = False
    with pytest.raises(ValueError, match="content_fingerprint"):
        validate_rpe_four_round_phase_payload(tampered)
