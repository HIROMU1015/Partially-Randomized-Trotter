from __future__ import annotations

from copy import deepcopy
import math

import pytest

from trotterlib.pf_c_system_size_validation import (
    configured_qiskit_delta_times,
    legacy_perturbation_conditioning,
    make_system_size_payload,
    validate_system_size_payload,
)


def _size_result(molecule_type: int) -> dict[str, object]:
    return {
        "molecule_type": molecule_type,
        "n_qubits": 2 * molecule_type,
        "configured_delta_grid_match": True,
        "single_dominant_phase_validation_pass": True,
        "corrected_perturbative_estimator_validation_pass": True,
        "operational_coefficient_usable": True,
        "perturbative_vs_exact_envelope_relative_difference": 0.01,
        "legacy_cosine_ill_conditioned_point_count": 0,
        "legacy_sine_ill_conditioned_point_count": 0,
    }


def test_configured_qiskit_delta_times_use_established_size_windows() -> None:
    assert configured_qiskit_delta_times(2) == (0.73, 0.732, 0.734, 0.736)
    assert configured_qiskit_delta_times(5) == (0.36, 0.362, 0.364, 0.366)
    assert configured_qiskit_delta_times(12) == (0.12, 0.122, 0.124)


def test_legacy_conditioning_flags_cosine_singularity_but_not_new_formula() -> None:
    diagnostic = legacy_perturbation_conditioning(
        1.0,
        0.5 * math.pi,
    )
    assert not diagnostic["legacy_cosine_formula_well_conditioned"]
    assert diagnostic["legacy_sine_formula_well_conditioned"]
    assert not diagnostic[
        "shift_invariant_formula_uses_trigonometric_denominator"
    ]


def test_system_size_payload_is_tamper_evident() -> None:
    payload = make_system_size_payload(
        [_size_result(2), _size_result(3)],
        request={"test": True},
        state_action_results=[
            {
                "molecule_type": 2,
                "n_qubits": 4,
                "state_action_validation_pass": True,
            }
        ],
        provenance={"test": True},
    )
    validate_system_size_payload(payload)
    assert payload["summary"]["all_operational_coefficients_usable"]
    assert payload["summary"]["maximum_n_qubits"] == 6
    assert payload["summary"]["all_state_action_validation_pass"]

    tampered = deepcopy(payload)
    tampered["size_results"][0]["operational_coefficient_usable"] = False
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_system_size_payload(tampered)
