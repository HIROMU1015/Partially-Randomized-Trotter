from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from qiskit.quantum_info import Operator

from trotterlib.df_hamiltonian import DFHamiltonian, PhysicalSector
from trotterlib.df_partial_randomized_pf import split_df_hamiltonian_by_ld
from trotterlib.df_partial_s2 import (
    DFPartialS2StepRequest,
    QiskitDFPartialS2CircuitBuilder,
    prepare_df_partial_s2,
)
from trotterlib.df_rte_circuit import DFRTEEventSequenceCircuitRequest
from trotterlib.df_rte_tail import extraction_to_normalized_rte_tail
from trotterlib.finite_rte_signal_validation import (
    validate_finite_rte_signal_payload,
    validate_finite_rte_signals,
    write_finite_rte_signal_validation,
)
from trotterlib.rte import (
    enumerate_rte_events,
    finite_rte_corrected_operator,
    make_rte_config,
)


def _hamiltonian() -> DFHamiltonian:
    return DFHamiltonian(
        constant=0.13,
        one_body=np.asarray(
            [[0.2, 0.03], [0.03, -0.1]],
            dtype=np.complex128,
        ),
        lambdas=np.asarray([0.2, -0.3]),
        g_matrices=(
            np.asarray([[1.0, 0.0], [0.0, 0.4]], dtype=np.complex128),
            np.asarray([[0.2, 0.0], [0.0, 2.0]], dtype=np.complex128),
        ),
        metadata={"name": "finite-rte-signal-validation-toy"},
    )


def test_sector_operator_validation_passes_and_json_is_tamper_evident(tmp_path) -> None:
    payload = validate_finite_rte_signals(
        _hamiltonian(),
        PhysicalSector.number_sector(n_qubits=2, n_electrons=1),
        ld=1,
        delta_time=0.05,
        q_values=(1, 2),
        rte_step_values=(1, 2),
        finite_taylor_orders=(0, 2),
        coefficient_atol=0.05,
        provenance={"test": True},
    )

    assert payload["summary"]["overall_pass"]
    assert payload["summary"]["point_count"] == 8
    assert payload["final_cost_evaluation_performed"] is False
    assert payload["partial_s2"]["threshold_dropped_component_count"] == 1
    assert payload["partial_s2"]["threshold_dropped_coefficient_l1"] == pytest.approx(
        0.04
    )
    assert payload["summary"]["maximum_explicit_trajectory_count_log10_avoided"] > 0
    validate_finite_rte_signal_payload(payload)

    target = tmp_path / "validation.json"
    write_finite_rte_signal_validation(payload, target)
    assert target.exists()

    tampered = deepcopy(payload)
    tampered["summary"]["overall_pass"] = False
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_finite_rte_signal_payload(tampered)


def test_operator_moment_acceleration_matches_explicit_partial_s2_mean() -> None:
    hamiltonian = _hamiltonian()
    preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, 1),
        identity_policy="extract_identity_phase",
    )
    config, distribution = make_rte_config(
        preparation.rte_preparation.symbolic_tail,
        evolution_time=0.05,
        rte_steps=1,
        finite_taylor_order=2,
        truncation_tolerance=1.0,
        seed=17,
    )
    events = enumerate_rte_events(
        preparation.rte_preparation.symbolic_tail.components,
        distribution,
    )
    builder = QiskitDFPartialS2CircuitBuilder()
    explicit_mean = np.zeros((4, 4), dtype=np.complex128)
    first_request = None
    for event in events:
        occurrence = DFRTEEventSequenceCircuitRequest(
            events=(event,),
            component_specs=preparation.rte_preparation.component_specs,
            tail_id=preparation.tail_extraction.tail_id,
            tail_hash=preparation.tail_extraction.tail_hash,
            occurrence_rte_steps=1,
        )
        request = DFPartialS2StepRequest(
            preparation=preparation,
            step_time=0.05,
            rte_config=config,
            rte_distribution=distribution,
            rte_occurrence=occurrence,
            seed=17,
        )
        if first_request is None:
            first_request = request
        explicit_mean += event.event_probability * np.asarray(
            Operator(builder.build_step(request).circuit).data
        )

    assert first_request is not None
    parts = builder.build_additive_circuits(first_request)
    forward = np.asarray(Operator(parts.forward_deterministic_half).data)
    reverse = np.asarray(Operator(parts.reverse_deterministic_half).data)
    dense_tail = extraction_to_normalized_rte_tail(
        preparation.tail_extraction
    )
    accelerated_corrected = (
        reverse
        @ finite_rte_corrected_operator(
            dense_tail.normalized_hamiltonian,
            config,
        )
        @ forward
    )
    attenuation = distribution.exact_finite_distribution ** -1

    np.testing.assert_allclose(
        explicit_mean,
        attenuation * accelerated_corrected,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        explicit_mean @ explicit_mean,
        attenuation**2 * accelerated_corrected @ accelerated_corrected,
        rtol=1e-12,
        atol=1e-12,
    )
