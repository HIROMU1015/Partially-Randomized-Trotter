from __future__ import annotations

import math
from copy import deepcopy

import numpy as np
import pytest
from qiskit.quantum_info import Operator, Pauli, Statevector

import trotterlib.pr2_s0_s1_validation as validation
from trotterlib.df_hamiltonian import DFHamiltonian, PhysicalSector
from trotterlib.df_partial_randomized_pf import split_df_hamiltonian_by_ld
from trotterlib.df_partial_s2 import (
    DFPartialS2StepRequest,
    QiskitDFPartialS2CircuitBuilder,
    make_df_partial_s2_step_request,
    prepare_df_partial_s2,
)
from trotterlib.df_partial_s2_repeated import (
    QiskitDFPartialS2RepeatedCircuitBuilder,
    make_df_partial_s2_repeated_request,
)
from trotterlib.df_rte_circuit import DFRTEEventSequenceCircuitRequest
from trotterlib.df_rte_tail import extraction_to_normalized_rte_tail
from trotterlib.rpe_hadamard_compiled_cost_benchmark import (
    QiskitRPEHadamardBenchmarkCircuitBuilder,
)
from trotterlib.rpe_hadamard_interrogation import (
    RPEHadamardInterrogationRequest,
)
from trotterlib.rte import (
    enumerate_rte_events,
    finite_rte_operator_moments,
    make_rte_config,
)


def _toy_hamiltonian(num_qubits: int = 2) -> DFHamiltonian:
    diagonal_a = np.linspace(0.4, 1.0, num_qubits)
    diagonal_b = np.linspace(1.1, 0.3, num_qubits)
    diagonal_c = np.linspace(-0.2, 0.8, num_qubits)
    diagonal_d = np.linspace(0.9, -0.4, num_qubits)
    return DFHamiltonian(
        constant=0.13,
        one_body=np.diag(np.linspace(0.2, -0.1, num_qubits)).astype(
            np.complex128
        ),
        lambdas=np.asarray([0.2, -0.3, 0.11, -0.07]),
        g_matrices=tuple(
            np.diag(values).astype(np.complex128)
            for values in (diagonal_a, diagonal_b, diagonal_c, diagonal_d)
        ),
        metadata={"name": "pr2-s0-s1-validation-toy", "tuple": (1, 2)},
    )


def _wrapper_hamiltonian() -> DFHamiltonian:
    return DFHamiltonian(
        constant=0.13,
        one_body=np.asarray([[0.2]], dtype=np.complex128),
        lambdas=np.asarray([0.4, -0.3]),
        g_matrices=(
            np.asarray([[1.0]], dtype=np.complex128),
            np.asarray([[0.7]], dtype=np.complex128),
        ),
        metadata={"name": "pr2-wrapper-toy"},
    )


def _random_preparation(hamiltonian: DFHamiltonian, rank: int = 1):
    return prepare_df_partial_s2(
        hamiltonian,
        validation.generation_partition(hamiltonian, rank),
        identity_policy="extract_identity_phase",
        coefficient_atol=0.0,
        partition_policy="explicit_ordered_partition",
    )


def _controlled_reference(unitary: np.ndarray) -> np.ndarray:
    identity = np.eye(unitary.shape[0], dtype=np.complex128)
    zero = np.zeros_like(identity)
    return np.block([[identity, zero], [zero, unitary]])


def _minimal_s0_payload(status: str = "BLOCKED_IMPLEMENTATION_INVALID"):
    payload = {
        "schema_version": validation.SCHEMA_S0,
        "status": status,
        "authorization_commit": validation.AUTHORIZATION_COMMIT,
        "authorization_manifest_sha256": (
            validation.AUTHORIZATION_MANIFEST_SHA256
        ),
        "amendment_v3_sha256": validation.AMENDMENT_V3_SHA256,
        "environment": {},
        "provenance": {"test": True},
        "automatic_next_stage": None,
        "S1_authorized": status == "S0_PASS_S1_AUTHORIZED",
        "held_out_signal_cost_ranking_evaluated": False,
        "molecular_calculations_executed": 0,
        "signal_evaluations_executed": 0,
        "circuits_compiled": 0,
        "trajectory_samples_drawn": 0,
        "quantum_shots_executed": 0,
    }
    payload["result_fingerprint"] = validation._fingerprint(payload)
    return payload


def test_generation_prefix_partition_is_ordered_disjoint_and_complete() -> None:
    hamiltonian = _toy_hamiltonian()
    partition = validation.generation_partition(hamiltonian, 2)

    assert partition.deterministic_block_indices == (0, 1)
    assert partition.randomized_block_indices == (2, 3)
    assert not set(partition.deterministic_block_indices).intersection(
        partition.randomized_block_indices
    )
    assert sorted(
        partition.deterministic_block_indices + partition.randomized_block_indices
    ) == list(range(hamiltonian.n_blocks))


def test_snapshot_round_trip_recomputes_hashes_and_detects_array_tampering(
    tmp_path,
) -> None:
    hamiltonian = _toy_hamiltonian()
    sector = PhysicalSector.number_sector(n_qubits=2, n_electrons=1)
    sector_state = np.asarray([1.0, 0.0], dtype=np.complex128)
    state = np.zeros(4, dtype=np.complex128)
    state[sector.basis_indices] = sector_state
    path = tmp_path / "snapshot.npz"

    record = validation.write_snapshot(
        path,
        hamiltonian,
        sector,
        state,
        sector_state,
        distance=1.0,
        role="development",
    )
    loaded, loaded_sector, loaded_state, metadata = validation.load_snapshot(path)

    assert validation.df_hamiltonian_hash(loaded) == record["hamiltonian_hash"]
    assert metadata["state_hash"] == record["state_hash"]
    np.testing.assert_array_equal(loaded_sector.basis_indices, sector.basis_indices)
    np.testing.assert_allclose(loaded_state, state)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        validation.write_snapshot(
            path,
            hamiltonian,
            sector,
            state,
            sector_state,
            distance=1.0,
            role="development",
        )

    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    arrays["one_body"] = arrays["one_body"].copy()
    arrays["one_body"][0, 0] += 1e-6
    tampered = tmp_path / "tampered.npz"
    np.savez_compressed(tampered, **arrays)
    with pytest.raises(ValueError, match="one-body hash mismatch"):
        validation.load_snapshot(tampered)


def test_corrected_event_mean_matches_explicit_enumeration_and_improves_with_k() -> None:
    hamiltonian = _toy_hamiltonian()
    preparation = _random_preparation(hamiltonian, 1)
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
    explicit_raw_mean = np.zeros((4, 4), dtype=np.complex128)
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
        first_request = first_request or request
        explicit_raw_mean += event.event_probability * np.asarray(
            Operator(builder.build_step(request).circuit).data
        )

    assert first_request is not None
    parts = builder.build_additive_circuits(first_request)
    forward = np.asarray(Operator(parts.forward_deterministic_half).data)
    reverse = np.asarray(Operator(parts.reverse_deterministic_half).data)
    normalized_tail = extraction_to_normalized_rte_tail(
        preparation.tail_extraction
    ).normalized_hamiltonian
    moments = finite_rte_operator_moments(normalized_tail, config)
    accelerated_corrected = reverse @ moments.corrected_operator @ forward

    np.testing.assert_allclose(
        explicit_raw_mean,
        accelerated_corrected / moments.normalization_product,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        moments.normalization_product * explicit_raw_mean,
        accelerated_corrected,
        rtol=1e-12,
        atol=1e-12,
    )

    config_k4, distribution_k4 = make_rte_config(
        preparation.rte_preparation.symbolic_tail,
        evolution_time=0.05,
        rte_steps=1,
        finite_taylor_order=4,
        truncation_tolerance=1.0,
        seed=17,
    )
    assert distribution_k4.step_truncation_residual_bound < (
        distribution.step_truncation_residual_bound
    )
    assert finite_rte_operator_moments(
        normalized_tail, config_k4
    ).normalization_product >= 1.0


def test_corrected_hoeffding_formula_has_multiplier_squared_and_ineligibility() -> None:
    multiplier = 1.7
    bias = 0.01
    expected = math.ceil(
        2.0
        * multiplier**2
        / (validation.AXIS_ERROR - bias) ** 2
        * math.log(2.0 / validation.AXIS_ALPHA)
    )

    assert validation.corrected_hoeffding_shots(multiplier, bias) == expected
    assert validation.corrected_hoeffding_shots(
        1.0, validation.AXIS_ERROR
    ) is None
    assert validation.corrected_hoeffding_shots(
        1.0, validation.AXIS_ERROR + 1e-9
    ) is None


@pytest.mark.parametrize("repetition_count", (1, 8))
def test_q1_q8_controlled_semantics_step_product_and_wrapper_signs(
    repetition_count: int,
) -> None:
    preparation = _random_preparation(_wrapper_hamiltonian(), 1)
    config, distribution = make_rte_config(
        preparation.rte_preparation.symbolic_tail,
        evolution_time=0.11,
        rte_steps=2,
        finite_taylor_order=2,
        truncation_tolerance=1.0,
        seed=31,
    )
    common = dict(
        preparation=preparation,
        step_time=0.11,
        repetition_count=repetition_count,
        rte_config=config,
        rte_distribution=distribution,
        seed=31,
        construction_policy="boundary_optimized",
    )
    uncontrolled_request = make_df_partial_s2_repeated_request(**common)
    controlled_request = make_df_partial_s2_repeated_request(
        **common,
        controlled=True,
        ancilla_qubit=preparation.num_system_qubits,
    )
    builder = QiskitDFPartialS2RepeatedCircuitBuilder()
    uncontrolled = builder.build(uncontrolled_request)
    controlled = builder.build(controlled_request)
    uncontrolled_operator = np.asarray(Operator(uncontrolled.circuit).data)
    controlled_operator = np.asarray(Operator(controlled.circuit).data)

    np.testing.assert_allclose(
        controlled_operator,
        _controlled_reference(uncontrolled_operator),
        rtol=1e-12,
        atol=1e-12,
    )
    explicit_product = np.eye(2, dtype=np.complex128)
    step_builder = QiskitDFPartialS2CircuitBuilder()
    for step_request in uncontrolled_request.iter_step_requests():
        step = np.asarray(Operator(step_builder.build_step(step_request).circuit).data)
        explicit_product = step @ explicit_product
    np.testing.assert_allclose(
        uncontrolled_operator, explicit_product, rtol=1e-12, atol=1e-12
    )

    system_state = np.asarray(
        [math.sqrt(0.3), 1j * math.sqrt(0.7)], dtype=np.complex128
    )
    signal = complex(
        np.vdot(system_state, uncontrolled_operator @ system_state)
    )
    wrapper_builder = QiskitRPEHadamardBenchmarkCircuitBuilder(
        maximum_repetition_count=8
    )
    initial = np.concatenate((system_state, np.zeros_like(system_state)))
    observable = Pauli("ZI")
    for axis, component, expected in (
        ("cosine", "real", signal.real),
        ("sine", "imaginary", signal.imag),
    ):
        wrapper = wrapper_builder.build(
            RPEHadamardInterrogationRequest(
                evolution=controlled,
                axis=axis,
                include_measurement=False,
            )
        )
        actual = float(
            np.real(
                Statevector(initial)
                .evolve(wrapper.circuit)
                .expectation_value(observable)
            )
        )
        assert actual == pytest.approx(expected, abs=1e-12)
        assert wrapper.signal_component == component
        assert wrapper.bit_value_mapping == ((0, 1), (1, -1))
        assert wrapper.quantum_shots_executed == 0


def test_deterministic_candidate_has_no_rte_and_unit_multiplier() -> None:
    hamiltonian = _toy_hamiltonian()
    preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, hamiltonian.n_blocks),
        identity_policy="extract_identity_phase",
    )
    request = make_df_partial_s2_step_request(preparation, step_time=0.1)

    assert preparation.is_deterministic_only
    assert request.rte_config is None
    assert request.rte_distribution is None
    assert request.rte_occurrence is None
    assert validation.corrected_hoeffding_shots(1.0, 0.0) is not None


def test_payload_fingerprint_stage_guard_and_non_overwrite(tmp_path) -> None:
    payload = _minimal_s0_payload()
    validation.validate_s0_payload(payload)

    tampered = deepcopy(payload)
    tampered["status"] = "S0_PASS_S1_AUTHORIZED"
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validation.validate_s0_payload(tampered)
    with pytest.raises(RuntimeError, match="not authorized"):
        validation.run_s1(payload, provenance={"test": True})

    target = tmp_path / "s0.json"
    validation.write_json_artifact(
        payload, target, validator=validation.validate_s0_payload
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        validation.write_json_artifact(
            payload, target, validator=validation.validate_s0_payload
        )


def test_environment_gate_is_exact_for_the_frozen_virtual_environment() -> None:
    record = validation.environment_record()

    assert record["exact_match"]
    assert record["versions"] == record["expected_versions"]
