from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from qiskit.quantum_info import Operator

import trotterlib.df_rte_tail as df_tail
import trotterlib.rte as rte_module
import trotterlib.rte_compiled_cost as compiled_cost_module
from trotterlib.df_hamiltonian import DFHamiltonian
from trotterlib.df_partial_randomized_pf import split_df_hamiltonian_by_ld
from trotterlib.df_partial_s2 import (
    QiskitDFPartialS2CircuitBuilder,
    make_df_partial_s2_step_request,
    prepare_df_partial_s2,
)
from trotterlib.df_partial_s2_repeated import (
    DFPartialS2RepeatedRequest,
    QiskitDFPartialS2RepeatedCircuitBuilder,
    make_df_partial_s2_repeated_request,
    repetition_count_for_rpe_round,
)
from trotterlib.rte import make_rte_config


def _hamiltonian(
    *,
    num_qubits: int = 2,
    identity_policy_case: bool = False,
) -> DFHamiltonian:
    diagonal_a = np.linspace(0.4, 1.0, num_qubits)
    diagonal_b = np.linspace(1.1, 0.3, num_qubits)
    return DFHamiltonian(
        constant=0.13,
        one_body=(
            np.zeros((num_qubits, num_qubits), dtype=np.complex128)
            if identity_policy_case
            else np.diag(np.linspace(0.2, -0.1, num_qubits)).astype(
                np.complex128
            )
        ),
        lambdas=np.asarray([0.2, -0.3]),
        g_matrices=(
            np.diag(diagonal_a).astype(np.complex128),
            np.diag(diagonal_b).astype(np.complex128),
        ),
        metadata={"name": "repeated-partial-s2"},
    )


def _randomized_case(*, identity_policy="extract_identity_phase"):
    hamiltonian = _hamiltonian()
    preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, 1),
        identity_policy=identity_policy,
    )
    config, distribution = make_rte_config(
        preparation.rte_preparation.symbolic_tail,
        evolution_time=0.17,
        rte_steps=2,
        truncation_tolerance=1.0,
        finite_taylor_order=2,
    )
    return hamiltonian, preparation, config, distribution


def _controlled_reference(unitary: np.ndarray) -> np.ndarray:
    identity = np.eye(unitary.shape[0], dtype=np.complex128)
    zero = np.zeros_like(identity)
    return np.block([[identity, zero], [zero, unitary]])


def _nontrivial_basis_case(step_time: float, repetition_count: int, *, controlled=False):
    hamiltonian = DFHamiltonian(
        constant=0.17,
        one_body=np.asarray([[0.4, 0.13], [0.13, -0.2]], dtype=np.complex128),
        lambdas=np.asarray([0.6, -0.35, 0.22]),
        g_matrices=(
            np.asarray([[1.0, 0.19], [0.19, 0.35]], dtype=np.complex128),
            np.asarray([[0.2, -0.23], [-0.23, 1.2]], dtype=np.complex128),
            np.asarray([[0.7, 0.31], [0.31, -0.4]], dtype=np.complex128),
        ),
        metadata={"name": "nontrivial-repeated-bases"},
    )
    preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, 2),
        identity_policy="extract_identity_phase",
    )
    config, distribution = make_rte_config(
        preparation.rte_preparation.symbolic_tail,
        evolution_time=step_time,
        rte_steps=2,
        truncation_tolerance=1.0,
        finite_taylor_order=2,
    )
    request = make_df_partial_s2_repeated_request(
        preparation,
        step_time=step_time,
        repetition_count=repetition_count,
        rte_config=config,
        rte_distribution=distribution,
        seed=41,
        controlled=controlled,
        ancilla_qubit=2 if controlled else None,
        construction_policy="boundary_optimized",
    )
    return hamiltonian, preparation, request


def test_repetition_count_one_matches_existing_step_exactly() -> None:
    _hamiltonian_value, preparation, config, distribution = _randomized_case()
    step = make_df_partial_s2_step_request(
        preparation,
        step_time=0.17,
        rte_steps=2,
        truncation_tolerance=1.0,
        finite_taylor_order=2,
        seed=7,
    )
    repeated = DFPartialS2RepeatedRequest.from_step_requests((step,))
    step_result = QiskitDFPartialS2CircuitBuilder().build_step(step)
    builder = QiskitDFPartialS2RepeatedCircuitBuilder()

    raw = builder.build(repeated, construction_policy="raw_concatenation")
    optimized = builder.build(repeated, construction_policy="boundary_optimized")

    assert np.allclose(Operator(raw.circuit).data, Operator(step_result.circuit).data)
    assert np.allclose(
        Operator(optimized.circuit).data,
        Operator(step_result.circuit).data,
    )
    assert raw.total_random_event_count == step_result.randomized_event_count
    assert raw.rte_steps_per_repetition == config.rte_steps
    assert raw.fused_boundary_count == 0
    assert repeated.step_seeds == (7,)


@pytest.mark.parametrize("bad_count", (0, -1, True, 1.0, "1"))
def test_repetition_count_rejects_nonpositive_and_noninteger_values(bad_count) -> None:
    _h, preparation, config, distribution = _randomized_case()
    with pytest.raises((TypeError, ValueError)):
        make_df_partial_s2_repeated_request(
            preparation,
            step_time=0.17,
            repetition_count=bad_count,
            rte_config=config,
            rte_distribution=distribution,
        )


def test_repetition_count_accepts_numpy_integer_and_rpe_mapping_is_strict() -> None:
    _h, preparation, config, distribution = _randomized_case()
    request = make_df_partial_s2_repeated_request(
        preparation,
        step_time=0.17,
        repetition_count=np.int64(2),
        rte_config=config,
        rte_distribution=distribution,
        seed=3,
    )
    assert request.repetition_count == 2
    assert repetition_count_for_rpe_round(np.int64(4)) == 16
    with pytest.raises(TypeError):
        repetition_count_for_rpe_round(True)


def test_from_step_requests_uses_strict_absolute_step_time_tolerance() -> None:
    hamiltonian = _hamiltonian()
    preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, hamiltonian.n_blocks),
    )
    first = make_df_partial_s2_step_request(preparation, step_time=1e9)
    outside_absolute_tolerance = make_df_partial_s2_step_request(
        preparation,
        step_time=1e9 + 1e-4,
    )
    with pytest.raises(ValueError, match="same step_time"):
        DFPartialS2RepeatedRequest.from_step_requests(
            (first, outside_absolute_tolerance)
        )


@pytest.mark.parametrize("repetition_count", (2, 3))
def test_deterministic_repetition_matches_ordered_step_product(
    repetition_count: int,
) -> None:
    hamiltonian = _hamiltonian()
    preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, hamiltonian.n_blocks),
    )
    step = make_df_partial_s2_step_request(preparation, step_time=0.19)
    request = DFPartialS2RepeatedRequest.from_step_requests(
        (step,) * repetition_count,
        construction_policy="boundary_optimized",
    )
    builder = QiskitDFPartialS2RepeatedCircuitBuilder()
    raw = builder.build(request, construction_policy="raw_concatenation")
    optimized = builder.build(request, construction_policy="boundary_optimized")
    one_step = np.asarray(
        Operator(QiskitDFPartialS2CircuitBuilder().build_step(step).circuit).data
    )
    expected = np.linalg.matrix_power(one_step, repetition_count)

    assert np.allclose(Operator(raw.circuit).data, expected, atol=1e-12)
    assert np.allclose(Operator(optimized.circuit).data, expected, atol=1e-12)
    assert optimized.fused_boundary_count == repetition_count - 1
    assert optimized.untranspiled_circuit_size < raw.untranspiled_circuit_size


def test_randomized_trajectory_order_and_boundary_optimization_match_step_product() -> None:
    _h, preparation, config, distribution = _randomized_case()
    request = make_df_partial_s2_repeated_request(
        preparation,
        step_time=0.17,
        repetition_count=3,
        rte_config=config,
        rte_distribution=distribution,
        seed=12,
        construction_policy="boundary_optimized",
    )
    step_builder = QiskitDFPartialS2CircuitBuilder()
    expected = np.eye(1 << preparation.num_system_qubits, dtype=np.complex128)
    for step_request in request.iter_step_requests():
        step_unitary = np.asarray(
            Operator(step_builder.build_step(step_request).circuit).data
        )
        expected = step_unitary @ expected
    repeated_builder = QiskitDFPartialS2RepeatedCircuitBuilder()
    raw = repeated_builder.build(request, construction_policy="raw_concatenation")
    optimized = repeated_builder.build(
        request,
        construction_policy="boundary_optimized",
    )

    assert np.allclose(Operator(raw.circuit).data, expected, atol=1e-12)
    assert np.allclose(Operator(optimized.circuit).data, expected, atol=1e-12)
    assert raw.matrix_product_convention == "U_(q-1)...U_1_U_0"
    assert tuple(step.step_index for step in raw.trajectory) == (0, 1, 2)
    reversed_seed_result = repeated_builder.build(
        replace(
            request,
            step_seeds=tuple(reversed(request.step_seeds)),
        ),
        construction_policy="raw_concatenation",
    )
    assert raw.trajectory_fingerprint != reversed_seed_result.trajectory_fingerprint


@pytest.mark.parametrize("repetition_count", (2, 3))
@pytest.mark.parametrize("step_time", (0.09, -0.09))
def test_nontrivial_df_basis_boundary_optimization_matches_dense_references(
    repetition_count: int,
    step_time: float,
) -> None:
    hamiltonian, preparation, request = _nontrivial_basis_case(
        step_time,
        repetition_count,
    )
    basis_hashes = tuple(
        block.basis_hash for block in preparation.deterministic_blocks
    )
    assert len(preparation.deterministic_blocks) == 3
    assert len(set(basis_hashes)) == 3
    assert all(
        block.basis_change_operations
        for block in preparation.deterministic_blocks
    )
    assert preparation.randomized_block_indices

    step_builder = QiskitDFPartialS2CircuitBuilder()
    expected = np.eye(1 << preparation.num_system_qubits, dtype=np.complex128)
    for step_request in request.iter_step_requests():
        expected = np.asarray(
            Operator(step_builder.build_step(step_request).circuit).data
        ) @ expected

    builder = QiskitDFPartialS2RepeatedCircuitBuilder()
    raw = builder.build(request, construction_policy="raw_concatenation")
    optimized = builder.build(request, construction_policy="boundary_optimized")
    controlled_request = _nontrivial_basis_case(
        step_time,
        repetition_count,
        controlled=True,
    )[2]
    controlled = builder.build(
        controlled_request,
        construction_policy="boundary_optimized",
    )

    assert np.allclose(Operator(raw.circuit).data, expected, atol=1e-12)
    assert np.allclose(
        Operator(optimized.circuit).data,
        Operator(raw.circuit).data,
        atol=1e-12,
    )
    assert np.allclose(
        Operator(controlled.circuit).data,
        _controlled_reference(expected),
        atol=1e-12,
    )
    assert optimized.fused_boundary_count == repetition_count - 1
    assert optimized.constant_phase == pytest.approx(
        -step_time * repetition_count * hamiltonian.constant
    )
    assert optimized.extracted_identity_phase == pytest.approx(
        -step_time
        * repetition_count
        * preparation.extracted_identity_coefficient
    )
    assert all(
        result.deterministic_block_order == preparation.deterministic_block_order
        for result in optimized.step_results
    )
    assert tuple(
        step.ordered_selected_component_ids for step in optimized.trajectory
    ) == tuple(
        tuple(event.selected_component_ids for event in occurrence.events)
        for occurrence in request.rte_occurrences
    )


def test_controlled_repetition_is_identity_on_zero_and_product_on_one() -> None:
    _h, preparation, config, distribution = _randomized_case()
    uncontrolled_request = make_df_partial_s2_repeated_request(
        preparation,
        step_time=0.17,
        repetition_count=2,
        rte_config=config,
        rte_distribution=distribution,
        seed=5,
    )
    controlled_request = make_df_partial_s2_repeated_request(
        preparation,
        step_time=0.17,
        repetition_count=2,
        rte_config=config,
        rte_distribution=distribution,
        seed=5,
        controlled=True,
        ancilla_qubit=2,
    )
    builder = QiskitDFPartialS2RepeatedCircuitBuilder()
    uncontrolled = np.asarray(
        Operator(
            builder.build(
                uncontrolled_request,
                construction_policy="boundary_optimized",
            ).circuit
        ).data
    )
    controlled = np.asarray(
        Operator(
            builder.build(
                controlled_request,
                construction_policy="boundary_optimized",
            ).circuit
        ).data
    )
    assert np.allclose(
        controlled,
        _controlled_reference(uncontrolled),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "identity_policy",
    ("extract_identity_phase", "faithful_identity_in_tail"),
)
def test_scalar_phase_metadata_is_applied_once_per_repetition(identity_policy) -> None:
    hamiltonian = _hamiltonian(identity_policy_case=True)
    preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, 1),
        identity_policy=identity_policy,
    )
    config, distribution = make_rte_config(
        preparation.rte_preparation.symbolic_tail,
        evolution_time=0.11,
        rte_steps=1,
        truncation_tolerance=1.0,
        finite_taylor_order=2,
    )
    request = make_df_partial_s2_repeated_request(
        preparation,
        step_time=0.11,
        repetition_count=3,
        rte_config=config,
        rte_distribution=distribution,
        seed=4,
    )
    result = QiskitDFPartialS2RepeatedCircuitBuilder().build(
        request,
        construction_policy="boundary_optimized",
    )
    assert result.constant_phase == pytest.approx(-3 * 0.11 * hamiltonian.constant)
    assert result.extracted_identity_phase == pytest.approx(
        -3 * 0.11 * preparation.extracted_identity_coefficient
    )
    if identity_policy == "faithful_identity_in_tail":
        assert result.extracted_identity_phase == 0.0


def test_attenuation_and_taylor_bounds_compose_independently() -> None:
    _h, preparation, config, distribution = _randomized_case()
    request = make_df_partial_s2_repeated_request(
        preparation,
        step_time=0.17,
        repetition_count=3,
        rte_config=config,
        rte_distribution=distribution,
        seed=9,
    )
    result = QiskitDFPartialS2RepeatedCircuitBuilder().build(request)
    attenuation = result.attenuation
    truncation = result.truncation

    assert attenuation.total_log_attenuation == pytest.approx(
        3 * math.log(attenuation.per_step_attenuation)
    )
    assert attenuation.total_attenuation == pytest.approx(
        attenuation.per_step_attenuation**3
    )
    assert not attenuation.underflowed
    assert truncation.partial_s2_randomized_residual_bound == pytest.approx(
        truncation.occurrence_truncation_residual_bound
    )
    expected = math.expm1(
        3 * config.rte_steps * math.log1p(config.truncation_residual_bound)
    )
    assert truncation.repeated_partial_s2_residual_bound == pytest.approx(expected)
    assert not truncation.product_formula_bias_included


def test_log_attenuation_remains_available_when_product_underflows() -> None:
    _h, preparation, _config, _distribution = _randomized_case()
    step_time = 1.2 / preparation.exact_rte_lambda_r
    config, distribution = make_rte_config(
        preparation.rte_preparation.symbolic_tail,
        evolution_time=step_time,
        rte_steps=1,
        truncation_tolerance=100.0,
        finite_taylor_order=2,
    )
    occurrence = preparation.rte_preparation.sample_occurrence_request(
        config,
        distribution,
        seed=2,
    )
    count = 1_000
    request = DFPartialS2RepeatedRequest(
        preparation=preparation,
        step_time=step_time,
        repetition_count=count,
        rte_config=config,
        rte_distribution=distribution,
        rte_occurrences=(occurrence,) * count,
        step_seeds=tuple(range(count)),
    )
    metadata = QiskitDFPartialS2RepeatedCircuitBuilder._attenuation(request)
    assert metadata.total_log_attenuation < -700.0
    assert metadata.underflowed
    assert metadata.total_attenuation is None


@pytest.mark.parametrize("num_qubits", (20, 26))
def test_high_qubit_repeated_construction_is_dense_free(
    num_qubits: int,
    monkeypatch,
) -> None:
    original_operator = df_tail.Operator

    def local_operator_only(value):
        from qiskit import QuantumCircuit

        if isinstance(value, QuantumCircuit):
            raise AssertionError("many-body Operator(circuit).data was requested")
        return original_operator(value)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a forbidden dense/enumeration/transpile helper was called")

    monkeypatch.setattr(df_tail, "Operator", local_operator_only)
    monkeypatch.setattr(rte_module, "enumerate_rte_events", forbidden)
    monkeypatch.setattr(compiled_cost_module, "transpile", forbidden)
    for name in (
        "basis_change_unitary",
        "diagonal_pauli_matrix",
        "component_dense_operator",
        "dense_extracted_df_tail",
        "dense_df_block_hamiltonian",
        "extraction_to_normalized_rte_tail",
    ):
        monkeypatch.setattr(df_tail, name, forbidden)

    hamiltonian = _hamiltonian(num_qubits=num_qubits)
    preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, 1),
    )
    config, distribution = make_rte_config(
        preparation.rte_preparation.symbolic_tail,
        evolution_time=1e-5,
        rte_steps=1,
        truncation_tolerance=1.0,
        finite_taylor_order=0,
    )
    request = make_df_partial_s2_repeated_request(
        preparation,
        step_time=1e-5,
        repetition_count=2,
        rte_config=config,
        rte_distribution=distribution,
        seed=8,
        controlled=True,
        ancilla_qubit=num_qubits,
        construction_policy="boundary_optimized",
    )
    result = QiskitDFPartialS2RepeatedCircuitBuilder().build(request)
    assert result.circuit_qubit_count == num_qubits + 1
    assert result.total_random_event_count == 2
    assert result.untranspiled_circuit_size > 0
