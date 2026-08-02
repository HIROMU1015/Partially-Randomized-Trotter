from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

import trotterlib.df_rte_tail as df_tail
from trotterlib.df_hamiltonian import DFHamiltonian
from trotterlib.df_partial_randomized_pf import (
    rank_df_fragments,
    split_df_hamiltonian_by_ld,
)
from trotterlib.df_partial_s2 import (
    DFDeterministicFragmentSpec,
    DFDeterministicOneBodySpec,
    DFPartialS2StepRequest,
    QiskitDFPartialS2CircuitBuilder,
    make_df_partial_s2_step_request,
    prepare_df_partial_s2,
)
from trotterlib.df_rte_circuit import DFRTEEventSequenceCircuitRequest
from trotterlib.df_rte_tail import extraction_to_normalized_rte_tail
from trotterlib.df_trotter.circuit import build_df_trotter_circuit
from trotterlib.df_trotter.model import Block, DFBlock, OneBodyGaussianBlock
from trotterlib.df_trotter.ops import (
    append_diagonal_primitives,
    apply_D_one_body,
    apply_D_squared,
    df_squared_diagonal_primitives,
    one_body_diagonal_primitives,
)
from trotterlib.rte import enumerate_rte_events, event_unitary, make_rte_config


def test_numeric_primitives_prune_only_exact_zero_angles() -> None:
    primitives = one_body_diagonal_primitives(
        np.asarray([0.0, 1e-15]),
        1.0,
    )

    assert primitives.rz == ((1, -1e-15),)


def _hamiltonian(
    *,
    constant: float = 0.13,
    one_body: bool = True,
) -> DFHamiltonian:
    one_body_matrix = (
        np.asarray([[0.2, 0.03], [0.03, -0.1]], dtype=np.complex128)
        if one_body
        else np.zeros((2, 2), dtype=np.complex128)
    )
    return DFHamiltonian(
        constant=constant,
        one_body=one_body_matrix,
        lambdas=np.asarray([0.2, -0.3]),
        g_matrices=(
            np.asarray([[1.0, 0.0], [0.0, 0.4]], dtype=np.complex128),
            np.asarray([[0.2, 0.0], [0.0, 2.0]], dtype=np.complex128),
        ),
        metadata={"name": "partial-s2-toy"},
    )


def _as_block(spec) -> Block:
    if isinstance(spec, DFDeterministicOneBodySpec):
        return Block.from_one_body_gaussian(
            OneBodyGaussianBlock(
                U_ops=spec.runtime_basis_operations,
                eps=np.asarray(spec.diagonal_eigenvalues),
            )
        )
    return Block.from_df(
        DFBlock(
            U_ops=spec.runtime_basis_operations,
            eta=np.asarray(spec.diagonal_eta),
            lam=spec.lam,
        )
    )


def _dense_step_reference(request: DFPartialS2StepRequest) -> np.ndarray:
    preparation = request.preparation
    forward = QuantumCircuit(preparation.num_system_qubits)
    reverse = QuantumCircuit(preparation.num_system_qubits)
    for spec in preparation.deterministic_blocks:
        _as_block(spec).apply(forward, request.step_time / 2.0)
    for spec in reversed(preparation.deterministic_blocks):
        _as_block(spec).apply(reverse, request.step_time / 2.0)
    rte = np.eye(1 << preparation.num_system_qubits, dtype=np.complex128)
    if request.rte_occurrence is not None:
        dense_tail = extraction_to_normalized_rte_tail(
            preparation.tail_extraction
        )
        operators = dict(
            zip(
                (item.component_id for item in dense_tail.components),
                dense_tail.operators,
                strict=True,
            )
        )
        for event in request.rte_occurrence.events:
            rte = event_unitary(event, operators) @ rte
    phase = np.exp(
        -1j
        * request.step_time
        * (
            preparation.constant_coefficient
            + preparation.extracted_identity_coefficient
        )
    )
    return (
        phase
        * np.asarray(Operator(reverse).data)
        @ rte
        @ np.asarray(Operator(forward).data)
    )


def _controlled_reference(unitary: np.ndarray) -> np.ndarray:
    identity = np.eye(unitary.shape[0], dtype=np.complex128)
    zero = np.zeros_like(identity)
    return np.block([[identity, zero], [zero, unitary]])


@pytest.mark.parametrize("ld", (0, 1, 2))
def test_partition_preparation_preserves_ranked_prefix_and_exact_tail(ld: int) -> None:
    hamiltonian = _hamiltonian()
    ranked = rank_df_fragments(hamiltonian)
    partition = split_df_hamiltonian_by_ld(
        hamiltonian,
        ld,
        ranked_fragments=ranked,
    )
    preparation = prepare_df_partial_s2(
        hamiltonian,
        partition,
        identity_policy="extract_identity_phase",
    )
    repeated = prepare_df_partial_s2(
        hamiltonian,
        partition,
        identity_policy="extract_identity_phase",
    )

    assert partition.deterministic_block_indices == tuple(
        item.original_index for item in ranked[:ld]
    )
    assert partition.randomized_block_indices == tuple(
        item.original_index for item in ranked[ld:]
    )
    assert not set(partition.deterministic_block_indices).intersection(
        partition.randomized_block_indices
    )
    assert set(
        (*partition.deterministic_block_indices, *partition.randomized_block_indices)
    ) == set(range(hamiltonian.n_blocks))
    assert preparation.deterministic_fragment_indices == (
        partition.deterministic_block_indices
    )
    assert preparation.randomized_block_indices == partition.randomized_block_indices
    assert preparation.ranking_proxy_lambda_r == partition.lambda_r
    assert preparation.exact_rte_lambda_r == preparation.tail_extraction.rte_lambda_r
    if partition.randomized_block_indices:
        assert preparation.ranking_proxy_lambda_r != pytest.approx(
            preparation.exact_rte_lambda_r
        )
        assert {
            item.df_fragment_id for item in preparation.tail_extraction.components
        } <= {
            f"df-fragment-{index}"
            for index in partition.randomized_block_indices
        }
    else:
        assert preparation.is_deterministic_only
        assert preparation.rte_preparation.component_specs == ()
    assert preparation.constant_coefficient == hamiltonian.constant
    assert preparation.deterministic_blocks[0].block_kind == "one_body"
    assert preparation.preparation_hash == repeated.preparation_hash
    assert preparation.partition_hash == repeated.partition_hash


@pytest.mark.parametrize("bad_ld", (True, 1.0, "1", -1, 3))
def test_ld_requires_strict_bounded_integer(bad_ld) -> None:
    with pytest.raises((TypeError, ValueError)):
        split_df_hamiltonian_by_ld(_hamiltonian(), bad_ld)


def test_ld_accepts_numpy_integer_and_rejects_incomplete_ranking() -> None:
    hamiltonian = _hamiltonian()
    partition = split_df_hamiltonian_by_ld(hamiltonian, np.int64(1))
    assert partition.ld == 1
    with pytest.raises(ValueError, match="cover every"):
        split_df_hamiltonian_by_ld(
            hamiltonian,
            1,
            ranked_fragments=rank_df_fragments(hamiltonian)[:1],
        )


@pytest.mark.parametrize(
    ("primitive_factory", "legacy_apply", "args"),
    (
        (
            one_body_diagonal_primitives,
            apply_D_one_body,
            (np.asarray([0.2, -0.3]), 0.41),
        ),
        (
            df_squared_diagonal_primitives,
            apply_D_squared,
            (np.asarray([1.0, -0.4]), 0.7, 0.41),
        ),
    ),
)
def test_shared_diagonal_primitives_match_existing_and_controlled_unitaries(
    primitive_factory,
    legacy_apply,
    args,
) -> None:
    legacy = QuantumCircuit(2)
    legacy_apply(legacy, *args)
    primitives = primitive_factory(*args)
    actual = QuantumCircuit(2)
    append_diagonal_primitives(actual, primitives)
    controlled = QuantumCircuit(3)
    append_diagonal_primitives(
        controlled,
        primitives,
        controlled=True,
        ancilla_qubit=2,
    )

    expected = np.asarray(Operator(legacy).data)
    assert np.allclose(Operator(actual).data, expected, atol=1e-12)
    assert np.allclose(
        Operator(controlled).data,
        _controlled_reference(expected),
        atol=1e-12,
    )
    assert primitives.global_phase != 0.0


def test_partial_s2_multiple_events_matches_dense_uncontrolled_and_controlled() -> None:
    hamiltonian = _hamiltonian()
    preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, 1),
        identity_policy="extract_identity_phase",
    )
    config, distribution = make_rte_config(
        preparation.rte_preparation.symbolic_tail,
        evolution_time=0.17,
        rte_steps=2,
        truncation_tolerance=1.0,
        finite_taylor_order=2,
    )
    events = enumerate_rte_events(
        preparation.rte_preparation.symbolic_tail.components,
        distribution,
        max_events=10_000,
    )
    order_zero_z = next(
        event
        for event in events
        if event.taylor_order == 0
        and len(event.application_sequence[-1].diagonal_pauli_support) == 1
    )
    order_two_zz = next(
        event
        for event in events
        if event.taylor_order == 2
        and event.application_sequence[-1].diagonal_pauli_support == (0, 1)
    )
    chosen = (order_zero_z, order_two_zz)

    def make_request(controlled: bool) -> DFPartialS2StepRequest:
        occurrence = DFRTEEventSequenceCircuitRequest(
            events=chosen,
            component_specs=preparation.rte_preparation.component_specs,
            controlled=controlled,
            ancilla_qubit=2 if controlled else None,
            tail_id=preparation.tail_extraction.tail_id,
            tail_hash=preparation.tail_extraction.tail_hash,
            occurrence_rte_steps=2,
        )
        return DFPartialS2StepRequest(
            preparation=preparation,
            step_time=0.17,
            rte_config=config,
            rte_distribution=distribution,
            rte_occurrence=occurrence,
            controlled=controlled,
            ancilla_qubit=2 if controlled else None,
            seed=7,
        )

    builder = QiskitDFPartialS2CircuitBuilder()
    uncontrolled_request = make_request(False)
    expected = _dense_step_reference(uncontrolled_request)
    uncontrolled = builder.build_step(uncontrolled_request)
    controlled = builder.build_step(make_request(True))

    assert np.allclose(Operator(uncontrolled.circuit).data, expected, atol=1e-12)
    assert np.allclose(
        Operator(controlled.circuit).data,
        _controlled_reference(expected),
        atol=1e-12,
    )
    assert uncontrolled.randomized_event_count == 2
    assert uncontrolled.rte_steps == 2
    assert uncontrolled.finite_taylor_order == 2
    assert uncontrolled.constant_phase == pytest.approx(-0.17 * hamiltonian.constant)
    assert uncontrolled.extracted_identity_phase == pytest.approx(
        -0.17 * preparation.extracted_identity_coefficient
    )
    assert uncontrolled.deterministic_block_order == (
        preparation.deterministic_block_order
    )
    assert order_two_zz.phase == -1
    assert {
        application.coefficient_sign
        for event in chosen
        for application in event.application_sequence
    } >= {-1, 1}


@pytest.mark.parametrize(
    ("identity_policy", "one_body", "constant"),
    (
        ("faithful_identity_in_tail", True, 0.13),
        ("extract_identity_phase", False, 0.0),
    ),
)
def test_identity_policy_one_body_and_constant_variants_match_dense(
    identity_policy: str,
    one_body: bool,
    constant: float,
) -> None:
    hamiltonian = _hamiltonian(constant=constant, one_body=one_body)
    preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, 0),
        identity_policy=identity_policy,
    )
    request = make_df_partial_s2_step_request(
        preparation,
        step_time=0.11,
        rte_steps=1,
        truncation_tolerance=1.0,
        finite_taylor_order=2,
        seed=4,
    )
    result = QiskitDFPartialS2CircuitBuilder().build_step(request)

    assert np.allclose(
        Operator(result.circuit).data,
        _dense_step_reference(request),
        atol=1e-12,
    )
    if identity_policy == "faithful_identity_in_tail":
        assert preparation.extracted_identity_coefficient == 0.0
    if not one_body:
        assert all(
            block.block_kind != "one_body"
            for block in preparation.deterministic_blocks
        )


def test_deterministic_only_limit_matches_existing_second_order_builder() -> None:
    hamiltonian = _hamiltonian()
    preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, hamiltonian.n_blocks),
    )
    request = make_df_partial_s2_step_request(preparation, step_time=0.19)
    result = QiskitDFPartialS2CircuitBuilder().build_step(request)
    existing = build_df_trotter_circuit(
        tuple(_as_block(spec) for spec in preparation.deterministic_blocks),
        time=0.19,
        num_qubits=hamiltonian.n_qubits,
        pf_label="2nd",
        energy_shift=hamiltonian.constant,
    )

    assert request.rte_config is None
    assert np.allclose(Operator(result.circuit).data, Operator(existing).data)
    assert result.randomized_event_count == 0
    assert result.attenuation_factor == 1.0
    assert result.exact_distribution_normalization == 1.0

    controlled_request = make_df_partial_s2_step_request(
        preparation,
        step_time=0.19,
        controlled=True,
        ancilla_qubit=2,
    )
    controlled = QiskitDFPartialS2CircuitBuilder().build_step(
        controlled_request
    )
    assert np.allclose(
        Operator(controlled.circuit).data,
        _controlled_reference(np.asarray(Operator(existing).data)),
        atol=1e-12,
    )
    forward = QiskitDFPartialS2CircuitBuilder().build_additive_circuits(
        controlled_request
    ).forward_deterministic_half
    for instruction in forward.data:
        qubits = {
            forward.find_bit(qubit).index for qubit in instruction.qubits
        }
        if instruction.operation.name in {"rz", "xx_plus_yy"}:
            assert 2 not in qubits
        if 2 in qubits:
            assert instruction.operation.name in {"p", "crz", "crzz"}


def test_step_validation_threshold_metadata_and_unsupported_formula() -> None:
    hamiltonian = _hamiltonian()
    preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, 1),
        coefficient_atol=0.05,
    )
    assert preparation.threshold_dropped_component_count > 0
    assert preparation.threshold_dropped_coefficient_l1 > 0.0
    assert preparation.threshold_operator_error_bound == (
        preparation.threshold_dropped_coefficient_l1
    )
    with pytest.raises(ValueError, match="only pf_label='2nd'"):
        make_df_partial_s2_step_request(
            preparation,
            step_time=0.1,
            truncation_tolerance=1.0,
            finite_taylor_order=2,
            pf_label="4th",  # type: ignore[arg-type]
        )
    valid = make_df_partial_s2_step_request(
        preparation,
        step_time=0.1,
        truncation_tolerance=1.0,
        finite_taylor_order=2,
        controlled=True,
        ancilla_qubit=2,
    )
    with pytest.raises(ValueError, match="controlled conditions"):
        replace(valid, controlled=False, ancilla_qubit=None)


@pytest.mark.parametrize("num_qubits", (20, 26))
def test_high_qubit_partial_s2_construction_is_dense_free(
    num_qubits: int,
    monkeypatch,
) -> None:
    original_operator = df_tail.Operator

    def local_operator_only(value):
        if isinstance(value, QuantumCircuit):
            raise AssertionError("many-body Operator(circuit).data was requested")
        return original_operator(value)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a dense DF helper was called")

    monkeypatch.setattr(df_tail, "Operator", local_operator_only)
    for name in (
        "basis_change_unitary",
        "diagonal_pauli_matrix",
        "component_dense_operator",
        "dense_extracted_df_tail",
        "dense_df_block_hamiltonian",
        "extraction_to_normalized_rte_tail",
    ):
        monkeypatch.setattr(df_tail, name, forbidden)

    diagonal_a = np.linspace(0.2, 1.0, num_qubits)
    diagonal_b = np.linspace(1.1, 0.3, num_qubits)
    hamiltonian = DFHamiltonian(
        constant=0.02,
        one_body=np.zeros((num_qubits, num_qubits), dtype=np.complex128),
        lambdas=np.asarray([0.4, -0.2]),
        g_matrices=(
            np.diag(diagonal_a).astype(np.complex128),
            np.diag(diagonal_b).astype(np.complex128),
        ),
        metadata={"synthetic": True},
    )
    preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, 1),
        identity_policy="extract_identity_phase",
    )
    request = make_df_partial_s2_step_request(
        preparation,
        step_time=1e-5,
        rte_steps=1,
        truncation_tolerance=1.0,
        finite_taylor_order=0,
        seed=8,
        controlled=True,
        ancilla_qubit=num_qubits,
    )
    result = QiskitDFPartialS2CircuitBuilder().build_step(request)

    assert result.circuit_qubit_count == num_qubits + 1
    assert result.randomized_event_count == 1
    assert result.untranspiled_circuit_size > 0
    assert result.compiler_independent_fingerprint
