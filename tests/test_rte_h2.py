from __future__ import annotations

from functools import lru_cache

import numpy as np
from openfermion import QubitOperator, get_sparse_operator
from scipy.linalg import expm

from trotterlib.partial_randomized_pf import build_sorted_pauli_hamiltonian
from trotterlib.rte import (
    InvolutoryTailTerm,
    enumerate_rte_events,
    exact_enumerated_event_mean_operator,
    finite_rte_multi_step_operator,
    finite_rte_distribution,
    finite_rte_corrected_operator,
    finite_taylor_operator,
    make_rte_config,
    normalize_involutory_tail,
    sample_event_mean_operator,
    sample_rte_events,
)


@lru_cache(maxsize=1)
def _h2_pauli_tail():
    hamiltonian = build_sorted_pauli_hamiltonian(2)
    selected_tail = hamiltonian.sorted_terms[-3:]
    terms = []
    for term in selected_tail:
        pauli = QubitOperator(term.pauli_term, 1.0)
        dense_pauli = get_sparse_operator(
            pauli, n_qubits=hamiltonian.num_qubits
        ).toarray()
        terms.append(
            InvolutoryTailTerm(
                component_id=f"h2-pauli-{term.rank}",
                coefficient=term.coeff,
                operator=dense_pauli,
                df_fragment_id=f"pauli-tail-source-{term.original_index}",
                basis_id=f"pauli-{term.pauli_term}",
            )
        )

    return (
        normalize_involutory_tail("H2-three-smallest-Pauli-tail", terms),
        selected_tail,
    )


def _operator_map(tail):
    return {
        component.component_id: operator
        for component, operator in zip(tail.components, tail.operators, strict=True)
    }


def test_h2_pauli_tail_finite_rte_matches_dense_exponential() -> None:
    """Small H2 reference without controlled circuits or quantum shots."""
    tail, selected_tail = _h2_pauli_tail()
    config, distribution = make_rte_config(
        tail,
        evolution_time=0.2,
        rte_steps=3,
        truncation_tolerance=1e-12,
        seed=20260801,
    )
    finite = finite_rte_multi_step_operator(tail.normalized_hamiltonian, config)
    exact = expm(-1j * config.evolution_time * tail.dense_hamiltonian)

    np.testing.assert_allclose(
        tail.lambda_r,
        sum(abs(term.coeff) for term in selected_tail),
        rtol=1e-14,
    )
    assert distribution.truncation_residual_bound <= config.truncation_tolerance
    assert np.linalg.norm(finite - exact, ord=2) < 5e-12


def test_h2_k2_enumeration_and_seeded_sample_mean() -> None:
    tail, _selected_tail = _h2_pauli_tail()
    tau = 0.12
    distribution = finite_rte_distribution(tau, 2)
    events = enumerate_rte_events(tail.components, distribution)
    operators = _operator_map(tail)
    exact_mean = exact_enumerated_event_mean_operator(events, operators)
    expected = finite_taylor_operator(tail.normalized_hamiltonian, tau, 2) / (
        distribution.exact_finite_distribution
    )
    sampled = sample_rte_events(
        tail.components,
        distribution,
        sample_count=6000,
        seed=20260801,
    )
    repeated = sample_rte_events(
        tail.components,
        distribution,
        sample_count=6000,
        seed=20260801,
    )
    estimate = sample_event_mean_operator(sampled, operators)
    small_sample = sample_rte_events(
        tail.components,
        distribution,
        sample_count=50,
        seed=20260801,
    )
    small_estimate = sample_event_mean_operator(small_sample, operators)

    assert len(events) == len(tail.components) + len(tail.components) ** 3
    np.testing.assert_allclose(sum(event.event_probability for event in events), 1.0)
    np.testing.assert_allclose(exact_mean, expected, rtol=1e-13, atol=1e-13)
    assert sampled == repeated
    assert np.linalg.norm(estimate.operator_mean - exact_mean) < (
        6.0 * estimate.frobenius_standard_error
    )
    assert np.linalg.norm(estimate.operator_mean - exact_mean) < np.linalg.norm(
        small_estimate.operator_mean - exact_mean
    )
    negative_id = next(
        component.component_id
        for component in tail.components
        if component.coefficient_sign == -1
    )
    negative_event = next(
        event for event in events if event.rotation_component_id == negative_id
    )
    assert negative_event.application_sequence[-1].coefficient_sign == -1
    assert negative_event.unsigned_rotation_angle == -negative_event.rotation_angle

    config, _ = make_rte_config(
        tail,
        evolution_time=0.24,
        rte_steps=2,
        finite_taylor_order=2,
        truncation_tolerance=1.0,
    )
    one_step = finite_taylor_operator(
        tail.normalized_hamiltonian,
        config.dimensionless_step_time,
        2,
    )
    np.testing.assert_allclose(
        finite_rte_corrected_operator(tail.normalized_hamiltonian, config),
        one_step @ one_step,
    )
