from __future__ import annotations

import math

import numpy as np
from scipy.linalg import expm

from trotterlib.df_hamiltonian import DFHamiltonian, PhysicalSector, df_linear_operator
from trotterlib.df_partial_randomized_pf import (
    select_df_h_d,
    split_df_hamiltonian_by_ld,
)
from trotterlib.rte import (
    InvolutoryTailTerm,
    enumerate_rte_events,
    event_unitary,
    finite_event_mean_operator,
    finite_rte_attenuation,
    finite_rte_distribution,
    finite_rte_multi_step_operator,
    finite_taylor_operator,
    make_rte_config,
    normalize_involutory_tail,
    sample_rte_events,
)


I2 = np.eye(2, dtype=np.complex128)
X = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
Y = np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
Z = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


def _toy_tail():
    return normalize_involutory_tail(
        "toy-xz",
        (
            InvolutoryTailTerm(
                "x",
                0.7,
                X,
                df_fragment_id="df-0",
                basis_id="basis-a",
            ),
            InvolutoryTailTerm(
                "z",
                -0.3,
                Z,
                df_fragment_id="df-1",
                basis_id="basis-b",
            ),
        ),
    )


def _operator_map(tail):
    return {
        component.component_id: operator
        for component, operator in zip(
            tail.components, tail.operators, strict=True
        )
    }


def test_finite_taylor_distribution_is_normalized() -> None:
    tau = 0.2
    distribution = finite_rte_distribution(tau, 4)
    expected_weights = (
        math.sqrt(1.0 + tau**2),
        (tau**2 / math.factorial(2))
        * math.sqrt(1.0 + tau**2 / 9.0),
        (tau**4 / math.factorial(4))
        * math.sqrt(1.0 + tau**2 / 25.0),
    )

    np.testing.assert_allclose(distribution.unnormalized_order_weights, expected_weights)
    np.testing.assert_allclose(
        distribution.exact_finite_distribution,
        sum(expected_weights),
    )
    np.testing.assert_allclose(sum(distribution.order_probabilities), 1.0)
    assert distribution.orders == (0, 2, 4)
    assert distribution.paper_upper_bound >= distribution.exact_finite_distribution


def test_all_finite_event_probabilities_sum_to_one() -> None:
    tail = _toy_tail()
    distribution = finite_rte_distribution(0.12, 4)
    events = enumerate_rte_events(tail.components, distribution)

    np.testing.assert_allclose(sum(event.event_probability for event in events), 1.0)
    assert len(events) == 2 + 2**3 + 2**5
    assert all(event.event_coefficient > 0.0 for event in events)
    assert {event.phase for event in events} == {1.0 + 0.0j, -1.0 + 0.0j}


def test_automatic_cutoff_meets_truncation_tolerance() -> None:
    tail = _toy_tail()
    config, distribution = make_rte_config(
        tail,
        evolution_time=0.7,
        rte_steps=3,
        truncation_tolerance=1e-10,
        seed=19,
    )

    assert distribution.truncation_residual_bound <= 1e-10
    assert config.truncation_residual_bound == distribution.truncation_residual_bound
    assert config.distribution_normalization == distribution.exact_finite_distribution
    assert config.finite_taylor_order % 2 == 0
    assert config.dimensionless_step_time == tail.lambda_r * config.step_time
    np.testing.assert_allclose(
        finite_rte_attenuation(config, tail_evolutions=2),
        distribution.exact_finite_distribution ** (-2 * config.rte_steps),
    )


def test_enumerated_event_mean_matches_dense_paired_taylor_reference() -> None:
    tail = normalize_involutory_tail(
        "toy-xyz",
        (
            InvolutoryTailTerm("x", 0.5, X, df_fragment_id="df-0"),
            InvolutoryTailTerm("y", -0.3, Y, df_fragment_id="df-1"),
            InvolutoryTailTerm("z", 0.2, Z, df_fragment_id="df-2"),
        ),
    )
    distribution = finite_rte_distribution(-0.17, 2)
    events = enumerate_rte_events(tail.components, distribution)
    mean = finite_event_mean_operator(events, _operator_map(tail))
    expected = finite_taylor_operator(
        tail.normalized_hamiltonian,
        distribution.dimensionless_step_time,
        distribution.finite_taylor_order,
    ) / distribution.exact_finite_distribution

    np.testing.assert_allclose(mean, expected, rtol=1e-13, atol=1e-13)


def test_more_integer_rte_steps_improve_fixed_order_dense_reference() -> None:
    tail = _toy_tail()
    exact = expm(-1j * 0.8 * tail.dense_hamiltonian)
    errors = []
    for rte_steps in (1, 2, 4, 8):
        config, _distribution = make_rte_config(
            tail,
            evolution_time=0.8,
            rte_steps=rte_steps,
            finite_taylor_order=0,
            truncation_tolerance=1.0,
        )
        finite = finite_rte_multi_step_operator(tail.normalized_hamiltonian, config)
        errors.append(float(np.linalg.norm(finite - exact, ord=2)))

    assert all(later < earlier for earlier, later in zip(errors, errors[1:]))


def test_identity_component_retains_global_phase() -> None:
    tail = normalize_involutory_tail(
        "identity",
        (InvolutoryTailTerm("identity", -0.4, I2, basis_id="identity"),),
    )
    config, distribution = make_rte_config(
        tail,
        evolution_time=0.6,
        rte_steps=2,
        truncation_tolerance=1e-12,
    )
    events = enumerate_rte_events(tail.components, distribution)
    finite = finite_rte_multi_step_operator(tail.normalized_hamiltonian, config)
    expected = np.exp(0.4j * 0.6) * I2

    assert tail.components[0].is_identity
    assert np.linalg.norm(finite - expected, ord=2) < 1e-12
    first = events[0]
    unitary = event_unitary(first, _operator_map(tail))
    np.testing.assert_allclose(unitary, unitary[0, 0] * I2, atol=1e-14)
    order_two = next(event for event in events if event.taylor_order == 2)
    assert order_two.basis_id == "identity"
    assert order_two.basis_reuse_intervals[0].start == 0
    assert order_two.basis_reuse_intervals[0].stop == 3


def test_fixed_seed_reproduces_event_sequence() -> None:
    tail = _toy_tail()
    distribution = finite_rte_distribution(0.3, 4)
    first = sample_rte_events(
        tail.components, distribution, sample_count=50, seed=912
    )
    second = sample_rte_events(
        tail.components, distribution, sample_count=50, seed=912
    )
    different = sample_rte_events(
        tail.components, distribution, sample_count=50, seed=913
    )

    assert first == second
    assert first != different


def _dense_df(hamiltonian: DFHamiltonian) -> np.ndarray:
    sector = PhysicalSector(
        n_qubits=hamiltonian.n_qubits,
        basis_indices=np.arange(1 << hamiltonian.n_qubits, dtype=np.int64),
    )
    operator, _counter = df_linear_operator(hamiltonian, sector, backend="python")
    identity = np.eye(sector.dimension, dtype=np.complex128)
    return np.column_stack([operator @ identity[:, index] for index in range(sector.dimension)])


def test_df_partition_tail_reconstructs_original_without_constant_or_one_body() -> None:
    hamiltonian = DFHamiltonian(
        constant=0.17,
        one_body=np.asarray([[0.2, 0.05], [0.05, -0.1]], dtype=np.complex128),
        lambdas=np.asarray([0.6, -0.25], dtype=float),
        g_matrices=(
            np.asarray([[0.8, 0.1], [0.1, -0.2]], dtype=np.complex128),
            np.asarray([[0.3, -0.05], [-0.05, 0.5]], dtype=np.complex128),
        ),
        metadata={},
    )
    partition = split_df_hamiltonian_by_ld(hamiltonian, 1)
    deterministic = select_df_h_d(hamiltonian, partition)
    randomized = DFHamiltonian(
        constant=0.0,
        one_body=np.zeros_like(hamiltonian.one_body),
        lambdas=np.asarray(
            [hamiltonian.lambdas[index] for index in partition.randomized_block_indices]
        ),
        g_matrices=tuple(
            hamiltonian.g_matrices[index]
            for index in partition.randomized_block_indices
        ),
        metadata={},
    )

    np.testing.assert_allclose(
        _dense_df(deterministic) + _dense_df(randomized),
        _dense_df(hamiltonian),
        rtol=1e-13,
        atol=1e-13,
    )
    assert deterministic.constant == hamiltonian.constant
    np.testing.assert_array_equal(deterministic.one_body, hamiltonian.one_body)
    assert randomized.constant == 0.0
    np.testing.assert_array_equal(randomized.one_body, 0.0)
