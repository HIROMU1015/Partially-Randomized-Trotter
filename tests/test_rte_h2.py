from __future__ import annotations

import numpy as np
from openfermion import QubitOperator, get_sparse_operator
from scipy.linalg import expm

from trotterlib.partial_randomized_pf import build_sorted_pauli_hamiltonian
from trotterlib.rte import (
    InvolutoryTailTerm,
    finite_rte_multi_step_operator,
    make_rte_config,
    normalize_involutory_tail,
)


def test_h2_pauli_tail_finite_rte_matches_dense_exponential() -> None:
    """Small H2 reference without controlled circuits or shot sampling."""
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

    tail = normalize_involutory_tail("H2-three-smallest-Pauli-tail", terms)
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
