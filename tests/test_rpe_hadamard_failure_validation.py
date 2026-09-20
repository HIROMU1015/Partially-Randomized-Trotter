from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from trotterlib.df_hamiltonian import DFHamiltonian, PhysicalSector
from trotterlib.rpe_hadamard_failure_validation import (
    validate_rpe_hadamard_failure_allocation,
    validate_rpe_hadamard_failure_payload,
    write_rpe_hadamard_failure_validation,
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
        metadata={"name": "hadamard-failure-toy"},
    )


def test_failure_validation_passes_and_is_tamper_evident(tmp_path) -> None:
    payload = validate_rpe_hadamard_failure_allocation(
        _hamiltonian(),
        PhysicalSector.number_sector(n_qubits=2, n_electrons=1),
        ld=1,
        delta_time=0.05,
        q_values=(1,),
        rte_steps_per_occurrence=1,
        finite_taylor_order=0,
        marginal_repetitions=20_000,
        marginal_seed=71,
        explicit_trajectory_seed=73,
        provenance={"test": True},
    )

    assert payload["summary"]["overall_pass"]
    assert payload["final_cost_evaluation_performed"] is False
    assert payload["full_rpe_phase_reconstruction_performed"] is False
    assert payload["exact_binomial"]["checks"][
        "combined_coordinate_failure_within_alpha_total"
    ]
    assert payload["explicit_fresh_iid_trajectory_batch"]["checks"][
        "fresh_seed_per_hadamard_shot"
    ]

    target = tmp_path / "failure.json"
    write_rpe_hadamard_failure_validation(payload, target)
    assert target.exists()

    tampered = deepcopy(payload)
    tampered["exact_binomial"]["checks"][
        "combined_coordinate_failure_within_alpha_total"
    ] = False
    with pytest.raises(ValueError, match="Exact probability check summary mismatch"):
        validate_rpe_hadamard_failure_payload(tampered)


def test_failure_validation_rejects_phase_budget_overflow() -> None:
    with pytest.raises(ValueError, match="Phase budgets exceed beta_rpe"):
        validate_rpe_hadamard_failure_allocation(
            _hamiltonian(),
            PhysicalSector.number_sector(n_qubits=2, n_electrons=1),
            ld=1,
            delta_time=0.05,
            q_values=(1,),
            rte_steps_per_occurrence=1,
            finite_taylor_order=0,
            beta_pf_budget=0.2,
            beta_rte_budget=0.2,
            beta_stat_budget=0.2,
            beta_rpe=0.4,
            marginal_repetitions=10,
        )
