from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from qiskit.quantum_info import Operator

from trotterlib.df_hamiltonian import DFHamiltonian
from trotterlib.df_partial_randomized_pf import split_df_hamiltonian_by_ld
from trotterlib.df_partial_s2 import prepare_df_partial_s2
from trotterlib.df_partial_s2_repeated import (
    DFPartialS2RepeatedRequest,
    QiskitDFPartialS2RepeatedCircuitBuilder,
    make_df_partial_s2_repeated_request,
)
from trotterlib.research_direction_full_scope import (
    apply_run_threshold_policy,
    finalize_wp05a_artifact,
    validate_wp05a_artifact,
)
from trotterlib.research_direction_sequence_policy import (
    register_support_restricted_bases,
)
from trotterlib.rte import make_rte_config


def _case():
    hamiltonian = DFHamiltonian(
        constant=0.07,
        one_body=np.asarray([[0.3, 0.08], [0.08, -0.2]], dtype=np.complex128),
        lambdas=np.asarray([0.7, -0.4]),
        g_matrices=(
            np.asarray([[0.9, 0.18], [0.18, 0.2]], dtype=np.complex128),
            np.asarray([[0.1, -0.23], [-0.23, 1.1]], dtype=np.complex128),
        ),
        metadata={"name": "wp05a-test"},
    )
    preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, 1),
        identity_policy="extract_identity_phase",
    )
    config, distribution = make_rte_config(
        preparation.rte_preparation.symbolic_tail,
        evolution_time=0.03,
        rte_steps=2,
        truncation_tolerance=1.0,
        finite_taylor_order=2,
    )
    request = make_df_partial_s2_repeated_request(
        preparation,
        step_time=0.03,
        repetition_count=2,
        rte_config=config,
        rte_distribution=distribution,
        seed=19,
        controlled=True,
        ancilla_qubit=2,
        construction_policy="boundary_optimized",
    )
    definitions, _proofs = register_support_restricted_bases(
        hamiltonian,
        preparation,
    )
    return request, definitions


def test_explicit_policy_propagates_through_complete_repeated_partial_s2() -> None:
    request, definitions = _case()
    selected_request = apply_run_threshold_policy(
        request,
        definitions,
        maximum_support_run_length=1_000,
        training_fingerprint="fixed-training",
    )
    builder = QiskitDFPartialS2RepeatedCircuitBuilder()
    full = builder.build(request, construction_policy="boundary_optimized")
    selected = builder.build(
        selected_request,
        construction_policy="boundary_optimized",
    )

    assert np.allclose(
        Operator(full.circuit).data,
        Operator(selected.circuit).data,
        atol=1e-12,
    )
    assert selected.rte_relative_phase == full.rte_relative_phase
    assert any(
        step.rte_support_restricted_application_count > 0
        for step in selected.step_results
    )
    assert all(
        step.rte_basis_plan_fingerprint is not None
        for step in selected.step_results
    )
    assert selected.compiler_independent_fingerprint != (
        full.compiler_independent_fingerprint
    )


def test_default_full_basis_request_remains_deterministic_and_plan_free() -> None:
    request, _definitions = _case()
    builder = QiskitDFPartialS2RepeatedCircuitBuilder()
    first = builder.build(request, construction_policy="boundary_optimized")
    second = builder.build(request, construction_policy="boundary_optimized")

    assert first.compiler_independent_fingerprint == (
        second.compiler_independent_fingerprint
    )
    assert all(step.rte_basis_plan_fingerprint is None for step in first.step_results)
    assert all(
        step.rte_support_restricted_application_count == 0
        for step in first.step_results
    )


def test_deterministic_repetition_rejects_nonempty_basis_plan() -> None:
    request, definitions = _case()
    selected = apply_run_threshold_policy(
        request,
        definitions,
        maximum_support_run_length=1_000,
        training_fingerprint="fixed-training",
    )
    hamiltonian = _deterministic_hamiltonian()
    deterministic_preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, 1),
    )
    with pytest.raises(ValueError, match="basis plans"):
        DFPartialS2RepeatedRequest(
            preparation=deterministic_preparation,
            step_time=0.03,
            repetition_count=2,
            rte_config=None,
            rte_distribution=None,
            rte_basis_plans=selected.rte_basis_plans,
            controlled=True,
            ancilla_qubit=2,
            step_seeds=(None, None),
        )


def _deterministic_hamiltonian() -> DFHamiltonian:
    return DFHamiltonian(
        constant=0.0,
        one_body=np.diag([0.2, -0.1]).astype(np.complex128),
        lambdas=np.asarray([0.3]),
        g_matrices=(np.diag([0.7, 0.4]).astype(np.complex128),),
        metadata={"name": "wp05a-deterministic-test"},
    )


def test_wp05a_artifact_is_tamper_evident_and_scope_guarded() -> None:
    body = {
        "policy_input": {"production_default_changed": False},
        "scope": {
            "final_total_cost_evaluation_performed": False,
            "decision_grade": False,
        },
        "checks": {"test": True},
        "overall_pass": True,
    }
    payload = finalize_wp05a_artifact(body, provenance={"test": True})
    validate_wp05a_artifact(payload)

    tampered = deepcopy(payload)
    tampered["scope"]["decision_grade"] = True
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_wp05a_artifact(tampered)
