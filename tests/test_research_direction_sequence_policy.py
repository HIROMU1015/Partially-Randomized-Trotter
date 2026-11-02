from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from trotterlib.df_rte_circuit import DFRTEBasisPlan
from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
from trotterlib.df_rte_tail import (
    extract_df_diagonal_tail,
    prepare_df_rte_event_inputs,
)
from trotterlib.df_trotter.model import DFBlock
from trotterlib.df_trotter.ops import U_to_qiskit_ops_jw
from trotterlib.research_direction_sequence_policy import (
    _choose_training_policy,
    finalize_wp06b_artifact,
    make_run_threshold_basis_plan,
    source_basis_run_lengths,
    validate_wp06b_artifact,
)
from trotterlib.research_direction_structure_pilot import (
    maximum_operator_difference,
    support_restricted_unitary_completion,
)
from trotterlib.rte import make_rte_config


def _orthogonal(size: int, seed: int = 84) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q_matrix, r_matrix = np.linalg.qr(rng.normal(size=(size, size)))
    signs = np.where(np.diag(r_matrix) < 0.0, -1.0, 1.0)
    return (q_matrix @ np.diag(signs)).astype(np.complex128)


def _controlled_sequence_case():
    full = _orthogonal(3)
    extraction = extract_df_diagonal_tail(
        "wp06b-test",
        (
            DFBlock(
                U_ops=tuple(U_to_qiskit_ops_jw(full)),
                eta=np.asarray((1.0, 0.7, -0.2)),
                lam=0.8,
            ),
        ),
        identity_policy="extract_identity_phase",
    )
    preparation = prepare_df_rte_event_inputs(extraction)
    config, distribution = make_rte_config(
        preparation.symbolic_tail,
        evolution_time=0.02,
        rte_steps=2,
        truncation_tolerance=1.0,
        finite_taylor_order=2,
    )
    request = preparation.sample_occurrence_request(
        config,
        distribution,
        seed=17,
        controlled=True,
        ancilla_qubit=3,
    )
    definitions = {}
    for application in (
        item
        for event in request.events
        for item in event.application_sequence
        if not item.is_identity
    ):
        key = (
            application.basis_id,
            application.basis_hash,
            application.diagonal_pauli_support,
        )
        if key in definitions:
            continue
        restricted = support_restricted_unitary_completion(
            full,
            application.diagonal_pauli_support,
        )
        definitions[key] = preparation.basis_registry.register(
            tuple(U_to_qiskit_ops_jw(restricted)),
            num_system_qubits=3,
            basis_id=(
                f"{application.basis_id}-support-"
                + "-".join(str(item) for item in application.diagonal_pauli_support)
            ),
        )
    return preparation, request, definitions


def test_run_threshold_plan_switches_whole_source_basis_run() -> None:
    _preparation, request, definitions = _controlled_sequence_case()
    run_lengths = source_basis_run_lengths(request.events)
    assert run_lengths
    assert len(set(run_lengths)) == 1

    guarded = make_run_threshold_basis_plan(
        request,
        definitions,
        maximum_support_run_length=max(run_lengths) - 1,
    )
    support_all = make_run_threshold_basis_plan(
        request,
        definitions,
        maximum_support_run_length=None,
    )
    assert all(
        choice.construction == "registered_full_basis"
        for choice in guarded.choices
    )
    assert all(
        choice.construction == "support_restricted_preserved_columns_v1"
        for choice in support_all.choices
    )


def test_explicit_support_plan_preserves_controlled_sequence_and_phase() -> None:
    preparation, request, definitions = _controlled_sequence_case()
    builder = QiskitDFRTEEventCircuitBuilder(
        basis_registry=preparation.basis_registry
    )
    full = builder.build_sequence(request)
    plan = make_run_threshold_basis_plan(
        request,
        definitions,
        maximum_support_run_length=None,
        training_fingerprint="test-training",
    )
    candidate = builder.build_sequence(request, basis_plan=plan)

    assert maximum_operator_difference(
        full.circuit,
        candidate.circuit,
        allow_global_phase=False,
    ) < 1e-12
    assert candidate.relative_ancilla_phase == full.relative_ancilla_phase
    assert candidate.support_restricted_application_count == len(plan.choices)
    assert candidate.basis_plan_fingerprint == plan.plan_fingerprint


def test_builder_rejects_basis_plan_source_mismatch() -> None:
    preparation, request, definitions = _controlled_sequence_case()
    plan = make_run_threshold_basis_plan(
        request,
        definitions,
        maximum_support_run_length=None,
    )
    bad_choice = replace(plan.choices[0], source_basis_hash="wrong")
    bad_plan = DFRTEBasisPlan(
        policy_id=plan.policy_id,
        selection_objective=plan.selection_objective,
        choices=(bad_choice, *plan.choices[1:]),
    )
    builder = QiskitDFRTEEventCircuitBuilder(
        basis_registry=preparation.basis_registry
    )
    with pytest.raises(ValueError, match="source hash"):
        builder.build_sequence(request, basis_plan=bad_plan)


def test_training_selector_applies_guard_before_lexicographic_objective() -> None:
    def costs(rz: int, cx: int, depth: int):
        return {
            "rz_count": rz,
            "rz_depth": rz,
            "cx_count": cx,
            "cx_depth": cx,
            "total_depth": depth,
            "circuit_size": rz + cx,
        }

    rows = [
        {
            "costs": {
                "full_basis_shared": costs(100, 50, 80),
                "support_run_le_1": costs(90, 49, 82),
                "support_run_le_2": costs(80, 48, 85),
                "support_run_le_3": costs(70, 47, 90),
                "support_all": costs(60, 46, 95),
            }
        },
        {
            "costs": {
                "full_basis_shared": costs(100, 50, 80),
                "support_run_le_1": costs(90, 49, 82),
                "support_run_le_2": costs(80, 48, 85),
                "support_run_le_3": costs(107, 47, 90),
                "support_all": costs(120, 46, 95),
            }
        },
    ]
    selected = _choose_training_policy(
        rows,
        (1, 2, 3, None),
        eta_decision=0.05,
    )
    assert selected["selected_policy"]["policy"] == "support_run_le_2"


def test_wp06b_artifact_is_tamper_evident_and_scope_guarded() -> None:
    body = {
        "scope": {
            "final_total_cost_evaluation_performed": False,
            "decision_grade": False,
            "production_default_changed": False,
        },
        "checks": {"test": True},
        "overall_pass": True,
    }
    payload = finalize_wp06b_artifact(body, provenance={"test": True})
    validate_wp06b_artifact(payload)

    tampered = deepcopy(payload)
    tampered["scope"]["production_default_changed"] = True
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_wp06b_artifact(tampered)
