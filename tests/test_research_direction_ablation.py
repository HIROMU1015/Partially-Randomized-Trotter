from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from trotterlib.df_hamiltonian import DFHamiltonian
from trotterlib.df_partial_randomized_pf import split_df_hamiltonian_by_ld
from trotterlib.df_partial_s2 import prepare_df_partial_s2
from trotterlib.research_direction_ablation import (
    AffineAxisCostModel,
    PairCompiledCostModel,
    evaluate_ablation_scenario,
    exact_binomial_minimum_shots,
    exact_coordinate_failure_probability,
    finalize_wp04_artifact,
    validate_wp04_artifact,
)


def _hamiltonian() -> DFHamiltonian:
    return DFHamiltonian(
        constant=0.1,
        one_body=np.asarray([[0.2, 0.01], [0.01, -0.1]], dtype=np.complex128),
        lambdas=np.asarray([0.3]),
        g_matrices=(
            np.asarray([[1.0, 0.0], [0.0, 0.5]], dtype=np.complex128),
        ),
        metadata={"name": "research-direction-ablation-toy"},
    )


def _axis(q1: float, q2: float) -> AffineAxisCostModel:
    return AffineAxisCostModel(q1, q2, 0.1, 0.2)


def _model(r_m: int, k_m: int, q1: float, q2: float) -> PairCompiledCostModel:
    return PairCompiledCostModel(
        rte_steps=r_m,
        finite_taylor_order=k_m,
        cosine=_axis(q1, q2),
        sine=_axis(q1, q2),
        dataset_fingerprint="0" * 64,
        proxy_fingerprint="1" * 64,
    )


def test_wp04_artifact_fingerprint_is_tamper_evident() -> None:
    payload = finalize_wp04_artifact(
        {
            "scope": {
                "final_total_cost_evaluation_performed": False,
                "decision_grade": False,
            },
            "configuration": {"factorial_cell_count": 1},
            "scenarios": [{"scenario_id": "test"}],
            "checks": {"test": True},
            "overall_pass": True,
        },
        provenance={"test": True},
    )
    validate_wp04_artifact(payload)
    tampered = deepcopy(payload)
    tampered["overall_pass"] = False
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_wp04_artifact(tampered)


def test_compiled_schedule_selection_is_separate_from_component_objective() -> None:
    hamiltonian = _hamiltonian()
    preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, 0),
        identity_policy="extract_identity_phase",
    )
    models = (
        _model(1, 0, 100.0, 200.0),
        _model(2, 2, 10.0, 20.0),
    )
    common = {
        "preparation": preparation,
        "ld": 0,
        "pf_coefficient": 1e-3,
        "cost_models": models,
        "beta_profile_label": "test",
        "beta_profile": (0.08, 0.08, 0.24),
        "alpha_policy": "uniform",
        "maximum_round_index": 1,
        "delta_time": 0.01,
    }
    component = evaluate_ablation_scenario(
        **common,
        schedule_policy="round_component_applications",
    )
    compiled = evaluate_ablation_scenario(
        **common,
        schedule_policy="round_compiled_rz",
    )

    assert component["selected_r_k_pairs"] == ["r1_k0"]
    assert compiled["selected_r_k_pairs"] == ["r2_k2"]
    assert (
        compiled["total_compiled_rz_point_estimate"]
        < component["total_compiled_rz_point_estimate"]
    )


def test_weighted_alpha_converges_and_preserves_total_probability() -> None:
    hamiltonian = _hamiltonian()
    preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, hamiltonian.n_blocks),
        identity_policy="extract_identity_phase",
    )
    scenario = evaluate_ablation_scenario(
        preparation,
        ld=hamiltonian.n_blocks,
        pf_coefficient=1e-3,
        cost_models=(_model(0, 0, 10.0, 20.0),),
        beta_profile_label="deterministic",
        beta_profile=(0.02, 0.0, 0.38),
        schedule_policy="round_compiled_rz",
        alpha_policy="cost_sensitivity_weighted",
        maximum_round_index=2,
        delta_time=0.01,
    )

    assert scenario["alpha_converged"]
    assert scenario["alpha_total_allocated"] == pytest.approx(0.05)
    assert len(scenario["rounds"]) == 3


def test_exact_binomial_minimum_is_no_larger_than_hoeffding_count() -> None:
    mean = 0.25
    epsilon = 0.15
    alpha = 0.05
    hoeffding = int(np.ceil(2.0 / epsilon**2 * np.log(2.0 / alpha)))
    exact = exact_binomial_minimum_shots(
        mean=mean,
        epsilon_coordinate=epsilon,
        alpha=alpha,
        hoeffding_shots=hoeffding,
    )

    assert exact <= hoeffding
    assert exact_coordinate_failure_probability(
        mean=mean,
        shots=hoeffding,
        epsilon_coordinate=epsilon,
    ) <= alpha
