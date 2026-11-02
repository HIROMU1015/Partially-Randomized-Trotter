from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from trotterlib.df_hamiltonian import DFHamiltonian
from trotterlib.df_partial_randomized_pf import split_df_hamiltonian_by_ld
from trotterlib.df_partial_s2 import prepare_df_partial_s2
from trotterlib.research_direction_prevalidation import (
    affine_prediction_with_standard_error,
    build_horizon_audit,
    finalize_artifact,
    select_analytic_round_schedule,
    validate_artifact,
)


def _hamiltonian() -> DFHamiltonian:
    return DFHamiltonian(
        constant=0.1,
        one_body=np.asarray([[0.2, 0.01], [0.01, -0.1]], dtype=np.complex128),
        lambdas=np.asarray([0.3]),
        g_matrices=(
            np.asarray([[1.0, 0.0], [0.0, 0.5]], dtype=np.complex128),
        ),
        metadata={"name": "research-direction-prevalidation-toy"},
    )


def test_artifact_fingerprint_is_tamper_evident() -> None:
    payload = finalize_artifact(
        stage="WP02",
        body={"overall_pass": True},
        provenance={"test": True},
    )
    validate_artifact(payload)
    tampered = deepcopy(payload)
    tampered["overall_pass"] = False
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_artifact(tampered)


def test_horizon_audit_separates_exact_schedule_coverage_and_pf_failure() -> None:
    rows = build_horizon_audit(
        precision_scenarios=(("main", 1.0e-3), ("strict", 1.0e-5)),
        delta_values=(0.01,),
        beta_rpe=0.4,
        beta_pf_budget=0.02,
        pf_coefficient=0.0134,
        direct_wrapper_q_max=8,
        existing_schedule_precision=1.0e-3,
        existing_schedule_deltas=(0.01,),
    )

    assert rows[0]["existing_round_schedule_exact_condition_match"]
    assert not rows[1]["existing_round_schedule_exact_condition_match"]
    assert rows[0]["minimal_power_of_two_horizon"]
    assert rows[1]["q_max"] > rows[0]["q_max"]
    assert not rows[1]["empirical_pf_screen_pass"]


def test_deterministic_endpoint_schedule_uses_canonical_zero_rte_pair() -> None:
    hamiltonian = _hamiltonian()
    preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, hamiltonian.n_blocks),
        identity_policy="extract_identity_phase",
    )
    schedule = select_analytic_round_schedule(
        preparation,
        delta_time=0.1,
        target_energy_precision=0.5,
        pf_coefficient=0.01,
    )

    assert schedule["all_rounds_feasible"]
    assert schedule["selected_r_k_pairs"] == ["r0_k0"]
    assert all(row["r_m"] == row["K_m"] == 0 for row in schedule["rounds"])


def test_affine_prediction_propagates_calibration_standard_errors() -> None:
    prediction, standard_error = affine_prediction_with_standard_error(
        q_m=4,
        q1_mean=10.0,
        q2_mean=18.0,
        q1_standard_error=1.0,
        q2_standard_error=2.0,
    )

    assert prediction == pytest.approx(34.0)
    assert standard_error == pytest.approx(np.hypot(-2.0, 6.0))
