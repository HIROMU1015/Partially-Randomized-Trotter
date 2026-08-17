from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import qiskit

from trotterlib.df_hamiltonian import DFHamiltonian
from trotterlib.df_partial_randomized_pf import split_df_hamiltonian_by_ld
from trotterlib.df_partial_s2 import prepare_df_partial_s2
from trotterlib.rpe_resource_accounting import (
    RPEErrorAllocation,
    RPEHadamardSamplingPolicy,
    RPEPFErrorModel,
    RPERoundCompiledCost,
    RPERoundSpecification,
    build_rpe_resource_summary,
    evaluate_rpe_round_candidate,
)
from trotterlib.rte import (
    CircuitCost,
    CompilerSettings,
    compose_truncation_residual_bounds,
    finite_rte_attenuation,
    finite_rte_distribution,
)



FRESH_IID_POLICY = RPEHadamardSamplingPolicy(
    rte_trajectory_mode="fresh_iid_per_hadamard_shot",
    independent_bounded_outcomes_within_each_round_axis=True,
)


def _compiler() -> CompilerSettings:
    return CompilerSettings(
        basis_gates=("rz", "sx", "x", "cx"),
        backend_name=None,
        coupling_map=None,
        optimization_level=1,
        layout_method=None,
        routing_method=None,
        transpiler_seed=17,
        qiskit_version=qiskit.__version__,
    )


def _cost(**overrides: float) -> CircuitCost:
    values = {
        "rz_count": 10.0,
        "rz_depth": 7.0,
        "cx_count": 5.0,
        "cx_depth": 4.0,
        "total_depth": 13.0,
        "circuit_size": 21.0,
    }
    values.update(overrides)
    return CircuitCost(
        **values,
        compiler=_compiler(),
        fidelity_level=5,
        estimate_kind="exact_compiled_repeated_partial_s2_expectation",
    )


class _RecordingProvider:
    def __init__(
        self,
        cosine: CircuitCost | None = None,
        sine: CircuitCost | None = None,
    ):
        self.cosine = cosine or _cost()
        self.sine = sine or self.cosine
        self.requests = []

    def evaluate(self, request):
        self.requests.append(request)
        return RPERoundCompiledCost(
            cosine_expected_cost=self.cosine,
            sine_expected_cost=self.sine,
            cosine_standard_error=None,
            sine_standard_error=None,
            evaluation_method="test_provider",
            classical_sample_count=37,
            cost_model_fingerprint="recording-provider-v1",
        )


def _preparation(*, deterministic: bool = False):
    hamiltonian = DFHamiltonian(
        constant=0.11,
        one_body=np.asarray([[0.2]], dtype=np.complex128),
        # For one orbital, identity extraction leaves a Z coefficient of
        # -lambda/2.  This gives exact_rte_lambda_r=1.2 while the DF ranking
        # proxy remains 2.4, so the test detects accidental proxy reuse.
        lambdas=np.asarray([2.4]),
        g_matrices=(np.asarray([[1.0]], dtype=np.complex128),),
        metadata={"name": "rpe-resource-accounting"},
    )
    return prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, 1 if deterministic else 0),
        identity_policy="extract_identity_phase",
    )


def _allocation(
    *,
    beta_pf_budget: float = 8e-5,
    beta_rte_budget: float = 0.020,
    beta_stat_budget: float = 0.37992,
    alpha_cosine: float = 0.0125,
    alpha_sine: float = 0.0125,
) -> RPEErrorAllocation:
    return RPEErrorAllocation(
        beta_pf_budget=beta_pf_budget,
        beta_rte_budget=beta_rte_budget,
        beta_stat_budget=beta_stat_budget,
        alpha_cosine=alpha_cosine,
        alpha_sine=alpha_sine,
    )


@pytest.mark.parametrize(
    ("rte_steps", "cutoff", "epsilon_z", "attenuation", "rho", "shots"),
    (
        (2, 0, 0.014787161342022707, 0.9857286727249772, 0.971152543801935, 157),
        (1, 2, 3.540678760820471e-5, 0.9444897513021748, 0.9444563099541524, 166),
        (3, 0, 0.009772793191628338, 0.9904535314471177, 0.980774033918367, 154),
    ),
)
def test_research_example_and_existing_rte_functions(
    rte_steps, cutoff, epsilon_z, attenuation, rho, shots
) -> None:
    preparation = _preparation()
    specification = RPERoundSpecification(round_index=2, delta_time=0.10)
    provider = _RecordingProvider()
    candidate = evaluate_rpe_round_candidate(
        preparation,
        specification,
        _allocation(),
        RPEPFErrorModel(
            coefficient=0.02,
            source="state_specific_phase_bias_surrogate",
            is_rigorous_bound=False,
        ),
        beta_rpe=0.40,
        rte_steps_per_occurrence=rte_steps,
        finite_taylor_order=cutoff,
        cost_metric="rz_count",
        cost_provider=provider,
        hadamard_sampling_policy=FRESH_IID_POLICY,
    )

    assert specification.q_m == 4
    assert specification.t_m == pytest.approx(0.4)
    assert candidate.tau_m == pytest.approx(1.2 * 0.1 / rte_steps)
    assert preparation.exact_rte_lambda_r == pytest.approx(1.2)
    assert preparation.ranking_proxy_lambda_r == pytest.approx(2.4)
    assert provider.requests[0].rte_config.lambda_r == pytest.approx(
        preparation.exact_rte_lambda_r
    )
    assert provider.requests[0].rte_config.lambda_r != pytest.approx(
        preparation.ranking_proxy_lambda_r
    )
    distribution = finite_rte_distribution(candidate.tau_m, cutoff)
    expected_error = compose_truncation_residual_bounds(
        ((distribution.step_truncation_residual_bound, rte_steps, 4),)
    )
    assert candidate.epsilon_step == distribution.step_truncation_residual_bound
    assert candidate.epsilon_z == pytest.approx(expected_error)
    assert candidate.epsilon_z == pytest.approx(epsilon_z)
    assert candidate.normalization == pytest.approx(
        distribution.exact_finite_distribution
    )
    assert candidate.attenuation == pytest.approx(
        finite_rte_attenuation(provider.requests[0].rte_config, tail_evolutions=4)
    )
    assert candidate.attenuation == pytest.approx(attenuation)
    assert candidate.rho_observed_lower_bound == pytest.approx(rho)
    assert candidate.cosine_shots == shots
    assert candidate.sine_shots == shots
    assert candidate.feasible
    assert candidate.pf_error_model.provenance_status == "empirical_surrogate"
    assert candidate.guarantee_status == "empirical_screening"


def test_infeasible_candidates_skip_cost_provider() -> None:
    preparation = _preparation()
    specification = RPERoundSpecification(round_index=2, delta_time=0.1)
    provider = _RecordingProvider()
    too_large = evaluate_rpe_round_candidate(
        preparation,
        RPERoundSpecification(round_index=2, delta_time=1.0),
        _allocation(beta_rte_budget=1.0, beta_stat_budget=0.01),
        RPEPFErrorModel(0.02, "external_certified_bound", True),
        beta_rpe=1.1,
        rte_steps_per_occurrence=1,
        finite_taylor_order=0,
        cost_metric="rz_count",
        cost_provider=provider,
        hadamard_sampling_policy=FRESH_IID_POLICY,
    )
    assert not too_large.feasible
    assert "finite_rte_error_not_below_one" in too_large.infeasibility_reasons
    assert provider.requests == []

    insufficient = evaluate_rpe_round_candidate(
        preparation,
        specification,
        _allocation(beta_pf_budget=1e-6),
        RPEPFErrorModel(0.02, "external_certified_bound", True),
        beta_rpe=0.4,
        rte_steps_per_occurrence=2,
        finite_taylor_order=0,
        cost_metric="rz_count",
        cost_provider=provider,
        hadamard_sampling_policy=FRESH_IID_POLICY,
    )
    assert not insufficient.feasible
    assert "product_formula_budget_exceeded" in insufficient.infeasibility_reasons
    assert provider.requests == []


def test_axes_metrics_summary_union_bound_and_mc_count_is_not_a_multiplier() -> None:
    preparation = _preparation()
    pf = RPEPFErrorModel(0.02, "external_certified_bound", True)
    first_provider = _RecordingProvider(
        _cost(rz_count=2.0, cx_count=11.0),
        _cost(rz_count=3.0, cx_count=13.0),
    )
    second_provider = _RecordingProvider(
        _cost(rz_count=5.0, cx_count=17.0),
        _cost(rz_count=7.0, cx_count=19.0),
    )
    first = evaluate_rpe_round_candidate(
        preparation,
        RPERoundSpecification(0, 0.1),
        _allocation(alpha_cosine=0.01, alpha_sine=0.02),
        pf,
        beta_rpe=0.4,
        rte_steps_per_occurrence=2,
        finite_taylor_order=2,
        cost_metric="rz_count",
        cost_provider=first_provider,
        hadamard_sampling_policy=FRESH_IID_POLICY,
    )
    second = evaluate_rpe_round_candidate(
        preparation,
        RPERoundSpecification(1, 0.1),
        _allocation(alpha_cosine=0.03, alpha_sine=0.04),
        pf,
        beta_rpe=0.4,
        rte_steps_per_occurrence=2,
        finite_taylor_order=2,
        cost_metric="rz_count",
        cost_provider=second_provider,
        hadamard_sampling_policy=FRESH_IID_POLICY,
    )
    assert first.cosine_shots != first.sine_shots
    assert first.round_total_cost == pytest.approx(
        first.cosine_shots * 2.0 + first.sine_shots * 3.0
    )
    assert first.round_total_cost != pytest.approx(
        37 * (first.cosine_shots * 2.0 + first.sine_shots * 3.0)
    )

    summary = build_rpe_resource_summary(
        (first, second), total_alpha_budget=0.11, cost_metric="rz_count"
    )
    assert summary.total_failure_probability_bound == pytest.approx(0.10)
    assert summary.union_bound_satisfied
    assert summary.total_cost == pytest.approx(
        first.round_total_cost + second.round_total_cost
    )
    assert summary.guarantee_status == "certified"
    assert summary.circuit_cost_scope == "compiled_time_evolution_subcircuit"

    failed = build_rpe_resource_summary(
        (first, second), total_alpha_budget=0.09, cost_metric="rz_count"
    )
    assert not failed.union_bound_satisfied

    metric_provider = _RecordingProvider(
        _cost(rz_count=2.0, cx_count=11.0), _cost(rz_count=3.0, cx_count=13.0)
    )
    cx_candidate = evaluate_rpe_round_candidate(
        preparation,
        RPERoundSpecification(0, 0.1),
        _allocation(),
        pf,
        beta_rpe=0.4,
        rte_steps_per_occurrence=2,
        finite_taylor_order=2,
        cost_metric="cx_count",
        cost_provider=metric_provider,
        hadamard_sampling_policy=FRESH_IID_POLICY,
    )
    assert cx_candidate.round_total_cost == pytest.approx(
        cx_candidate.cosine_shots * 11.0 + cx_candidate.sine_shots * 13.0
    )


def test_empty_randomized_tail_uses_deterministic_conventions() -> None:
    preparation = _preparation(deterministic=True)
    provider = _RecordingProvider()
    candidate = evaluate_rpe_round_candidate(
        preparation,
        RPERoundSpecification(2, 0.1),
        _allocation(beta_rte_budget=0.0, beta_stat_budget=0.37992),
        RPEPFErrorModel(0.02, "state_specific_phase_bias_surrogate", False),
        beta_rpe=0.4,
        rte_steps_per_occurrence=0,
        finite_taylor_order=0,
        cost_metric="total_depth",
        cost_provider=provider,
        hadamard_sampling_policy=FRESH_IID_POLICY,
    )
    assert candidate.feasible
    assert candidate.tau_m == 0.0
    assert candidate.epsilon_step == 0.0
    assert candidate.epsilon_z == 0.0
    assert candidate.normalization == 1.0
    assert candidate.attenuation == 1.0
    assert candidate.rho_observed_lower_bound == 1.0
    assert provider.requests[0].rte_config is None
    assert provider.requests[0].rte_distribution is None
    assert candidate.guarantee_status == "empirical_screening"


def test_invalid_round_and_metric_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        RPERoundSpecification(-1, 0.1)
    with pytest.raises(ValueError):
        RPERoundSpecification(0, 0.0)
    with pytest.raises(ValueError, match="round_index"):
        RPERoundSpecification(63, 0.1)
    with pytest.raises(ValueError, match="Derived t_m"):
        RPERoundSpecification(62, 1e308)
    with pytest.raises(ValueError):
        evaluate_rpe_round_candidate(
            _preparation(),
            RPERoundSpecification(0, 0.1),
            _allocation(),
            RPEPFErrorModel(0.02, "external_certified_bound", True),
            beta_rpe=0.4,
            rte_steps_per_occurrence=2,
            finite_taylor_order=2,
            cost_metric="not_a_metric",
            cost_provider=_RecordingProvider(),
            hadamard_sampling_policy=FRESH_IID_POLICY,
        )
