from __future__ import annotations

import math

import numpy as np
import pytest

from trotterlib.df_hamiltonian import DFHamiltonian
from trotterlib.df_partial_randomized_pf import split_df_hamiltonian_by_ld
from trotterlib.df_partial_s2 import prepare_df_partial_s2
from trotterlib.rpe_resource_accounting import (
    RPE_CONDITIONAL_GUARANTEE_SCOPE,
    RPE_STRICT_BRANCH_CERTIFICATION_LIMIT,
    RPEErrorAllocation,
    RPEPFErrorModel,
    RPERoundCompiledCost,
    RPERoundSpecification,
    build_rpe_resource_summary,
    evaluate_rpe_round_candidate,
)
from trotterlib.rte import CircuitCost, CompilerSettings


def _deterministic_preparation():
    hamiltonian = DFHamiltonian(
        constant=0.0,
        one_body=np.asarray([[0.2]], dtype=np.complex128),
        lambdas=np.asarray([1.0]),
        g_matrices=(np.asarray([[1.0]], dtype=np.complex128),),
        metadata={"name": "rpe-pi-over-three-boundary"},
    )
    return prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, 1),
        identity_policy="extract_identity_phase",
    )


def _allocation(*, beta_stat_budget: float = 0.2) -> RPEErrorAllocation:
    return RPEErrorAllocation(
        beta_pf_budget=0.0,
        beta_rte_budget=0.0,
        beta_stat_budget=beta_stat_budget,
        alpha_cosine=0.01,
        alpha_sine=0.01,
    )


class _StaticCostProvider:
    def __init__(self) -> None:
        compiler = CompilerSettings(
            basis_gates=("rz", "sx", "x", "cx"),
            backend_name=None,
            coupling_map=None,
            optimization_level=1,
            layout_method=None,
            routing_method=None,
            transpiler_seed=17,
            qiskit_version="boundary-test",
        )
        self.cost = CircuitCost(
            rz_count=2.0,
            rz_depth=2.0,
            cx_count=3.0,
            cx_depth=3.0,
            total_depth=5.0,
            circuit_size=7.0,
            compiler=compiler,
            fidelity_level=5,
            estimate_kind="exact_compiled_repeated_partial_s2_expectation",
        )

    def evaluate(self, request) -> RPERoundCompiledCost:
        return RPERoundCompiledCost(
            cosine_expected_cost=self.cost,
            sine_expected_cost=self.cost,
            cosine_standard_error=None,
            sine_standard_error=None,
            evaluation_method="rpe_boundary_test_provider",
            classical_sample_count=None,
            cost_model_fingerprint="rpe-boundary-test-provider-v1",
        )


def _candidate(
    beta_rpe: float,
    *,
    round_index: int = 0,
    allocation: RPEErrorAllocation | None = None,
    pf_error_model: RPEPFErrorModel | None = None,
    cost_provider: _StaticCostProvider | None = None,
):
    return evaluate_rpe_round_candidate(
        _deterministic_preparation(),
        RPERoundSpecification(round_index=round_index, delta_time=0.1),
        allocation or _allocation(),
        pf_error_model
        or RPEPFErrorModel(
            coefficient=0.0,
            source="test_rigorous_upper_bound",
            is_rigorous_bound=True,
        ),
        beta_rpe=beta_rpe,
        rte_steps_per_occurrence=0,
        finite_taylor_order=0,
        cost_metric="rz_count",
        cost_provider=cost_provider,
    )


@pytest.mark.parametrize(
    ("beta_rpe", "expected_status"),
    (
        (
            math.nextafter(math.pi / 3.0, 0.0),
            "certified",
        ),
        (math.pi / 3.0, "not_certified"),
        (
            math.nextafter(math.pi / 3.0, math.inf),
            "not_certified",
        ),
    ),
    ids=("strictly-below", "equal", "strictly-above"),
)
def test_rpe_certification_requires_strict_pi_over_three_margin(
    beta_rpe: float,
    expected_status: str,
) -> None:
    candidate = _candidate(beta_rpe)

    # Crossing the PR Lemma B.1 branch margin changes only the guarantee:
    # candidates at and above pi/3 remain usable for numerical accounting.
    assert candidate.feasible
    assert candidate.guarantee_status == expected_status
    assert candidate.unattenuated_signal_radius_lower_bound == 1.0
    assert candidate.guarantee_scope == RPE_CONDITIONAL_GUARANTEE_SCOPE
    assert (
        "exact_effective_partial_s2_eigenstate_input_assumed"
        in candidate.assumptions
    )
    assert (
        "unit_unattenuated_survival_signal_radius_assumed"
        in candidate.assumptions
    )
    assert "alias_free_target_energy_branch_assumed" in candidate.assumptions

    if expected_status == "certified":
        assert candidate.strict_branch_margin_satisfied
        assert candidate.branch_certification_margin > 0.0
        assert not candidate.certification_reasons
    else:
        assert not candidate.strict_branch_margin_satisfied
        assert candidate.branch_certification_margin <= 0.0
        assert (
            "rpe_branch_margin_not_strictly_below_pi_over_three"
            in candidate.certification_reasons
        )


def test_pi_over_three_dominates_empirical_screening_status() -> None:
    candidate = _candidate(
        math.pi / 3.0,
        pf_error_model=RPEPFErrorModel(
            coefficient=0.0,
            source="state_specific_phase_bias_surrogate",
            is_rigorous_bound=False,
        ),
    )

    assert candidate.feasible
    assert candidate.guarantee_status == "not_certified"
    assert candidate.certification_reasons == (
        "rpe_branch_margin_not_strictly_below_pi_over_three",
        "pf_error_model_is_empirical",
    )


def test_certification_rechecks_phase_budget_without_feasibility_tolerance() -> None:
    beta_rpe = 0.4
    beta_stat_budget = math.nextafter(beta_rpe, math.inf)
    candidate = _candidate(
        beta_rpe,
        allocation=_allocation(beta_stat_budget=beta_stat_budget),
    )

    assert candidate.feasible
    assert not candidate.infeasibility_reasons
    assert candidate.strict_branch_margin_satisfied
    assert candidate.guarantee_status == "not_certified"
    assert candidate.certification_reasons == (
        "phase_budget_sum_exceeded_without_tolerance",
    )


def test_certification_uses_accurate_phase_budget_summation() -> None:
    allocation = RPEErrorAllocation(
        beta_pf_budget=0.01,
        beta_rte_budget=0.02,
        beta_stat_budget=0.3,
        alpha_cosine=0.01,
        alpha_sine=0.01,
    )
    sequential_sum = (
        allocation.beta_pf_budget
        + allocation.beta_rte_budget
        + allocation.beta_stat_budget
    )
    assert sequential_sum < math.fsum((0.01, 0.02, 0.3))

    candidate = _candidate(sequential_sum, allocation=allocation)

    assert candidate.feasible
    assert candidate.guarantee_status == "not_certified"
    assert candidate.certification_reasons == (
        "phase_budget_sum_exceeded_without_tolerance",
    )


@pytest.mark.parametrize(
    ("beta_rpe", "expected_status"),
    (
        (
            math.nextafter(
                RPE_STRICT_BRANCH_CERTIFICATION_LIMIT,
                0.0,
            ),
            "certified",
        ),
        (RPE_STRICT_BRANCH_CERTIFICATION_LIMIT, "not_certified"),
    ),
    ids=("certified", "branch-boundary"),
)
def test_summary_propagates_branch_status_and_conditional_energy_bound(
    beta_rpe: float,
    expected_status: str,
) -> None:
    provider = _StaticCostProvider()
    rounds = tuple(
        _candidate(
            beta_rpe,
            round_index=round_index,
            cost_provider=provider,
        )
        for round_index in range(2)
    )

    summary = build_rpe_resource_summary(
        rounds,
        total_alpha_budget=0.05,
        cost_metric="rz_count",
    )

    assert summary.union_bound_satisfied
    assert summary.guarantee_status == expected_status
    assert summary.maximum_round_index == 1
    assert summary.maximum_evolution_time == pytest.approx(0.2)
    assert summary.nominal_energy_resolution == pytest.approx(beta_rpe / 0.2)
    if expected_status == "certified":
        assert summary.conditional_energy_error_bound == pytest.approx(
            beta_rpe / 0.2
        )
        assert summary.conditional_success_probability_lower_bound == pytest.approx(
            0.96
        )
        assert not summary.certification_reasons
    else:
        assert summary.conditional_energy_error_bound is None
        assert summary.conditional_success_probability_lower_bound is None
        assert (
            "rpe_branch_margin_not_strictly_below_pi_over_three"
            in summary.certification_reasons
        )


def test_summary_union_certification_does_not_use_feasibility_tolerance() -> None:
    provider = _StaticCostProvider()
    rounds = tuple(
        _candidate(0.4, round_index=round_index, cost_provider=provider)
        for round_index in range(2)
    )
    represented_failure_bound = math.fsum(
        item.round_failure_probability_bound for item in rounds
    )

    summary = build_rpe_resource_summary(
        rounds,
        total_alpha_budget=math.nextafter(represented_failure_bound, 0.0),
        cost_metric="rz_count",
    )

    assert not summary.union_bound_satisfied
    assert summary.guarantee_status == "not_certified"
    assert summary.conditional_energy_error_bound is None
    assert summary.certification_reasons == (
        "all_round_union_bound_exceeded_without_tolerance",
    )


def test_summary_union_bound_sums_axis_allocations_directly() -> None:
    provider = _StaticCostProvider()
    first = _candidate(0.4, round_index=0, cost_provider=provider)
    second = _candidate(
        0.4,
        round_index=1,
        allocation=RPEErrorAllocation(0.0, 0.0, 0.2, 0.02, 0.2),
        cost_provider=provider,
    )
    rounded_pairwise_sum = math.fsum(
        item.allocation.alpha_cosine + item.allocation.alpha_sine
        for item in (first, second)
    )
    direct_sum = math.fsum((0.01, 0.01, 0.02, 0.2))
    assert rounded_pairwise_sum < direct_sum

    summary = build_rpe_resource_summary(
        (first, second),
        total_alpha_budget=rounded_pairwise_sum,
        cost_metric="rz_count",
    )

    assert summary.total_failure_probability_bound == direct_sum
    assert not summary.union_bound_satisfied
    assert summary.guarantee_status == "not_certified"
