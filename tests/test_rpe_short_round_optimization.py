from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import qiskit

import trotterlib.rpe_short_round_optimization as optimizer_module
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
from trotterlib.rpe_short_round_optimization import (
    RPERoundOptimizationInput,
    optimize_rpe_short_round,
    optimize_rpe_short_rounds,
)
from trotterlib.rte import CircuitCost, CompilerSettings


FRESH_IID_POLICY = RPEHadamardSamplingPolicy(
    rte_trajectory_mode="fresh_iid_per_hadamard_shot",
    independent_bounded_outcomes_within_each_round_axis=True,
)


def _preparation(*, deterministic_only: bool = False):
    hamiltonian = DFHamiltonian(
        constant=0.11,
        one_body=np.asarray([[0.2]], dtype=np.complex128),
        lambdas=np.asarray([2.4]),
        g_matrices=(np.asarray([[1.0]], dtype=np.complex128),),
        metadata={"name": f"short-rpe-grid-{deterministic_only}"},
    )
    return prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(
            hamiltonian, hamiltonian.n_blocks if deterministic_only else 0
        ),
        identity_policy="extract_identity_phase",
    )


def _allocation(
    *, alpha_cosine: float = 0.01, alpha_sine: float = 0.01
) -> RPEErrorAllocation:
    return RPEErrorAllocation(
        beta_pf_budget=0.0,
        beta_rte_budget=0.05,
        beta_stat_budget=0.20,
        alpha_cosine=alpha_cosine,
        alpha_sine=alpha_sine,
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


class _GridCostProvider:
    def __init__(
        self,
        values: dict[tuple[int, int], float] | None = None,
        *,
        fingerprint: str | None = "short-grid-cost-model-v1",
        fail_pair: tuple[int, int] | None = None,
        fingerprint_by_pair: bool = False,
    ) -> None:
        self.values = values or {}
        self.fingerprint = fingerprint
        self.fail_pair = fail_pair
        self.fingerprint_by_pair = fingerprint_by_pair
        self.requests = []
        self.compiler = _compiler()

    def evaluate(self, request) -> RPERoundCompiledCost:
        pair = (
            request.rte_steps_per_occurrence,
            request.finite_taylor_order,
        )
        self.requests.append(request)
        if pair == self.fail_pair:
            raise ValueError("synthetic workload guard rejection")
        value = float(self.values.get(pair, 1.0))
        cost = CircuitCost(
            rz_count=value,
            rz_depth=value,
            cx_count=value,
            cx_depth=value,
            total_depth=value,
            circuit_size=value,
            compiler=self.compiler,
            fidelity_level=5,
            estimate_kind="exact_compiled_repeated_partial_s2_expectation",
        )
        fingerprint = self.fingerprint
        if fingerprint is not None and self.fingerprint_by_pair:
            fingerprint = f"{fingerprint}-{pair[0]}-{pair[1]}"
        return RPERoundCompiledCost(
            cosine_expected_cost=cost,
            sine_expected_cost=cost,
            cosine_standard_error=None,
            sine_standard_error=None,
            evaluation_method="short_grid_test_provider",
            classical_sample_count=None,
            cost_model_fingerprint=fingerprint,
            metadata=(("provider_version", "short_grid_test_provider_v1"),),
        )


def _optimize(**overrides):
    arguments = dict(
        preparation=_preparation(),
        specification=RPERoundSpecification(0, 0.02),
        allocation=_allocation(),
        pf_error_model=RPEPFErrorModel(0.0, "test_rigorous_bound", True),
        beta_rpe=0.30,
        r_candidates=(2, 1),
        k_candidates=(2, 0),
        allowed_guarantee_statuses=("certified",),
        cost_metric="rz_count",
        cost_provider=_GridCostProvider(
            {(1, 0): 4.0, (1, 2): 1.0, (2, 0): 3.0, (2, 2): 2.0}
        ),
        hadamard_sampling_policy=FRESH_IID_POLICY,
    )
    arguments.update(overrides)
    return optimize_rpe_short_round(**arguments)


def test_complete_grid_matches_independent_brute_force_minimum() -> None:
    provider = _GridCostProvider(
        {(1, 0): 4.0, (1, 2): 1.0, (2, 0): 3.0, (2, 2): 2.0}
    )
    result = _optimize(cost_provider=provider)

    brute_force = []
    for r_m, k_m in ((1, 0), (1, 2), (2, 0), (2, 2)):
        candidate = evaluate_rpe_round_candidate(
            _preparation(),
            RPERoundSpecification(0, 0.02),
            _allocation(),
            RPEPFErrorModel(0.0, "test_rigorous_bound", True),
            beta_rpe=0.30,
            rte_steps_per_occurrence=r_m,
            finite_taylor_order=k_m,
            cost_metric="rz_count",
            cost_provider=_GridCostProvider(provider.values),
            hadamard_sampling_policy=FRESH_IID_POLICY,
        )
        if candidate.feasible and candidate.guarantee_status == "certified":
            brute_force.append(candidate)
    expected = min(
        brute_force,
        key=lambda candidate: (
            candidate.round_total_cost,
            candidate.r_m,
            candidate.k_m,
        ),
    )

    assert result.search_complete
    assert result.minimum_over_declared_grid
    assert result.requested_r_candidates == (2, 1)
    assert result.requested_k_candidates == (2, 0)
    assert result.effective_search_pairs == ((1, 0), (1, 2), (2, 0), (2, 2))
    assert result.selected_pair == (expected.r_m, expected.k_m)
    assert result.selected_candidate.round_total_cost == expected.round_total_cost
    assert len(result.evaluations) == 4
    assert result.cost_model_fingerprint == "short-grid-cost-model-v1"
    assert len(result.search_configuration_fingerprint) == 64
    assert "no_pruning_no_early_stop" in result.enumeration_rule
    assert result.statistically_certified_ranking is False


def test_lower_cost_infeasible_candidate_is_never_selected(monkeypatch) -> None:
    provider = _GridCostProvider({(1, 0): 1.0, (2, 0): 2.0})
    base = {}
    for r_m in (1, 2):
        base[r_m] = evaluate_rpe_round_candidate(
            _preparation(),
            RPERoundSpecification(0, 0.02),
            _allocation(),
            RPEPFErrorModel(0.0, "test_rigorous_bound", True),
            beta_rpe=0.30,
            rte_steps_per_occurrence=r_m,
            finite_taylor_order=0,
            cost_metric="rz_count",
            cost_provider=provider,
            hadamard_sampling_policy=FRESH_IID_POLICY,
        )
    cheap_but_infeasible = replace(
        base[1],
        feasible=False,
        infeasibility_reasons=("synthetic_infeasible_candidate",),
        round_total_cost=0.0,
    )

    def _fake_evaluate(*args, **kwargs):
        r_m = kwargs["rte_steps_per_occurrence"]
        return cheap_but_infeasible if r_m == 1 else base[2]

    monkeypatch.setattr(
        optimizer_module, "evaluate_rpe_round_candidate", _fake_evaluate
    )
    result = _optimize(
        r_candidates=(1, 2),
        k_candidates=(0,),
        cost_provider=provider,
    )

    assert result.selected_pair == (2, 0)
    first = result.evaluations[0]
    assert not first.selection_eligible
    assert "candidate_numerically_infeasible" in first.selection_exclusion_reasons
    assert first.candidate.infeasibility_reasons == (
        "synthetic_infeasible_candidate",
    )


def test_allowed_guarantee_statuses_change_empirical_selection() -> None:
    empirical_model = RPEPFErrorModel(
        0.0, "state_specific_phase_bias_surrogate", False
    )
    certified_only = _optimize(pf_error_model=empirical_model)
    empirical_allowed = _optimize(
        pf_error_model=empirical_model,
        allowed_guarantee_statuses=("certified", "empirical_screening"),
    )

    assert certified_only.search_complete
    assert certified_only.selected_candidate is None
    assert all(
        "guarantee_status_not_allowed" in evaluation.selection_exclusion_reasons
        for evaluation in certified_only.evaluations
    )
    assert empirical_allowed.selected_candidate is not None
    assert empirical_allowed.selected_candidate.guarantee_status == (
        "empirical_screening"
    )


def test_deterministic_tail_evaluates_only_canonical_pair() -> None:
    provider = _GridCostProvider()
    result = _optimize(
        preparation=_preparation(deterministic_only=True),
        allocation=RPEErrorAllocation(0.0, 0.0, 0.2, 0.01, 0.01),
        cost_provider=provider,
    )

    assert result.requested_r_candidates == (2, 1)
    assert result.requested_k_candidates == (2, 0)
    assert result.effective_search_pairs == ((0, 0),)
    assert result.selected_pair == (0, 0)
    assert len(provider.requests) == 1

    canonical_request = _optimize(
        preparation=_preparation(deterministic_only=True),
        allocation=RPEErrorAllocation(0.0, 0.0, 0.2, 0.01, 0.01),
        r_candidates=(0,),
        k_candidates=(0,),
        cost_provider=_GridCostProvider(),
    )
    assert canonical_request.requested_r_candidates == (0,)
    assert canonical_request.effective_search_pairs == ((0, 0),)


def test_exact_cost_tie_uses_lexicographic_r_k() -> None:
    result = _optimize(
        cost_provider=_GridCostProvider(
            {(1, 0): 0.0, (1, 2): 0.0, (2, 0): 0.0, (2, 2): 0.0}
        )
    )

    assert all(
        evaluation.candidate.round_total_cost == 0.0
        for evaluation in result.evaluations
        if evaluation.candidate is not None
    )
    assert result.selected_pair == (1, 0)
    assert "lexicographic_r_k" in result.tie_break_rule


def test_q_greater_than_four_is_rejected_before_provider_call() -> None:
    provider = _GridCostProvider()
    with pytest.raises(ValueError, match="q_m<=4"):
        _optimize(
            specification=RPERoundSpecification(3, 0.02),
            cost_provider=provider,
        )
    assert provider.requests == []


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"r_candidates": (0,)}, "positive"),
        ({"r_candidates": (1, 1)}, "duplicates"),
        ({"k_candidates": (1,)}, "even"),
        ({"k_candidates": (0, 0)}, "duplicates"),
        ({"allowed_guarantee_statuses": ()}, "must not be empty"),
        ({"maximum_candidate_count": 3}, "exceeding"),
    ),
)
def test_search_inputs_are_validated_before_enumeration(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        _optimize(**overrides)


def test_provider_workload_failure_cannot_masquerade_as_grid_minimum() -> None:
    result = _optimize(
        cost_provider=_GridCostProvider(fail_pair=(2, 2)),
    )

    assert not result.search_complete
    assert not result.minimum_over_declared_grid
    assert result.selected_candidate is None
    assert "declared_grid_evaluation_incomplete" in result.search_failure_reasons
    failure = result.evaluations[-1].evaluation_failure
    assert failure is not None
    assert failure.exception_type == "ValueError"
    assert "workload guard" in failure.message


@pytest.mark.parametrize(
    ("provider", "expected_reason"),
    (
        (
            _GridCostProvider(fingerprint=None),
            "unverified_cost_model_fingerprint",
        ),
        (
            _GridCostProvider(fingerprint_by_pair=True),
            "mixed_cost_model_fingerprints",
        ),
    ),
)
def test_unverified_or_mixed_cost_models_do_not_produce_an_optimum(
    provider, expected_reason
) -> None:
    result = _optimize(cost_provider=provider)

    assert not result.search_complete
    assert result.selected_candidate is None
    assert expected_reason in result.search_failure_reasons


def test_selected_rounds_connect_to_existing_resource_summary() -> None:
    provider = _GridCostProvider({(1, 0): 2.0, (1, 2): 1.0})
    round_inputs = tuple(
        RPERoundOptimizationInput(
            RPERoundSpecification(m, 0.02),
            _allocation(alpha_cosine=0.01, alpha_sine=0.01),
        )
        for m in range(3)
    )
    result = optimize_rpe_short_rounds(
        _preparation(),
        round_inputs,
        RPEPFErrorModel(0.0, "test_rigorous_bound", True),
        beta_rpe=0.30,
        r_candidates=(1,),
        k_candidates=(0, 2),
        allowed_guarantee_statuses=("certified",),
        total_alpha_budget=0.07,
        cost_metric="rz_count",
        cost_provider=provider,
        hadamard_sampling_policy=FRESH_IID_POLICY,
    )

    assert result.search_complete
    assert result.selected_candidates is not None
    assert len(result.selected_candidates) == 3
    assert result.resource_summary is not None
    assert result.resource_summary.rounds == result.selected_candidates
    assert result.resource_summary.guarantee_status == "certified"
    assert result.summary_failure_reason is None


def test_multi_round_input_requires_fixed_delta_time() -> None:
    inputs = (
        RPERoundOptimizationInput(RPERoundSpecification(0, 0.02), _allocation()),
        RPERoundOptimizationInput(RPERoundSpecification(1, 0.03), _allocation()),
    )
    provider = _GridCostProvider()

    with pytest.raises(ValueError, match="same delta_time"):
        optimize_rpe_short_rounds(
            _preparation(),
            inputs,
            RPEPFErrorModel(0.0, "test_rigorous_bound", True),
            beta_rpe=0.30,
            r_candidates=(1,),
            k_candidates=(2,),
            allowed_guarantee_statuses=("certified",),
            total_alpha_budget=0.05,
            cost_metric="rz_count",
            cost_provider=provider,
            hadamard_sampling_policy=FRESH_IID_POLICY,
        )
    assert provider.requests == []


@pytest.mark.parametrize(
    ("policy", "expected_status", "expected_reason"),
    (
        (FRESH_IID_POLICY, "certified", None),
        (
            RPEHadamardSamplingPolicy(
                "reuse_across_hadamard_shots",
                True,
            ),
            "not_certified",
            "rte_trajectory_reuse_not_covered_by_hoeffding_guarantee",
        ),
        (
            RPEHadamardSamplingPolicy("unspecified", True),
            "not_certified",
            "rte_trajectory_sampling_policy_unspecified",
        ),
        (
            RPEHadamardSamplingPolicy(
                "fresh_iid_per_hadamard_shot",
                False,
            ),
            "not_certified",
            "independent_bounded_hadamard_outcomes_not_assumed",
        ),
    ),
)
def test_randomized_tail_certification_requires_explicit_fresh_iid_policy(
    policy, expected_status, expected_reason
) -> None:
    candidate = evaluate_rpe_round_candidate(
        _preparation(),
        RPERoundSpecification(0, 0.02),
        _allocation(),
        RPEPFErrorModel(0.0, "test_rigorous_bound", True),
        beta_rpe=0.30,
        rte_steps_per_occurrence=1,
        finite_taylor_order=2,
        cost_metric="rz_count",
        cost_provider=_GridCostProvider(),
        rte_seed=123,
        hadamard_sampling_policy=policy,
    )

    assert candidate.feasible
    assert candidate.guarantee_status == expected_status
    if expected_reason is None:
        assert not candidate.certification_reasons
        assert (
            "fresh_iid_rte_trajectory_per_hadamard_shot_assumed"
            in candidate.assumptions
        )
        assert (
            "independent_bounded_hadamard_outcomes_within_each_round_axis_assumed"
            in candidate.assumptions
        )
    else:
        assert expected_reason in candidate.certification_reasons


def test_deterministic_tail_does_not_require_a_trajectory_sampling_mode() -> None:
    policy = RPEHadamardSamplingPolicy(
        "reuse_across_hadamard_shots",
        True,
    )
    candidate = evaluate_rpe_round_candidate(
        _preparation(deterministic_only=True),
        RPERoundSpecification(0, 0.02),
        RPEErrorAllocation(0.0, 0.0, 0.2, 0.01, 0.01),
        RPEPFErrorModel(0.0, "test_rigorous_bound", True),
        beta_rpe=0.30,
        rte_steps_per_occurrence=0,
        finite_taylor_order=0,
        cost_metric="rz_count",
        cost_provider=_GridCostProvider(),
        hadamard_sampling_policy=policy,
    )

    assert candidate.guarantee_status == "certified"
    assert (
        "rte_trajectory_reuse_not_covered_by_hoeffding_guarantee"
        not in candidate.certification_reasons
    )
    assert (
        "fresh_iid_rte_trajectory_per_hadamard_shot_assumed"
        not in candidate.assumptions
    )
    assert (
        "independent_bounded_hadamard_outcomes_within_each_round_axis_assumed"
        in candidate.assumptions
    )


def test_classical_rte_seed_does_not_substitute_for_quantum_shot_policy() -> None:
    candidate = evaluate_rpe_round_candidate(
        _preparation(),
        RPERoundSpecification(0, 0.02),
        _allocation(),
        RPEPFErrorModel(0.0, "test_rigorous_bound", True),
        beta_rpe=0.30,
        rte_steps_per_occurrence=1,
        finite_taylor_order=2,
        cost_metric="rz_count",
        cost_provider=_GridCostProvider(),
        rte_seed=987654,
    )

    assert candidate.rte_seed == 987654
    assert candidate.guarantee_status == "not_certified"
    assert "rte_trajectory_sampling_policy_unspecified" in (
        candidate.certification_reasons
    )
    assert "independent_bounded_hadamard_outcomes_not_assumed" in (
        candidate.certification_reasons
    )


def test_multi_round_summary_rejects_mixed_quantum_sampling_policies() -> None:
    common = dict(
        preparation=_preparation(),
        allocation=_allocation(),
        pf_error_model=RPEPFErrorModel(0.0, "test_rigorous_bound", True),
        beta_rpe=0.30,
        rte_steps_per_occurrence=1,
        finite_taylor_order=2,
        cost_metric="rz_count",
        cost_provider=_GridCostProvider(),
    )
    first = evaluate_rpe_round_candidate(
        specification=RPERoundSpecification(0, 0.02),
        hadamard_sampling_policy=FRESH_IID_POLICY,
        **common,
    )
    second = evaluate_rpe_round_candidate(
        specification=RPERoundSpecification(1, 0.02),
        hadamard_sampling_policy=RPEHadamardSamplingPolicy(
            "reuse_across_hadamard_shots", True
        ),
        **common,
    )

    with pytest.raises(ValueError, match="Hadamard sampling policy"):
        build_rpe_resource_summary(
            (first, second), total_alpha_budget=0.05, cost_metric="rz_count"
        )
