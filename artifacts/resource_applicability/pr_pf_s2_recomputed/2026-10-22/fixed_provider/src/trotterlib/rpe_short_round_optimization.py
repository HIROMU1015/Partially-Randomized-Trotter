"""Complete inner grid search for directly constructible short RPE rounds.

The optimizer in this module fixes the DF preparation, ``delta_time``, phase
and failure-probability allocations, and the compiled-cost model.  It only
selects ``(r_m, K_m)`` for rounds with ``q_m=2**m <= 4``.  It intentionally
does not optimize ``L_D`` or ``delta_time`` and does not extrapolate long
rounds.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

from .df_partial_s2 import DFPartialS2Preparation
from .rpe_resource_accounting import (
    RPE_COST_METRICS,
    RPE_NUMERICAL_PHASE_MODEL_LIMIT,
    RPECostMetric,
    RPEErrorAllocation,
    RPEGuaranteeStatus,
    RPEHadamardSamplingPolicy,
    RPEPFErrorModel,
    RPEResourceSummary,
    RPERoundCandidate,
    RPERoundCostProvider,
    RPERoundSpecification,
    build_rpe_resource_summary,
    evaluate_rpe_round_candidate,
)
from .rte import require_integer_count


RPERoundSelectionExclusionReason: TypeAlias = Literal[
    "evaluation_failed",
    "candidate_numerically_infeasible",
    "missing_round_total_cost",
    "guarantee_status_not_allowed",
]

RPE_SHORT_ROUND_OPTIMIZER_VERSION = "rpe_short_round_complete_grid_v1"
RPE_SHORT_ROUND_MAXIMUM_Q = 4
RPE_SHORT_ROUND_ENUMERATION_RULE = (
    "sorted_r_then_k_complete_cartesian_product_no_pruning_no_early_stop"
)
RPE_SHORT_ROUND_TIE_BREAK_RULE = (
    "minimum_represented_round_total_cost_then_lexicographic_r_k"
)
RPE_SHORT_ROUND_SELECTION_STATISTIC = "minimum_compiled_cost_point_estimate"
DEFAULT_MAXIMUM_SHORT_ROUND_CANDIDATES = 10_000

_GUARANTEE_STATUS_ORDER: tuple[RPEGuaranteeStatus, ...] = (
    "certified",
    "empirical_screening",
    "not_certified",
)


@dataclass(frozen=True)
class RPERoundOptimizationInput:
    """One fixed round specification and its fixed error allocation."""

    specification: RPERoundSpecification
    allocation: RPEErrorAllocation

    def __post_init__(self) -> None:
        if not isinstance(self.specification, RPERoundSpecification):
            raise TypeError("specification must be an RPERoundSpecification.")
        if not isinstance(self.allocation, RPEErrorAllocation):
            raise TypeError("allocation must be an RPEErrorAllocation.")


@dataclass(frozen=True)
class RPERoundEvaluationFailure:
    """Operational failure while evaluating one structurally valid pair."""

    r_m: int
    k_m: int
    stage: str
    exception_type: str
    message: str


@dataclass(frozen=True)
class RPERoundSearchEvaluation:
    """One declared grid point and its selection decision."""

    r_m: int
    k_m: int
    candidate: RPERoundCandidate | None
    evaluation_failure: RPERoundEvaluationFailure | None
    selection_eligible: bool
    selection_exclusion_reasons: tuple[RPERoundSelectionExclusionReason, ...]

    def __post_init__(self) -> None:
        if (self.candidate is None) == (self.evaluation_failure is None):
            raise ValueError(
                "Exactly one of candidate and evaluation_failure is required."
            )
        if self.selection_eligible and self.selection_exclusion_reasons:
            raise ValueError(
                "An eligible evaluation cannot have selection-exclusion reasons."
            )
        if self.evaluation_failure is not None and self.selection_eligible:
            raise ValueError("A failed evaluation cannot be selection-eligible.")


@dataclass(frozen=True)
class RPERoundOptimizationResult:
    """Auditable result of one complete short-round Cartesian search."""

    specification: RPERoundSpecification
    allocation: RPEErrorAllocation
    requested_r_candidates: tuple[int, ...]
    requested_k_candidates: tuple[int, ...]
    effective_search_pairs: tuple[tuple[int, int], ...]
    evaluations: tuple[RPERoundSearchEvaluation, ...]
    selected_candidate: RPERoundCandidate | None
    allowed_guarantee_statuses: tuple[RPEGuaranteeStatus, ...]
    cost_metric: RPECostMetric
    cost_model_fingerprint: str | None
    search_configuration_fingerprint: str
    search_complete: bool
    search_failure_reasons: tuple[str, ...]
    enumeration_rule: str = RPE_SHORT_ROUND_ENUMERATION_RULE
    tie_break_rule: str = RPE_SHORT_ROUND_TIE_BREAK_RULE
    selection_statistic: str = RPE_SHORT_ROUND_SELECTION_STATISTIC
    statistically_certified_ranking: bool = False
    optimizer_version: str = RPE_SHORT_ROUND_OPTIMIZER_VERSION

    @property
    def selected_pair(self) -> tuple[int, int] | None:
        if self.selected_candidate is None:
            return None
        return (self.selected_candidate.r_m, self.selected_candidate.k_m)

    @property
    def eligible_candidates(self) -> tuple[RPERoundCandidate, ...]:
        return tuple(
            evaluation.candidate
            for evaluation in self.evaluations
            if evaluation.selection_eligible and evaluation.candidate is not None
        )

    @property
    def minimum_over_declared_grid(self) -> bool:
        return self.search_complete and self.selected_candidate is not None


@dataclass(frozen=True)
class RPEShortRoundOptimizationResult:
    """Selected contiguous short rounds and their existing resource summary."""

    round_results: tuple[RPERoundOptimizationResult, ...]
    selected_candidates: tuple[RPERoundCandidate, ...] | None
    resource_summary: RPEResourceSummary | None
    search_complete: bool
    summary_failure_reason: str | None = None
    optimizer_version: str = RPE_SHORT_ROUND_OPTIMIZER_VERSION


def _normalize_statuses(
    statuses: tuple[RPEGuaranteeStatus, ...] | list[RPEGuaranteeStatus],
) -> tuple[RPEGuaranteeStatus, ...]:
    values = tuple(statuses)
    if not values:
        raise ValueError("allowed_guarantee_statuses must not be empty.")
    if any(not isinstance(value, str) for value in values):
        raise TypeError("Every allowed guarantee status must be a string.")
    if len(set(values)) != len(values):
        raise ValueError("allowed_guarantee_statuses must not contain duplicates.")
    unknown = set(values).difference(_GUARANTEE_STATUS_ORDER)
    if unknown:
        raise ValueError(
            "Unsupported allowed guarantee status values: "
            f"{tuple(sorted(unknown))}."
        )
    return tuple(value for value in _GUARANTEE_STATUS_ORDER if value in values)


def _normalize_search_grid(
    r_candidates: tuple[int, ...] | list[int],
    k_candidates: tuple[int, ...] | list[int],
    *,
    minimum_r: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    raw_r = tuple(r_candidates)
    raw_k = tuple(k_candidates)
    if not raw_r or not raw_k:
        raise ValueError("r_candidates and k_candidates must both be non-empty.")
    normalized_r = tuple(
        require_integer_count(value, name="r candidate", minimum=minimum_r)
        for value in raw_r
    )
    normalized_k = tuple(
        require_integer_count(value, name="K candidate") for value in raw_k
    )
    if any(value % 2 for value in normalized_k):
        raise ValueError("Every K candidate must be a non-negative even integer.")
    if len(set(normalized_r)) != len(normalized_r):
        raise ValueError("r_candidates must not contain duplicates.")
    if len(set(normalized_k)) != len(normalized_k):
        raise ValueError("k_candidates must not contain duplicates.")
    return (
        normalized_r,
        normalized_k,
        tuple(sorted(normalized_r)),
        tuple(sorted(normalized_k)),
    )


def _float_payload(value: float) -> str:
    return float(value).hex()


def _search_configuration_fingerprint(
    *,
    preparation: DFPartialS2Preparation,
    specification: RPERoundSpecification,
    allocation: RPEErrorAllocation,
    pf_error_model: RPEPFErrorModel,
    beta_rpe: float,
    requested_r_candidates: tuple[int, ...],
    requested_k_candidates: tuple[int, ...],
    effective_search_pairs: tuple[tuple[int, int], ...],
    allowed_guarantee_statuses: tuple[RPEGuaranteeStatus, ...],
    cost_metric: str,
    cost_model_fingerprint: str | None,
    rte_seed: int,
    hadamard_sampling_policy: RPEHadamardSamplingPolicy,
    maximum_candidate_count: int,
) -> str:
    independent_outcomes = (
        hadamard_sampling_policy.independent_bounded_outcomes_within_each_round_axis
    )
    payload = {
        "optimizer_version": RPE_SHORT_ROUND_OPTIMIZER_VERSION,
        "enumeration_rule": RPE_SHORT_ROUND_ENUMERATION_RULE,
        "tie_break_rule": RPE_SHORT_ROUND_TIE_BREAK_RULE,
        "selection_statistic": RPE_SHORT_ROUND_SELECTION_STATISTIC,
        "hamiltonian_hash": preparation.hamiltonian_hash,
        "partition_hash": preparation.partition_hash,
        "preparation_hash": preparation.preparation_hash,
        "ld": preparation.ld,
        "delta_time": _float_payload(specification.delta_time),
        "round_index": specification.round_index,
        "allocation": {
            "beta_pf_budget": _float_payload(allocation.beta_pf_budget),
            "beta_rte_budget": _float_payload(allocation.beta_rte_budget),
            "beta_stat_budget": _float_payload(allocation.beta_stat_budget),
            "alpha_cosine": _float_payload(allocation.alpha_cosine),
            "alpha_sine": _float_payload(allocation.alpha_sine),
        },
        "pf_error_model": {
            "coefficient": _float_payload(pf_error_model.coefficient),
            "source": pf_error_model.source,
            "is_rigorous_bound": pf_error_model.is_rigorous_bound,
        },
        "beta_rpe": _float_payload(beta_rpe),
        "requested_r_candidates": requested_r_candidates,
        "requested_k_candidates": requested_k_candidates,
        "effective_search_pairs": effective_search_pairs,
        "allowed_guarantee_statuses": allowed_guarantee_statuses,
        "cost_metric": cost_metric,
        "cost_model_fingerprint": cost_model_fingerprint,
        "rte_seed": rte_seed,
        "hadamard_sampling_policy": {
            "rte_trajectory_mode": hadamard_sampling_policy.rte_trajectory_mode,
            "independent_bounded_outcomes_within_each_round_axis": (
                independent_outcomes
            ),
        },
        "maximum_candidate_count": maximum_candidate_count,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _cost_comparability_failures(
    evaluations: tuple[RPERoundSearchEvaluation, ...],
) -> tuple[tuple[str, ...], str | None]:
    costed = tuple(
        evaluation.candidate
        for evaluation in evaluations
        if evaluation.candidate is not None
        and evaluation.candidate.round_total_cost is not None
    )
    if not costed:
        return (), None
    reasons: list[str] = []
    fingerprints = {candidate.cost_model_fingerprint for candidate in costed}
    common_fingerprint = (
        next(iter(fingerprints))
        if len(fingerprints) == 1 and None not in fingerprints
        else None
    )
    if None in fingerprints:
        reasons.append("unverified_cost_model_fingerprint")
    if len(fingerprints) != 1:
        reasons.append("mixed_cost_model_fingerprints")
    scopes = {candidate.circuit_cost_scope for candidate in costed}
    if len(scopes) != 1:
        reasons.append("mixed_circuit_cost_scopes")
    provider_versions = {
        dict(candidate.cost_metadata).get("provider_version")
        for candidate in costed
    }
    if len(provider_versions) != 1:
        reasons.append("mixed_compiled_cost_provider_versions")
    evaluation_methods = {candidate.cost_evaluation_method for candidate in costed}
    if len(evaluation_methods) != 1:
        reasons.append("mixed_compiled_cost_evaluation_methods")
    compilers = {
        candidate.cosine_expected_cost.compiler
        for candidate in costed
        if candidate.cosine_expected_cost is not None
    }
    if len(compilers) != 1 or any(
        candidate.cosine_expected_cost is None
        or candidate.sine_expected_cost is None
        or candidate.cosine_expected_cost.compiler
        != candidate.sine_expected_cost.compiler
        for candidate in costed
    ):
        reasons.append("mixed_compiler_settings")
    return tuple(reasons), common_fingerprint


def optimize_rpe_short_round(
    preparation: DFPartialS2Preparation,
    specification: RPERoundSpecification,
    allocation: RPEErrorAllocation,
    pf_error_model: RPEPFErrorModel,
    *,
    beta_rpe: float,
    r_candidates: tuple[int, ...] | list[int],
    k_candidates: tuple[int, ...] | list[int],
    allowed_guarantee_statuses: (
        tuple[RPEGuaranteeStatus, ...] | list[RPEGuaranteeStatus]
    ),
    cost_metric: str,
    cost_provider: RPERoundCostProvider,
    rte_seed: int = 0,
    hadamard_sampling_policy: RPEHadamardSamplingPolicy | None = None,
    maximum_candidate_count: int = DEFAULT_MAXIMUM_SHORT_ROUND_CANDIDATES,
) -> RPERoundOptimizationResult:
    """Exhaustively minimize one short round over the declared ``(r,K)`` grid."""
    if not isinstance(preparation, DFPartialS2Preparation):
        raise TypeError("preparation must be a DFPartialS2Preparation.")
    if not isinstance(specification, RPERoundSpecification):
        raise TypeError("specification must be an RPERoundSpecification.")
    if not isinstance(allocation, RPEErrorAllocation):
        raise TypeError("allocation must be an RPEErrorAllocation.")
    if not isinstance(pf_error_model, RPEPFErrorModel):
        raise TypeError("pf_error_model must be an RPEPFErrorModel.")
    if specification.q_m > RPE_SHORT_ROUND_MAXIMUM_Q:
        raise ValueError(
            f"Short-round optimization requires q_m<=4; got q_m={specification.q_m}."
        )
    if cost_provider is None:
        raise TypeError("cost_provider is required for compiled-cost optimization.")
    if cost_metric not in RPE_COST_METRICS:
        raise ValueError(
            f"Unsupported RPE cost metric: {cost_metric}. "
            f"Expected one of {RPE_COST_METRICS}."
        )
    normalized_cost_metric = cast(RPECostMetric, cost_metric)
    beta_rpe_value = float(beta_rpe)
    if not math.isfinite(beta_rpe_value) or beta_rpe_value <= 0.0:
        raise ValueError("beta_rpe must be finite and positive.")
    if beta_rpe_value > RPE_NUMERICAL_PHASE_MODEL_LIMIT:
        raise ValueError("beta_rpe must not exceed pi/2 in this phase model.")
    requested_r, requested_k, normalized_r, normalized_k = (
        _normalize_search_grid(
            r_candidates,
            k_candidates,
            minimum_r=0 if preparation.is_deterministic_only else 1,
        )
    )
    allowed_statuses = _normalize_statuses(allowed_guarantee_statuses)
    maximum_count = require_integer_count(
        maximum_candidate_count,
        name="maximum_candidate_count",
        minimum=1,
    )
    seed = require_integer_count(rte_seed, name="rte_seed")
    if hadamard_sampling_policy is None:
        hadamard_sampling_policy = RPEHadamardSamplingPolicy()
    if not isinstance(hadamard_sampling_policy, RPEHadamardSamplingPolicy):
        raise TypeError(
            "hadamard_sampling_policy must be an RPEHadamardSamplingPolicy."
        )

    effective_pairs = (
        ((0, 0),)
        if preparation.is_deterministic_only
        else tuple((r_m, k_m) for r_m in normalized_r for k_m in normalized_k)
    )
    if len(effective_pairs) > maximum_count:
        raise ValueError(
            f"Declared search has {len(effective_pairs)} candidates, exceeding "
            f"maximum_candidate_count={maximum_count}."
        )

    evaluations: list[RPERoundSearchEvaluation] = []
    for r_m, k_m in effective_pairs:
        try:
            candidate = evaluate_rpe_round_candidate(
                preparation,
                specification,
                allocation,
                pf_error_model,
                beta_rpe=beta_rpe,
                rte_steps_per_occurrence=r_m,
                finite_taylor_order=k_m,
                cost_metric=normalized_cost_metric,
                cost_provider=cost_provider,
                rte_seed=seed,
                hadamard_sampling_policy=hadamard_sampling_policy,
            )
        except (ValueError, OverflowError) as exc:
            failure = RPERoundEvaluationFailure(
                r_m=r_m,
                k_m=k_m,
                stage="candidate_or_compiled_cost_evaluation",
                exception_type=type(exc).__name__,
                message=str(exc),
            )
            evaluations.append(
                RPERoundSearchEvaluation(
                    r_m=r_m,
                    k_m=k_m,
                    candidate=None,
                    evaluation_failure=failure,
                    selection_eligible=False,
                    selection_exclusion_reasons=("evaluation_failed",),
                )
            )
            continue

        exclusion_reasons: list[RPERoundSelectionExclusionReason] = []
        if not candidate.feasible:
            exclusion_reasons.append("candidate_numerically_infeasible")
        if candidate.round_total_cost is None:
            exclusion_reasons.append("missing_round_total_cost")
        if candidate.guarantee_status not in allowed_statuses:
            exclusion_reasons.append("guarantee_status_not_allowed")
        evaluations.append(
            RPERoundSearchEvaluation(
                r_m=r_m,
                k_m=k_m,
                candidate=candidate,
                evaluation_failure=None,
                selection_eligible=not exclusion_reasons,
                selection_exclusion_reasons=tuple(exclusion_reasons),
            )
        )

    evaluations_tuple = tuple(evaluations)
    comparability_reasons, cost_model_fingerprint = (
        _cost_comparability_failures(evaluations_tuple)
    )
    has_evaluation_failure = any(
        evaluation.evaluation_failure is not None
        for evaluation in evaluations_tuple
    )
    search_failure_reasons = list(comparability_reasons)
    if has_evaluation_failure:
        search_failure_reasons.append("declared_grid_evaluation_incomplete")
    search_complete = not search_failure_reasons

    selected: RPERoundCandidate | None = None
    eligible = tuple(
        evaluation.candidate
        for evaluation in evaluations_tuple
        if evaluation.selection_eligible and evaluation.candidate is not None
    )
    if search_complete and eligible:
        selected = min(
            eligible,
            key=lambda candidate: (
                candidate.round_total_cost,
                candidate.r_m,
                candidate.k_m,
            ),
        )
        if selected.round_total_cost is None:  # pragma: no cover - eligibility
            raise RuntimeError("Selected candidate must have a compiled cost.")

    fingerprint = _search_configuration_fingerprint(
        preparation=preparation,
        specification=specification,
        allocation=allocation,
        pf_error_model=pf_error_model,
        beta_rpe=beta_rpe_value,
        requested_r_candidates=requested_r,
        requested_k_candidates=requested_k,
        effective_search_pairs=effective_pairs,
        allowed_guarantee_statuses=allowed_statuses,
        cost_metric=normalized_cost_metric,
        cost_model_fingerprint=cost_model_fingerprint,
        rte_seed=seed,
        hadamard_sampling_policy=hadamard_sampling_policy,
        maximum_candidate_count=maximum_count,
    )
    return RPERoundOptimizationResult(
        specification=specification,
        allocation=allocation,
        requested_r_candidates=requested_r,
        requested_k_candidates=requested_k,
        effective_search_pairs=effective_pairs,
        evaluations=evaluations_tuple,
        selected_candidate=selected,
        allowed_guarantee_statuses=allowed_statuses,
        cost_metric=normalized_cost_metric,
        cost_model_fingerprint=cost_model_fingerprint,
        search_configuration_fingerprint=fingerprint,
        search_complete=search_complete,
        search_failure_reasons=tuple(search_failure_reasons),
    )


def optimize_rpe_short_rounds(
    preparation: DFPartialS2Preparation,
    round_inputs: (
        tuple[RPERoundOptimizationInput, ...]
        | list[RPERoundOptimizationInput]
    ),
    pf_error_model: RPEPFErrorModel,
    *,
    beta_rpe: float,
    r_candidates: tuple[int, ...] | list[int],
    k_candidates: tuple[int, ...] | list[int],
    allowed_guarantee_statuses: (
        tuple[RPEGuaranteeStatus, ...] | list[RPEGuaranteeStatus]
    ),
    total_alpha_budget: float,
    cost_metric: str,
    cost_provider: RPERoundCostProvider,
    rte_seed: int = 0,
    hadamard_sampling_policy: RPEHadamardSamplingPolicy | None = None,
    maximum_candidate_count: int = DEFAULT_MAXIMUM_SHORT_ROUND_CANDIDATES,
) -> RPEShortRoundOptimizationResult:
    """Optimize contiguous rounds ``m=0,...,M<=2`` and build their summary."""
    inputs = tuple(round_inputs)
    if not inputs:
        raise ValueError("At least one round optimization input is required.")
    if any(not isinstance(item, RPERoundOptimizationInput) for item in inputs):
        raise TypeError("Every round input must be RPERoundOptimizationInput.")
    indices = tuple(item.specification.round_index for item in inputs)
    if len(set(indices)) != len(indices):
        raise ValueError("At most one optimization input is allowed per round.")
    if tuple(sorted(indices)) != tuple(range(max(indices) + 1)):
        raise ValueError("Round inputs must contain every round from 0 through M.")
    if max(item.specification.q_m for item in inputs) > RPE_SHORT_ROUND_MAXIMUM_Q:
        raise ValueError("Short-round optimization requires every q_m<=4.")
    delta_times = {item.specification.delta_time for item in inputs}
    if len(delta_times) != 1:
        raise ValueError("Every short-round input must use the same delta_time.")
    alpha_budget = float(total_alpha_budget)
    if not math.isfinite(alpha_budget) or not 0.0 < alpha_budget < 1.0:
        raise ValueError("total_alpha_budget must be finite and lie in (0, 1).")

    ordered_inputs = tuple(
        sorted(inputs, key=lambda item: item.specification.round_index)
    )
    round_results = tuple(
        optimize_rpe_short_round(
            preparation,
            item.specification,
            item.allocation,
            pf_error_model,
            beta_rpe=beta_rpe,
            r_candidates=r_candidates,
            k_candidates=k_candidates,
            allowed_guarantee_statuses=allowed_guarantee_statuses,
            cost_metric=cost_metric,
            cost_provider=cost_provider,
            rte_seed=rte_seed,
            hadamard_sampling_policy=hadamard_sampling_policy,
            maximum_candidate_count=maximum_candidate_count,
        )
        for item in ordered_inputs
    )
    all_selected = all(
        result.search_complete and result.selected_candidate is not None
        for result in round_results
    )
    selected_candidates: tuple[RPERoundCandidate, ...] | None = None
    summary: RPEResourceSummary | None = None
    summary_failure_reason: str | None = None
    if all_selected:
        selected_candidates = tuple(
            result.selected_candidate
            for result in round_results
            if result.selected_candidate is not None
        )
        try:
            summary = build_rpe_resource_summary(
                selected_candidates,
                total_alpha_budget=alpha_budget,
                cost_metric=cost_metric,
            )
        except (ValueError, OverflowError) as exc:
            summary_failure_reason = str(exc)
    return RPEShortRoundOptimizationResult(
        round_results=round_results,
        selected_candidates=selected_candidates,
        resource_summary=summary,
        search_complete=all(result.search_complete for result in round_results),
        summary_failure_reason=summary_failure_reason,
    )
