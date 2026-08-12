"""Finite-RTE resource accounting for directly constructible RPE rounds.

This layer connects the existing finite-RTE mathematics and Level-5-R
time-evolution-subcircuit costs to attenuation-aware Hadamard-test shot
counts.  It deliberately does not construct a full RPE circuit or claim a
rigorous success guarantee when the product-formula input is empirical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

from .df_partial_s2 import DFPartialS2Preparation
from .rte import (
    CircuitCost,
    RTEConfig,
    RTEFiniteDistribution,
    compose_truncation_residual_bounds,
    finite_rte_attenuation,
    finite_rte_distribution,
    make_rte_config,
    require_integer_count,
)


RPECostMetric: TypeAlias = Literal[
    "rz_count",
    "cx_count",
    "rz_depth",
    "cx_depth",
    "total_depth",
    "circuit_size",
]
RPEGuaranteeStatus: TypeAlias = Literal[
    "certified",
    "empirical_screening",
    "not_certified",
]
RPEPFProvenance: TypeAlias = Literal[
    "rigorous_upper_bound",
    "empirical_surrogate",
]
RPECircuitCostScope: TypeAlias = Literal[
    "compiled_time_evolution_subcircuit",
    "full_rpe_circuit",
    "validated_long_circuit_proxy",
]

RPE_COST_METRICS: tuple[RPECostMetric, ...] = (
    "rz_count",
    "cx_count",
    "rz_depth",
    "cx_depth",
    "total_depth",
    "circuit_size",
)
RPE_RESOURCE_ACCOUNTING_VERSION = "finite_rpe_resource_accounting_v2"
RPE_NUMERICAL_PHASE_MODEL_LIMIT = math.pi / 2.0
RPE_STRICT_BRANCH_CERTIFICATION_LIMIT = math.pi / 3.0
RPE_CONDITIONAL_GUARANTEE_SCOPE = (
    "effective_partial_s2_eigenstate_unit_radius_alias_free_v1"
)
RPE_UNIT_UNATTENUATED_SIGNAL_RADIUS = 1.0
MAXIMUM_RPE_ROUND_INDEX = 62
DEFAULT_RPE_CIRCUIT_COST_SCOPE: RPECircuitCostScope = (
    "compiled_time_evolution_subcircuit"
)
_KNOWN_EMPIRICAL_PF_SOURCES = frozenset(
    {
        "state_specific_phase_bias_surrogate",
        "df_phase_bias_surrogate_v3",
    }
)


def _finite_nonnegative(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return normalized


def _finite_positive(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return normalized


def _conservative_nonnegative_sum(values: tuple[float, ...]) -> float:
    """Return the smallest float not below the exact represented-value sum."""
    exact = sum((Fraction.from_float(value) for value in values), Fraction())
    try:
        rounded = float(exact)
    except OverflowError:
        return math.inf
    if math.isfinite(rounded) and Fraction.from_float(rounded) < exact:
        rounded = math.nextafter(rounded, math.inf)
    return rounded


def _normalize_cost_metric(metric: str) -> RPECostMetric:
    if metric not in RPE_COST_METRICS:
        raise ValueError(
            f"Unsupported RPE cost metric: {metric}. "
            f"Expected one of {RPE_COST_METRICS}."
        )
    return metric  # type: ignore[return-value]


def circuit_cost_metric(cost: CircuitCost, metric: str) -> float:
    """Read one explicitly selected non-negative expected-cost metric."""
    normalized = _normalize_cost_metric(metric)
    value = float(getattr(cost, normalized))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            f"Expected compiled cost metric {normalized} must be finite and "
            "non-negative."
        )
    return value


@dataclass(frozen=True)
class RPERoundSpecification:
    """One round with ``q_m=2**m`` and ``t_m=q_m*delta_time``."""

    round_index: int
    delta_time: float

    def __post_init__(self) -> None:
        round_index = require_integer_count(self.round_index, name="round_index")
        if round_index > MAXIMUM_RPE_ROUND_INDEX:
            raise ValueError(
                f"round_index must not exceed {MAXIMUM_RPE_ROUND_INDEX}; "
                "larger rounds require a separately validated proxy."
            )
        object.__setattr__(self, "round_index", round_index)
        object.__setattr__(
            self,
            "delta_time",
            _finite_positive(self.delta_time, name="delta_time"),
        )
        if not math.isfinite((1 << round_index) * self.delta_time):
            raise ValueError("Derived t_m must be finite.")

    @property
    def m(self) -> int:
        return self.round_index

    @property
    def q_m(self) -> int:
        return 1 << self.round_index

    @property
    def t_m(self) -> float:
        return float(self.q_m * self.delta_time)


@dataclass(frozen=True)
class RPEErrorAllocation:
    """Allocated round phase errors and axis-specific failure probabilities."""

    beta_pf_budget: float
    beta_rte_budget: float
    beta_stat_budget: float
    alpha_cosine: float
    alpha_sine: float

    def __post_init__(self) -> None:
        for name in (
            "beta_pf_budget",
            "beta_rte_budget",
            "beta_stat_budget",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name=name),
            )
        for name in ("alpha_cosine", "alpha_sine"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be finite and lie strictly in (0, 1).")
            object.__setattr__(self, name, value)

    @property
    def beta_total(self) -> float:
        return _conservative_nonnegative_sum(
            (
                self.beta_pf_budget,
                self.beta_rte_budget,
                self.beta_stat_budget,
            )
        )

    @property
    def round_failure_probability_bound(self) -> float:
        """Union bound for the cosine and sine coordinate events."""
        return _conservative_nonnegative_sum(
            (self.alpha_cosine, self.alpha_sine)
        )

    @property
    def alpha_m_c(self) -> float:
        return self.alpha_cosine

    @property
    def alpha_m_s(self) -> float:
        return self.alpha_sine


@dataclass(frozen=True)
class RPEPFErrorModel:
    """Externally classified coefficient for ``epsilon_PF=C*delta**2``.

    The coefficient is not inferred here.  Callers must explicitly say
    whether it is a rigorous upper bound.  A state-specific fitted surrogate
    has ``provenance_status='empirical_surrogate'`` and gives downstream
    ``guarantee_status='empirical_screening'`` when no independent hard
    certification condition fails.
    """

    coefficient: float
    source: str
    is_rigorous_bound: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coefficient",
            _finite_nonnegative(self.coefficient, name="coefficient"),
        )
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string.")
        normalized_source = self.source.strip()
        object.__setattr__(self, "source", normalized_source)
        if not isinstance(self.is_rigorous_bound, bool):
            raise TypeError("is_rigorous_bound must be boolean.")
        if (
            normalized_source.lower() in _KNOWN_EMPIRICAL_PF_SOURCES
            and self.is_rigorous_bound
        ):
            raise ValueError(
                f"{normalized_source} is an empirical screening surrogate and "
                "cannot be marked as a rigorous bound."
            )

    @property
    def provenance_status(self) -> RPEPFProvenance:
        return (
            "rigorous_upper_bound"
            if self.is_rigorous_bound
            else "empirical_surrogate"
        )


@dataclass(frozen=True)
class RPERoundCostRequest:
    """Provider-neutral input for a round's compiled circuit-cost estimate."""

    preparation: DFPartialS2Preparation
    specification: RPERoundSpecification
    allocation: RPEErrorAllocation
    rte_steps_per_occurrence: int
    finite_taylor_order: int
    rte_config: RTEConfig | None
    rte_distribution: RTEFiniteDistribution | None


@dataclass(frozen=True)
class RPERoundCompiledCost:
    """Axis-specific compiled expectations returned by a cost provider."""

    cosine_expected_cost: CircuitCost
    sine_expected_cost: CircuitCost
    cosine_standard_error: CircuitCost | None
    sine_standard_error: CircuitCost | None
    evaluation_method: str
    classical_sample_count: int | None
    circuit_cost_scope: RPECircuitCostScope = DEFAULT_RPE_CIRCUIT_COST_SCOPE
    cost_model_fingerprint: str | None = None
    metadata: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.cosine_expected_cost, CircuitCost):
            raise TypeError("cosine_expected_cost must be a CircuitCost.")
        if not isinstance(self.sine_expected_cost, CircuitCost):
            raise TypeError("sine_expected_cost must be a CircuitCost.")
        if self.cosine_expected_cost.compiler != self.sine_expected_cost.compiler:
            raise ValueError(
                "Cosine and sine compiled costs must use the same compiler settings."
            )
        for name in ("cosine_standard_error", "sine_standard_error"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, CircuitCost):
                raise TypeError(f"{name} must be a CircuitCost or None.")
            if (
                value is not None
                and value.compiler != self.cosine_expected_cost.compiler
            ):
                raise ValueError(
                    f"{name} must use the expected costs' compiler settings."
                )
        if not isinstance(self.evaluation_method, str) or not self.evaluation_method:
            raise ValueError("evaluation_method must be a non-empty string.")
        if self.classical_sample_count is not None:
            object.__setattr__(
                self,
                "classical_sample_count",
                require_integer_count(
                    self.classical_sample_count,
                    name="classical_sample_count",
                    minimum=1,
                ),
            )
        if self.circuit_cost_scope not in (
            "compiled_time_evolution_subcircuit",
            "full_rpe_circuit",
            "validated_long_circuit_proxy",
        ):
            raise ValueError("Unsupported circuit_cost_scope.")
        if self.cost_model_fingerprint is not None and (
            not isinstance(self.cost_model_fingerprint, str)
            or not self.cost_model_fingerprint
        ):
            raise ValueError("cost_model_fingerprint must be non-empty or None.")
        if any(not isinstance(key, str) or not key for key, _ in self.metadata):
            raise ValueError("Every metadata key must be a non-empty string.")
        metadata_keys = tuple(key for key, _ in self.metadata)
        if len(set(metadata_keys)) != len(metadata_keys):
            raise ValueError("Compiled-cost metadata keys must be unique.")
        metadata = dict(self.metadata)
        provider_version = metadata.get("provider_version")
        if provider_version is not None and (
            not isinstance(provider_version, str) or not provider_version
        ):
            raise ValueError("provider_version metadata must be non-empty or absent.")
        backend_context = metadata.get("backend_context_canonical")
        if backend_context is not None and not isinstance(backend_context, bool):
            raise TypeError("backend_context_canonical metadata must be boolean.")


@runtime_checkable
class RPERoundCostProvider(Protocol):
    """Replaceable compiled-cost boundary for current and future models."""

    def evaluate(self, request: RPERoundCostRequest) -> RPERoundCompiledCost: ...


@dataclass(frozen=True)
class RPERoundCandidate:
    """Finite-RTE, shot-count, and compiled-cost record for one ``(r_m,K_m)``."""

    specification: RPERoundSpecification
    allocation: RPEErrorAllocation
    pf_error_model: RPEPFErrorModel
    beta_rpe: float
    hamiltonian_hash: str
    partition_hash: str
    preparation_hash: str
    ld: int
    coefficient_atol: float
    threshold_operator_error_bound: float
    rte_seed: int
    rte_steps_per_occurrence: int
    finite_taylor_order: int
    exact_rte_lambda_r: float
    tau_m: float
    epsilon_step: float
    epsilon_z: float
    normalization: float
    attenuation: float
    unattenuated_signal_radius_lower_bound: float
    rho_observed_lower_bound: float
    epsilon_pf: float
    beta_pf: float
    beta_rte: float | None
    phase_budget_residual: float
    epsilon_coordinate: float | None
    cosine_shots: int | None
    sine_shots: int | None
    cosine_expected_cost: CircuitCost | None
    sine_expected_cost: CircuitCost | None
    cosine_standard_error: CircuitCost | None
    sine_standard_error: CircuitCost | None
    cosine_expected_metric: float | None
    sine_expected_metric: float | None
    round_total_cost: float | None
    cost_metric: RPECostMetric
    cost_evaluation_method: str | None
    classical_cost_sample_count: int | None
    circuit_cost_scope: RPECircuitCostScope
    cost_model_fingerprint: str | None
    feasible: bool
    infeasibility_reasons: tuple[str, ...]
    guarantee_status: RPEGuaranteeStatus
    certification_reasons: tuple[str, ...]
    assumptions: tuple[str, ...]
    cost_metadata: tuple[tuple[str, Any], ...] = ()
    accounting_version: str = RPE_RESOURCE_ACCOUNTING_VERSION
    guarantee_scope: str = RPE_CONDITIONAL_GUARANTEE_SCOPE

    @property
    def m(self) -> int:
        return self.specification.round_index

    @property
    def q_m(self) -> int:
        return self.specification.q_m

    @property
    def t_m(self) -> float:
        return self.specification.t_m

    @property
    def r_m(self) -> int:
        return self.rte_steps_per_occurrence

    @property
    def k_m(self) -> int:
        return self.finite_taylor_order

    @property
    def b_k(self) -> float:
        return self.normalization

    @property
    def n_m_c(self) -> int | None:
        return self.cosine_shots

    @property
    def n_m_s(self) -> int | None:
        return self.sine_shots

    @property
    def round_failure_probability_bound(self) -> float:
        return self.allocation.round_failure_probability_bound

    @property
    def statistical_phase_budget_ceiling(self) -> float:
        """Phase budget left after the allocated PF and finite-RTE budgets."""
        return self.phase_budget_residual

    @property
    def strict_branch_margin_satisfied(self) -> bool:
        """Whether the PR Lemma B.1 strict ``pi/3`` condition can hold."""
        return self.beta_rpe < RPE_STRICT_BRANCH_CERTIFICATION_LIMIT

    @property
    def branch_certification_margin(self) -> float:
        return float(RPE_STRICT_BRANCH_CERTIFICATION_LIMIT - self.beta_rpe)


@dataclass(frozen=True)
class RPEResourceSummary:
    """All-round total for one selected directly constructed candidate per round."""

    rounds: tuple[RPERoundCandidate, ...]
    total_cost: float
    cost_metric: RPECostMetric
    circuit_cost_scope: RPECircuitCostScope
    total_failure_probability_bound: float
    total_alpha_budget: float
    union_bound_satisfied: bool
    guarantee_status: RPEGuaranteeStatus
    certification_reasons: tuple[str, ...]
    assumptions: tuple[str, ...]
    accounting_version: str = RPE_RESOURCE_ACCOUNTING_VERSION
    guarantee_scope: str = RPE_CONDITIONAL_GUARANTEE_SCOPE

    @property
    def g_total(self) -> float:
        return self.total_cost

    @property
    def maximum_round_index(self) -> int:
        return max(item.m for item in self.rounds)

    @property
    def maximum_evolution_time(self) -> float:
        return max(item.t_m for item in self.rounds)

    @property
    def nominal_energy_resolution(self) -> float:
        """``beta_rpe/t_M`` scale; not itself a target-precision guarantee."""
        return float(self.rounds[0].beta_rpe / self.maximum_evolution_time)

    @property
    def conditional_energy_error_bound(self) -> float | None:
        """High-probability energy bound under the recorded assumptions."""
        if self.guarantee_status != "certified":
            return None
        return self.nominal_energy_resolution

    @property
    def conditional_success_probability_lower_bound(self) -> float | None:
        if self.guarantee_status != "certified":
            return None
        return float(max(0.0, 1.0 - self.total_failure_probability_bound))


def _provider_evaluate(
    provider: RPERoundCostProvider,
    request: RPERoundCostRequest,
) -> RPERoundCompiledCost:
    method = getattr(provider, "evaluate", None)
    result = (
        method(request)
        if callable(method)
        else provider(request)  # type: ignore[operator]
    )
    if not isinstance(result, RPERoundCompiledCost):
        raise TypeError("cost_provider must return RPERoundCompiledCost.")
    return result


def _hoeffding_shots(epsilon_coordinate: float, alpha: float) -> int:
    if not math.isfinite(epsilon_coordinate) or epsilon_coordinate <= 0.0:
        raise ValueError("epsilon_coordinate must be finite and positive.")
    value = (2.0 / (epsilon_coordinate * epsilon_coordinate)) * math.log(
        2.0 / alpha
    )
    if not math.isfinite(value):
        raise OverflowError("Hoeffding shot count overflowed.")
    return int(math.ceil(value))


def evaluate_rpe_round_candidate(
    preparation: DFPartialS2Preparation,
    specification: RPERoundSpecification,
    allocation: RPEErrorAllocation,
    pf_error_model: RPEPFErrorModel,
    *,
    beta_rpe: float,
    rte_steps_per_occurrence: int,
    finite_taylor_order: int,
    cost_metric: str,
    cost_provider: RPERoundCostProvider | None = None,
    rte_seed: int = 0,
) -> RPERoundCandidate:
    """Evaluate one short, second-order partial-S2 RPE round candidate.

    Invalid resource candidates are returned with explicit reasons and are
    never passed to the compiled-cost provider.
    """
    if not isinstance(preparation, DFPartialS2Preparation):
        raise TypeError("preparation must be a DFPartialS2Preparation.")
    if preparation.product_formula != "2nd":  # pragma: no cover - guarded by type
        raise ValueError("RPE accounting currently supports only partial-S2.")
    if not isinstance(specification, RPERoundSpecification):
        raise TypeError("specification must be an RPERoundSpecification.")
    if not isinstance(allocation, RPEErrorAllocation):
        raise TypeError("allocation must be an RPEErrorAllocation.")
    if not isinstance(pf_error_model, RPEPFErrorModel):
        raise TypeError("pf_error_model must be an RPEPFErrorModel.")
    normalized_metric = _normalize_cost_metric(cost_metric)
    beta_rpe_value = _finite_positive(beta_rpe, name="beta_rpe")
    if beta_rpe_value > RPE_NUMERICAL_PHASE_MODEL_LIMIT:
        raise ValueError("beta_rpe must not exceed pi/2 in this phase model.")
    cutoff = require_integer_count(
        finite_taylor_order,
        name="finite_taylor_order",
    )
    if cutoff % 2:
        raise ValueError("finite_taylor_order must be a non-negative even integer.")
    deterministic_only = preparation.is_deterministic_only
    rte_steps = require_integer_count(
        rte_steps_per_occurrence,
        name="rte_steps_per_occurrence",
        minimum=0 if deterministic_only else 1,
    )
    if deterministic_only and (rte_steps != 0 or cutoff != 0):
        raise ValueError(
            "A deterministic-only DF tail requires the canonical candidate "
            "rte_steps_per_occurrence=0 and finite_taylor_order=0."
        )
    rte_seed = require_integer_count(rte_seed, name="rte_seed")

    exact_lambda = float(preparation.exact_rte_lambda_r)
    if not math.isfinite(exact_lambda) or exact_lambda < 0.0:
        raise ValueError("preparation.exact_rte_lambda_r must be non-negative.")
    threshold_error = _finite_nonnegative(
        preparation.threshold_operator_error_bound,
        name="preparation.threshold_operator_error_bound",
    )

    rte_config: RTEConfig | None
    rte_distribution: RTEFiniteDistribution | None
    if deterministic_only:
        tau = 0.0
        epsilon_step = 0.0
        epsilon_z = 0.0
        normalization = 1.0
        attenuation = 1.0
        rte_config = None
        rte_distribution = None
    else:
        tau = exact_lambda * specification.delta_time / rte_steps
        rte_distribution = finite_rte_distribution(tau, cutoff)
        epsilon_step = rte_distribution.step_truncation_residual_bound
        epsilon_z = compose_truncation_residual_bounds(
            ((epsilon_step, rte_steps, specification.q_m),)
        )
        normalization = rte_distribution.exact_finite_distribution
        if not math.isfinite(epsilon_step):
            rte_config = None
            attenuation = 0.0
        else:
            rte_config, canonical_distribution = make_rte_config(
                preparation.rte_preparation.symbolic_tail,
                evolution_time=specification.delta_time,
                rte_steps=rte_steps,
                truncation_tolerance=max(epsilon_step, math.ulp(0.0)),
                finite_taylor_order=cutoff,
                seed=rte_seed,
            )
            # Keep a single canonical finite distribution at the provider boundary.
            rte_distribution = canonical_distribution
            attenuation = finite_rte_attenuation(
                rte_config,
                tail_evolutions=specification.q_m,
            )

    reference_radius = RPE_UNIT_UNATTENUATED_SIGNAL_RADIUS
    rho_lower = float(attenuation * (reference_radius - epsilon_z))
    epsilon_pf = float(
        pf_error_model.coefficient * specification.delta_time**2
    )
    beta_pf = float(specification.t_m * epsilon_pf)
    beta_rte = (
        float(math.asin(epsilon_z / reference_radius))
        if math.isfinite(epsilon_z) and 0.0 <= epsilon_z < reference_radius
        else None
    )
    phase_residual = float(
        beta_rpe_value - allocation.beta_pf_budget - allocation.beta_rte_budget
    )

    reasons: list[str] = []
    if not math.isfinite(epsilon_z) or epsilon_z >= reference_radius:
        reasons.append("finite_rte_error_not_below_one")
    if not math.isfinite(rho_lower) or rho_lower <= 0.0:
        reasons.append("nonpositive_observed_radius_lower_bound")
    if phase_residual <= 0.0:
        reasons.append("nonpositive_phase_budget_residual")
    if allocation.beta_stat_budget <= 0.0:
        reasons.append("nonpositive_statistical_phase_budget")
    if allocation.beta_total > beta_rpe_value + 1e-15:
        reasons.append("phase_budget_sum_exceeded")
    if beta_pf > allocation.beta_pf_budget + 1e-15:
        reasons.append("product_formula_budget_exceeded")
    if beta_rte is None or beta_rte > allocation.beta_rte_budget + 1e-15:
        reasons.append("finite_rte_phase_budget_exceeded")

    epsilon_coordinate: float | None = None
    cosine_shots: int | None = None
    sine_shots: int | None = None
    if rho_lower > 0.0 and allocation.beta_stat_budget > 0.0:
        coordinate = float(
            rho_lower
            * math.sin(allocation.beta_stat_budget)
            / math.sqrt(2.0)
        )
        if not math.isfinite(coordinate) or coordinate <= 0.0:
            reasons.append("nonpositive_coordinate_error_budget")
        else:
            epsilon_coordinate = coordinate
            try:
                cosine_shots = _hoeffding_shots(
                    coordinate, allocation.alpha_cosine
                )
                sine_shots = _hoeffding_shots(coordinate, allocation.alpha_sine)
            except OverflowError:
                reasons.append("shot_count_overflow")

    reasons_tuple = tuple(dict.fromkeys(reasons))
    feasible = not reasons_tuple
    cosine_cost: CircuitCost | None = None
    sine_cost: CircuitCost | None = None
    cosine_se: CircuitCost | None = None
    sine_se: CircuitCost | None = None
    cosine_metric: float | None = None
    sine_metric: float | None = None
    round_total: float | None = None
    evaluation_method: str | None = None
    classical_samples: int | None = None
    scope: RPECircuitCostScope = DEFAULT_RPE_CIRCUIT_COST_SCOPE
    cost_model_fingerprint: str | None = None
    cost_metadata: tuple[tuple[str, Any], ...] = ()
    if feasible and cost_provider is not None:
        if cosine_shots is None or sine_shots is None:  # pragma: no cover
            raise RuntimeError("A feasible candidate must have both shot counts.")
        compiled = _provider_evaluate(
            cost_provider,
            RPERoundCostRequest(
                preparation=preparation,
                specification=specification,
                allocation=allocation,
                rte_steps_per_occurrence=rte_steps,
                finite_taylor_order=cutoff,
                rte_config=rte_config,
                rte_distribution=rte_distribution,
            ),
        )
        cosine_cost = compiled.cosine_expected_cost
        sine_cost = compiled.sine_expected_cost
        cosine_se = compiled.cosine_standard_error
        sine_se = compiled.sine_standard_error
        cosine_metric = circuit_cost_metric(cosine_cost, normalized_metric)
        sine_metric = circuit_cost_metric(sine_cost, normalized_metric)
        round_total = float(
            cosine_shots * cosine_metric + sine_shots * sine_metric
        )
        if not math.isfinite(round_total):
            raise OverflowError("Round compiled cost overflowed.")
        evaluation_method = compiled.evaluation_method
        classical_samples = compiled.classical_sample_count
        scope = compiled.circuit_cost_scope
        cost_model_fingerprint = compiled.cost_model_fingerprint
        cost_metadata = compiled.metadata

    assumptions_list = [
        "second_order_partial_s2",
        "one_random_tail_occurrence_per_partial_s2_step",
        "fixed_df_representation_rank_and_fragment_order",
        "fixed_ld_and_delta_time",
        "exact_effective_partial_s2_eigenstate_input_assumed",
        "unit_unattenuated_survival_signal_radius_assumed",
        "alias_free_target_energy_branch_assumed",
        "finite_rte_error_converted_with_arcsin_unit_radius_model",
        "classical_compiled_cost_samples_are_not_quantum_shot_multipliers",
    ]
    if scope == "compiled_time_evolution_subcircuit":
        assumptions_list.extend(
            (
                (
                    "state_preparation_hadamard_measurement_noise_and_"
                    "backend_runs_excluded"
                ),
                "ordinary_controlled_diag_I_U_for_level5r_provider",
            )
        )
    if threshold_error == 0.0:
        assumptions_list.append("zero_df_threshold_operator_error_bound")
    else:
        assumptions_list.append("df_threshold_error_not_included_in_phase_budget")
    if dict(cost_metadata).get("backend_context_canonical") is False:
        assumptions_list.append("compiled_cost_backend_context_uncanonical")
    assumptions = tuple(assumptions_list)

    certification_reasons_list: list[str] = []
    if not feasible:
        certification_reasons_list.append("candidate_numerically_infeasible")
    if beta_rpe_value >= RPE_STRICT_BRANCH_CERTIFICATION_LIMIT:
        certification_reasons_list.append(
            "rpe_branch_margin_not_strictly_below_pi_over_three"
        )
    if threshold_error != 0.0:
        certification_reasons_list.append(
            "nonzero_df_threshold_operator_error_bound"
        )
    # Numerical feasibility retains the versioned 1e-15 comparison tolerance.
    # Certification deliberately rechecks every bound without that relaxation.
    if pf_error_model.is_rigorous_bound:
        if allocation.beta_total > beta_rpe_value:
            certification_reasons_list.append(
                "phase_budget_sum_exceeded_without_tolerance"
            )
        if beta_pf > allocation.beta_pf_budget:
            certification_reasons_list.append(
                "product_formula_budget_exceeded_without_tolerance"
            )
        if beta_rte is None or beta_rte > allocation.beta_rte_budget:
            certification_reasons_list.append(
                "finite_rte_phase_budget_exceeded_without_tolerance"
            )
    hard_certification_failure = bool(certification_reasons_list)
    if not pf_error_model.is_rigorous_bound:
        certification_reasons_list.append("pf_error_model_is_empirical")
    certification_reasons = tuple(dict.fromkeys(certification_reasons_list))
    if hard_certification_failure:
        guarantee_status: RPEGuaranteeStatus = "not_certified"
    elif not pf_error_model.is_rigorous_bound:
        guarantee_status = "empirical_screening"
    else:
        guarantee_status = "certified"
    return RPERoundCandidate(
        specification=specification,
        allocation=allocation,
        pf_error_model=pf_error_model,
        beta_rpe=beta_rpe_value,
        hamiltonian_hash=preparation.hamiltonian_hash,
        partition_hash=preparation.partition_hash,
        preparation_hash=preparation.preparation_hash,
        ld=preparation.ld,
        coefficient_atol=float(preparation.coefficient_atol),
        threshold_operator_error_bound=threshold_error,
        rte_seed=rte_seed,
        rte_steps_per_occurrence=rte_steps,
        finite_taylor_order=cutoff,
        exact_rte_lambda_r=exact_lambda,
        tau_m=float(tau),
        epsilon_step=float(epsilon_step),
        epsilon_z=float(epsilon_z),
        normalization=float(normalization),
        attenuation=float(attenuation),
        unattenuated_signal_radius_lower_bound=reference_radius,
        rho_observed_lower_bound=rho_lower,
        epsilon_pf=epsilon_pf,
        beta_pf=beta_pf,
        beta_rte=beta_rte,
        phase_budget_residual=phase_residual,
        epsilon_coordinate=epsilon_coordinate,
        cosine_shots=cosine_shots,
        sine_shots=sine_shots,
        cosine_expected_cost=cosine_cost,
        sine_expected_cost=sine_cost,
        cosine_standard_error=cosine_se,
        sine_standard_error=sine_se,
        cosine_expected_metric=cosine_metric,
        sine_expected_metric=sine_metric,
        round_total_cost=round_total,
        cost_metric=normalized_metric,
        cost_evaluation_method=evaluation_method,
        classical_cost_sample_count=classical_samples,
        circuit_cost_scope=scope,
        cost_model_fingerprint=cost_model_fingerprint,
        feasible=feasible,
        infeasibility_reasons=reasons_tuple,
        guarantee_status=guarantee_status,
        certification_reasons=certification_reasons,
        assumptions=assumptions,
        cost_metadata=cost_metadata,
    )


def build_rpe_resource_summary(
    candidates: tuple[RPERoundCandidate, ...] | list[RPERoundCandidate],
    *,
    total_alpha_budget: float,
    cost_metric: str,
) -> RPEResourceSummary:
    """Sum one feasible, compiled-cost-bearing candidate for each round."""
    rounds = tuple(candidates)
    if not rounds:
        raise ValueError("At least one RPE round candidate is required.")
    if any(not isinstance(item, RPERoundCandidate) for item in rounds):
        raise TypeError("Every summary item must be an RPERoundCandidate.")
    normalized_metric = _normalize_cost_metric(cost_metric)
    alpha_budget = float(total_alpha_budget)
    if not math.isfinite(alpha_budget) or not 0.0 < alpha_budget < 1.0:
        raise ValueError("total_alpha_budget must be finite and lie in (0, 1).")
    indices = tuple(item.specification.round_index for item in rounds)
    if len(set(indices)) != len(indices):
        raise ValueError(
            "RPE resource summary requires at most one candidate per round."
        )
    if tuple(sorted(indices)) != tuple(range(max(indices) + 1)):
        raise ValueError("All RPE rounds from 0 through max(round_index) are required.")
    if any(not item.feasible for item in rounds):
        raise ValueError("RPE resource summary cannot include infeasible candidates.")
    if any(item.cost_metric != normalized_metric for item in rounds):
        raise ValueError("Every round must use the summary cost_metric.")
    if any(item.round_total_cost is None for item in rounds):
        raise ValueError("Every round must include a compiled-cost estimate.")
    scopes = {item.circuit_cost_scope for item in rounds}
    if len(scopes) != 1:
        raise ValueError("Every round must use the same circuit-cost scope.")

    reference = rounds[0]
    shared_context = (
        reference.hamiltonian_hash,
        reference.partition_hash,
        reference.preparation_hash,
        reference.ld,
        reference.coefficient_atol,
        reference.threshold_operator_error_bound,
        reference.rte_seed,
        reference.specification.delta_time,
        reference.beta_rpe,
        reference.pf_error_model,
        reference.accounting_version,
        reference.guarantee_scope,
    )
    if any(
        (
            item.hamiltonian_hash,
            item.partition_hash,
            item.preparation_hash,
            item.ld,
            item.coefficient_atol,
            item.threshold_operator_error_bound,
            item.rte_seed,
            item.specification.delta_time,
            item.beta_rpe,
            item.pf_error_model,
            item.accounting_version,
            item.guarantee_scope,
        )
        != shared_context
        for item in rounds[1:]
    ):
        raise ValueError(
            "Every round must share one DF preparation, RTE seed, delta_time, "
            "beta_rpe, PF error model, accounting version, and guarantee scope."
        )
    reference_cost = reference.cosine_expected_cost
    if reference_cost is None:  # pragma: no cover - checked above via round total
        raise RuntimeError("A compiled-cost-bearing candidate must store axis costs.")
    reference_compiler = reference_cost.compiler
    if any(
        item.cosine_expected_cost is None
        or item.sine_expected_cost is None
        or item.cosine_expected_cost.compiler != reference_compiler
        or item.sine_expected_cost.compiler != reference_compiler
        for item in rounds
    ):
        raise ValueError("Every round must use the same compiler settings.")
    provider_versions = {
        dict(item.cost_metadata).get("provider_version") for item in rounds
    }
    if len(provider_versions) != 1:
        raise ValueError(
            "Every round must use the same compiled-cost provider version."
        )
    cost_model_fingerprints = {item.cost_model_fingerprint for item in rounds}
    if len(cost_model_fingerprints) != 1:
        raise ValueError("Every round must use the same compiled-cost model.")
    if len(rounds) > 1 and None in cost_model_fingerprints:
        raise ValueError(
            "A multi-round summary requires a verified cost_model_fingerprint."
        )

    failure_bound = _conservative_nonnegative_sum(
        tuple(
            alpha
            for item in rounds
            for alpha in (
                item.allocation.alpha_cosine,
                item.allocation.alpha_sine,
            )
        )
    )
    union_ok = failure_bound <= alpha_budget
    total_cost = float(
        math.fsum(
            item.round_total_cost
            for item in rounds
            if item.round_total_cost is not None
        )
    )
    if not math.isfinite(total_cost):
        raise OverflowError("All-round compiled cost overflowed.")
    summary_certification_reasons = list(
        dict.fromkeys(
            reason for item in rounds for reason in item.certification_reasons
        )
    )
    if not union_ok:
        summary_certification_reasons.append(
            "all_round_union_bound_exceeded_without_tolerance"
        )
    any_not_certified = any(
        item.guarantee_status == "not_certified" for item in rounds
    )
    any_empirical = any(
        item.guarantee_status == "empirical_screening" for item in rounds
    )
    all_certified = all(item.guarantee_status == "certified" for item in rounds)
    if any_not_certified or not union_ok:
        guarantee: RPEGuaranteeStatus = "not_certified"
    elif any_empirical:
        guarantee = "empirical_screening"
    elif all_certified:
        guarantee = "certified"
    else:  # pragma: no cover - exhaustive over the three statuses
        raise RuntimeError("Unexpected RPE guarantee-status combination.")
    assumptions = tuple(
        dict.fromkeys(value for item in rounds for value in item.assumptions)
    )
    return RPEResourceSummary(
        rounds=tuple(sorted(rounds, key=lambda item: item.specification.round_index)),
        total_cost=total_cost,
        cost_metric=normalized_metric,
        circuit_cost_scope=next(iter(scopes)),
        total_failure_probability_bound=failure_bound,
        total_alpha_budget=alpha_budget,
        union_bound_satisfied=union_ok,
        guarantee_status=guarantee,
        certification_reasons=tuple(summary_certification_reasons),
        assumptions=assumptions,
        accounting_version=reference.accounting_version,
        guarantee_scope=reference.guarantee_scope,
    )


summarize_rpe_resources = build_rpe_resource_summary
