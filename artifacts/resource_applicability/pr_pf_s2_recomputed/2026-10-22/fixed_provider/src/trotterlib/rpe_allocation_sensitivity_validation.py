"""Validate provisional RPE phase/failure-budget allocation sensitivity.

The validation reuses fixed one-shot Hadamard compiled costs.  It changes only
the phase-error and failure-probability allocations, so no circuit is rebuilt
or transpiled inside the sweep.  The output is a diagnostic comparison over an
explicitly validated q grid, not a final multi-round resource estimate.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .df_hamiltonian import DFHamiltonian
from .df_partial_randomized_pf import split_df_hamiltonian_by_ld
from .df_partial_s2 import DFPartialS2Preparation, prepare_df_partial_s2
from .rpe_hadamard_compiled_cost_proxy import (
    RPEHadamardCompiledCostProxyValidationResult,
)
from .rpe_hadamard_interrogation import RPE_HADAMARD_INTERROGATION_SCOPE
from .rpe_hadamard_validated_proxy_provider import (
    ValidatedRPEHadamardCompiledCostProxyProvider,
)
from .rpe_resource_accounting import (
    RPE_COST_METRICS,
    RPEErrorAllocation,
    RPEHadamardSamplingPolicy,
    RPEPFErrorModel,
    RPERoundCandidate,
    RPERoundCompiledCost,
    RPERoundCostRequest,
    RPERoundSpecification,
    evaluate_rpe_round_candidate,
)
from .rpe_round_cost_connection_validation import (
    validate_rpe_round_cost_connection_payload,
)
from .rte import CircuitCost, CompilerSettings, require_integer_count
from .rte_compiled_cost import compiler_settings_hash


RPE_ALLOCATION_SENSITIVITY_SCHEMA_VERSION = (
    "rpe_allocation_sensitivity_validation_v1"
)
RPE_ALLOCATION_SENSITIVITY_METHOD = (
    "fixed_q1_q2_q4_direct_q8_proxy_beta_alpha_sensitivity_v1"
)
DEFAULT_BETA_PROFILES: tuple[tuple[str, float, float, float], ...] = (
    ("more_conservative", 0.12, 0.12, 0.16),
    ("current_provisional", 0.08, 0.08, 0.24),
    ("moderate", 0.04, 0.04, 0.32),
    ("guarded_provisional", 0.02, 0.02, 0.36),
    ("tight_diagnostic", 0.01, 0.01, 0.38),
)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _cost_from_payload(
    payload: Mapping[str, Any],
    compiler: CompilerSettings,
    *,
    estimate_kind: str,
) -> CircuitCost:
    return CircuitCost(
        **{metric: float(payload[metric]) for metric in RPE_COST_METRICS},
        compiler=compiler,
        fidelity_level=5,
        estimate_kind=estimate_kind,  # type: ignore[arg-type]
    )


def _compiler_payload(compiler: CompilerSettings) -> dict[str, Any]:
    return {
        "basis_gates": list(compiler.basis_gates),
        "backend_name": compiler.backend_name,
        "coupling_map": compiler.coupling_map,
        "optimization_level": compiler.optimization_level,
        "layout_method": compiler.layout_method,
        "routing_method": compiler.routing_method,
        "transpiler_seed": compiler.transpiler_seed,
        "qiskit_version": compiler.qiskit_version,
    }


def _candidate_payload(candidate: RPERoundCandidate) -> dict[str, Any]:
    return {
        "round_index": candidate.m,
        "q_m": candidate.q_m,
        "feasible": candidate.feasible,
        "infeasible_reasons": list(candidate.infeasibility_reasons),
        "epsilon_z": candidate.epsilon_z,
        "attenuation": candidate.attenuation,
        "rho_observed_lower_bound": candidate.rho_observed_lower_bound,
        "epsilon_pf": candidate.epsilon_pf,
        "beta_pf_actual": candidate.beta_pf,
        "beta_rte_actual": candidate.beta_rte,
        "epsilon_coordinate": candidate.epsilon_coordinate,
        "alpha_cosine": candidate.allocation.alpha_cosine,
        "alpha_sine": candidate.allocation.alpha_sine,
        "cosine_shots": candidate.cosine_shots,
        "sine_shots": candidate.sine_shots,
        "cosine_expected_metric": candidate.cosine_expected_metric,
        "sine_expected_metric": candidate.sine_expected_metric,
        "round_total_cost": candidate.round_total_cost,
        "circuit_cost_scope": candidate.circuit_cost_scope,
        "cost_evaluation_method": candidate.cost_evaluation_method,
        "cost_model_fingerprint": candidate.cost_model_fingerprint,
    }


@dataclass(frozen=True)
class _FixedDirectAndValidatedProxyProvider:
    direct_connection_payload: Mapping[str, Any]
    proxy_provider: ValidatedRPEHadamardCompiledCostProxyProvider
    compiler: CompilerSettings
    preparation: DFPartialS2Preparation
    _direct_rounds: Mapping[int, Mapping[str, Any]] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        validate_rpe_round_cost_connection_payload(self.direct_connection_payload)
        if not isinstance(self.compiler, CompilerSettings):
            raise TypeError("compiler must be a CompilerSettings instance.")
        if not isinstance(self.preparation, DFPartialS2Preparation):
            raise TypeError("preparation must be a DFPartialS2Preparation.")
        request = self.direct_connection_payload["request"]
        if request["compiler"] != _compiler_payload(self.compiler):
            raise ValueError("Direct-cost compiler context does not match.")
        if (
            self.direct_connection_payload["hamiltonian"]["hamiltonian_hash"]
            != self.preparation.hamiltonian_hash
        ):
            raise ValueError("Direct-cost Hamiltonian does not match preparation.")
        if int(request["ld"]) != self.preparation.ld:
            raise ValueError("Direct-cost L_D does not match preparation.")
        rounds = {
            int(item["q_m"]): item["hadamard_interrogation_candidate"]
            for item in self.direct_connection_payload["rounds"]
        }
        if len(rounds) != len(self.direct_connection_payload["rounds"]):
            raise ValueError("Direct-cost source contains duplicate q_m values.")
        if any(not item["feasible"] for item in rounds.values()):
            raise ValueError("Direct-cost source contains an infeasible round.")
        object.__setattr__(self, "_direct_rounds", rounds)

    @property
    def direct_q_values(self) -> tuple[int, ...]:
        return tuple(sorted(self._direct_rounds))

    def evaluate(self, request: RPERoundCostRequest) -> RPERoundCompiledCost:
        q_m = request.specification.q_m
        if q_m not in self._direct_rounds:
            return self.proxy_provider.evaluate(request)
        if request.preparation.preparation_hash != self.preparation.preparation_hash:
            raise ValueError("Preparation does not match the fixed direct-cost source.")
        source_request = self.direct_connection_payload["request"]
        if request.specification.delta_time != float(source_request["delta_time"]):
            raise ValueError("delta_time does not match the fixed cost source.")
        if request.rte_steps_per_occurrence != int(
            source_request["rte_steps_per_occurrence"]
        ):
            raise ValueError("r_m does not match the fixed cost source.")
        if request.finite_taylor_order != int(
            source_request["finite_taylor_order"]
        ):
            raise ValueError("K_m does not match the fixed cost source.")
        source = self._direct_rounds[q_m]
        expected_kind = "monte_carlo_compiled_rpe_hadamard_interrogation_expectation"
        se_kind = "compiled_cost_standard_error"
        fingerprint = _fingerprint(
            {
                "source_validation_fingerprint": self.direct_connection_payload[
                    "validation_fingerprint"
                ],
                "q_m": q_m,
                "scope": RPE_HADAMARD_INTERROGATION_SCOPE,
            }
        )
        return RPERoundCompiledCost(
            cosine_expected_cost=_cost_from_payload(
                source["cosine_expected_cost"],
                self.compiler,
                estimate_kind=expected_kind,
            ),
            sine_expected_cost=_cost_from_payload(
                source["sine_expected_cost"],
                self.compiler,
                estimate_kind=expected_kind,
            ),
            cosine_standard_error=_cost_from_payload(
                source["cosine_standard_error"],
                self.compiler,
                estimate_kind=se_kind,
            ),
            sine_standard_error=_cost_from_payload(
                source["sine_standard_error"],
                self.compiler,
                estimate_kind=se_kind,
            ),
            evaluation_method="fixed_direct_hadamard_compiled_cost",
            classical_sample_count=int(source["classical_cost_sample_count"]),
            circuit_cost_scope=RPE_HADAMARD_INTERROGATION_SCOPE,
            cost_model_fingerprint=fingerprint,
            metadata=(
                ("provider_version", "fixed_direct_and_validated_proxy_v1"),
                (
                    "source_validation_fingerprint",
                    self.direct_connection_payload["validation_fingerprint"],
                ),
                ("source_q_m", q_m),
                ("backend_context_canonical", True),
            ),
        )


def _normalize_beta_profiles(
    profiles: Sequence[tuple[str, float, float, float]],
    *,
    beta_rpe: float,
) -> tuple[tuple[str, float, float, float], ...]:
    normalized = []
    labels = set()
    for label, beta_pf, beta_rte, beta_stat in profiles:
        if not isinstance(label, str) or not label.strip() or label in labels:
            raise ValueError("Beta-profile labels must be unique and non-empty.")
        values = tuple(float(value) for value in (beta_pf, beta_rte, beta_stat))
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Every beta-profile component must be positive.")
        if not math.isclose(sum(values), beta_rpe, rel_tol=0.0, abs_tol=1e-14):
            raise ValueError("Every beta profile must sum to beta_rpe.")
        labels.add(label)
        normalized.append((label.strip(), *values))
    if not normalized:
        raise ValueError("At least one beta profile is required.")
    return tuple(normalized)


def _evaluate_scenario(
    preparation: DFPartialS2Preparation,
    provider: _FixedDirectAndValidatedProxyProvider,
    *,
    q_values: tuple[int, ...],
    delta_time: float,
    beta_rpe: float,
    beta_profile: tuple[str, float, float, float],
    alpha_by_axis: Mapping[tuple[int, str], float],
    pf_model: RPEPFErrorModel,
    rte_steps_per_occurrence: int,
    finite_taylor_order: int,
    rte_seed: int,
    cost_metric: str,
    alpha_policy: str,
) -> dict[str, Any]:
    label, beta_pf, beta_rte, beta_stat = beta_profile
    sampling = RPEHadamardSamplingPolicy(
        rte_trajectory_mode="fresh_iid_per_hadamard_shot",
        independent_bounded_outcomes_within_each_round_axis=True,
    )
    candidates = []
    for q_m in q_values:
        candidate = evaluate_rpe_round_candidate(
            preparation,
            RPERoundSpecification(q_m.bit_length() - 1, delta_time),
            RPEErrorAllocation(
                beta_pf_budget=beta_pf,
                beta_rte_budget=beta_rte,
                beta_stat_budget=beta_stat,
                alpha_cosine=alpha_by_axis[(q_m, "cosine")],
                alpha_sine=alpha_by_axis[(q_m, "sine")],
            ),
            pf_model,
            beta_rpe=beta_rpe,
            rte_steps_per_occurrence=rte_steps_per_occurrence,
            finite_taylor_order=finite_taylor_order,
            cost_metric=cost_metric,
            cost_provider=provider,
            rte_seed=rte_seed,
            hadamard_sampling_policy=sampling,
        )
        candidates.append(candidate)
    total_cost = math.fsum(
        candidate.round_total_cost or math.inf for candidate in candidates
    )
    total_shots = sum(
        (candidate.cosine_shots or 0) + (candidate.sine_shots or 0)
        for candidate in candidates
    )
    max_beta_pf = max(candidate.beta_pf for candidate in candidates)
    finite_beta_rte = [
        candidate.beta_rte
        for candidate in candidates
        if candidate.beta_rte is not None
    ]
    max_beta_rte = max(finite_beta_rte, default=math.inf)
    headroom_pf = beta_pf / max_beta_pf if max_beta_pf > 0.0 else math.inf
    headroom_rte = beta_rte / max_beta_rte if max_beta_rte > 0.0 else math.inf
    round_identity = all(
        candidate.round_total_cost is not None
        and candidate.cosine_shots is not None
        and candidate.sine_shots is not None
        and candidate.cosine_expected_metric is not None
        and candidate.sine_expected_metric is not None
        and math.isclose(
            candidate.round_total_cost,
            candidate.cosine_shots * candidate.cosine_expected_metric
            + candidate.sine_shots * candidate.sine_expected_metric,
            rel_tol=1e-14,
            abs_tol=1e-8,
        )
        for candidate in candidates
    )
    return {
        "scenario_id": f"{label}:{alpha_policy}",
        "beta_profile": label,
        "alpha_policy": alpha_policy,
        "beta_pf_budget": beta_pf,
        "beta_rte_budget": beta_rte,
        "beta_stat_budget": beta_stat,
        "beta_total": beta_pf + beta_rte + beta_stat,
        "alpha_total_allocated": math.fsum(alpha_by_axis.values()),
        "maximum_actual_beta_pf": max_beta_pf,
        "maximum_actual_beta_rte": max_beta_rte,
        "beta_pf_headroom_factor": headroom_pf,
        "beta_rte_headroom_factor": headroom_rte,
        "minimum_phase_headroom_factor": min(headroom_pf, headroom_rte),
        "all_rounds_feasible": all(candidate.feasible for candidate in candidates),
        "all_round_cost_identities_pass": round_identity,
        "total_shots": total_shots,
        "comparison_total_cost": total_cost,
        "rounds": [_candidate_payload(candidate) for candidate in candidates],
    }


def validate_rpe_allocation_sensitivity(
    hamiltonian: DFHamiltonian,
    compiler: CompilerSettings,
    direct_connection_payload: Mapping[str, Any],
    proxy_validation: RPEHadamardCompiledCostProxyValidationResult,
    *,
    ld: int = 3,
    delta_time: float = 0.1,
    q_values: Sequence[int] = (1, 2, 4, 8),
    rte_steps_per_occurrence: int = 4,
    finite_taylor_order: int = 2,
    rte_seed: int = 20260818,
    beta_rpe: float = 0.40,
    beta_profiles: Sequence[tuple[str, float, float, float]] = DEFAULT_BETA_PROFILES,
    alpha_total: float = 0.05,
    pf_coefficient: float = 0.01342567,
    pf_coefficient_source: str = "paper_d6_empirical_surrogate",
    cost_metric: str = "rz_count",
    minimum_provisional_headroom_factor: float = 100.0,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare beta and alpha allocations on fixed validated one-shot costs."""
    started = time.perf_counter()
    if cost_metric not in RPE_COST_METRICS:
        raise ValueError(f"Unsupported cost metric: {cost_metric}.")
    if not math.isfinite(delta_time) or delta_time <= 0.0:
        raise ValueError("delta_time must be finite and positive.")
    if not math.isfinite(alpha_total) or not 0.0 < alpha_total < 1.0:
        raise ValueError("alpha_total must lie strictly in (0, 1).")
    if (
        not math.isfinite(minimum_provisional_headroom_factor)
        or minimum_provisional_headroom_factor <= 1.0
    ):
        raise ValueError("minimum provisional headroom must exceed one.")
    ld_value = require_integer_count(ld, name="ld")
    r_value = require_integer_count(
        rte_steps_per_occurrence,
        name="rte_steps_per_occurrence",
        minimum=1,
    )
    k_value = require_integer_count(finite_taylor_order, name="finite_taylor_order")
    q_grid = tuple(sorted(set(require_integer_count(q, name="q", minimum=1) for q in q_values)))
    if any(q & (q - 1) for q in q_grid):
        raise ValueError("q_values must be powers of two.")
    profiles = _normalize_beta_profiles(beta_profiles, beta_rpe=beta_rpe)
    preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, ld_value),
        identity_policy="extract_identity_phase",
    )
    proxy_provider = ValidatedRPEHadamardCompiledCostProxyProvider(
        validation=proxy_validation,
        compiler=compiler,
    )
    provider = _FixedDirectAndValidatedProxyProvider(
        direct_connection_payload=direct_connection_payload,
        proxy_provider=proxy_provider,
        compiler=compiler,
        preparation=preparation,
    )
    expected_q = tuple(sorted((*provider.direct_q_values, *proxy_provider.validated_q_m_values)))
    if q_grid != expected_q:
        raise ValueError(
            f"q_values={q_grid} must equal the validated source grid {expected_q}."
        )
    source_request = direct_connection_payload["request"]
    for name, observed, expected in (
        ("delta_time", float(source_request["delta_time"]), delta_time),
        ("r_m", int(source_request["rte_steps_per_occurrence"]), r_value),
        ("K_m", int(source_request["finite_taylor_order"]), k_value),
    ):
        if observed != expected:
            raise ValueError(f"{name} differs from the direct-cost source.")

    pf_model = RPEPFErrorModel(
        coefficient=pf_coefficient,
        source=pf_coefficient_source,
        is_rigorous_bound=False,
    )
    uniform_alpha = alpha_total / (2 * len(q_grid))
    scenarios = []
    for profile in profiles:
        uniform_map = {
            (q, axis): uniform_alpha
            for q in q_grid
            for axis in ("cosine", "sine")
        }
        uniform = _evaluate_scenario(
            preparation,
            provider,
            q_values=q_grid,
            delta_time=delta_time,
            beta_rpe=beta_rpe,
            beta_profile=profile,
            alpha_by_axis=uniform_map,
            pf_model=pf_model,
            rte_steps_per_occurrence=r_value,
            finite_taylor_order=k_value,
            rte_seed=rte_seed,
            cost_metric=cost_metric,
            alpha_policy="uniform",
        )
        weights: dict[tuple[int, str], float] = {}
        for round_payload in uniform["rounds"]:
            epsilon_coordinate = float(round_payload["epsilon_coordinate"])
            for axis in ("cosine", "sine"):
                cost = float(round_payload[f"{axis}_expected_metric"])
                weights[(round_payload["q_m"], axis)] = (
                    2.0 * cost / (epsilon_coordinate * epsilon_coordinate)
                )
        weight_sum = math.fsum(weights.values())
        weighted_map = {
            key: alpha_total * weight / weight_sum
            for key, weight in weights.items()
        }
        weighted = _evaluate_scenario(
            preparation,
            provider,
            q_values=q_grid,
            delta_time=delta_time,
            beta_rpe=beta_rpe,
            beta_profile=profile,
            alpha_by_axis=weighted_map,
            pf_model=pf_model,
            rte_steps_per_occurrence=r_value,
            finite_taylor_order=k_value,
            rte_seed=rte_seed,
            cost_metric=cost_metric,
            alpha_policy="cost_sensitivity_weighted",
        )
        weighted["alpha_weight_formula"] = (
            "alpha_m_b=alpha_total*w_m_b/sum(w), "
            "w_m_b=2*g_m_b/epsilon_coordinate_m^2"
        )
        weighted["relative_cost_reduction_vs_same_beta_uniform"] = float(
            (uniform["comparison_total_cost"] - weighted["comparison_total_cost"])
            / uniform["comparison_total_cost"]
        )
        scenarios.extend((uniform, weighted))

    eligible = [
        scenario
        for scenario in scenarios
        if scenario["alpha_policy"] == "cost_sensitivity_weighted"
        and scenario["all_rounds_feasible"]
        and scenario["minimum_phase_headroom_factor"]
        >= minimum_provisional_headroom_factor
    ]
    if not eligible:
        raise RuntimeError("No beta profile satisfies the provisional guard.")
    selected = min(eligible, key=lambda item: item["comparison_total_cost"])
    baseline = next(
        item
        for item in scenarios
        if item["scenario_id"] == "current_provisional:uniform"
    )
    selected_vs_baseline = float(
        (baseline["comparison_total_cost"] - selected["comparison_total_cost"])
        / baseline["comparison_total_cost"]
    )
    weighted_never_worse = all(
        next(
            item
            for item in scenarios
            if item["beta_profile"] == label and item["alpha_policy"] == "cost_sensitivity_weighted"
        )["comparison_total_cost"]
        <= next(
            item
            for item in scenarios
            if item["beta_profile"] == label and item["alpha_policy"] == "uniform"
        )["comparison_total_cost"]
        for label, *_ in profiles
    )
    checks = {
        "direct_source_validation_passed": bool(
            direct_connection_payload["summary"]["overall_pass"]
        ),
        "proxy_holdout_validation_passed": proxy_validation.overall_pass,
        "source_compiler_fingerprints_match": bool(
            compiler_settings_hash(compiler)
            == proxy_validation.proxy.compiler_settings_fingerprint
        ),
        "validated_q_grid_is_exact": q_grid == expected_q,
        "all_scenarios_feasible": all(
            item["all_rounds_feasible"] for item in scenarios
        ),
        "all_beta_sums_equal_beta_rpe": all(
            math.isclose(item["beta_total"], beta_rpe, rel_tol=0.0, abs_tol=1e-14)
            for item in scenarios
        ),
        "all_alpha_sums_equal_alpha_total": all(
            math.isclose(
                item["alpha_total_allocated"],
                alpha_total,
                rel_tol=0.0,
                abs_tol=1e-14,
            )
            for item in scenarios
        ),
        "all_round_cost_identities_pass": all(
            item["all_round_cost_identities_pass"] for item in scenarios
        ),
        "weighted_alpha_never_worse_than_uniform": weighted_never_worse,
        "selected_profile_meets_explicit_headroom_guard": bool(
            selected["minimum_phase_headroom_factor"]
            >= minimum_provisional_headroom_factor
        ),
        "selected_profile_reduces_cost_vs_current_uniform": bool(
            selected_vs_baseline > 0.0
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": RPE_ALLOCATION_SENSITIVITY_SCHEMA_VERSION,
        "method": RPE_ALLOCATION_SENSITIVITY_METHOD,
        "scope": {
            "validated_q_values": list(q_grid),
            "one_shot_costs_recompiled_during_sweep": False,
            "q1_q2_q4_cost_source": "fixed_direct_hadamard_compiled_cost",
            "q8_cost_source": "unused_holdout_validated_affine_proxy",
            "state_preparation_included": False,
            "physical_q8_signal_or_phase_evaluated": False,
            "backend_execution_included": False,
            "full_rpe_phase_reconstruction_included": False,
            "final_total_cost_evaluation_performed": False,
            "interpretation": "allocation_sensitivity_diagnostic_only",
        },
        "system": {
            "model": "H4 linear chain",
            "geometry_angstrom": 1.0,
            "basis": "STO-3G",
            "num_system_qubits": hamiltonian.n_qubits,
            "df_rank": hamiltonian.n_blocks,
            "ld": ld_value,
            "hamiltonian_hash": preparation.hamiltonian_hash,
            "partition_hash": preparation.partition_hash,
            "preparation_hash": preparation.preparation_hash,
        },
        "configuration": {
            "delta_time": delta_time,
            "q_values": list(q_grid),
            "rte_steps_per_occurrence": r_value,
            "finite_taylor_order": k_value,
            "rte_seed": rte_seed,
            "beta_rpe": beta_rpe,
            "beta_profiles": [
                {
                    "label": label,
                    "beta_pf": beta_pf,
                    "beta_rte": beta_rte,
                    "beta_stat": beta_stat,
                }
                for label, beta_pf, beta_rte, beta_stat in profiles
            ],
            "alpha_total": alpha_total,
            "alpha_policies": ["uniform", "cost_sensitivity_weighted"],
            "pf_coefficient": pf_coefficient,
            "pf_coefficient_source": pf_coefficient_source,
            "pf_coefficient_is_rigorous_bound": False,
            "cost_metric": cost_metric,
            "minimum_provisional_headroom_factor": (
                minimum_provisional_headroom_factor
            ),
            "compiler": _compiler_payload(compiler),
        },
        "source_evidence": {
            "direct_connection_validation_fingerprint": (
                direct_connection_payload["validation_fingerprint"]
            ),
            "proxy_validation_fingerprint": proxy_validation.validation_fingerprint,
            "proxy_fit_fingerprint": proxy_validation.proxy.fit_fingerprint,
            "proxy_cost_model_fingerprint": proxy_provider.cost_model_fingerprint,
        },
        "scenarios": scenarios,
        "selection": {
            "rule": (
                "minimum comparison cost among cost-sensitivity-weighted profiles "
                "whose PF and RTE allocated budgets each exceed the maximum "
                "representative actual contribution by the explicit headroom factor"
            ),
            "selected_scenario_id": selected["scenario_id"],
            "selected_beta_profile": selected["beta_profile"],
            "selected_alpha_policy": selected["alpha_policy"],
            "selected_comparison_total_cost": selected["comparison_total_cost"],
            "selected_total_shots": selected["total_shots"],
            "minimum_phase_headroom_factor": selected[
                "minimum_phase_headroom_factor"
            ],
            "relative_cost_reduction_vs_current_uniform": selected_vs_baseline,
            "status": "provisional_local_choice_not_global_optimum",
        },
        "summary": {
            "overall_pass": all(checks.values()),
            "checks": checks,
            "scenario_count": len(scenarios),
            "baseline_scenario_id": baseline["scenario_id"],
            "baseline_comparison_total_cost": baseline["comparison_total_cost"],
            "selected_scenario_id": selected["scenario_id"],
            "selected_comparison_total_cost": selected["comparison_total_cost"],
            "selected_relative_cost_reduction": selected_vs_baseline,
            "beta_allocation_is_cost_sensitive": bool(
                selected_vs_baseline >= 0.10
            ),
            "alpha_reallocation_is_secondary_at_selected_beta": bool(
                selected["relative_cost_reduction_vs_same_beta_uniform"] < 0.10
            ),
        },
        "provenance": dict(provenance or {}),
        "performance": {"elapsed_seconds": time.perf_counter() - started},
    }
    payload["content_fingerprint"] = _fingerprint(payload)
    validate_rpe_allocation_sensitivity_payload(payload)
    return payload


def validate_rpe_allocation_sensitivity_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != RPE_ALLOCATION_SENSITIVITY_SCHEMA_VERSION:
        raise ValueError("Unsupported allocation-sensitivity schema.")
    if payload.get("method") != RPE_ALLOCATION_SENSITIVITY_METHOD:
        raise ValueError("Unsupported allocation-sensitivity method.")
    stored = payload.get("content_fingerprint")
    if not isinstance(stored, str):
        raise ValueError("Missing content_fingerprint.")
    unsigned = dict(payload)
    del unsigned["content_fingerprint"]
    if stored != _fingerprint(unsigned):
        raise ValueError("content_fingerprint does not match the payload.")
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Allocation sensitivity requires scenarios.")
    if payload["summary"]["scenario_count"] != len(scenarios):
        raise ValueError("Scenario count mismatch.")
    checks = payload["summary"]["checks"]
    if payload["summary"]["overall_pass"] != all(checks.values()):
        raise ValueError("Summary pass value does not match its checks.")
    selected_id = payload["selection"]["selected_scenario_id"]
    if sum(item["scenario_id"] == selected_id for item in scenarios) != 1:
        raise ValueError("Selected scenario is missing or duplicated.")
    if payload["scope"]["final_total_cost_evaluation_performed"] is not False:
        raise ValueError("Sensitivity artifact cannot claim a final total cost.")


def write_rpe_allocation_sensitivity_validation(
    payload: Mapping[str, Any], path: str | Path
) -> None:
    validate_rpe_allocation_sensitivity_payload(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
