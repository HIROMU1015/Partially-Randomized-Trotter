"""Limited four-round accounting with a validated mixed Hadamard cost source.

This is an accounting and short-round statistical check, not full RPE phase
reconstruction or a final optimized total-cost estimate.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from .df_hamiltonian import DFHamiltonian
from .df_partial_randomized_pf import split_df_hamiltonian_by_ld
from .df_partial_s2 import prepare_df_partial_s2
from .rpe_allocation_sensitivity_validation import (
    _FixedDirectAndValidatedProxyProvider,
    _compiler_payload,
    validate_rpe_allocation_sensitivity_payload,
)
from .rpe_hadamard_compiled_cost_proxy import (
    RPEHadamardCompiledCostProxyValidationResult,
)
from .rpe_hadamard_failure_validation import (
    _axis_distribution,
    _round_exact_phase_probability,
    validate_rpe_hadamard_failure_payload,
)
from .rpe_hadamard_interrogation import RPE_HADAMARD_INTERROGATION_SCOPE
from .rpe_hadamard_validated_proxy_provider import (
    ValidatedRPEHadamardCompiledCostProxyProvider,
)
from .rpe_resource_accounting import (
    RPEErrorAllocation,
    RPEHadamardSamplingPolicy,
    RPEPFErrorModel,
    RPERoundCompiledCost,
    RPERoundCostRequest,
    RPERoundSpecification,
    build_rpe_resource_summary,
    evaluate_rpe_round_candidate,
)
from .rpe_round_cost_connection_validation import (
    validate_rpe_round_cost_connection_payload,
)
from .rte import CompilerSettings
from .rte_compiled_cost import compiler_settings_hash


SCHEMA_VERSION = "rpe_four_round_accounting_validation_v1"
METHOD = "selected_allocation_mixed_cost_summary_and_short_round_binomial_v1"
PROVIDER_VERSION = "fixed_direct_q1_q2_q4_and_validated_proxy_q8_v1"


def _fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _same_number(left: float | int | None, right: float | int | None) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=1e-13, abs_tol=1e-11)


@dataclass(frozen=True)
class _MixedValidatedCostProvider:
    """One portfolio fingerprint, with each q's original source still recorded."""

    underlying: _FixedDirectAndValidatedProxyProvider
    portfolio_fingerprint: str

    def evaluate(self, request: RPERoundCostRequest) -> RPERoundCompiledCost:
        result = self.underlying.evaluate(request)
        metadata = dict(result.metadata)
        source_version = metadata.pop("provider_version", None)
        metadata.update(
            {
                "provider_version": PROVIDER_VERSION,
                "source_provider_version": source_version,
                "source_cost_model_fingerprint": result.cost_model_fingerprint,
                "portfolio_q_m": request.specification.q_m,
            }
        )
        return replace(
            result,
            cost_model_fingerprint=self.portfolio_fingerprint,
            metadata=tuple(sorted(metadata.items())),
        )


def validate_rpe_four_round_accounting(
    hamiltonian: DFHamiltonian,
    compiler: CompilerSettings,
    allocation_payload: Mapping[str, Any],
    direct_connection_payload: Mapping[str, Any],
    proxy_validation: RPEHadamardCompiledCostProxyValidationResult,
    previous_failure_payload: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Re-evaluate the selected four rounds and audit their limited sum."""
    validate_rpe_allocation_sensitivity_payload(allocation_payload)
    validate_rpe_round_cost_connection_payload(direct_connection_payload)
    validate_rpe_hadamard_failure_payload(previous_failure_payload)
    if not allocation_payload["summary"]["overall_pass"]:
        raise ValueError("Allocation source did not pass its validation.")
    if not previous_failure_payload["summary"]["overall_pass"]:
        raise ValueError("Previous short-round failure source did not pass.")
    if not proxy_validation.overall_pass:
        raise ValueError("The q=8 proxy holdout did not pass.")

    system = allocation_payload["system"]
    config = allocation_payload["configuration"]
    q_grid = tuple(int(q) for q in config["q_values"])
    if q_grid != (1, 2, 4, 8):
        raise ValueError("This limited validation requires q=(1,2,4,8).")
    if system["num_system_qubits"] != hamiltonian.n_qubits:
        raise ValueError("System size differs from the allocation source.")
    if system["df_rank"] != hamiltonian.n_blocks:
        raise ValueError("DF rank differs from the allocation source.")
    if config["compiler"] != _compiler_payload(compiler):
        raise ValueError("Compiler context differs from the allocation source.")
    if previous_failure_payload["hamiltonian"]["hamiltonian_hash"] != system["hamiltonian_hash"]:
        raise ValueError("Physical-signal Hamiltonian differs from the allocation source.")
    prior_request = previous_failure_payload["request"]
    if (
        tuple(prior_request["q_values"]) != (1, 2, 4)
        or prior_request["ld"] != system["ld"]
        or prior_request["delta_time"] != config["delta_time"]
        or prior_request["rte_steps_per_occurrence"] != config["rte_steps_per_occurrence"]
        or prior_request["finite_taylor_order"] != config["finite_taylor_order"]
        or prior_request["beta_rpe"] != config["beta_rpe"]
    ):
        raise ValueError("Physical-signal conditions differ from the selected rounds.")

    preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, int(system["ld"])),
        identity_policy="extract_identity_phase",
    )
    if (
        preparation.hamiltonian_hash != system["hamiltonian_hash"]
        or preparation.partition_hash != system["partition_hash"]
        or preparation.preparation_hash != system["preparation_hash"]
    ):
        raise ValueError("DF preparation differs from the allocation source.")

    proxy_provider = ValidatedRPEHadamardCompiledCostProxyProvider(
        proxy_validation, compiler
    )
    underlying = _FixedDirectAndValidatedProxyProvider(
        direct_connection_payload,
        proxy_provider,
        compiler,
        preparation,
    )
    if underlying.direct_q_values != (1, 2, 4) or proxy_provider.validated_q_m_values != (8,):
        raise ValueError("Mixed cost sources do not cover precisely q=(1,2,4,8).")
    expected_sources = {
        "direct_connection_validation_fingerprint": direct_connection_payload[
            "validation_fingerprint"
        ],
        "proxy_validation_fingerprint": proxy_validation.validation_fingerprint,
        "proxy_fit_fingerprint": proxy_validation.proxy.fit_fingerprint,
        "proxy_cost_model_fingerprint": proxy_provider.cost_model_fingerprint,
    }
    if allocation_payload["source_evidence"] != expected_sources:
        raise ValueError("Allocation input fingerprints differ from live sources.")
    portfolio_fingerprint = _fingerprint(
        {
            "provider_version": PROVIDER_VERSION,
            "direct_validation_fingerprint": direct_connection_payload[
                "validation_fingerprint"
            ],
            "proxy_validation_fingerprint": proxy_validation.validation_fingerprint,
            "proxy_cost_model_fingerprint": proxy_provider.cost_model_fingerprint,
            "compiler_fingerprint": compiler_settings_hash(compiler),
            "q_routes": {"direct": [1, 2, 4], "validated_proxy": [8]},
            "scope": RPE_HADAMARD_INTERROGATION_SCOPE,
        }
    )
    provider = _MixedValidatedCostProvider(underlying, portfolio_fingerprint)

    selected_id = allocation_payload["selection"]["selected_scenario_id"]
    selected = next(
        item for item in allocation_payload["scenarios"]
        if item["scenario_id"] == selected_id
    )
    if tuple(int(item["q_m"]) for item in selected["rounds"]) != q_grid:
        raise ValueError("Selected scenario rounds do not match the q grid.")
    pf_model = RPEPFErrorModel(
        coefficient=float(config["pf_coefficient"]),
        source=config["pf_coefficient_source"],
        is_rigorous_bound=False,
    )
    sampling = RPEHadamardSamplingPolicy(
        rte_trajectory_mode="fresh_iid_per_hadamard_shot",
        independent_bounded_outcomes_within_each_round_axis=True,
    )
    candidates = []
    agreement = []
    for source_round in selected["rounds"]:
        q = int(source_round["q_m"])
        candidate = evaluate_rpe_round_candidate(
            preparation,
            RPERoundSpecification(q.bit_length() - 1, float(config["delta_time"])),
            RPEErrorAllocation(
                beta_pf_budget=float(selected["beta_pf_budget"]),
                beta_rte_budget=float(selected["beta_rte_budget"]),
                beta_stat_budget=float(selected["beta_stat_budget"]),
                alpha_cosine=float(source_round["alpha_cosine"]),
                alpha_sine=float(source_round["alpha_sine"]),
            ),
            pf_model,
            beta_rpe=float(config["beta_rpe"]),
            rte_steps_per_occurrence=int(config["rte_steps_per_occurrence"]),
            finite_taylor_order=int(config["finite_taylor_order"]),
            cost_metric=config["cost_metric"],
            cost_provider=provider,
            rte_seed=int(config["rte_seed"]),
            hadamard_sampling_policy=sampling,
        )
        candidates.append(candidate)
        agreement.append(
            candidate.feasible == source_round["feasible"]
            and candidate.cosine_shots == source_round["cosine_shots"]
            and candidate.sine_shots == source_round["sine_shots"]
            and all(
                _same_number(getattr(candidate, attribute), source_round[key])
                for attribute, key in (
                    ("epsilon_coordinate", "epsilon_coordinate"),
                    ("rho_observed_lower_bound", "rho_observed_lower_bound"),
                    ("cosine_expected_metric", "cosine_expected_metric"),
                    ("sine_expected_metric", "sine_expected_metric"),
                    ("round_total_cost", "round_total_cost"),
                )
            )
        )
    summary = build_rpe_resource_summary(
        candidates,
        total_alpha_budget=float(config["alpha_total"]),
        cost_metric=config["cost_metric"],
    )

    previous_signals = {
        int(item["q_m"]): item
        for item in previous_failure_payload["exact_binomial"]["rounds"]
    }
    exact_rounds = []
    for candidate in candidates[:3]:
        source = previous_signals[candidate.q_m]
        signal_data = source["attenuated_event_mean_signal"]
        signal = complex(float(signal_data["real"]), float(signal_data["imag"]))
        epsilon = candidate.epsilon_coordinate
        if epsilon is None or candidate.cosine_shots is None or candidate.sine_shots is None:
            raise ValueError("Selected round lacks finite statistical resources.")
        cosine = _axis_distribution(signal.real, candidate.cosine_shots, epsilon)
        sine = _axis_distribution(signal.imag, candidate.sine_shots, epsilon)
        phase_probability, implication_violations = _round_exact_phase_probability(
            cosine, sine, signal, candidate.allocation.beta_stat_budget
        )
        exact_rounds.append(
            {
                "q_m": candidate.q_m,
                "signal_radius": abs(signal),
                "cosine_shots": candidate.cosine_shots,
                "sine_shots": candidate.sine_shots,
                "epsilon_coordinate": epsilon,
                "cosine_coordinate_failure_probability": cosine[
                    "exact_coordinate_failure_probability"
                ],
                "sine_coordinate_failure_probability": sine[
                    "exact_coordinate_failure_probability"
                ],
                "exact_phase_failure_probability": phase_probability,
                "coordinate_success_but_phase_failure_grid_points": implication_violations,
                "alpha_cosine": candidate.allocation.alpha_cosine,
                "alpha_sine": candidate.allocation.alpha_sine,
            }
        )

    exact_coordinate_probability = 1.0 - math.prod(
        (1.0 - item["cosine_coordinate_failure_probability"])
        * (1.0 - item["sine_coordinate_failure_probability"])
        for item in exact_rounds
    )
    exact_phase_probability = 1.0 - math.prod(
        1.0 - item["exact_phase_failure_probability"] for item in exact_rounds
    )
    short_alpha = math.fsum(
        item[axis]
        for item in exact_rounds
        for axis in ("alpha_cosine", "alpha_sine")
    )
    manual_cost = math.fsum(
        candidate.cosine_shots * candidate.cosine_expected_metric
        + candidate.sine_shots * candidate.sine_expected_metric
        for candidate in candidates
    )
    checks = {
        "selected_rounds_reproduce_prior_sensitivity_result": all(agreement),
        "all_four_rounds_feasible": all(item.feasible for item in candidates),
        "same_hadamard_scope_all_rounds": summary.circuit_cost_scope
        == RPE_HADAMARD_INTERROGATION_SCOPE,
        "mixed_cost_portfolio_has_validated_q_routes": all(
            item.cost_evaluation_method
            == ("holdout_validated_affine_proxy" if item.q_m == 8 else "fixed_direct_hadamard_compiled_cost")
            for item in candidates
        ),
        "union_bound_within_total_alpha": summary.union_bound_satisfied,
        "manual_shot_times_one_shot_cost_matches_summary": _same_number(
            manual_cost, summary.total_cost
        ),
        "summary_matches_prior_comparison_sum": _same_number(
            summary.total_cost, selected["comparison_total_cost"]
        ),
        "short_round_exact_coordinate_failures_within_allocations": all(
            item["cosine_coordinate_failure_probability"] <= item["alpha_cosine"]
            and item["sine_coordinate_failure_probability"] <= item["alpha_sine"]
            for item in exact_rounds
        ),
        "short_round_exact_phase_failures_within_axis_unions": all(
            item["exact_phase_failure_probability"]
            <= item["alpha_cosine"] + item["alpha_sine"]
            for item in exact_rounds
        ),
        "short_round_coordinate_success_implies_phase_success": all(
            item["coordinate_success_but_phase_failure_grid_points"] == 0
            for item in exact_rounds
        ),
        "short_round_exact_failures_within_short_round_alpha": (
            exact_coordinate_probability <= short_alpha
            and exact_phase_probability <= short_alpha
        ),
        "not_a_certified_final_result": summary.guarantee_status != "certified",
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "scope": {
            "description": "limited_q1_q2_q4_q8_accounting_diagnostic_only",
            "state_preparation_included": False,
            "q8_physical_signal_or_phase_evaluated": False,
            "q8_failure_check": "analytic_hoeffding_axis_bounds_only",
            "new_explicit_fresh_iid_trajectory_batch_performed": False,
            "full_rpe_phase_reconstruction_performed": False,
            "final_total_cost_evaluation_performed": False,
            "backend_execution_performed": False,
        },
        "system": dict(system),
        "configuration": {
            "selected_scenario_id": selected_id,
            "beta_pf_budget": selected["beta_pf_budget"],
            "beta_rte_budget": selected["beta_rte_budget"],
            "beta_stat_budget": selected["beta_stat_budget"],
            "beta_rpe": config["beta_rpe"],
            "alpha_total": config["alpha_total"],
            "delta_time": config["delta_time"],
            "rte_steps_per_occurrence": config["rte_steps_per_occurrence"],
            "finite_taylor_order": config["finite_taylor_order"],
            "cost_metric": config["cost_metric"],
            "compiler": config["compiler"],
            "mixed_cost_portfolio_fingerprint": portfolio_fingerprint,
        },
        "source_evidence": {
            "allocation_content_fingerprint": allocation_payload["content_fingerprint"],
            "direct_connection_validation_fingerprint": direct_connection_payload[
                "validation_fingerprint"
            ],
            "proxy_validation_fingerprint": proxy_validation.validation_fingerprint,
            "previous_failure_validation_fingerprint": previous_failure_payload[
                "validation_fingerprint"
            ],
        },
        "rounds": [
            {
                "m": item.m,
                "q_m": item.q_m,
                "alpha_cosine": item.allocation.alpha_cosine,
                "alpha_sine": item.allocation.alpha_sine,
                "cosine_shots": item.cosine_shots,
                "sine_shots": item.sine_shots,
                "cosine_one_shot_rz": item.cosine_expected_metric,
                "sine_one_shot_rz": item.sine_expected_metric,
                "round_rz_cost": item.round_total_cost,
                "cost_evaluation_method": item.cost_evaluation_method,
                "source_cost_model_fingerprint": dict(item.cost_metadata)[
                    "source_cost_model_fingerprint"
                ],
                "guarantee_status": item.guarantee_status,
            }
            for item in summary.rounds
        ],
        "limited_aggregation": {
            "total_quantum_shots": sum(
                item.cosine_shots + item.sine_shots for item in candidates
            ),
            "total_rz_cost": summary.total_cost,
            "manual_recomputed_rz_cost": manual_cost,
            "total_axis_alpha_union_bound": summary.total_failure_probability_bound,
            "total_alpha_budget": summary.total_alpha_budget,
            "union_bound_satisfied": summary.union_bound_satisfied,
            "guarantee_status": summary.guarantee_status,
            "certification_reasons": list(summary.certification_reasons),
            "cost_scope": summary.circuit_cost_scope,
            "interpretation": "fixed_condition_four_round_accounting_not_final_total_cost",
        },
        "short_round_exact_binomial": {
            "q_values": [1, 2, 4],
            "reused_physical_signal_from_previous_validation": True,
            "new_beta_stat_and_axis_shots_applied": True,
            "rounds": exact_rounds,
            "combined_coordinate_failure_probability": exact_coordinate_probability,
            "combined_phase_failure_probability": exact_phase_probability,
            "allocated_short_round_alpha": short_alpha,
        },
        "summary": {"overall_pass": all(checks.values()), "checks": checks},
        "provenance": dict(provenance or {}),
    }
    payload["content_fingerprint"] = _fingerprint(payload)
    validate_rpe_four_round_accounting_payload(payload)
    return payload


def validate_rpe_four_round_accounting_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("method") != METHOD:
        raise ValueError("Unsupported four-round accounting artifact.")
    unsigned = dict(payload)
    stored = unsigned.pop("content_fingerprint", None)
    if stored != _fingerprint(unsigned):
        raise ValueError("Four-round accounting content_fingerprint mismatch.")
    if payload["summary"]["overall_pass"] != all(payload["summary"]["checks"].values()):
        raise ValueError("Four-round accounting summary/checks mismatch.")
    if payload["scope"]["final_total_cost_evaluation_performed"] is not False:
        raise ValueError("Limited aggregation cannot claim a final total cost.")


def write_rpe_four_round_accounting(payload: Mapping[str, Any], path: str | Path) -> None:
    validate_rpe_four_round_accounting_payload(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
