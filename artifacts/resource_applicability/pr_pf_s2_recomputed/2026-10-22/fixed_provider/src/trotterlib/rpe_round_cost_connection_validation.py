"""Validate the short-round signal-to-shot-to-compiled-cost connection.

This module deliberately stops at individual short RPE rounds.  It checks that
the finite-RTE signal validation, resource-accounting shot formula, and direct
compiled-cost providers use compatible conventions.  It does not aggregate a
full RPE experiment or claim a final total cost.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .df_hamiltonian import DFHamiltonian, PhysicalSector
from .df_partial_randomized_pf import (
    df_hamiltonian_hash,
    split_df_hamiltonian_by_ld,
)
from .df_partial_s2 import prepare_df_partial_s2
from .df_rpe_hadamard_compiled_cost import DFRPEHadamardCompiledCostProvider
from .df_rpe_resource import DFLevel5RCompiledCostProvider
from .finite_rte_signal_validation import validate_finite_rte_signals
from .rpe_resource_accounting import (
    RPE_COST_METRICS,
    RPEErrorAllocation,
    RPEHadamardSamplingPolicy,
    RPEPFErrorModel,
    RPERoundCandidate,
    RPERoundSpecification,
    circuit_cost_metric,
    evaluate_rpe_round_candidate,
)
from .rte import CircuitCost, CompilerSettings, require_integer_count
from .rte_compiled_cost import TranspiledCircuitCostCache


RPE_ROUND_COST_CONNECTION_SCHEMA_VERSION = (
    "rpe_round_cost_connection_validation_v1"
)
RPE_ROUND_COST_CONNECTION_METHOD = (
    "direct_short_round_signal_shot_and_scope_connection_v1"
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


def _cost_payload(cost: CircuitCost | None) -> dict[str, float] | None:
    if cost is None:
        return None
    return {
        metric: float(circuit_cost_metric(cost, metric))
        for metric in RPE_COST_METRICS
    }


def _relative_difference(left: float, right: float) -> float | None:
    if right == 0.0:
        return None
    return float(abs(left - right) / abs(right))


def _candidate_payload(candidate: RPERoundCandidate) -> dict[str, Any]:
    return {
        "feasible": candidate.feasible,
        "infeasible_reasons": list(candidate.infeasibility_reasons),
        "guarantee_status": candidate.guarantee_status,
        "epsilon_z": candidate.epsilon_z,
        "attenuation": candidate.attenuation,
        "rho_observed_lower_bound": candidate.rho_observed_lower_bound,
        "epsilon_coordinate": candidate.epsilon_coordinate,
        "cosine_shots": candidate.cosine_shots,
        "sine_shots": candidate.sine_shots,
        "cosine_expected_cost": _cost_payload(candidate.cosine_expected_cost),
        "sine_expected_cost": _cost_payload(candidate.sine_expected_cost),
        "cosine_standard_error": _cost_payload(candidate.cosine_standard_error),
        "sine_standard_error": _cost_payload(candidate.sine_standard_error),
        "round_total_cost": candidate.round_total_cost,
        "cost_metric": candidate.cost_metric,
        "cost_evaluation_method": candidate.cost_evaluation_method,
        "classical_cost_sample_count": candidate.classical_cost_sample_count,
        "circuit_cost_scope": candidate.circuit_cost_scope,
        "cost_model_fingerprint": candidate.cost_model_fingerprint,
        "cost_metadata": dict(candidate.cost_metadata),
    }


def _physical_state_result(point: Mapping[str, Any]) -> Mapping[str, Any]:
    matches = [
        item
        for item in point["state_results"]
        if item["state_label"] == "physical_df_ground_state"
    ]
    if len(matches) != 1:
        raise RuntimeError("Expected one physical ground-state signal result.")
    return matches[0]


def validate_rpe_round_cost_connection(
    hamiltonian: DFHamiltonian,
    sector: PhysicalSector,
    compiler: CompilerSettings,
    *,
    ld: int,
    delta_time: float,
    q_values: Sequence[int] = (1, 2, 4),
    rte_steps_per_occurrence: int = 4,
    finite_taylor_order: int = 2,
    compiled_cost_sample_count: int = 8,
    signal_seed: int = 20260818,
    compiled_cost_seed: int = 20260901,
    beta_rpe: float = 0.40,
    beta_pf_budget: float = 0.08,
    beta_rte_budget: float = 0.08,
    beta_stat_budget: float = 0.24,
    alpha_total: float = 0.05,
    pf_coefficient: float = 0.01342567,
    pf_coefficient_source: str = "paper_d6_empirical_surrogate",
    cost_metric: str = "rz_count",
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the representative direct short-round connection validation."""
    started = time.perf_counter()
    if hamiltonian.n_qubits != sector.n_qubits:
        raise ValueError("Hamiltonian and sector n_qubits differ.")
    ld_value = require_integer_count(ld, name="ld")
    r_value = require_integer_count(
        rte_steps_per_occurrence,
        name="rte_steps_per_occurrence",
        minimum=1,
    )
    k_value = require_integer_count(finite_taylor_order, name="finite_taylor_order")
    if k_value % 2:
        raise ValueError("finite_taylor_order must be non-negative and even.")
    sample_count = require_integer_count(
        compiled_cost_sample_count,
        name="compiled_cost_sample_count",
        minimum=2,
    )
    q_grid = tuple(sorted(set(require_integer_count(q, name="q", minimum=1) for q in q_values)))
    if not q_grid or any(q > 4 or q & (q - 1) for q in q_grid):
        raise ValueError("q_values must be a non-empty subset of (1, 2, 4).")
    if not math.isfinite(delta_time) or delta_time <= 0.0:
        raise ValueError("delta_time must be finite and positive.")
    if cost_metric not in RPE_COST_METRICS:
        raise ValueError(f"Unsupported cost metric: {cost_metric}.")
    if not math.isfinite(alpha_total) or not 0.0 < alpha_total < 1.0:
        raise ValueError("alpha_total must lie strictly in (0, 1).")

    alpha_axis = float(alpha_total / (2 * len(q_grid)))
    allocation = RPEErrorAllocation(
        beta_pf_budget=beta_pf_budget,
        beta_rte_budget=beta_rte_budget,
        beta_stat_budget=beta_stat_budget,
        alpha_cosine=alpha_axis,
        alpha_sine=alpha_axis,
    )
    pf_model = RPEPFErrorModel(
        coefficient=pf_coefficient,
        source=pf_coefficient_source,
        is_rigorous_bound=False,
    )
    sampling_policy = RPEHadamardSamplingPolicy(
        rte_trajectory_mode="fresh_iid_per_hadamard_shot",
        independent_bounded_outcomes_within_each_round_axis=True,
    )

    signal_payload = validate_finite_rte_signals(
        hamiltonian,
        sector,
        ld=ld_value,
        delta_time=delta_time,
        q_values=q_grid,
        rte_step_values=(r_value,),
        finite_taylor_orders=(k_value,),
        beta_rpe=beta_rpe,
        beta_pf_budget=beta_pf_budget,
        beta_rte_budget=beta_rte_budget,
        beta_stat_budget=beta_stat_budget,
        alpha_total=alpha_total,
        seed=signal_seed,
        provenance={
            "parent_validation": RPE_ROUND_COST_CONNECTION_METHOD,
            "outer_provenance": dict(provenance or {}),
        },
    )
    preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, ld_value),
        identity_policy="extract_identity_phase",
    )

    evolution_cache = TranspiledCircuitCostCache()
    hadamard_cache = TranspiledCircuitCostCache()
    evolution_provider = DFLevel5RCompiledCostProvider(
        compiler=compiler,
        evaluation_method="monte_carlo",
        sample_count=sample_count,
        seed=compiled_cost_seed,
        maximum_samples=sample_count,
        maximum_retained_provenance_records=sample_count,
        cache=evolution_cache,
    )
    hadamard_provider = DFRPEHadamardCompiledCostProvider(
        compiler=compiler,
        evaluation_method="monte_carlo",
        sample_count=sample_count,
        seed=compiled_cost_seed,
        maximum_samples=sample_count,
        maximum_retained_trajectory_records=sample_count,
        cache=hadamard_cache,
    )

    signal_points = {int(point["q_m"]): point for point in signal_payload["points"]}
    rounds: list[dict[str, Any]] = []
    for q_m in q_grid:
        specification = RPERoundSpecification(int(math.log2(q_m)), delta_time)
        common = {
            "beta_rpe": beta_rpe,
            "rte_steps_per_occurrence": r_value,
            "finite_taylor_order": k_value,
            "cost_metric": cost_metric,
            "rte_seed": signal_seed,
            "hadamard_sampling_policy": sampling_policy,
        }
        evolution_candidate = evaluate_rpe_round_candidate(
            preparation,
            specification,
            allocation,
            pf_model,
            cost_provider=evolution_provider,
            **common,
        )
        hadamard_candidate = evaluate_rpe_round_candidate(
            preparation,
            specification,
            allocation,
            pf_model,
            cost_provider=hadamard_provider,
            **common,
        )
        point = signal_points[q_m]
        physical = _physical_state_result(point)
        unit_shots = physical["provisional_shots_unit_radius_per_axis"]
        reference_shots = physical[
            "provisional_shots_reference_radius_per_axis"
        ]
        if unit_shots is None or reference_shots is None:
            reference_round_cost = None
            radius_cost_relative_difference = None
        else:
            cosine_metric = circuit_cost_metric(
                hadamard_candidate.cosine_expected_cost,
                cost_metric,
            )
            sine_metric = circuit_cost_metric(
                hadamard_candidate.sine_expected_cost,
                cost_metric,
            )
            reference_round_cost = float(
                reference_shots * cosine_metric + reference_shots * sine_metric
            )
            radius_cost_relative_difference = _relative_difference(
                reference_round_cost,
                float(hadamard_candidate.round_total_cost),
            )

        overhead: dict[str, Any] = {}
        for metric in RPE_COST_METRICS:
            base = circuit_cost_metric(evolution_candidate.cosine_expected_cost, metric)
            cosine = circuit_cost_metric(
                hadamard_candidate.cosine_expected_cost,
                metric,
            )
            sine = circuit_cost_metric(
                hadamard_candidate.sine_expected_cost,
                metric,
            )
            overhead[metric] = {
                "time_evolution_subcircuit": float(base),
                "cosine_hadamard_interrogation": float(cosine),
                "sine_hadamard_interrogation": float(sine),
                "cosine_minus_time_evolution": float(cosine - base),
                "sine_minus_time_evolution": float(sine - base),
                "cosine_relative_difference": _relative_difference(cosine, base),
                "sine_relative_difference": _relative_difference(sine, base),
            }

        expected_unit_total = None
        if hadamard_candidate.cosine_shots is not None and hadamard_candidate.sine_shots is not None:
            expected_unit_total = float(
                hadamard_candidate.cosine_shots
                * circuit_cost_metric(
                    hadamard_candidate.cosine_expected_cost,
                    cost_metric,
                )
                + hadamard_candidate.sine_shots
                * circuit_cost_metric(
                    hadamard_candidate.sine_expected_cost,
                    cost_metric,
                )
            )

        checks = {
            "both_candidates_feasible": bool(
                evolution_candidate.feasible and hadamard_candidate.feasible
            ),
            "analytic_inputs_match": bool(
                evolution_candidate.cosine_shots == hadamard_candidate.cosine_shots
                and evolution_candidate.sine_shots == hadamard_candidate.sine_shots
                and math.isclose(
                    evolution_candidate.epsilon_z,
                    hadamard_candidate.epsilon_z,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
                and math.isclose(
                    evolution_candidate.attenuation,
                    hadamard_candidate.attenuation,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
            ),
            "resource_shots_match_signal_unit_radius": bool(
                hadamard_candidate.cosine_shots == unit_shots
                and hadamard_candidate.sine_shots == unit_shots
            ),
            "round_cost_identity_pass": bool(
                expected_unit_total is not None
                and hadamard_candidate.round_total_cost is not None
                and math.isclose(
                    expected_unit_total,
                    hadamard_candidate.round_total_cost,
                    rel_tol=1e-14,
                    abs_tol=1e-9,
                )
            ),
            "cost_scopes_are_distinct_and_correct": bool(
                evolution_candidate.circuit_cost_scope
                == "compiled_time_evolution_subcircuit"
                and hadamard_candidate.circuit_cost_scope
                == "single_hadamard_interrogation_without_state_preparation"
            ),
            "classical_sample_count_not_used_as_shot_multiplier": bool(
                hadamard_candidate.classical_cost_sample_count == sample_count
                and expected_unit_total == hadamard_candidate.round_total_cost
            ),
        }
        rounds.append(
            {
                "round_index": specification.round_index,
                "q_m": q_m,
                "t_m": specification.t_m,
                "signal": {
                    "reference_signal_radius": physical["reference_signal_radius"],
                    "reference_radius_deviation_from_one": physical[
                        "reference_radius_deviation_from_one"
                    ],
                    "round_signal_error_bound": point["round_signal_error_bound"],
                    "attenuation": point["attenuation"],
                    "unit_radius_shots_per_axis": unit_shots,
                    "reference_radius_shots_per_axis": reference_shots,
                },
                "time_evolution_candidate": _candidate_payload(evolution_candidate),
                "hadamard_interrogation_candidate": _candidate_payload(
                    hadamard_candidate
                ),
                "scope_comparison": overhead,
                "reference_radius_round_cost": reference_round_cost,
                "reference_vs_unit_radius_round_cost_relative_difference": (
                    radius_cost_relative_difference
                ),
                "checks": checks,
            }
        )

    maximum_radius_deviation = max(
        round_result["signal"]["reference_radius_deviation_from_one"]
        for round_result in rounds
    )
    maximum_radius_cost_difference = max(
        value
        for value in (
            item["reference_vs_unit_radius_round_cost_relative_difference"]
            for item in rounds
        )
        if value is not None
    )
    all_connection_checks_pass = all(
        all(item["checks"].values()) for item in rounds
    )
    summary = {
        "round_count": len(rounds),
        "all_finite_rte_signal_checks_pass": signal_payload["summary"][
            "overall_pass"
        ],
        "all_connection_checks_pass": all_connection_checks_pass,
        "maximum_physical_reference_radius_deviation_from_one": (
            maximum_radius_deviation
        ),
        "maximum_reference_vs_unit_radius_round_cost_relative_difference": (
            maximum_radius_cost_difference
        ),
        "unit_radius_specialization_changes_integer_shots": any(
            item["signal"]["unit_radius_shots_per_axis"]
            != item["signal"]["reference_radius_shots_per_axis"]
            for item in rounds
        ),
        "overall_pass": bool(
            signal_payload["summary"]["overall_pass"]
            and all_connection_checks_pass
        ),
        "interpretation": (
            "short_round_connection_validated_not_full_rpe_total_cost"
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": RPE_ROUND_COST_CONNECTION_SCHEMA_VERSION,
        "validation_method": RPE_ROUND_COST_CONNECTION_METHOD,
        "final_cost_evaluation_performed": False,
        "scope": {
            "signal": "physical_full_h_ground_state_survival_signal",
            "direct_cost_reference": "compiled_time_evolution_subcircuit",
            "accounting_cost": (
                "single_hadamard_interrogation_without_state_preparation"
            ),
            "state_preparation_included": False,
            "backend_execution_included": False,
            "quantum_shots_executed": 0,
            "full_rpe_phase_reconstruction_included": False,
        },
        "hamiltonian": {
            "hamiltonian_hash": df_hamiltonian_hash(hamiltonian),
            "n_qubits": hamiltonian.n_qubits,
            "df_rank": hamiltonian.n_blocks,
            "metadata": dict(hamiltonian.metadata),
        },
        "request": {
            "ld": ld_value,
            "delta_time": float(delta_time),
            "q_values": list(q_grid),
            "rte_steps_per_occurrence": r_value,
            "finite_taylor_order": k_value,
            "compiled_cost_sample_count": sample_count,
            "signal_seed": signal_seed,
            "compiled_cost_seed": compiled_cost_seed,
            "cost_metric": cost_metric,
            "beta_rpe": beta_rpe,
            "allocation": {
                "beta_pf_budget": beta_pf_budget,
                "beta_rte_budget": beta_rte_budget,
                "beta_stat_budget": beta_stat_budget,
                "alpha_total": alpha_total,
                "alpha_per_round_axis": alpha_axis,
            },
            "pf_error_model": {
                "coefficient": pf_model.coefficient,
                "source": pf_model.source,
                "is_rigorous_bound": pf_model.is_rigorous_bound,
            },
            "compiler": {
                "basis_gates": list(compiler.basis_gates),
                "backend_name": compiler.backend_name,
                "coupling_map": compiler.coupling_map,
                "optimization_level": compiler.optimization_level,
                "layout_method": compiler.layout_method,
                "routing_method": compiler.routing_method,
                "transpiler_seed": compiler.transpiler_seed,
                "qiskit_version": compiler.qiskit_version,
            },
        },
        "finite_rte_signal_validation_fingerprint": signal_payload[
            "validation_fingerprint"
        ],
        "rounds": rounds,
        "summary": summary,
        "performance": {
            "elapsed_seconds": time.perf_counter() - started,
        },
        "provenance": dict(provenance or {}),
    }
    payload["validation_fingerprint"] = _fingerprint(payload)
    validate_rpe_round_cost_connection_payload(payload)
    return payload


def validate_rpe_round_cost_connection_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != RPE_ROUND_COST_CONNECTION_SCHEMA_VERSION:
        raise ValueError("Unsupported RPE round-cost connection schema.")
    if payload.get("validation_method") != RPE_ROUND_COST_CONNECTION_METHOD:
        raise ValueError("Unsupported RPE round-cost connection method.")
    if payload.get("final_cost_evaluation_performed") is not False:
        raise ValueError("Connection validation cannot contain a final cost result.")
    rounds = payload.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise ValueError("Connection validation requires at least one round.")
    if payload.get("summary", {}).get("round_count") != len(rounds):
        raise ValueError("Connection validation round count mismatch.")
    expected_connection = all(
        all(bool(value) for value in item.get("checks", {}).values())
        and bool(item.get("checks"))
        for item in rounds
    )
    if payload.get("summary", {}).get("all_connection_checks_pass") != expected_connection:
        raise ValueError("Connection validation check summary mismatch.")
    expected_overall = bool(
        payload.get("summary", {}).get("all_finite_rte_signal_checks_pass")
        and expected_connection
    )
    if payload.get("summary", {}).get("overall_pass") != expected_overall:
        raise ValueError("Connection validation overall status mismatch.")
    fingerprint = payload.get("validation_fingerprint")
    without_fingerprint = dict(payload)
    without_fingerprint.pop("validation_fingerprint", None)
    if fingerprint != _fingerprint(without_fingerprint):
        raise ValueError("Connection validation fingerprint mismatch.")


def write_rpe_round_cost_connection_validation(
    payload: Mapping[str, Any], path: str | Path
) -> None:
    validate_rpe_round_cost_connection_payload(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
