"""Validate a q>4 Hadamard cost proxy and its accounting connection.

This validation directly compiles complete Hadamard-interrogation wrappers for
calibration and unused holdout repetition counts.  It then fits the existing
affine proxy, validates it at the holdout count, and checks that the fixed
validated prediction is consumed exactly once per RPE shot.  It deliberately
does not sum a full multi-round RPE experiment.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .df_hamiltonian import DFHamiltonian, PhysicalSector
from .df_partial_randomized_pf import split_df_hamiltonian_by_ld
from .df_partial_s2 import prepare_df_partial_s2
from .rpe_hadamard_compiled_cost_benchmark import (
    RPEHadamardCompiledCostBenchmarkDataset,
    RPEHadamardCompiledCostBenchmarkRequest,
    generate_rpe_hadamard_compiled_cost_benchmark_dataset,
)
from .rpe_hadamard_compiled_cost_proxy import (
    RPEHadamardCompiledCostProxy,
    RPEHadamardCompiledCostProxyFitRequest,
    RPEHadamardCompiledCostProxyValidationRequest,
    RPEHadamardCompiledCostProxyValidationResult,
    RPEHadamardProxyMetricTolerance,
    fit_rpe_hadamard_compiled_cost_proxy,
    validate_rpe_hadamard_compiled_cost_proxy,
)
from .rpe_hadamard_validated_proxy_provider import (
    ValidatedRPEHadamardCompiledCostProxyProvider,
)
from .rpe_resource_accounting import (
    RPE_COST_METRICS,
    RPEErrorAllocation,
    RPEHadamardSamplingPolicy,
    RPEPFErrorModel,
    RPERoundCandidate,
    RPERoundCostRequest,
    RPERoundSpecification,
    circuit_cost_metric,
    evaluate_rpe_round_candidate,
)
from .rte import (
    CircuitCost,
    CompilerSettings,
    finite_rte_distribution,
    make_rte_config,
    require_integer_count,
)
from .rte_compiled_cost import TranspiledCircuitCostCache


RPE_HADAMARD_PROXY_RESOURCE_VALIDATION_SCHEMA_VERSION = (
    "rpe_hadamard_proxy_resource_validation_v1"
)
RPE_HADAMARD_PROXY_RESOURCE_VALIDATION_METHOD = (
    "unused_q8_holdout_and_single_round_accounting_connection_v1"
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


def _candidate_payload(candidate: RPERoundCandidate) -> dict[str, Any]:
    return {
        "round_index": candidate.m,
        "q_m": candidate.q_m,
        "t_m": candidate.t_m,
        "feasible": candidate.feasible,
        "infeasibility_reasons": list(candidate.infeasibility_reasons),
        "epsilon_z": candidate.epsilon_z,
        "attenuation": candidate.attenuation,
        "rho_observed_lower_bound": candidate.rho_observed_lower_bound,
        "epsilon_coordinate": candidate.epsilon_coordinate,
        "cosine_shots": candidate.cosine_shots,
        "sine_shots": candidate.sine_shots,
        "cosine_expected_cost": _cost_payload(candidate.cosine_expected_cost),
        "sine_expected_cost": _cost_payload(candidate.sine_expected_cost),
        "round_total_cost": candidate.round_total_cost,
        "cost_metric": candidate.cost_metric,
        "cost_evaluation_method": candidate.cost_evaluation_method,
        "classical_cost_sample_count": candidate.classical_cost_sample_count,
        "circuit_cost_scope": candidate.circuit_cost_scope,
        "cost_model_fingerprint": candidate.cost_model_fingerprint,
        "cost_metadata": dict(candidate.cost_metadata),
    }


@dataclass(frozen=True)
class RPEHadamardProxyResourceValidationBundle:
    dataset: RPEHadamardCompiledCostBenchmarkDataset
    proxy: RPEHadamardCompiledCostProxy
    proxy_validation: RPEHadamardCompiledCostProxyValidationResult
    connection: dict[str, Any]

    def write(
        self,
        *,
        dataset_path: str | Path,
        proxy_path: str | Path,
        proxy_validation_path: str | Path,
        connection_path: str | Path,
    ) -> None:
        self.dataset.write_json(dataset_path)
        self.proxy.write_json(proxy_path)
        self.proxy_validation.write_json(proxy_validation_path)
        output = Path(connection_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.connection, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def validate_rpe_hadamard_proxy_resource_connection(
    hamiltonian: DFHamiltonian,
    sector: PhysicalSector,
    compiler: CompilerSettings,
    *,
    ld: int,
    delta_time: float,
    calibration_q_values: Sequence[int] = (1, 2, 4),
    holdout_q_values: Sequence[int] = (8,),
    rte_steps_per_occurrence: int = 4,
    finite_taylor_order: int = 2,
    sample_count: int = 64,
    benchmark_seed: int = 20260904,
    rte_seed: int = 20260818,
    beta_rpe: float = 0.40,
    beta_pf_budget: float = 0.08,
    beta_rte_budget: float = 0.08,
    beta_stat_budget: float = 0.24,
    alpha_total: float = 0.05,
    accounted_round_count: int = 4,
    pf_coefficient: float = 0.01342567,
    pf_coefficient_source: str = "paper_d6_empirical_surrogate",
    cost_metric: str = "rz_count",
    relative_error_tolerance: float = 0.05,
    rz_relative_standard_error_tolerance: float = 0.02,
    model_label: str = "H4 linear chain",
    geometry_angstrom: float = 1.0,
    basis_label: str = "STO-3G",
    provenance: Mapping[str, Any] | None = None,
) -> RPEHadamardProxyResourceValidationBundle:
    """Run the representative medium-q proxy and accounting validation."""
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
    n_samples = require_integer_count(sample_count, name="sample_count", minimum=2)
    n_rounds = require_integer_count(
        accounted_round_count,
        name="accounted_round_count",
        minimum=1,
    )
    calibration = tuple(sorted(set(int(q) for q in calibration_q_values)))
    holdout = tuple(sorted(set(int(q) for q in holdout_q_values)))
    if not calibration or not holdout or set(calibration).intersection(holdout):
        raise ValueError("Calibration and holdout q grids must be non-empty/disjoint.")
    if any(q < 1 or q & (q - 1) for q in (*calibration, *holdout)):
        raise ValueError("All q values must be positive powers of two.")
    if not any(q > 4 for q in holdout):
        raise ValueError("At least one q>4 holdout point is required.")
    if cost_metric not in RPE_COST_METRICS:
        raise ValueError(f"Unsupported cost metric: {cost_metric}.")
    for name, value in (
        ("relative_error_tolerance", relative_error_tolerance),
        (
            "rz_relative_standard_error_tolerance",
            rz_relative_standard_error_tolerance,
        ),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    if not isinstance(model_label, str) or not model_label.strip():
        raise ValueError("model_label must be non-empty.")
    if not isinstance(basis_label, str) or not basis_label.strip():
        raise ValueError("basis_label must be non-empty.")
    if not math.isfinite(geometry_angstrom) or geometry_angstrom <= 0.0:
        raise ValueError("geometry_angstrom must be finite and positive.")

    preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, ld_value),
        identity_policy="extract_identity_phase",
    )
    tau = preparation.exact_rte_lambda_r * delta_time / r_value
    distribution = finite_rte_distribution(tau, k_value)
    rte_config, rte_distribution = make_rte_config(
        preparation.rte_preparation.symbolic_tail,
        evolution_time=delta_time,
        rte_steps=r_value,
        truncation_tolerance=max(
            distribution.step_truncation_residual_bound,
            math.ulp(0.0),
        ),
        finite_taylor_order=k_value,
        seed=rte_seed,
    )

    benchmark_result = generate_rpe_hadamard_compiled_cost_benchmark_dataset(
        RPEHadamardCompiledCostBenchmarkRequest(
            preparation=preparation,
            delta_time=delta_time,
            calibration_repetition_counts=calibration,
            holdout_repetition_counts=holdout,
            rte_steps_per_occurrence=r_value,
            finite_taylor_order=k_value,
            rte_config=rte_config,
            rte_distribution=rte_distribution,
            compiler=compiler,
            evaluation_method="monte_carlo",
            sample_count=n_samples,
            seed=benchmark_seed,
            generation_id="h4-q8-proxy-resource-validation-2026-09-01",
            maximum_repetition_count=max(holdout),
            maximum_samples=n_samples,
            maximum_retained_trajectory_records=n_samples,
            cache=TranspiledCircuitCostCache(),
        )
    )
    dataset = benchmark_result.dataset
    if not dataset.complete:
        raise RuntimeError(
            "Hadamard benchmark generation was incomplete: "
            + "; ".join(dataset.incomplete_reasons)
        )

    proxy = fit_rpe_hadamard_compiled_cost_proxy(
        RPEHadamardCompiledCostProxyFitRequest(
            dataset=dataset,
            weighting="uniform",
        )
    )
    tolerances = tuple(
        RPEHadamardProxyMetricTolerance(
            metric=metric,
            absolute_tolerance=0.0,
            relative_tolerance=relative_error_tolerance,
            standard_error_multiplier=0.0,
        )
        for metric in RPE_COST_METRICS
    )
    proxy_validation = validate_rpe_hadamard_compiled_cost_proxy(
        RPEHadamardCompiledCostProxyValidationRequest(
            proxy=proxy,
            dataset=dataset,
            metric_tolerances=tolerances,
        )
    )

    source_rz_precision = []
    for point in (*proxy.calibration_points, *proxy_validation.holdout_points):
        mean = point.mean("rz_count")
        standard_error = point.standard_error("rz_count")
        relative = None if standard_error is None or mean == 0.0 else standard_error / mean
        source_rz_precision.append(
            {
                "partition": point.partition,
                "q_m": point.q_m,
                "axis": point.axis,
                "mean": mean,
                "standard_error": standard_error,
                "relative_standard_error": relative,
            }
        )
    finite_relative_se = [
        item["relative_standard_error"]
        for item in source_rz_precision
        if item["relative_standard_error"] is not None
    ]
    max_rz_relative_se = max(finite_relative_se, default=math.inf)

    provider = ValidatedRPEHadamardCompiledCostProxyProvider(
        validation=proxy_validation,
        compiler=compiler,
    )
    alpha_axis = alpha_total / (2 * n_rounds)
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
    resource_candidates = []
    connection_checks = []
    for q_m in holdout:
        candidate = evaluate_rpe_round_candidate(
            preparation,
            RPERoundSpecification(q_m.bit_length() - 1, delta_time),
            allocation,
            pf_model,
            beta_rpe=beta_rpe,
            rte_steps_per_occurrence=r_value,
            finite_taylor_order=k_value,
            cost_metric=cost_metric,
            cost_provider=provider,
            rte_seed=rte_seed,
            hadamard_sampling_policy=sampling_policy,
        )
        predicted_cosine = proxy.predict(q_m, axis="cosine", metric=cost_metric)
        predicted_sine = proxy.predict(q_m, axis="sine", metric=cost_metric)
        expected_total = None
        if candidate.cosine_shots is not None and candidate.sine_shots is not None:
            expected_total = float(
                candidate.cosine_shots * predicted_cosine
                + candidate.sine_shots * predicted_sine
            )
        checks = {
            "candidate_feasible": candidate.feasible,
            "cosine_cost_equals_fixed_proxy": bool(
                candidate.cosine_expected_metric == predicted_cosine
            ),
            "sine_cost_equals_fixed_proxy": bool(
                candidate.sine_expected_metric == predicted_sine
            ),
            "round_cost_identity_pass": bool(
                expected_total is not None
                and candidate.round_total_cost is not None
                and math.isclose(
                    candidate.round_total_cost,
                    expected_total,
                    rel_tol=1e-14,
                    abs_tol=1e-9,
                )
            ),
            "scope_is_single_hadamard_without_state_preparation": bool(
                candidate.circuit_cost_scope
                == "single_hadamard_interrogation_without_state_preparation"
            ),
            "validation_bound_to_cost_model": bool(
                candidate.cost_model_fingerprint == provider.cost_model_fingerprint
                and dict(candidate.cost_metadata).get("validation_fingerprint")
                == proxy_validation.validation_fingerprint
            ),
            "classical_samples_not_multiplied_by_quantum_shots": bool(
                candidate.classical_cost_sample_count is None
                and expected_total == candidate.round_total_cost
            ),
        }
        resource_candidates.append(_candidate_payload(candidate))
        connection_checks.append({"q_m": q_m, "checks": checks})

    rejected_unvalidated_q = max(holdout) * 2
    try:
        provider.evaluate(
            RPERoundCostRequest(
                preparation=preparation,
                specification=RPERoundSpecification(
                    rejected_unvalidated_q.bit_length() - 1,
                    delta_time,
                ),
                allocation=allocation,
                rte_steps_per_occurrence=r_value,
                finite_taylor_order=k_value,
                rte_config=rte_config,
                rte_distribution=rte_distribution,
            )
        )
    except ValueError as exc:
        unvalidated_rejection = {
            "q_m": rejected_unvalidated_q,
            "rejected": True,
            "reason": str(exc),
        }
    else:
        unvalidated_rejection = {
            "q_m": rejected_unvalidated_q,
            "rejected": False,
            "reason": None,
        }

    validation_relative_errors = [
        entry.relative_error
        for entry in proxy_validation.entries
        if entry.relative_error is not None
    ]
    maximum_relative_error = max(validation_relative_errors, default=math.inf)
    checks = {
        "benchmark_complete": dataset.complete,
        "holdout_not_used_for_fit": not proxy.holdout_used_for_fit,
        "proxy_holdout_all_metrics_pass_5pct": proxy_validation.overall_pass,
        "maximum_holdout_relative_error_within_tolerance": bool(
            maximum_relative_error <= relative_error_tolerance
        ),
        "rz_source_relative_standard_error_within_tolerance": bool(
            max_rz_relative_se <= rz_relative_standard_error_tolerance
        ),
        "all_accounting_connection_checks_pass": all(
            all(item["checks"].values()) for item in connection_checks
        ),
        "unvalidated_q_is_rejected": unvalidated_rejection["rejected"],
    }
    summary = {
        "overall_pass": all(checks.values()),
        "checks": checks,
        "maximum_holdout_relative_error": maximum_relative_error,
        "maximum_rz_source_relative_standard_error": max_rz_relative_se,
        "validated_q_m_values": list(proxy_validation.validated_q_m_values),
        "connected_round_count": len(resource_candidates),
    }
    connection: dict[str, Any] = {
        "schema_version": RPE_HADAMARD_PROXY_RESOURCE_VALIDATION_SCHEMA_VERSION,
        "method": RPE_HADAMARD_PROXY_RESOURCE_VALIDATION_METHOD,
        "system": {
            "model": model_label.strip(),
            "geometry_angstrom": geometry_angstrom,
            "basis": basis_label.strip(),
            "num_system_qubits": hamiltonian.n_qubits,
            "physical_sector": {
                "n_electrons": sector.n_electrons,
                "nelec_alpha": sector.nelec_alpha,
                "nelec_beta": sector.nelec_beta,
                "sz_value": sector.sz_value,
            },
            "df_rank": len(hamiltonian.lambdas),
            "ld": ld_value,
            "hamiltonian_hash": preparation.hamiltonian_hash,
            "partition_hash": preparation.partition_hash,
            "preparation_hash": preparation.preparation_hash,
        },
        "configuration": {
            "delta_time": delta_time,
            "calibration_q_values": list(calibration),
            "holdout_q_values": list(holdout),
            "rte_steps_per_occurrence": r_value,
            "finite_taylor_order": k_value,
            "sample_count_per_q": n_samples,
            "benchmark_seed": benchmark_seed,
            "rte_seed": rte_seed,
            "fit_weighting": "uniform",
            "relative_error_tolerance": relative_error_tolerance,
            "rz_relative_standard_error_tolerance": (
                rz_relative_standard_error_tolerance
            ),
            "beta_rpe": beta_rpe,
            "phase_budget": {
                "beta_pf": beta_pf_budget,
                "beta_rte": beta_rte_budget,
                "beta_stat": beta_stat_budget,
            },
            "alpha_total": alpha_total,
            "accounted_round_count": n_rounds,
            "alpha_per_axis_per_round": alpha_axis,
            "pf_coefficient": pf_coefficient,
            "pf_coefficient_source": pf_coefficient_source,
            "pf_coefficient_is_rigorous_bound": False,
            "cost_metric": cost_metric,
            "compiler": {
                "basis_gates": list(compiler.basis_gates),
                "optimization_level": compiler.optimization_level,
                "transpiler_seed": compiler.transpiler_seed,
                "qiskit_version": compiler.qiskit_version,
                "coupling_map": compiler.coupling_map,
            },
        },
        "evidence_fingerprints": {
            "benchmark_dataset": dataset.dataset_fingerprint,
            "proxy_fit": proxy.fit_fingerprint,
            "proxy": proxy.proxy_fingerprint,
            "proxy_validation": proxy_validation.validation_fingerprint,
            "provider_cost_model": provider.cost_model_fingerprint,
        },
        "source_rz_precision": source_rz_precision,
        "holdout_validation_entries": [
            entry.to_dict() for entry in proxy_validation.entries
        ],
        "resource_candidates": resource_candidates,
        "connection_checks": connection_checks,
        "unvalidated_q_rejection": unvalidated_rejection,
        "summary": summary,
        "scope": {
            "state_preparation_included": False,
            "backend_execution_included": False,
            "quantum_shots_executed": 0,
            "physical_q8_signal_or_phase_evaluated": False,
            "full_multi_round_resource_summary_evaluated": False,
            "final_total_cost_evaluation_performed": False,
            "accuracy_guaranteed_beyond_validated_q": False,
        },
        "provenance": dict(provenance or {}),
        "performance": {"elapsed_seconds": time.perf_counter() - started},
    }
    connection["content_fingerprint"] = _fingerprint(connection)
    return RPEHadamardProxyResourceValidationBundle(
        dataset=dataset,
        proxy=proxy,
        proxy_validation=proxy_validation,
        connection=connection,
    )


def validate_rpe_hadamard_proxy_resource_connection_payload(
    payload: Mapping[str, Any],
) -> None:
    if payload.get("schema_version") != (
        RPE_HADAMARD_PROXY_RESOURCE_VALIDATION_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported proxy-resource validation schema.")
    stored = payload.get("content_fingerprint")
    if not isinstance(stored, str):
        raise ValueError("Missing content_fingerprint.")
    unsigned = dict(payload)
    del unsigned["content_fingerprint"]
    if stored != _fingerprint(unsigned):
        raise ValueError("content_fingerprint does not match the payload.")
    checks = payload["summary"]["checks"]
    if payload["summary"]["overall_pass"] != all(checks.values()):
        raise ValueError("Summary pass value does not match its checks.")


def write_rpe_hadamard_proxy_resource_connection(
    payload: Mapping[str, Any],
    path: str | Path,
) -> None:
    validate_rpe_hadamard_proxy_resource_connection_payload(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
