"""WP06-b sequence-aware DF basis policy and additive proxy recalibration.

This module keeps the policy selection separate from its holdout and from the
later full Hadamard-interrogation validation.  It exercises the production RTE
builder's explicit basis-plan path, but does not silently change its default.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from qiskit.quantum_info import Operator

from .df_hamiltonian import DFHamiltonian
from .df_partial_s2 import DFPartialS2Preparation
from .df_rte_circuit import (
    DFRTEApplicationBasisChoice,
    DFRTEBasisPlan,
    DFRTEComponentCircuitSpec,
    DFRTEEventSequenceCircuitRequest,
)
from .df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
from .df_rte_tail import DFBasisDefinition, describe_basis_change_operations
from .df_trotter.decompose import diag_hermitian
from .df_trotter.ops import U_to_qiskit_ops_jw
from .research_direction_structure_pilot import (
    maximum_operator_difference,
    support_restricted_unitary_completion,
)
from .rte import (
    CompilerSettings,
    _make_event,
    make_rte_config,
    step_taylor_truncation_residual_bound,
)
from .rte_compiled_cost import transpile_and_measure_cost


SCHEMA_VERSION = "research_direction_sequence_policy_v1"
METHOD = "wp06b_sequence_aware_basis_policy_additive_proxy_bridge_v1"
METRICS = (
    "rz_count",
    "rz_depth",
    "cx_count",
    "cx_depth",
    "total_depth",
    "circuit_size",
)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _derived_seed(master_seed: int, *parts: object) -> int:
    encoded = json.dumps(
        [int(master_seed), *parts],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def _fragment_index(fragment_id: str) -> int:
    prefix = "df-fragment-"
    if not fragment_id.startswith(prefix):
        raise ValueError("DF component lacks a canonical fragment index.")
    return int(fragment_id[len(prefix) :])


def _support_key(
    basis_id: str,
    basis_hash: str,
    support: Sequence[int],
) -> tuple[str, str, tuple[int, ...]]:
    return basis_id, basis_hash, tuple(int(index) for index in support)


def register_support_restricted_bases(
    hamiltonian: DFHamiltonian,
    preparation: DFPartialS2Preparation,
) -> tuple[
    dict[tuple[str, str, tuple[int, ...]], DFBasisDefinition],
    list[dict[str, Any]],
]:
    """Register certified support completions alongside established bases."""
    registry = preparation.rte_preparation.basis_registry
    definitions: dict[
        tuple[str, str, tuple[int, ...]], DFBasisDefinition
    ] = {}
    proof_records: list[dict[str, Any]] = []
    full_unitaries: dict[tuple[str, str], np.ndarray] = {}
    specs = tuple(
        spec
        for spec in preparation.rte_preparation.component_specs
        if isinstance(spec, DFRTEComponentCircuitSpec)
    )
    for spec in specs:
        source_key = (spec.basis_id, str(spec.basis_hash))
        if source_key not in full_unitaries:
            unitary, _eigenvalues = diag_hermitian(
                hamiltonian.g_matrices[_fragment_index(spec.df_fragment_id)],
                sort=preparation.diagonal_sort,
                assume_hermitian=True,
            )
            operations = tuple(U_to_qiskit_ops_jw(unitary))
            if describe_basis_change_operations(operations) != (
                spec.basis_change_operations
            ):
                raise ValueError("Rebuilt source basis differs from preparation.")
            full_unitaries[source_key] = np.asarray(unitary)
        key = _support_key(
            spec.basis_id,
            str(spec.basis_hash),
            spec.diagonal_pauli_support,
        )
        if key in definitions:
            continue
        full = full_unitaries[source_key]
        support = spec.diagonal_pauli_support
        restricted = support_restricted_unitary_completion(full, support)
        residual = float(np.max(np.abs(full[:, support] - restricted[:, support])))
        suffix = "-".join(str(index) for index in support)
        basis_id = f"{spec.basis_id}-support-{suffix}-wp06b-v1"
        definition = registry.register(
            tuple(U_to_qiskit_ops_jw(restricted)),
            num_system_qubits=hamiltonian.n_qubits,
            basis_id=basis_id,
        )
        definitions[key] = definition
        proof_records.append(
            {
                "source_basis_id": spec.basis_id,
                "source_basis_hash": spec.basis_hash,
                "selected_basis_id": definition.basis_id,
                "selected_basis_hash": definition.basis_hash,
                "support": list(support),
                "preserved_columns_max_abs_residual": residual,
                "source_basis_operation_count": len(
                    registry.definition(spec.basis_id).runtime_operations
                ),
                "support_basis_operation_count": len(
                    definition.runtime_operations
                ),
            }
        )
    return definitions, proof_records


def _flatten_applications(events: Sequence[Any]) -> list[Any]:
    return [
        application
        for event in events
        for application in event.application_sequence
    ]


def source_basis_run_lengths(events: Sequence[Any]) -> tuple[int, ...]:
    """Return each non-identity application's maximal source-basis run length."""
    applications = _flatten_applications(events)
    lengths: list[int] = []
    cursor = 0
    while cursor < len(applications):
        application = applications[cursor]
        if application.is_identity:
            cursor += 1
            continue
        key = (application.basis_id, application.basis_hash)
        stop = cursor + 1
        while stop < len(applications):
            candidate = applications[stop]
            if candidate.is_identity or (
                candidate.basis_id,
                candidate.basis_hash,
            ) != key:
                break
            stop += 1
        lengths.extend([stop - cursor] * (stop - cursor))
        cursor = stop
    return tuple(lengths)


def make_run_threshold_basis_plan(
    request: DFRTEEventSequenceCircuitRequest,
    support_definitions: Mapping[
        tuple[str, str, tuple[int, ...]], DFBasisDefinition
    ],
    *,
    maximum_support_run_length: int | None,
    training_fingerprint: str | None = None,
) -> DFRTEBasisPlan:
    """Choose support bases only for source-basis runs up to a fixed length."""
    if maximum_support_run_length is not None and maximum_support_run_length < 0:
        raise ValueError("maximum_support_run_length must be non-negative or None.")
    applications = [
        application
        for application in _flatten_applications(request.events)
        if not application.is_identity
    ]
    run_lengths = source_basis_run_lengths(request.events)
    if len(applications) != len(run_lengths):
        raise RuntimeError("Run-length metadata does not align with applications.")
    choices: list[DFRTEApplicationBasisChoice] = []
    for application, run_length in zip(applications, run_lengths, strict=True):
        use_support = (
            maximum_support_run_length is None
            or run_length <= maximum_support_run_length
        )
        if use_support:
            key = _support_key(
                str(application.basis_id),
                str(application.basis_hash),
                application.diagonal_pauli_support,
            )
            selected = support_definitions[key]
            construction = "support_restricted_preserved_columns_v1"
        else:
            selected = None
            construction = "registered_full_basis"
        choices.append(
            DFRTEApplicationBasisChoice(
                source_basis_id=str(application.basis_id),
                source_basis_hash=str(application.basis_hash),
                selected_basis_id=(
                    str(application.basis_id)
                    if selected is None
                    else selected.basis_id
                ),
                selected_basis_hash=(
                    str(application.basis_hash)
                    if selected is None
                    else selected.basis_hash
                ),
                diagonal_pauli_support=tuple(
                    int(index) for index in application.diagonal_pauli_support
                ),
                construction=construction,
                preserved_columns_max_abs_residual=0.0,
            )
        )
    threshold_label = (
        "all" if maximum_support_run_length is None else str(maximum_support_run_length)
    )
    return DFRTEBasisPlan(
        policy_id=f"support_for_source_basis_run_le_{threshold_label}",
        selection_objective="training_lexicographic_rz_cx_depth_with_5pct_guard",
        choices=tuple(choices),
        training_fingerprint=training_fingerprint,
    )


def _threshold_label(threshold: int | None) -> str:
    return "support_all" if threshold is None else f"support_run_le_{threshold}"


def _cost_record(circuit: Any, compiler: CompilerSettings) -> dict[str, int]:
    cost = transpile_and_measure_cost(circuit, compiler)
    return {metric: int(getattr(cost, metric)) for metric in METRICS}


def _statistics(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.size < 1:
        raise ValueError("At least one value is required.")
    variance = 0.0 if array.size == 1 else float(np.var(array, ddof=1))
    return {
        "mean": float(np.mean(array)),
        "standard_error": float(math.sqrt(variance / array.size)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _request_for_seed(
    preparation: DFPartialS2Preparation,
    *,
    delta_time: float,
    sequence_length: int,
    finite_taylor_order: int,
    seed: int,
) -> tuple[DFRTEEventSequenceCircuitRequest, str]:
    tau = preparation.exact_rte_lambda_r * delta_time / sequence_length
    residual = step_taylor_truncation_residual_bound(tau, finite_taylor_order)
    tolerance = max(math.nextafter(residual, math.inf), residual * (1.0 + 1e-12))
    config, distribution = make_rte_config(
        preparation.rte_preparation.symbolic_tail,
        evolution_time=delta_time,
        rte_steps=sequence_length,
        truncation_tolerance=tolerance,
        finite_taylor_order=finite_taylor_order,
    )
    request = preparation.rte_preparation.sample_occurrence_request(
        config,
        distribution,
        seed=seed,
        controlled=True,
        ancilla_qubit=preparation.num_system_qubits,
        cancel_adjacent_equal_bases=True,
    )
    digest = hashlib.sha256(
        json.dumps(
            [event.to_dict() for event in request.events],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return request, digest


def _compile_policy_grid(
    preparation: DFPartialS2Preparation,
    compiler: CompilerSettings,
    support_definitions: Mapping[
        tuple[str, str, tuple[int, ...]], DFBasisDefinition
    ],
    *,
    delta_time: float,
    finite_taylor_order: int,
    lengths: Sequence[int],
    sample_count: int,
    master_seed: int,
    partition: str,
    thresholds: Sequence[int | None],
    training_fingerprint: str | None,
) -> list[dict[str, Any]]:
    builder = QiskitDFRTEEventCircuitBuilder(
        basis_registry=preparation.rte_preparation.basis_registry
    )
    rows: list[dict[str, Any]] = []
    for length in lengths:
        for sample_index in range(sample_count):
            seed = _derived_seed(
                master_seed,
                partition,
                int(length),
                sample_index,
            )
            request, event_digest = _request_for_seed(
                preparation,
                delta_time=delta_time,
                sequence_length=int(length),
                finite_taylor_order=finite_taylor_order,
                seed=seed,
            )
            full = builder.build_sequence(request)
            costs = {"full_basis_shared": _cost_record(full.circuit, compiler)}
            policy_metadata: dict[str, Any] = {
                "full_basis_shared": {
                    "full_basis_application_count": (
                        full.full_basis_application_count
                    ),
                    "support_restricted_application_count": 0,
                    "basis_switch_count": full.basis_switch_count,
                    "cancelled_basis_change_pairs": (
                        full.cancelled_basis_change_pairs
                    ),
                }
            }
            for threshold in thresholds:
                label = _threshold_label(threshold)
                plan = make_run_threshold_basis_plan(
                    request,
                    support_definitions,
                    maximum_support_run_length=threshold,
                    training_fingerprint=training_fingerprint,
                )
                built = builder.build_sequence(request, basis_plan=plan)
                costs[label] = _cost_record(built.circuit, compiler)
                policy_metadata[label] = {
                    "full_basis_application_count": (
                        built.full_basis_application_count
                    ),
                    "support_restricted_application_count": (
                        built.support_restricted_application_count
                    ),
                    "basis_switch_count": built.basis_switch_count,
                    "cancelled_basis_change_pairs": (
                        built.cancelled_basis_change_pairs
                    ),
                }
            rows.append(
                {
                    "partition": partition,
                    "sequence_length": int(length),
                    "sample_index": sample_index,
                    "seed": seed,
                    "event_digest": event_digest,
                    "event_orders": [event.taylor_order for event in request.events],
                    "nonidentity_application_count": len(
                        source_basis_run_lengths(request.events)
                    ),
                    "source_basis_run_lengths": list(
                        source_basis_run_lengths(request.events)
                    ),
                    "costs": costs,
                    "policy_metadata": policy_metadata,
                }
            )
    return rows


def _policy_summary(
    rows: Sequence[Mapping[str, Any]],
    label: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {"policy": label, "metrics": {}}
    for metric in METRICS:
        baseline = [float(row["costs"]["full_basis_shared"][metric]) for row in rows]
        candidate = [float(row["costs"][label][metric]) for row in rows]
        deltas = [right - left for left, right in zip(baseline, candidate, strict=True)]
        baseline_stats = _statistics(baseline)
        candidate_stats = _statistics(candidate)
        delta_stats = _statistics(deltas)
        denominator = baseline_stats["mean"]
        result["metrics"][metric] = {
            "baseline": baseline_stats,
            "candidate": candidate_stats,
            "paired_delta": delta_stats,
            "candidate_relative_change": (
                None
                if denominator == 0.0
                else candidate_stats["mean"] / denominator - 1.0
            ),
        }
    return result


def _partition_summary(
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
) -> dict[str, Any]:
    lengths = sorted({int(row["sequence_length"]) for row in rows})
    return {
        "sample_count": len(rows),
        "sequence_lengths": lengths,
        "pooled": {label: _policy_summary(rows, label) for label in labels},
        "by_sequence_length": {
            str(length): {
                label: _policy_summary(
                    [row for row in rows if int(row["sequence_length"]) == length],
                    label,
                )
                for label in labels
            }
            for length in lengths
        },
        "event_stream_digest": hashlib.sha256(
            "".join(str(row["event_digest"]) for row in rows).encode()
        ).hexdigest(),
    }


def _choose_training_policy(
    rows: Sequence[Mapping[str, Any]],
    thresholds: Sequence[int | None],
    *,
    eta_decision: float,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    all_labels = ["full_basis_shared", *(_threshold_label(item) for item in thresholds)]
    for index, label in enumerate(all_labels):
        rz_values = [float(row["costs"][label]["rz_count"]) for row in rows]
        cx_values = [float(row["costs"][label]["cx_count"]) for row in rows]
        depth_values = [float(row["costs"][label]["total_depth"]) for row in rows]
        relative_increases = []
        for row, value in zip(rows, rz_values, strict=True):
            baseline = float(row["costs"]["full_basis_shared"]["rz_count"])
            relative_increases.append(0.0 if baseline == 0.0 else value / baseline - 1.0)
        feasible = max(relative_increases) <= eta_decision
        candidates.append(
            {
                "policy": label,
                "threshold": (
                    0
                    if label == "full_basis_shared"
                    else thresholds[index - 1]
                ),
                "feasible_under_per_trajectory_rz_guard": feasible,
                "maximum_per_trajectory_rz_relative_increase": max(
                    relative_increases
                ),
                "objective_totals": {
                    "rz_count": math.fsum(rz_values),
                    "cx_count": math.fsum(cx_values),
                    "total_depth": math.fsum(depth_values),
                },
                "conservative_tie_break_rank": index,
            }
        )
    feasible = [item for item in candidates if item["feasible_under_per_trajectory_rz_guard"]]
    selected = min(
        feasible,
        key=lambda item: (
            item["objective_totals"]["rz_count"],
            item["objective_totals"]["cx_count"],
            item["objective_totals"]["total_depth"],
            item["conservative_tie_break_rank"],
        ),
    )
    training_fingerprint = fingerprint(
        {
            "eta_decision": eta_decision,
            "candidate_results": candidates,
            "selected_policy": selected,
        }
    )
    return {
        "eta_decision": eta_decision,
        "selection_rule": (
            "minimize_pooled_RZ_then_CX_then_total_depth_subject_to_each_"
            "training_trajectory_RZ_not_exceeding_full_by_more_than_eta"
        ),
        "candidate_results": candidates,
        "selected_policy": selected,
        "training_fingerprint": training_fingerprint,
    }


def _operator_equivalence_probes(
    preparation: DFPartialS2Preparation,
    support_definitions: Mapping[
        tuple[str, str, tuple[int, ...]], DFBasisDefinition
    ],
    *,
    delta_time: float,
    finite_taylor_order: int,
    selected_threshold: int | None,
    training_fingerprint: str,
    master_seed: int,
    lengths: Sequence[int] = (1, 3),
) -> list[dict[str, Any]]:
    builder = QiskitDFRTEEventCircuitBuilder(
        basis_registry=preparation.rte_preparation.basis_registry
    )
    rows = []

    def append_probe(
        request: DFRTEEventSequenceCircuitRequest,
        *,
        probe_kind: str,
        seed: int | None,
    ) -> None:
        event_digest = hashlib.sha256(
            json.dumps(
                [event.to_dict() for event in request.events],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        full = builder.build_sequence(request)
        selected_plan = make_run_threshold_basis_plan(
            request,
            support_definitions,
            maximum_support_run_length=selected_threshold,
            training_fingerprint=training_fingerprint,
        )
        support_plan = make_run_threshold_basis_plan(
            request,
            support_definitions,
            maximum_support_run_length=None,
            training_fingerprint=training_fingerprint,
        )
        selected = builder.build_sequence(request, basis_plan=selected_plan)
        support = builder.build_sequence(request, basis_plan=support_plan)
        rows.append(
            {
                "probe_kind": probe_kind,
                "sequence_length": len(request.events),
                "seed": seed,
                "event_digest": event_digest,
                "event_orders": [event.taylor_order for event in request.events],
                "selected_vs_full_max_abs_difference": maximum_operator_difference(
                    full.circuit,
                    selected.circuit,
                    allow_global_phase=False,
                ),
                "support_all_vs_full_max_abs_difference": (
                    maximum_operator_difference(
                        full.circuit,
                        support.circuit,
                        allow_global_phase=False,
                    )
                ),
                "relative_ancilla_phase_full": full.relative_ancilla_phase,
                "relative_ancilla_phase_selected": selected.relative_ancilla_phase,
                "relative_ancilla_phase_support_all": support.relative_ancilla_phase,
            }
        )

    for length in lengths:
        seed = _derived_seed(master_seed, "equivalence", int(length))
        request, _event_digest = _request_for_seed(
            preparation,
            delta_time=delta_time,
            sequence_length=int(length),
            finite_taylor_order=finite_taylor_order,
            seed=seed,
        )
        append_probe(request, probe_kind="sampled_physical_sequence", seed=seed)

    tau = preparation.exact_rte_lambda_r * delta_time
    residual = step_taylor_truncation_residual_bound(tau, finite_taylor_order)
    tolerance = max(math.nextafter(residual, math.inf), residual * (1.0 + 1e-12))
    _config, distribution = make_rte_config(
        preparation.rte_preparation.symbolic_tail,
        evolution_time=delta_time,
        rte_steps=1,
        truncation_tolerance=tolerance,
        finite_taylor_order=finite_taylor_order,
    )
    order_index = distribution.orders.index(finite_taylor_order)
    components = preparation.rte_preparation.symbolic_tail.components
    component_indices = tuple(
        sorted(
            range(len(components)),
            key=lambda index: components[index].probability,
            reverse=True,
        )[: finite_taylor_order + 1]
    )
    forced_event = _make_event(
        component_indices,
        components,
        distribution,
        order_index,
    )
    forced_request = DFRTEEventSequenceCircuitRequest(
        events=(forced_event,),
        component_specs=preparation.rte_preparation.component_specs,
        controlled=True,
        ancilla_qubit=preparation.num_system_qubits,
        cancel_adjacent_equal_bases=True,
        tail_id=preparation.rte_preparation.symbolic_tail.tail_id,
        tail_hash=preparation.rte_preparation.symbolic_tail.tail_hash,
        occurrence_rte_steps=1,
    )
    append_probe(
        forced_request,
        probe_kind="forced_nonzero_taylor_phase_structure_probe",
        seed=None,
    )
    return rows


def _legacy_slope(
    proxy: Mapping[str, Any],
    metric: str,
) -> float:
    matches = [
        float(model["slope"])
        for model in proxy["models"]
        if model["axis"] == "cosine" and model["metric"] == metric
    ]
    if len(matches) != 1:
        raise ValueError("Legacy proxy lacks one cosine metric slope.")
    return matches[0]


def _proxy_recalibration(
    transfer_rows: Sequence[Mapping[str, Any]],
    selected_label: str,
    legacy_proxies: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[int, dict[str, dict[str, float]]]]:
    by_r: dict[str, Any] = {}
    corrections: dict[int, dict[str, dict[str, float]]] = {}
    for r in sorted(legacy_proxies):
        rows = [row for row in transfer_rows if int(row["sequence_length"]) == r]
        summary = _policy_summary(rows, selected_label)
        metrics: dict[str, Any] = {}
        correction_metrics: dict[str, dict[str, float]] = {}
        for metric in METRICS:
            old_slope = _legacy_slope(legacy_proxies[r], metric)
            delta = float(summary["metrics"][metric]["paired_delta"]["mean"])
            delta_se = float(
                summary["metrics"][metric]["paired_delta"]["standard_error"]
            )
            new_slope = old_slope + delta
            metrics[metric] = {
                "legacy_q_slope": old_slope,
                "central_occurrence_paired_delta": delta,
                "central_occurrence_paired_delta_standard_error": delta_se,
                "additive_bridge_q_slope": new_slope,
                "relative_slope_change": (
                    None if old_slope == 0.0 else new_slope / old_slope - 1.0
                ),
                "intercept_policy": "retain_legacy_wrapper_intercept",
            }
            correction_metrics[metric] = {"mean": delta, "standard_error": delta_se}
        by_r[str(r)] = {
            "rte_steps": r,
            "sample_count": len(rows),
            "metrics": metrics,
        }
        corrections[r] = correction_metrics
    rz_changes = [
        abs(float(row["metrics"]["rz_count"]["relative_slope_change"]))
        for row in by_r.values()
    ]
    return (
        {
            "model_formula": "legacy_intercept+(legacy_slope+paired_central_delta)*q",
            "full_wrapper_retranspiled": False,
            "status": "additive_bridge_candidate_for_WP05",
            "by_rte_steps": by_r,
            "maximum_absolute_rz_q_slope_relative_change": max(rz_changes),
        },
        corrections,
    )


def _intervals_overlap(left: Sequence[float], right: Sequence[float]) -> bool:
    return max(float(left[0]), float(right[0])) <= min(
        float(left[1]), float(right[1])
    )


def _ranking_bridge(
    wp04: Mapping[str, Any],
    corrections: Mapping[int, Mapping[str, Mapping[str, float]]],
) -> dict[str, Any]:
    full_setting = wp04["full_setting"]
    scenarios = {row["scenario_id"]: row for row in wp04["scenarios"]}
    ld3 = scenarios[full_setting["scenario_ids"]["3"]]
    ld12 = scenarios[full_setting["scenario_ids"]["12"]]
    correction_by_r: dict[int, float] = {}
    uncertainty_weight_by_r: dict[int, float] = {}
    rows = []
    for round_row in ld3["rounds"]:
        r = int(round_row["r_m"])
        q = int(round_row["q_m"])
        delta = float(corrections[r]["rz_count"]["mean"])
        delta_se = float(corrections[r]["rz_count"]["standard_error"])
        axis_shots = sum(int(axis["shots"]) for axis in round_row["axes"].values())
        correction = axis_shots * q * delta
        correction_by_r[r] = correction_by_r.get(r, 0.0) + correction
        uncertainty_weight_by_r[r] = (
            uncertainty_weight_by_r.get(r, 0.0) + axis_shots * q * delta_se
        )
        adjusted_axes = {
            axis_name: float(axis["predicted_rz_count_per_interrogation"])
            + q * delta
            for axis_name, axis in round_row["axes"].items()
        }
        if any(value < 0.0 for value in adjusted_axes.values()):
            raise ValueError("Additive bridge produced a negative RZ prediction.")
        rows.append(
            {
                "round_index": int(round_row["round_index"]),
                "q_m": q,
                "r_m": r,
                "paired_central_rz_delta_per_repetition": delta,
                "adjusted_predicted_rz_count_per_interrogation": adjusted_axes,
                "shot_weighted_total_correction": correction,
            }
        )
    total_correction = math.fsum(correction_by_r.values())
    correction_95_half_width = 1.96 * math.fsum(
        abs(value) for value in uncertainty_weight_by_r.values()
    )
    old_ld3 = float(ld3["total_compiled_rz_point_estimate"])
    new_ld3 = old_ld3 + total_correction
    ld12_total = float(ld12["total_compiled_rz_point_estimate"])
    old_interval = ld3["scenario_intervals"]["local_5_percent_plus_calibration"]
    new_interval = [
        float(old_interval[0]) + total_correction - correction_95_half_width,
        float(old_interval[1]) + total_correction + correction_95_half_width,
    ]
    ld12_interval = ld12["scenario_intervals"]["local_5_percent_plus_calibration"]
    return {
        "bridge_scope": (
            "fixed_WP04_rounds_shots_alpha_and_wrapper_intercepts_with_only_"
            "central_RTE_q_slope_corrected"
        ),
        "old_ld3_rz_point_estimate": old_ld3,
        "additive_rz_correction": total_correction,
        "additive_correction_95_half_width": correction_95_half_width,
        "adjusted_ld3_rz_point_estimate": new_ld3,
        "ld12_rz_point_estimate_unchanged": ld12_total,
        "adjusted_ld3_relative_change": new_ld3 / old_ld3 - 1.0,
        "old_point_preference": "L_D=12",
        "adjusted_point_preference": "L_D=3" if new_ld3 < ld12_total else "L_D=12",
        "point_ranking_reversed": new_ld3 < ld12_total,
        "adjusted_ld12_reduction_relative_to_ld3": 1.0 - ld12_total / new_ld3,
        "adjusted_ld3_local_interval": new_interval,
        "ld12_local_interval": list(ld12_interval),
        "local_intervals_overlap": _intervals_overlap(new_interval, ld12_interval),
        "round_corrections": rows,
        "alpha_reoptimized": False,
        "shot_counts_reoptimized": False,
        "full_wrapper_retranspiled": False,
    }


def evaluate_wp06b_sequence_policy(
    hamiltonian: DFHamiltonian,
    preparation: DFPartialS2Preparation,
    compiler: CompilerSettings,
    wp04: Mapping[str, Any],
    legacy_proxies: Mapping[int, Mapping[str, Any]],
    *,
    delta_time: float = 0.02,
    finite_taylor_order: int = 2,
    eta_decision: float = 0.05,
    equivalence_atol: float = 1e-10,
    training_lengths: Sequence[int] = (1, 2, 4),
    holdout_lengths: Sequence[int] = (3, 6),
    schedule_lengths: Sequence[int] = (1, 2, 4, 8, 16, 32),
    training_sample_count: int = 8,
    holdout_sample_count: int = 12,
    transfer_sample_count: int = 8,
    training_seed: int = 2026092201,
    holdout_seed: int = 2026092202,
    transfer_seed: int = 2026092203,
    equivalence_seed: int = 2026092204,
) -> dict[str, Any]:
    """Select one fixed run-threshold policy and validate it on unused streams."""
    if preparation.ld != 3:
        raise ValueError("WP06-b is fixed to the H4 L_D=3 randomized candidate.")
    if set(schedule_lengths) != set(legacy_proxies):
        raise ValueError("Legacy proxy coverage must match the schedule lengths.")
    if len({training_seed, holdout_seed, transfer_seed, equivalence_seed}) != 4:
        raise ValueError("WP06-b stream seeds must be distinct.")
    thresholds: tuple[int | None, ...] = (1, 2, 3, None)
    support_definitions, proof_records = register_support_restricted_bases(
        hamiltonian,
        preparation,
    )
    training_rows = _compile_policy_grid(
        preparation,
        compiler,
        support_definitions,
        delta_time=delta_time,
        finite_taylor_order=finite_taylor_order,
        lengths=training_lengths,
        sample_count=training_sample_count,
        master_seed=training_seed,
        partition="training",
        thresholds=thresholds,
        training_fingerprint=None,
    )
    selection = _choose_training_policy(
        training_rows,
        thresholds,
        eta_decision=eta_decision,
    )
    selected_label = str(selection["selected_policy"]["policy"])
    selected_threshold = selection["selected_policy"]["threshold"]
    training_fingerprint = str(selection["training_fingerprint"])
    holdout_rows = _compile_policy_grid(
        preparation,
        compiler,
        support_definitions,
        delta_time=delta_time,
        finite_taylor_order=finite_taylor_order,
        lengths=holdout_lengths,
        sample_count=holdout_sample_count,
        master_seed=holdout_seed,
        partition="holdout",
        thresholds=thresholds,
        training_fingerprint=training_fingerprint,
    )
    transfer_rows = _compile_policy_grid(
        preparation,
        compiler,
        support_definitions,
        delta_time=delta_time,
        finite_taylor_order=finite_taylor_order,
        lengths=schedule_lengths,
        sample_count=transfer_sample_count,
        master_seed=transfer_seed,
        partition="schedule_transfer",
        thresholds=(selected_threshold,),
        training_fingerprint=training_fingerprint,
    )
    labels = ["full_basis_shared", *(_threshold_label(item) for item in thresholds)]
    training_summary = _partition_summary(training_rows, labels)
    holdout_summary = _partition_summary(holdout_rows, labels)
    transfer_summary = _partition_summary(
        transfer_rows,
        ["full_basis_shared", selected_label],
    )
    probes = _operator_equivalence_probes(
        preparation,
        support_definitions,
        delta_time=delta_time,
        finite_taylor_order=finite_taylor_order,
        selected_threshold=selected_threshold,
        training_fingerprint=training_fingerprint,
        master_seed=equivalence_seed,
    )
    maximum_equivalence_residual = max(
        max(
            float(row["selected_vs_full_max_abs_difference"]),
            float(row["support_all_vs_full_max_abs_difference"]),
        )
        for row in probes
    )
    selected_holdout = holdout_summary["pooled"][selected_label]
    holdout_rz_change = float(
        selected_holdout["metrics"]["rz_count"]["candidate_relative_change"]
    )
    oracle_rz = [
        min(float(costs[label]["rz_count"]) for label in labels)
        for costs in (row["costs"] for row in holdout_rows)
    ]
    selected_rz = [
        float(row["costs"][selected_label]["rz_count"]) for row in holdout_rows
    ]
    full_rz = [
        float(row["costs"]["full_basis_shared"]["rz_count"])
        for row in holdout_rows
    ]
    holdout_oracle_regret = (
        math.fsum(selected_rz) - math.fsum(oracle_rz)
    ) / math.fsum(full_rz)
    recalibration, corrections = _proxy_recalibration(
        transfer_rows,
        selected_label,
        legacy_proxies,
    )
    ranking = _ranking_bridge(wp04, corrections)
    maximum_proof_residual = max(
        float(row["preserved_columns_max_abs_residual"])
        for row in proof_records
    )
    checks = {
        "support_basis_certificates_pass": maximum_proof_residual
        <= equivalence_atol,
        "controlled_operator_equivalence_passes": maximum_equivalence_residual
        <= equivalence_atol,
        "training_holdout_transfer_streams_are_independent": len(
            {training_seed, holdout_seed, transfer_seed, equivalence_seed}
        )
        == 4,
        "policy_fixed_before_holdout": all(
            selected_label in row["costs"] for row in holdout_rows
        ),
        "holdout_rz_guard_passes": holdout_rz_change <= eta_decision,
        "holdout_oracle_regret_within_eta": holdout_oracle_regret
        <= eta_decision,
        "all_schedule_r_values_recalibrated": set(
            int(value) for value in recalibration["by_rte_steps"]
        )
        == set(schedule_lengths),
        "ranking_bridge_predictions_remain_nonnegative": all(
            value >= 0.0
            for row in ranking["round_corrections"]
            for value in row[
                "adjusted_predicted_rz_count_per_interrogation"
            ].values()
        ),
        "final_total_cost_evaluation_not_claimed": True,
    }
    material_proxy_change = (
        float(recalibration["maximum_absolute_rz_q_slope_relative_change"])
        >= eta_decision
    )
    next_action = (
        "WP05_full_controlled_interrogation_connection_with_selected_policy"
        if all(checks.values())
        else "refine_WP06b_policy_before_WP05"
    )
    return {
        "configuration": {
            "molecule": "H4_chain",
            "geometry_angstrom": 1.0,
            "basis": "STO-3G",
            "n_qubits": hamiltonian.n_qubits,
            "df_rank": len(hamiltonian.lambdas),
            "ld": preparation.ld,
            "delta_time": delta_time,
            "finite_taylor_order": finite_taylor_order,
            "eta_decision": eta_decision,
            "equivalence_atol": equivalence_atol,
            "training_lengths": list(training_lengths),
            "holdout_lengths": list(holdout_lengths),
            "schedule_lengths": list(schedule_lengths),
            "training_sample_count_per_length": training_sample_count,
            "holdout_sample_count_per_length": holdout_sample_count,
            "transfer_sample_count_per_length": transfer_sample_count,
            "seeds": {
                "training": training_seed,
                "holdout": holdout_seed,
                "schedule_transfer": transfer_seed,
                "equivalence": equivalence_seed,
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
        "physical_instance": {
            "hamiltonian_hash": preparation.hamiltonian_hash,
            "partition_hash": preparation.partition_hash,
            "preparation_hash": preparation.preparation_hash,
            "tail_hash": preparation.tail_extraction.tail_hash,
            "exact_rte_lambda_r": preparation.exact_rte_lambda_r,
        },
        "support_basis_library": {
            "definition_count": len(support_definitions),
            "maximum_preserved_columns_max_abs_residual": maximum_proof_residual,
            "definitions": proof_records,
        },
        "training_selection": selection,
        "training_results": training_summary,
        "holdout_results": {
            **holdout_summary,
            "selected_policy": selected_label,
            "selected_policy_pooled_rz_relative_change": holdout_rz_change,
            "selected_policy_oracle_regret_over_full_rz": holdout_oracle_regret,
        },
        "operator_equivalence_probes": probes,
        "maximum_operator_equivalence_residual": maximum_equivalence_residual,
        "schedule_transfer_results": transfer_summary,
        "proxy_recalibration": recalibration,
        "candidate_ranking_bridge": ranking,
        "decision": {
            "selected_policy": selected_label,
            "selected_maximum_support_run_length": selected_threshold,
            "selected_policy_approved_for_WP05_validation_path": all(
                checks.values()
            ),
            "production_default_changed": False,
            "material_rz_q_slope_change": material_proxy_change,
            "point_ranking_reversed_in_additive_bridge": ranking[
                "point_ranking_reversed"
            ],
            "directional_result": (
                "undetermined_intervals_overlap"
                if ranking["local_intervals_overlap"]
                else "point_and_local_interval_separate_in_additive_bridge_only"
            ),
            "next_action": next_action,
            "wp05_status": (
                "ready_with_selected_policy_as_explicit_validation_input"
                if all(checks.values())
                else "waiting_for_policy_refinement"
            ),
        },
        "scope": {
            "production_builder_explicit_basis_plan_path_exercised": True,
            "production_default_changed": False,
            "physical_event_distribution_sampled": True,
            "independent_holdout_used": True,
            "full_partial_s2_step_retranspiled": False,
            "full_hadamard_interrogation_retranspiled": False,
            "state_preparation_included": False,
            "backend_execution_included": False,
            "decision_grade": False,
            "final_total_cost_evaluation_performed": False,
        },
        "limitations": [
            "The selected policy is trained and held out on one H4 L_D=3 snapshot and one topology-free Qiskit compiler context.",
            "The proxy update is an additive central-RTE slope bridge; WP05 must retranspile complete controlled Hadamard interrogations.",
            "The L_D=3 versus L_D=12 bridge retains WP04 schedules, shots, alpha allocation, and wrapper intercepts.",
            "No state preparation, backend, noise, quantum shots, final total cost, or scientific superiority is evaluated.",
        ],
        "checks": checks,
        "overall_pass": all(checks.values()),
        "summary": {
            "status": "WP06b_sequence_policy_and_additive_proxy_bridge_complete",
            "selected_policy": selected_label,
            "holdout_rz_relative_change": holdout_rz_change,
            "holdout_oracle_regret_over_full_rz": holdout_oracle_regret,
            "maximum_operator_equivalence_residual": maximum_equivalence_residual,
            "maximum_absolute_rz_q_slope_relative_change": recalibration[
                "maximum_absolute_rz_q_slope_relative_change"
            ],
            "point_ranking_reversed_in_additive_bridge": ranking[
                "point_ranking_reversed"
            ],
            "local_intervals_overlap_after_bridge": ranking[
                "local_intervals_overlap"
            ],
            "next_action": next_action,
        },
    }


def finalize_wp06b_artifact(
    body: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "stage": "WP06-b",
        **dict(body),
        "provenance": dict(provenance),
    }
    payload["content_fingerprint"] = fingerprint(payload)
    validate_wp06b_artifact(payload)
    return payload


def validate_wp06b_artifact(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported WP06-b schema.")
    if payload.get("method") != METHOD or payload.get("stage") != "WP06-b":
        raise ValueError("Unsupported WP06-b method or stage.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("WP06-b content_fingerprint mismatch.")
    checks = payload.get("checks", {})
    if payload.get("overall_pass") != (bool(checks) and all(checks.values())):
        raise ValueError("WP06-b overall status does not match its checks.")
    scope = payload.get("scope", {})
    if scope.get("final_total_cost_evaluation_performed") is not False:
        raise ValueError("WP06-b cannot claim a final total-cost evaluation.")
    if scope.get("decision_grade") is not False:
        raise ValueError("WP06-b must remain non-decision-grade.")
    if scope.get("production_default_changed") is not False:
        raise ValueError("WP06-b must not silently change the production default.")


def write_wp06b_artifact(payload: Mapping[str, Any], path: str | Path) -> None:
    validate_wp06b_artifact(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
