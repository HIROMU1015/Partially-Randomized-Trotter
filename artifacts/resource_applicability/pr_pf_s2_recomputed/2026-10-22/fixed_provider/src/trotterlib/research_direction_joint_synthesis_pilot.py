"""P-A pilot for interval-aware joint synthesis of DF event sequences."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

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
from .parallel_validation_executor import atomic_write_json
from .research_direction_sequence_policy import (
    METRICS,
    _cost_record,
    _derived_seed,
    _request_for_seed,
    _statistics,
    fingerprint,
    make_run_threshold_basis_plan,
    register_support_restricted_bases,
)
from .research_direction_structure_pilot import maximum_operator_difference
from .rte import CompilerSettings


SCHEMA_VERSION = "research_direction_joint_synthesis_pilot_v1"
METHOD = "pa_interval_union_basis_joint_synthesis_holdout_v1"
STAGE = "P-A-joint-circuit-sequence-synthesis"
POLICY_LABELS = (
    "full_basis_shared",
    "event_support_restricted",
    "support_run_le_1",
    "interval_union_dp",
)

DELTA_TIME = 0.02
FINITE_TAYLOR_ORDER = 2
TRAINING_LENGTHS = (2, 4, 6)
HOLDOUT_LENGTHS = (3, 5, 8)
TRAINING_SAMPLES_PER_LENGTH = 6
HOLDOUT_SAMPLES_PER_LENGTH = 8
TRAINING_SEED = 2026092501
HOLDOUT_SEED = 2026092502
EQUIVALENCE_SEED = 2026092503

# Theme-selection gates fixed before the production batch.
MINIMUM_POOLED_RZ_IMPROVEMENT_OVER_CURRENT = 0.02
MAXIMUM_PER_TRAJECTORY_RZ_INCREASE_OVER_CURRENT = 0.05
MAXIMUM_ORACLE_REGRET_OVER_FULL_RZ = 0.01
MINIMUM_CHANGED_TRAJECTORY_FRACTION = 0.20
EQUIVALENCE_ATOL = 1.0e-10


def preserved_columns_unitary_completion(
    full_unitary: np.ndarray,
    support: Sequence[int],
    *,
    atol: float = 1.0e-10,
) -> np.ndarray:
    """Preserve any nonempty column union and deterministically complete it."""
    unitary = np.asarray(full_unitary)
    if unitary.ndim != 2 or unitary.shape[0] != unitary.shape[1]:
        raise ValueError("full_unitary must be square.")
    n = unitary.shape[0]
    normalized = tuple(sorted({int(index) for index in support}))
    if not normalized or len(normalized) > n:
        raise ValueError("support must contain between one and n columns.")
    if any(index < 0 or index >= n for index in normalized):
        raise ValueError("support column is outside the unitary.")
    if np.max(np.abs(unitary.conj().T @ unitary - np.eye(n))) > atol:
        raise ValueError("full_unitary is not unitary within tolerance.")

    dtype = np.result_type(unitary.dtype, np.float64)
    completed = np.zeros((n, n), dtype=dtype)
    filled: list[int] = []
    for index in normalized:
        completed[:, index] = unitary[:, index]
        filled.append(index)
    for target in range(n):
        if target in normalized:
            continue
        accepted = False
        for seed in range(n):
            vector = np.eye(n, dtype=dtype)[:, seed].copy()
            for _pass in range(2):
                for index in filled:
                    vector -= completed[:, index] * np.vdot(
                        completed[:, index], vector
                    )
            norm = float(np.linalg.norm(vector))
            if norm > atol:
                completed[:, target] = vector / norm
                filled.append(target)
                accepted = True
                break
        if not accepted:
            raise ValueError("Could not complete the preserved-column unitary.")
    residual = float(
        np.max(np.abs(completed.conj().T @ completed - np.eye(n)))
    )
    if residual > 10.0 * atol:
        raise ValueError("Completed union-support basis is not unitary.")
    return completed


def _fragment_index(fragment_id: str) -> int:
    prefix = "df-fragment-"
    if not fragment_id.startswith(prefix):
        raise ValueError("DF component lacks a canonical fragment index.")
    return int(fragment_id[len(prefix) :])


def _source_unitaries(
    hamiltonian: DFHamiltonian,
    preparation: DFPartialS2Preparation,
) -> dict[tuple[str, str], np.ndarray]:
    result: dict[tuple[str, str], np.ndarray] = {}
    for spec in preparation.rte_preparation.component_specs:
        if not isinstance(spec, DFRTEComponentCircuitSpec):
            continue
        key = (spec.basis_id, str(spec.basis_hash))
        if key in result:
            continue
        unitary, _ = diag_hermitian(
            hamiltonian.g_matrices[_fragment_index(spec.df_fragment_id)],
            sort=preparation.diagonal_sort,
            assume_hermitian=True,
        )
        if describe_basis_change_operations(tuple(U_to_qiskit_ops_jw(unitary))) != (
            spec.basis_change_operations
        ):
            raise ValueError("Rebuilt source basis differs from preparation.")
        result[key] = np.asarray(unitary)
    return result


def _flatten_with_runs(
    request: DFRTEEventSequenceCircuitRequest,
) -> tuple[list[Any], list[list[int]]]:
    applications: list[Any] = []
    runs: list[list[int]] = []
    active_key: tuple[str, str] | None = None
    active_run: list[int] = []
    for event in request.events:
        for application in event.application_sequence:
            if application.is_identity:
                if active_run:
                    runs.append(active_run)
                    active_run = []
                    active_key = None
                continue
            index = len(applications)
            applications.append(application)
            key = (str(application.basis_id), str(application.basis_hash))
            if active_run and key != active_key:
                runs.append(active_run)
                active_run = []
            active_key = key
            active_run.append(index)
    if active_run:
        runs.append(active_run)
    return applications, runs


def _union_definition(
    preparation: DFPartialS2Preparation,
    source_unitaries: Mapping[tuple[str, str], np.ndarray],
    application: Any,
    support_union: tuple[int, ...],
    cache: dict[tuple[str, str, tuple[int, ...]], DFBasisDefinition],
) -> tuple[DFBasisDefinition, float]:
    source_key = (str(application.basis_id), str(application.basis_hash))
    key = (*source_key, support_union)
    if key not in cache:
        full = source_unitaries[source_key]
        restricted = preserved_columns_unitary_completion(full, support_union)
        label = "-".join(str(index) for index in support_union)
        cache[key] = preparation.rte_preparation.basis_registry.register(
            tuple(U_to_qiskit_ops_jw(restricted)),
            num_system_qubits=preparation.num_system_qubits,
            basis_id=f"{source_key[0]}-interval-union-{label}-pa-v1",
        )
    full = source_unitaries[source_key]
    restricted = preserved_columns_unitary_completion(full, support_union)
    residual = float(
        np.max(np.abs(full[:, support_union] - restricted[:, support_union]))
    )
    return cache[key], residual


def make_interval_union_basis_plan(
    request: DFRTEEventSequenceCircuitRequest,
    preparation: DFPartialS2Preparation,
    source_unitaries: Mapping[tuple[str, str], np.ndarray],
    union_cache: dict[tuple[str, str, tuple[int, ...]], DFBasisDefinition],
) -> tuple[DFRTEBasisPlan, dict[str, Any]]:
    """Minimize basis-operation count over contiguous full/union intervals."""
    applications, runs = _flatten_with_runs(request)
    choices: list[DFRTEApplicationBasisChoice | None] = [None] * len(applications)
    run_records: list[dict[str, Any]] = []
    total_transitions = 0
    maximum_residual = 0.0

    for run_indices in runs:
        run_apps = [applications[index] for index in run_indices]
        length = len(run_apps)
        # value: proxy cost, segment count, support-size sum, mode penalty, segments
        best: list[tuple[int, int, int, int, list[dict[str, Any]]] | None] = [
            (0, 0, 0, 0, [])
        ] + [None] * length
        for stop in range(1, length + 1):
            candidates = []
            for start in range(stop):
                prefix = best[start]
                if prefix is None:
                    continue
                segment_apps = run_apps[start:stop]
                support_union = tuple(
                    sorted(
                        {
                            int(index)
                            for app in segment_apps
                            for index in app.diagonal_pauli_support
                        }
                    )
                )
                source_definition = preparation.rte_preparation.basis_registry.definition(
                    str(segment_apps[0].basis_id)
                )
                union_definition, residual = _union_definition(
                    preparation,
                    source_unitaries,
                    segment_apps[0],
                    support_union,
                    union_cache,
                )
                maximum_residual = max(maximum_residual, residual)
                options = (
                    (
                        "full",
                        source_definition,
                        1,
                        len(source_definition.runtime_operations),
                    ),
                    (
                        "support_union",
                        union_definition,
                        0,
                        len(union_definition.runtime_operations),
                    ),
                )
                for mode, definition, mode_penalty, operation_count in options:
                    total_transitions += 1
                    segment = {
                        "start": start,
                        "stop": stop,
                        "length": stop - start,
                        "mode": mode,
                        "support_union": list(support_union),
                        "support_union_size": len(support_union),
                        "basis_id": definition.basis_id,
                        "basis_hash": definition.basis_hash,
                        "basis_operation_count": operation_count,
                    }
                    candidates.append(
                        (
                            prefix[0] + 2 * operation_count,
                            prefix[1] + 1,
                            prefix[2] + len(support_union),
                            prefix[3] + mode_penalty,
                            [*prefix[4], segment],
                        )
                    )
            best[stop] = min(
                candidates,
                key=lambda item: (item[0], item[1], item[2], item[3]),
            )
        selected = best[length]
        if selected is None:
            raise RuntimeError("Interval dynamic program failed to cover a run.")
        for segment in selected[4]:
            definition = preparation.rte_preparation.basis_registry.definition(
                segment["basis_id"]
            )
            for local_index in range(segment["start"], segment["stop"]):
                global_index = run_indices[local_index]
                application = applications[global_index]
                construction = (
                    "registered_full_basis"
                    if segment["mode"] == "full"
                    else "support_restricted_preserved_columns_v1"
                )
                choices[global_index] = DFRTEApplicationBasisChoice(
                    source_basis_id=str(application.basis_id),
                    source_basis_hash=str(application.basis_hash),
                    selected_basis_id=definition.basis_id,
                    selected_basis_hash=definition.basis_hash,
                    diagonal_pauli_support=tuple(
                        int(index)
                        for index in application.diagonal_pauli_support
                    ),
                    construction=construction,
                    preserved_columns_max_abs_residual=maximum_residual,
                )
        run_records.append(
            {
                "source_basis_id": str(run_apps[0].basis_id),
                "run_length": length,
                "selected_proxy_basis_operation_count": selected[0],
                "segment_count": selected[1],
                "segments": selected[4],
            }
        )
    if any(choice is None for choice in choices):
        raise RuntimeError("Interval plan did not assign every application.")
    plan = DFRTEBasisPlan(
        policy_id="interval_union_dp_basis_operation_count_v1",
        selection_objective=(
            "minimize_twice_basis_operation_count_then_segments_then_union_size"
        ),
        choices=tuple(choice for choice in choices if choice is not None),
    )
    return plan, {
        "run_count": len(runs),
        "runs": run_records,
        "dp_transition_count": total_transitions,
        "maximum_preserved_column_residual": maximum_residual,
        "selected_segment_count": sum(row["segment_count"] for row in run_records),
        "selected_multi_application_interval_count": sum(
            int(segment["length"] > 1)
            for row in run_records
            for segment in row["segments"]
        ),
        "selected_support_union_interval_count": sum(
            int(segment["mode"] == "support_union")
            for row in run_records
            for segment in row["segments"]
        ),
    }


def _compile_partition(
    hamiltonian: DFHamiltonian,
    preparation: DFPartialS2Preparation,
    compiler: CompilerSettings,
    *,
    lengths: Sequence[int],
    samples_per_length: int,
    master_seed: int,
    partition: str,
) -> list[dict[str, Any]]:
    support_definitions, _ = register_support_restricted_bases(
        hamiltonian, preparation
    )
    source_unitaries = _source_unitaries(hamiltonian, preparation)
    union_cache: dict[
        tuple[str, str, tuple[int, ...]], DFBasisDefinition
    ] = {}
    builder = QiskitDFRTEEventCircuitBuilder(
        basis_registry=preparation.rte_preparation.basis_registry
    )
    rows = []
    for length in lengths:
        for sample_index in range(samples_per_length):
            seed = _derived_seed(master_seed, partition, length, sample_index)
            request, event_digest = _request_for_seed(
                preparation,
                delta_time=DELTA_TIME,
                sequence_length=length,
                finite_taylor_order=FINITE_TAYLOR_ORDER,
                seed=seed,
            )
            current_plan = make_run_threshold_basis_plan(
                request,
                support_definitions,
                maximum_support_run_length=1,
            )
            event_support_plan = make_run_threshold_basis_plan(
                request,
                support_definitions,
                maximum_support_run_length=None,
            )
            interval_plan, interval_metadata = make_interval_union_basis_plan(
                request, preparation, source_unitaries, union_cache
            )
            built = {
                "full_basis_shared": builder.build_sequence(request),
                "event_support_restricted": builder.build_sequence(
                    request, basis_plan=event_support_plan
                ),
                "support_run_le_1": builder.build_sequence(
                    request, basis_plan=current_plan
                ),
                "interval_union_dp": builder.build_sequence(
                    request, basis_plan=interval_plan
                ),
            }
            costs = {
                label: _cost_record(result.circuit, compiler)
                for label, result in built.items()
            }
            oracle_label = min(
                POLICY_LABELS,
                key=lambda label: (
                    costs[label]["rz_count"],
                    costs[label]["cx_count"],
                    costs[label]["total_depth"],
                    POLICY_LABELS.index(label),
                ),
            )
            rows.append(
                {
                    "partition": partition,
                    "sequence_length": length,
                    "sample_index": sample_index,
                    "seed": seed,
                    "event_digest": event_digest,
                    "event_orders": [event.taylor_order for event in request.events],
                    "costs": costs,
                    "compiled_oracle_policy": oracle_label,
                    "interval_metadata": interval_metadata,
                    "current_plan_fingerprint": current_plan.plan_fingerprint,
                    "interval_plan_fingerprint": interval_plan.plan_fingerprint,
                    "interval_choice_differs_from_current": (
                        tuple(
                            choice.selected_basis_hash
                            for choice in interval_plan.choices
                        )
                        != tuple(
                            choice.selected_basis_hash
                            for choice in current_plan.choices
                        )
                    ),
                }
            )
    return rows


def _policy_summary(
    rows: Sequence[Mapping[str, Any]], label: str
) -> dict[str, Any]:
    result: dict[str, Any] = {"policy": label, "metrics": {}}
    for metric in METRICS:
        full = [float(row["costs"]["full_basis_shared"][metric]) for row in rows]
        current = [float(row["costs"]["support_run_le_1"][metric]) for row in rows]
        candidate = [float(row["costs"][label][metric]) for row in rows]
        candidate_vs_full = [right - left for left, right in zip(full, candidate)]
        candidate_vs_current = [
            right - left for left, right in zip(current, candidate)
        ]
        result["metrics"][metric] = {
            "full": _statistics(full),
            "current": _statistics(current),
            "candidate": _statistics(candidate),
            "candidate_minus_full": _statistics(candidate_vs_full),
            "candidate_minus_current": _statistics(candidate_vs_current),
            "candidate_relative_to_full": (
                None if math.fsum(full) == 0.0 else math.fsum(candidate) / math.fsum(full) - 1.0
            ),
            "candidate_relative_to_current": (
                None
                if math.fsum(current) == 0.0
                else math.fsum(candidate) / math.fsum(current) - 1.0
            ),
        }
    return result


def _partition_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lengths = sorted({int(row["sequence_length"]) for row in rows})
    return {
        "sample_count": len(rows),
        "sequence_lengths": lengths,
        "policies": {
            label: _policy_summary(rows, label) for label in POLICY_LABELS
        },
        "by_sequence_length": {
            str(length): {
                label: _policy_summary(
                    [row for row in rows if row["sequence_length"] == length],
                    label,
                )
                for label in POLICY_LABELS
            }
            for length in lengths
        },
        "event_stream_digest": fingerprint(
            {"event_digests": [row["event_digest"] for row in rows]}
        ),
    }


def _equivalence_probes(
    hamiltonian: DFHamiltonian,
    preparation: DFPartialS2Preparation,
) -> list[dict[str, Any]]:
    source_unitaries = _source_unitaries(hamiltonian, preparation)
    union_cache: dict[
        tuple[str, str, tuple[int, ...]], DFBasisDefinition
    ] = {}
    builder = QiskitDFRTEEventCircuitBuilder(
        basis_registry=preparation.rte_preparation.basis_registry
    )
    probes = []
    for length in HOLDOUT_LENGTHS:
        seed = _derived_seed(EQUIVALENCE_SEED, "equivalence", length)
        request, digest = _request_for_seed(
            preparation,
            delta_time=DELTA_TIME,
            sequence_length=length,
            finite_taylor_order=FINITE_TAYLOR_ORDER,
            seed=seed,
        )
        plan, metadata = make_interval_union_basis_plan(
            request, preparation, source_unitaries, union_cache
        )
        full = builder.build_sequence(request)
        candidate = builder.build_sequence(request, basis_plan=plan)
        probes.append(
            {
                "sequence_length": length,
                "seed": seed,
                "event_digest": digest,
                "operator_max_abs_difference": maximum_operator_difference(
                    full.circuit,
                    candidate.circuit,
                    allow_global_phase=False,
                ),
                "relative_ancilla_phase_matches": (
                    full.relative_ancilla_phase == candidate.relative_ancilla_phase
                ),
                "interval_metadata": metadata,
            }
        )
    return probes


def evaluate_joint_synthesis_pilot(
    hamiltonian: DFHamiltonian,
    preparation: DFPartialS2Preparation,
    compiler: CompilerSettings,
) -> dict[str, Any]:
    """Run fixed training diagnostics and an independent P-A holdout."""
    training_rows = _compile_partition(
        hamiltonian,
        preparation,
        compiler,
        lengths=TRAINING_LENGTHS,
        samples_per_length=TRAINING_SAMPLES_PER_LENGTH,
        master_seed=TRAINING_SEED,
        partition="training_diagnostic",
    )
    holdout_rows = _compile_partition(
        hamiltonian,
        preparation,
        compiler,
        lengths=HOLDOUT_LENGTHS,
        samples_per_length=HOLDOUT_SAMPLES_PER_LENGTH,
        master_seed=HOLDOUT_SEED,
        partition="holdout",
    )
    training_summary = _partition_summary(training_rows)
    holdout_summary = _partition_summary(holdout_rows)
    probes = _equivalence_probes(hamiltonian, preparation)

    current_rz = [float(row["costs"]["support_run_le_1"]["rz_count"]) for row in holdout_rows]
    candidate_rz = [float(row["costs"]["interval_union_dp"]["rz_count"]) for row in holdout_rows]
    full_rz = [float(row["costs"]["full_basis_shared"]["rz_count"]) for row in holdout_rows]
    oracle_rz = [
        min(float(row["costs"][label]["rz_count"]) for label in POLICY_LABELS)
        for row in holdout_rows
    ]
    pooled_change = math.fsum(candidate_rz) / math.fsum(current_rz) - 1.0
    per_trajectory_increases = [
        0.0 if current == 0.0 else candidate / current - 1.0
        for current, candidate in zip(current_rz, candidate_rz)
    ]
    oracle_regret = (math.fsum(candidate_rz) - math.fsum(oracle_rz)) / math.fsum(full_rz)
    changed_fraction = sum(
        int(row["interval_choice_differs_from_current"]) for row in holdout_rows
    ) / len(holdout_rows)
    multi_interval_count = sum(
        int(row["interval_metadata"]["selected_multi_application_interval_count"] > 0)
        for row in holdout_rows
    )
    support_union_count = sum(
        int(row["interval_metadata"]["selected_support_union_interval_count"] > 0)
        for row in holdout_rows
    )
    maximum_residual = max(
        float(row["operator_max_abs_difference"]) for row in probes
    )
    gates = {
        "operator_equivalence_pass": maximum_residual <= EQUIVALENCE_ATOL
        and all(row["relative_ancilla_phase_matches"] for row in probes),
        "pooled_rz_improvement_over_current_at_least_2pct": pooled_change
        <= -MINIMUM_POOLED_RZ_IMPROVEMENT_OVER_CURRENT,
        "per_trajectory_rz_increase_over_current_at_most_5pct": max(
            per_trajectory_increases
        )
        <= MAXIMUM_PER_TRAJECTORY_RZ_INCREASE_OVER_CURRENT,
        "compiled_oracle_regret_over_full_rz_at_most_1pct": oracle_regret
        <= MAXIMUM_ORACLE_REGRET_OVER_FULL_RZ,
        "interval_choice_changes_at_least_20pct_of_holdout": changed_fraction
        >= MINIMUM_CHANGED_TRAJECTORY_FRACTION,
        "multi_application_and_union_intervals_observed": multi_interval_count > 0
        and support_union_count > 0,
    }
    hypothesis_supported = all(gates.values())
    checks = {
        "fixed_snapshot_matches": preparation.ld == 3
        and hamiltonian.n_qubits == 8
        and len(hamiltonian.lambdas) == 12,
        "independent_streams": len({TRAINING_SEED, HOLDOUT_SEED, EQUIVALENCE_SEED})
        == 3,
        "holdout_grid_complete": len(holdout_rows)
        == len(HOLDOUT_LENGTHS) * HOLDOUT_SAMPLES_PER_LENGTH,
        "all_costs_nonnegative": all(
            value >= 0
            for row in [*training_rows, *holdout_rows]
            for policy in POLICY_LABELS
            for value in row["costs"][policy].values()
        ),
    }
    return {
        "configuration": {
            "molecule": "H4_chain",
            "geometry_angstrom": 1.0,
            "basis": "STO-3G",
            "n_qubits": 8,
            "df_rank": 12,
            "ld": 3,
            "delta_time": DELTA_TIME,
            "finite_taylor_order": FINITE_TAYLOR_ORDER,
            "training_lengths": list(TRAINING_LENGTHS),
            "holdout_lengths": list(HOLDOUT_LENGTHS),
            "training_samples_per_length": TRAINING_SAMPLES_PER_LENGTH,
            "holdout_samples_per_length": HOLDOUT_SAMPLES_PER_LENGTH,
            "seeds": {
                "training": TRAINING_SEED,
                "holdout": HOLDOUT_SEED,
                "equivalence": EQUIVALENCE_SEED,
            },
            "compiler": {
                "basis_gates": list(compiler.basis_gates),
                "optimization_level": compiler.optimization_level,
                "transpiler_seed": compiler.transpiler_seed,
                "coupling_map": compiler.coupling_map,
                "qiskit_version": compiler.qiskit_version,
            },
            "candidate_definition": (
                "dynamic programming over contiguous intervals within each "
                "source-basis run; each interval chooses the registered full "
                "basis or a preserved support-union completion"
            ),
            "candidate_objective": (
                "twice_basis_operation_count_then_segment_count_then_union_size"
            ),
            "gates": {
                "minimum_pooled_rz_improvement_over_current": (
                    MINIMUM_POOLED_RZ_IMPROVEMENT_OVER_CURRENT
                ),
                "maximum_per_trajectory_rz_increase_over_current": (
                    MAXIMUM_PER_TRAJECTORY_RZ_INCREASE_OVER_CURRENT
                ),
                "maximum_oracle_regret_over_full_rz": (
                    MAXIMUM_ORACLE_REGRET_OVER_FULL_RZ
                ),
                "minimum_changed_trajectory_fraction": (
                    MINIMUM_CHANGED_TRAJECTORY_FRACTION
                ),
                "equivalence_atol": EQUIVALENCE_ATOL,
            },
        },
        "physical_instance": {
            "hamiltonian_hash": preparation.hamiltonian_hash,
            "partition_hash": preparation.partition_hash,
            "preparation_hash": preparation.preparation_hash,
            "tail_hash": preparation.tail_extraction.tail_hash,
        },
        "baseline_contract": {
            "policies": list(POLICY_LABELS),
            "adjacent_basis_cancellation_enabled_for_all_policies": True,
            "compiled_oracle_scope": "minimum among the four listed policies only",
        },
        "training_diagnostic": training_summary,
        "holdout": holdout_summary,
        "operator_equivalence_probes": probes,
        "summary": {
            "holdout_pooled_rz_relative_change_vs_current": pooled_change,
            "maximum_holdout_trajectory_rz_relative_increase_vs_current": max(
                per_trajectory_increases
            ),
            "compiled_oracle_regret_over_full_rz": oracle_regret,
            "changed_holdout_trajectory_fraction": changed_fraction,
            "holdout_trajectories_with_multi_application_interval": multi_interval_count,
            "holdout_trajectories_with_support_union_interval": support_union_count,
            "maximum_operator_equivalence_residual": maximum_residual,
            "pa_hypothesis_supported_in_scope": hypothesis_supported,
        },
        "pilot_gates": gates,
        "decision": {
            "status": (
                "advance_pa_to_joint_synthesis_research"
                if hypothesis_supported
                else "do_not_advance_pa_on_current_h4_sequence_grid"
            ),
            "pa_primary_theme_candidate": hypothesis_supported,
            "next_action": "compare_PB_PC_PA_and_select_primary_theme",
        },
        "checks": checks,
        "overall_pass": all(checks.values()),
        "scope": {
            "new_circuit_compilation_performed": True,
            "full_partial_s2_or_hadamard_wrapper_compiled": False,
            "production_default_changed": False,
            "backend_or_noise_evaluated": False,
            "rpe_or_final_total_cost_evaluated": False,
            "scientific_superiority_claimed": False,
        },
    }


def finalize_joint_synthesis_pilot_artifact(
    body: Mapping[str, Any], *, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "stage": STAGE,
        **dict(body),
        "provenance": dict(provenance),
    }
    payload["content_fingerprint"] = fingerprint(payload)
    validate_joint_synthesis_pilot_artifact(payload)
    return payload


def validate_joint_synthesis_pilot_artifact(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported P-A joint-synthesis schema.")
    if payload.get("method") != METHOD or payload.get("stage") != STAGE:
        raise ValueError("Unsupported P-A joint-synthesis method or stage.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("P-A joint-synthesis artifact fingerprint mismatch.")
    checks = payload.get("checks", {})
    if payload.get("overall_pass") != (bool(checks) and all(checks.values())):
        raise ValueError("P-A overall status does not match checks.")
    scope = payload.get("scope", {})
    for key in (
        "full_partial_s2_or_hadamard_wrapper_compiled",
        "production_default_changed",
        "backend_or_noise_evaluated",
        "rpe_or_final_total_cost_evaluated",
        "scientific_superiority_claimed",
    ):
        if scope.get(key) is not False:
            raise ValueError(f"P-A artifact overstates scope: {key}.")


def write_joint_synthesis_pilot_artifact(
    payload: Mapping[str, Any], path: str | Path
) -> None:
    validate_joint_synthesis_pilot_artifact(payload)
    output = Path(path)
    if output.exists():
        raise ValueError(f"Refusing to replace existing artifact: {output}")
    atomic_write_json(output, payload)
