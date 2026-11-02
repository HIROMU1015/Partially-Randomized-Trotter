"""Preregistered nondegenerate mechanism validation for P-A interval synthesis."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .df_partial_s2 import DFPartialS2Preparation
from .df_rte_circuit import (
    DFRTEApplicationBasisChoice,
    DFRTEBasisPlan,
    DFRTEEventSequenceCircuitRequest,
)
from .df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
from .df_rte_tail import DFBasisDefinition
from .parallel_validation_executor import atomic_write_json
from .research_direction_full_scope import fingerprint
from .research_direction_joint_synthesis_pilot import (
    _flatten_with_runs,
    _source_unitaries,
    _union_definition,
    make_interval_union_basis_plan,
)
from .research_direction_sequence_policy import METRICS, _cost_record
from .research_direction_structure_pilot import maximum_operator_difference
from .rte import CompilerSettings, _make_event, finite_rte_distribution


EXPECTED_SCHEMA_VERSION = "pa_joint_synthesis_mechanism_expected_tasks_v1"
SCHEMA_VERSION = "pa_joint_synthesis_mechanism_validation_v1"
METHOD = "forced_support_order2_one_segment_baseline_blind_v1"
STAGE = "P-A-nondegenerate-mechanism-validation"

DELTA_TIME = 0.02
FINITE_TAYLOR_ORDER = 2
TRAINING_FRAGMENTS = (3, 5, 7)
BLIND_FRAGMENTS = (4, 6, 8)
EQUIVALENCE_FRAGMENT = 4

MINIMUM_BLIND_SPLIT_BASIS_COUNT = 2
MINIMUM_BLIND_SPLIT_PROFILE_COUNT = 2
MINIMUM_CHANGED_BLIND_ROW_FRACTION = 0.25
MINIMUM_POOLED_RZ_IMPROVEMENT = 0.02
MAXIMUM_PER_ROW_RZ_INCREASE = 0.05
EQUIVALENCE_ATOL = 1.0e-10

# Each event tuple is in circuit application order: product, product, rotation.
# Every event is constructed at Taylor order 2 from real H4 DF components.
SUPPORT_PROFILES: dict[
    str, tuple[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]], ...]
] = {
    "singleton_far_blocks": (
        ((0,), (0,), (0,)),
        ((7,), (7,), (7,)),
    ),
    "singleton_pair_far_blocks": (
        ((0,), (1,), (0,)),
        ((6,), (7,), (6,)),
    ),
    "zz_far_blocks": (
        ((0, 1), (0, 1), (0, 1)),
        ((6, 7), (6, 7), (6, 7)),
    ),
    "mixed_local_far_blocks": (
        ((0,), (0, 1), (1,)),
        ((6,), (6, 7), (7,)),
    ),
    "three_separated_zz_blocks": (
        ((0, 1), (0, 1), (0, 1)),
        ((3, 4), (3, 4), (3, 4)),
        ((6, 7), (6, 7), (6, 7)),
    ),
}


def _event_digest(request: DFRTEEventSequenceCircuitRequest) -> str:
    encoded = json.dumps(
        [event.to_dict() for event in request.events],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _fragment_component_indices(
    preparation: DFPartialS2Preparation,
    fragment_index: int,
) -> tuple[str, str, dict[tuple[int, ...], int]]:
    fragment_id = f"df-fragment-{int(fragment_index)}"
    components = preparation.rte_preparation.symbolic_tail.components
    matches = [
        (index, component)
        for index, component in enumerate(components)
        if not component.is_identity and component.df_fragment_id == fragment_id
    ]
    if not matches:
        raise ValueError(f"No non-identity components for {fragment_id}.")
    basis_ids = {str(component.basis_id) for _, component in matches}
    basis_hashes = {str(component.basis_hash) for _, component in matches}
    if len(basis_ids) != 1 or len(basis_hashes) != 1:
        raise ValueError(f"{fragment_id} does not have one source basis.")
    by_support: dict[tuple[int, ...], int] = {}
    for index, component in matches:
        support = tuple(int(value) for value in component.diagonal_pauli_support)
        if support in by_support:
            raise ValueError(f"Duplicate support {support} in {fragment_id}.")
        by_support[support] = index
    return next(iter(basis_ids)), next(iter(basis_hashes)), by_support


def build_forced_order2_request(
    preparation: DFPartialS2Preparation,
    *,
    fragment_index: int,
    profile_id: str,
) -> DFRTEEventSequenceCircuitRequest:
    """Build one fixed legal order-2 stream without stochastic order sampling."""
    if profile_id not in SUPPORT_PROFILES:
        raise ValueError(f"Unknown support profile: {profile_id}.")
    _basis_id, _basis_hash, by_support = _fragment_component_indices(
        preparation, fragment_index
    )
    profile = SUPPORT_PROFILES[profile_id]
    tail = preparation.rte_preparation.symbolic_tail
    tau = tail.lambda_r * DELTA_TIME / len(profile)
    distribution = finite_rte_distribution(tau, FINITE_TAYLOR_ORDER)
    order_index = distribution.orders.index(FINITE_TAYLOR_ORDER)
    components = tail.components
    events = []
    for application_supports in profile:
        if len(application_supports) != 3:
            raise ValueError("An order-2 profile event must contain three applications.")
        try:
            application_indices = [
                by_support[tuple(support)] for support in application_supports
            ]
        except KeyError as exc:
            raise ValueError(
                f"Profile {profile_id} uses unavailable support {exc.args[0]}."
            ) from exc
        # _make_event takes rotation first and products after it, then emits
        # products followed by rotation in circuit order.
        component_indices = (
            application_indices[2],
            application_indices[0],
            application_indices[1],
        )
        events.append(
            _make_event(
                component_indices,
                components,
                distribution,
                order_index,
            )
        )
    return DFRTEEventSequenceCircuitRequest(
        events=tuple(events),
        component_specs=preparation.rte_preparation.component_specs,
        controlled=True,
        ancilla_qubit=preparation.num_system_qubits,
        cancel_adjacent_equal_bases=True,
        tail_id=tail.tail_id,
        tail_hash=tail.tail_hash,
        occurrence_rte_steps=len(events),
    )


def _task_record(
    preparation: DFPartialS2Preparation,
    *,
    partition: str,
    fragment_index: int,
    profile_id: str,
) -> dict[str, Any]:
    request = build_forced_order2_request(
        preparation,
        fragment_index=fragment_index,
        profile_id=profile_id,
    )
    applications, runs = _flatten_with_runs(request)
    basis_ids = {str(app.basis_id) for app in applications}
    basis_hashes = {str(app.basis_hash) for app in applications}
    return {
        "task_key": f"{partition}__fragment_{fragment_index}__{profile_id}",
        "partition": partition,
        "fragment_index": int(fragment_index),
        "profile_id": profile_id,
        "event_count": len(request.events),
        "application_count": len(applications),
        "event_orders": [int(event.taylor_order) for event in request.events],
        "source_basis_ids": sorted(basis_ids),
        "source_basis_hashes": sorted(basis_hashes),
        "source_run_count": len(runs),
        "source_run_lengths": [len(run) for run in runs],
        "event_digest": _event_digest(request),
    }


def build_expected_task_manifest(
    preparation: DFPartialS2Preparation,
    *,
    source_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    tasks = []
    for partition, fragments in (
        ("training_diagnostic", TRAINING_FRAGMENTS),
        ("blind_holdout", BLIND_FRAGMENTS),
    ):
        for fragment_index in fragments:
            for profile_id in SUPPORT_PROFILES:
                tasks.append(
                    _task_record(
                        preparation,
                        partition=partition,
                        fragment_index=fragment_index,
                        profile_id=profile_id,
                    )
                )
    payload: dict[str, Any] = {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "method": METHOD,
        "created_before_compilation": True,
        "configuration": {
            "molecule": "H4_linear_chain",
            "geometry_angstrom": 1.0,
            "basis": "STO-3G",
            "n_qubits": 8,
            "df_rank": 12,
            "ld": 3,
            "delta_time": DELTA_TIME,
            "finite_taylor_order": FINITE_TAYLOR_ORDER,
            "training_fragments": list(TRAINING_FRAGMENTS),
            "blind_fragments": list(BLIND_FRAGMENTS),
            "profiles": {
                profile_id: [
                    [[int(index) for index in support] for support in event]
                    for event in events
                ]
                for profile_id, events in SUPPORT_PROFILES.items()
            },
            "primary_gates": {
                "minimum_blind_split_basis_count": (
                    MINIMUM_BLIND_SPLIT_BASIS_COUNT
                ),
                "minimum_blind_split_profile_count": (
                    MINIMUM_BLIND_SPLIT_PROFILE_COUNT
                ),
                "minimum_changed_blind_row_fraction": (
                    MINIMUM_CHANGED_BLIND_ROW_FRACTION
                ),
                "minimum_pooled_rz_improvement": MINIMUM_POOLED_RZ_IMPROVEMENT,
                "maximum_per_row_rz_increase": MAXIMUM_PER_ROW_RZ_INCREASE,
                "equivalence_atol": EQUIVALENCE_ATOL,
            },
        },
        "task_count": len(tasks),
        "tasks": tasks,
        "source_evidence": dict(source_evidence),
        "decision_rule": {
            "pass": (
                "All frozen blind structural, compiled-RZ, and operator-equivalence "
                "gates pass without threshold changes."
            ),
            "pass_status": "advance_pa_after_nondegenerate_mechanism_validation",
            "fail_status": "stop_pa_interval_dp_as_primary_and_return_to_pc",
            "threshold_relaxation_after_results": False,
        },
    }
    payload["content_fingerprint"] = fingerprint(payload)
    validate_expected_task_manifest(payload)
    return payload


def validate_expected_task_manifest(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise ValueError("Unsupported P-A mechanism expected-task schema.")
    if payload.get("method") != METHOD:
        raise ValueError("Unexpected P-A mechanism method.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("P-A mechanism expected-task fingerprint mismatch.")
    if payload.get("created_before_compilation") is not True:
        raise ValueError("Expected tasks were not marked pre-compilation.")
    tasks = list(payload.get("tasks", ()))
    expected_count = (
        (len(TRAINING_FRAGMENTS) + len(BLIND_FRAGMENTS))
        * len(SUPPORT_PROFILES)
    )
    if payload.get("task_count") != expected_count or len(tasks) != expected_count:
        raise ValueError("P-A mechanism expected task count is incomplete.")
    keys = [str(task.get("task_key")) for task in tasks]
    if len(set(keys)) != len(keys):
        raise ValueError("P-A mechanism expected task keys are not unique.")
    for task in tasks:
        event_count = int(task["event_count"])
        if task.get("event_orders") != [FINITE_TAYLOR_ORDER] * event_count:
            raise ValueError("Expected task does not contain only Taylor-order-2 events.")
        if int(task["application_count"]) != 3 * event_count:
            raise ValueError("Expected task application count is inconsistent.")
        if task.get("source_run_count") != 1:
            raise ValueError("Forced-support task must contain one source-basis run.")
        if len(task.get("source_basis_ids", ())) != 1:
            raise ValueError("Forced-support task must use one source basis.")
        if task.get("source_run_lengths") != [3 * event_count]:
            raise ValueError("Forced-support task run length is inconsistent.")


def make_one_segment_per_source_run_basis_plan(
    request: DFRTEEventSequenceCircuitRequest,
    preparation: DFPartialS2Preparation,
    source_unitaries: Mapping[tuple[str, str], Any],
    union_cache: dict[tuple[str, str, tuple[int, ...]], DFBasisDefinition],
) -> tuple[DFRTEBasisPlan, dict[str, Any]]:
    """Choose full or support-union once for each maximal source-basis run."""
    applications, runs = _flatten_with_runs(request)
    choices: list[DFRTEApplicationBasisChoice | None] = [None] * len(applications)
    records = []
    maximum_residual = 0.0
    for run_indices in runs:
        run_apps = [applications[index] for index in run_indices]
        support_union = tuple(
            sorted(
                {
                    int(index)
                    for application in run_apps
                    for index in application.diagonal_pauli_support
                }
            )
        )
        source = preparation.rte_preparation.basis_registry.definition(
            str(run_apps[0].basis_id)
        )
        union, residual = _union_definition(
            preparation,
            source_unitaries,
            run_apps[0],
            support_union,
            union_cache,
        )
        maximum_residual = max(maximum_residual, residual)
        mode, selected = min(
            (("full", source), ("support_union", union)),
            key=lambda item: (
                2 * len(item[1].runtime_operations),
                1,
                len(support_union),
                int(item[0] == "full"),
            ),
        )
        construction = (
            "registered_full_basis"
            if mode == "full"
            else "support_restricted_preserved_columns_v1"
        )
        for global_index in run_indices:
            application = applications[global_index]
            choices[global_index] = DFRTEApplicationBasisChoice(
                source_basis_id=str(application.basis_id),
                source_basis_hash=str(application.basis_hash),
                selected_basis_id=selected.basis_id,
                selected_basis_hash=selected.basis_hash,
                diagonal_pauli_support=tuple(
                    int(index) for index in application.diagonal_pauli_support
                ),
                construction=construction,
                preserved_columns_max_abs_residual=residual,
            )
        records.append(
            {
                "source_basis_id": str(run_apps[0].basis_id),
                "run_length": len(run_apps),
                "mode": mode,
                "support_union": list(support_union),
                "support_union_size": len(support_union),
                "basis_operation_count": len(selected.runtime_operations),
            }
        )
    if any(choice is None for choice in choices):
        raise RuntimeError("One-segment baseline did not cover every application.")
    plan = DFRTEBasisPlan(
        policy_id="one_segment_per_source_run_full_or_support_union_v1",
        selection_objective=(
            "same_four_level_objective_as_interval_v1_restricted_to_one_segment"
        ),
        choices=tuple(choice for choice in choices if choice is not None),
    )
    return plan, {
        "run_count": len(runs),
        "selected_segment_count": len(runs),
        "runs": records,
        "maximum_preserved_column_residual": maximum_residual,
    }


def _compile_row(
    preparation: DFPartialS2Preparation,
    compiler: CompilerSettings,
    task: Mapping[str, Any],
    *,
    source_unitaries: Mapping[tuple[str, str], Any],
    union_cache: dict[tuple[str, str, tuple[int, ...]], DFBasisDefinition],
) -> dict[str, Any]:
    request = build_forced_order2_request(
        preparation,
        fragment_index=int(task["fragment_index"]),
        profile_id=str(task["profile_id"]),
    )
    if _event_digest(request) != task["event_digest"]:
        raise ValueError(f"Expected event digest mismatch for {task['task_key']}.")
    baseline_plan, baseline_metadata = make_one_segment_per_source_run_basis_plan(
        request, preparation, source_unitaries, union_cache
    )
    candidate_plan, candidate_metadata = make_interval_union_basis_plan(
        request, preparation, source_unitaries, union_cache
    )
    builder = QiskitDFRTEEventCircuitBuilder(
        basis_registry=preparation.rte_preparation.basis_registry
    )
    baseline = builder.build_sequence(request, basis_plan=baseline_plan)
    candidate = builder.build_sequence(request, basis_plan=candidate_plan)
    costs = {
        "one_segment_per_source_run": _cost_record(baseline.circuit, compiler),
        "interval_union_dp": _cost_record(candidate.circuit, compiler),
    }
    return {
        **dict(task),
        "baseline_plan_fingerprint": baseline_plan.plan_fingerprint,
        "candidate_plan_fingerprint": candidate_plan.plan_fingerprint,
        "baseline_metadata": baseline_metadata,
        "candidate_metadata": candidate_metadata,
        "within_run_split_count": (
            int(candidate_metadata["selected_segment_count"])
            - int(candidate_metadata["run_count"])
        ),
        "candidate_differs_from_baseline": tuple(
            choice.selected_basis_hash for choice in candidate_plan.choices
        )
        != tuple(choice.selected_basis_hash for choice in baseline_plan.choices),
        "costs": costs,
    }


def _partition_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    baseline_rz = [
        float(row["costs"]["one_segment_per_source_run"]["rz_count"])
        for row in rows
    ]
    candidate_rz = [
        float(row["costs"]["interval_union_dp"]["rz_count"]) for row in rows
    ]
    per_row_changes = [
        candidate / baseline - 1.0
        for baseline, candidate in zip(baseline_rz, candidate_rz, strict=True)
    ]
    split_rows = [row for row in rows if int(row["within_run_split_count"]) > 0]
    metrics = {}
    for metric in METRICS:
        baseline = math.fsum(
            float(row["costs"]["one_segment_per_source_run"][metric]) for row in rows
        )
        candidate = math.fsum(
            float(row["costs"]["interval_union_dp"][metric]) for row in rows
        )
        metrics[metric] = {
            "baseline_sum": baseline,
            "candidate_sum": candidate,
            "candidate_relative_to_baseline": candidate / baseline - 1.0,
        }
    return {
        "row_count": len(rows),
        "split_row_count": len(split_rows),
        "split_basis_ids": sorted(
            {str(row["source_basis_ids"][0]) for row in split_rows}
        ),
        "split_profile_ids": sorted(
            {str(row["profile_id"]) for row in split_rows}
        ),
        "changed_row_count": sum(
            int(bool(row["candidate_differs_from_baseline"])) for row in rows
        ),
        "improved_rz_row_count": sum(
            int(candidate < baseline)
            for baseline, candidate in zip(baseline_rz, candidate_rz, strict=True)
        ),
        "maximum_per_row_rz_relative_change": max(per_row_changes),
        "minimum_per_row_rz_relative_change": min(per_row_changes),
        "metrics": metrics,
    }


def evaluate_mechanism_validation(
    hamiltonian: Any,
    preparation: DFPartialS2Preparation,
    compiler: CompilerSettings,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile the frozen tasks and apply only the preregistered blind gates."""
    validate_expected_task_manifest(expected)
    source_unitaries = _source_unitaries(hamiltonian, preparation)
    union_cache: dict[
        tuple[str, str, tuple[int, ...]], DFBasisDefinition
    ] = {}
    rows = [
        _compile_row(
            preparation,
            compiler,
            task,
            source_unitaries=source_unitaries,
            union_cache=union_cache,
        )
        for task in expected["tasks"]
    ]
    training_rows = [
        row for row in rows if row["partition"] == "training_diagnostic"
    ]
    blind_rows = [row for row in rows if row["partition"] == "blind_holdout"]
    training = _partition_summary(training_rows)
    blind = _partition_summary(blind_rows)
    probes = _operator_probes_with_source(
        hamiltonian,
        preparation,
    )
    maximum_operator_difference_value = max(
        max(
            float(probe["baseline_operator_max_abs_difference"]),
            float(probe["candidate_operator_max_abs_difference"]),
        )
        for probe in probes
    )
    blind_changed_fraction = blind["changed_row_count"] / blind["row_count"]
    blind_pooled_rz_change = blind["metrics"]["rz_count"][
        "candidate_relative_to_baseline"
    ]
    gates = {
        "all_blind_events_are_taylor_order_2": all(
            order == FINITE_TAYLOR_ORDER
            for row in blind_rows
            for order in row["event_orders"]
        ),
        "within_run_split_transfers_to_at_least_2_blind_bases": len(
            blind["split_basis_ids"]
        )
        >= MINIMUM_BLIND_SPLIT_BASIS_COUNT,
        "within_run_split_transfers_to_at_least_2_blind_profiles": len(
            blind["split_profile_ids"]
        )
        >= MINIMUM_BLIND_SPLIT_PROFILE_COUNT,
        "candidate_plan_differs_on_at_least_25pct_of_blind_rows": (
            blind_changed_fraction >= MINIMUM_CHANGED_BLIND_ROW_FRACTION
        ),
        "pooled_blind_rz_improves_by_at_least_2pct": (
            blind_pooled_rz_change <= -MINIMUM_POOLED_RZ_IMPROVEMENT
        ),
        "maximum_blind_row_rz_increase_is_at_most_5pct": (
            float(blind["maximum_per_row_rz_relative_change"])
            <= MAXIMUM_PER_ROW_RZ_INCREASE
        ),
        "operator_equivalence_and_relative_phase_pass": (
            maximum_operator_difference_value <= EQUIVALENCE_ATOL
            and all(
                bool(probe["baseline_relative_ancilla_phase_matches"])
                and bool(probe["candidate_relative_ancilla_phase_matches"])
                for probe in probes
            )
        ),
    }
    passed = all(gates.values())
    status = (
        "advance_pa_after_nondegenerate_mechanism_validation"
        if passed
        else "stop_pa_interval_dp_as_primary_and_return_to_pc"
    )
    return {
        "configuration": dict(expected["configuration"]),
        "expected_task_fingerprint": expected["content_fingerprint"],
        "physical_instance": {
            "hamiltonian_hash": preparation.hamiltonian_hash,
            "partition_hash": preparation.partition_hash,
            "preparation_hash": preparation.preparation_hash,
            "tail_hash": preparation.tail_extraction.tail_hash,
        },
        "compiler": {
            "basis_gates": list(compiler.basis_gates),
            "optimization_level": compiler.optimization_level,
            "transpiler_seed": compiler.transpiler_seed,
            "coupling_map": compiler.coupling_map,
            "qiskit_version": compiler.qiskit_version,
        },
        "baseline_contract": {
            "baseline": "one_segment_per_source_run",
            "candidate": "interval_union_dp",
            "same_four_level_proxy_objective": True,
            "only_candidate_can_partition_within_a_source_run": True,
            "adjacent_equal_basis_cancellation_enabled": True,
        },
        "training_diagnostic": training,
        "blind_holdout": blind,
        "rows": rows,
        "operator_probes": probes,
        "gates": gates,
        "overall_pass": passed,
        "decision": {
            "status": status,
            "primary_theme": (
                "P-A_DF_event_sequence_joint_synthesis"
                if passed
                else "P-C_geometry_energy_difference"
            ),
            "thresholds_changed_after_results": False,
            "next_action": (
                "formalize_predictive_structure_and_reserve_a_new_physical_holdout"
                if passed
                else "stop_interval_dp_primary_claim_and_continue_with_P-C"
            ),
        },
        "scope": {
            "forced_legal_order2_event_streams": True,
            "natural_sampling_frequency_estimated": False,
            "new_physical_system_evaluated": False,
            "routed_or_noisy_backend_evaluated": False,
            "full_wrapper_or_rpe_total_cost_evaluated": False,
            "h12_evaluated": False,
            "literature_novelty_established": False,
            "global_circuit_optimality_established": False,
            "scientific_superiority_claimed": False,
        },
        "checks": {
            "expected_manifest_valid": True,
            "expected_task_count_matches": len(rows) == expected["task_count"],
            "training_blind_disjoint_fragments": not (
                set(TRAINING_FRAGMENTS) & set(BLIND_FRAGMENTS)
            ),
            "all_task_digests_reproduced": all(
                row["event_digest"] == task["event_digest"]
                for row, task in zip(rows, expected["tasks"], strict=True)
            ),
        },
    }


def _operator_probes_with_source(
    hamiltonian: Any,
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
    for profile_id in SUPPORT_PROFILES:
        request = build_forced_order2_request(
            preparation,
            fragment_index=EQUIVALENCE_FRAGMENT,
            profile_id=profile_id,
        )
        baseline_plan, _ = make_one_segment_per_source_run_basis_plan(
            request, preparation, source_unitaries, union_cache
        )
        candidate_plan, metadata = make_interval_union_basis_plan(
            request, preparation, source_unitaries, union_cache
        )
        full = builder.build_sequence(request)
        baseline = builder.build_sequence(request, basis_plan=baseline_plan)
        candidate = builder.build_sequence(request, basis_plan=candidate_plan)
        probes.append(
            {
                "profile_id": profile_id,
                "fragment_index": EQUIVALENCE_FRAGMENT,
                "event_digest": _event_digest(request),
                "within_run_split_count": (
                    int(metadata["selected_segment_count"])
                    - int(metadata["run_count"])
                ),
                "baseline_operator_max_abs_difference": maximum_operator_difference(
                    full.circuit,
                    baseline.circuit,
                    allow_global_phase=False,
                ),
                "candidate_operator_max_abs_difference": maximum_operator_difference(
                    full.circuit,
                    candidate.circuit,
                    allow_global_phase=False,
                ),
                "baseline_relative_ancilla_phase_matches": (
                    full.relative_ancilla_phase == baseline.relative_ancilla_phase
                ),
                "candidate_relative_ancilla_phase_matches": (
                    full.relative_ancilla_phase == candidate.relative_ancilla_phase
                ),
            }
        )
    return probes


def finalize_mechanism_validation_artifact(
    body: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "stage": STAGE,
        **dict(body),
        "provenance": dict(provenance),
    }
    payload["content_fingerprint"] = fingerprint(payload)
    validate_mechanism_validation_artifact(payload)
    return payload


def validate_mechanism_validation_artifact(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported P-A mechanism validation schema.")
    if payload.get("method") != METHOD or payload.get("stage") != STAGE:
        raise ValueError("Unexpected P-A mechanism method or stage.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("P-A mechanism artifact fingerprint mismatch.")
    checks = payload.get("checks", {})
    if not checks or not all(bool(value) for value in checks.values()):
        raise ValueError("P-A mechanism artifact consistency checks failed.")
    gates = payload.get("gates", {})
    passed = bool(gates) and all(bool(value) for value in gates.values())
    if payload.get("overall_pass") != passed:
        raise ValueError("P-A mechanism overall pass does not match gates.")
    expected_status = (
        "advance_pa_after_nondegenerate_mechanism_validation"
        if passed
        else "stop_pa_interval_dp_as_primary_and_return_to_pc"
    )
    if payload.get("decision", {}).get("status") != expected_status:
        raise ValueError("P-A mechanism decision does not follow frozen gates.")
    if payload.get("decision", {}).get("thresholds_changed_after_results") is not False:
        raise ValueError("P-A mechanism thresholds changed after results.")
    scope = payload.get("scope", {})
    for key in (
        "natural_sampling_frequency_estimated",
        "new_physical_system_evaluated",
        "routed_or_noisy_backend_evaluated",
        "full_wrapper_or_rpe_total_cost_evaluated",
        "h12_evaluated",
        "literature_novelty_established",
        "global_circuit_optimality_established",
        "scientific_superiority_claimed",
    ):
        if scope.get(key) is not False:
            raise ValueError(f"P-A mechanism scope overstated: {key}.")
    if scope.get("forced_legal_order2_event_streams") is not True:
        raise ValueError("P-A mechanism artifact lost its forced-stream scope.")


def write_expected_task_manifest(
    payload: Mapping[str, Any],
    path: str | Path,
) -> None:
    validate_expected_task_manifest(payload)
    output = Path(path)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != dict(payload):
            raise ValueError(f"Refusing to replace different expected tasks: {output}")
        return
    atomic_write_json(output, payload)


def write_mechanism_validation_artifact(
    payload: Mapping[str, Any],
    path: str | Path,
) -> None:
    validate_mechanism_validation_artifact(payload)
    output = Path(path)
    if output.exists():
        raise ValueError(f"Refusing to replace existing artifact: {output}")
    atomic_write_json(output, payload)
