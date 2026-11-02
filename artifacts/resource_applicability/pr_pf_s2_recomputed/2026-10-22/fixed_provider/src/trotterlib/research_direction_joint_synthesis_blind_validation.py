"""Preregistered blind transfer validation for the frozen P-A v1 policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import resource
import time
from typing import Any, Callable, Mapping, Sequence

from .df_hamiltonian import DFHamiltonian
from .df_partial_s2 import DFPartialS2Preparation
from .df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
from .df_rte_tail import DFBasisDefinition
from .research_direction_joint_synthesis_pilot import (
    EQUIVALENCE_ATOL,
    MAXIMUM_ORACLE_REGRET_OVER_FULL_RZ,
    MAXIMUM_PER_TRAJECTORY_RZ_INCREASE_OVER_CURRENT,
    MINIMUM_CHANGED_TRAJECTORY_FRACTION,
    MINIMUM_POOLED_RZ_IMPROVEMENT_OVER_CURRENT,
    POLICY_LABELS,
    _partition_summary,
    _source_unitaries,
    make_interval_union_basis_plan,
)
from .research_direction_sequence_policy import (
    _cost_record,
    _derived_seed,
    _request_for_seed,
    fingerprint,
    make_run_threshold_basis_plan,
    register_support_restricted_bases,
)
from .research_direction_structure_pilot import maximum_operator_difference
from .rte import CompilerSettings


SCHEMA_VERSION = "research_direction_joint_synthesis_blind_validation_v1"
EXPECTED_MANIFEST_SCHEMA = "pa_v1_blind_expected_tasks_v1"
CHECKPOINT_SCHEMA = "pa_v1_blind_checkpoint_v1"
METHOD = "pa_v1_preregistered_h5_physical_h4_opt2_transfer_v1"
STAGE = "P-A-blind-transfer-validation"

HOLDOUT_LENGTHS = (3, 5, 8)
HOLDOUT_SAMPLES_PER_LENGTH = 8
FINITE_TAYLOR_ORDER = 2

FROZEN_CORE_HASHES = {
    "src/trotterlib/research_direction_joint_synthesis_pilot.py": (
        "6f239f0c1f11fda55f4949b730b2087eff3b3538bc003da035767f69dad0d92b"
    ),
    "src/trotterlib/df_rte_qiskit.py": (
        "c710de23b7bfef525199ab28b897c21b2a78723a01b29da6067ff64354dcd436"
    ),
}
FROZEN_PILOT_FILE_SHA256 = (
    "b5165af6561ca7b1ba8dfe6920bbdd01a2c2c3699d8fd79ed721eba000b8b64f"
)
FROZEN_PILOT_CONTENT_FINGERPRINT = (
    "1a9840a4ee46daa6e3749272593acf3e8ae29be9fcecc1a05b0bd9ea817fbc37"
)
FROZEN_H4_EVENT_STREAM_DIGEST = (
    "22f753022216bbb9ce31e36cd4cbbb910f23c74eb962acbc57e13209d8aebfdd"
)
FROZEN_H4_PROBE_DIGESTS = {
    3: "466355626759c64128991997270ffcb02fca4edfd20393437e66279add32065f",
    5: "c72f22a511452b7883667160fe87fdfae475bea928cfa785bc4ffd5eaa3d39f1",
    8: "b2b7bcebf3aa7f5819c3cd8bf9793317336a9ac235052d4bec745c47e757168a",
}


@dataclass(frozen=True)
class BlindStratumSpec:
    stratum_id: str
    molecule: str
    geometry_angstrom: float
    basis: str
    n_qubits: int
    df_rank: int
    ld: int
    delta_time: float
    optimization_level: int
    holdout_master_seed: int
    operator_master_seed: int
    holdout_partition: str
    operator_partition: str
    expected_hamiltonian_hash: str
    expected_partition_hash: str
    expected_preparation_hash: str
    expected_event_stream_digest: str | None = None


H5_PHYSICAL_TRANSFER = BlindStratumSpec(
    stratum_id="h5_physical_transfer",
    molecule="H5_chain_charge_plus_1_multiplicity_3",
    geometry_angstrom=1.0,
    basis="STO-3G",
    n_qubits=10,
    df_rank=9,
    ld=4,
    delta_time=0.025,
    optimization_level=1,
    holdout_master_seed=2026092601,
    operator_master_seed=2026092602,
    holdout_partition="h5_physical_transfer_blind",
    operator_partition="h5_physical_transfer_equivalence",
    expected_hamiltonian_hash=(
        "3b7f161147cc72936ad74863933e14b58dc45d9fb32ab1be9b8ab4f70e6ff60e"
    ),
    expected_partition_hash=(
        "94e7767b3a802aa84975f60db13daf59d16f8a97382a9204b00becf1777f834e"
    ),
    expected_preparation_hash=(
        "4a0ceb1fccaa8ed3a5584d77e7e85b9c0bad8dabe1028da385470b7def88e0c0"
    ),
)

H4_COMPILER_TRANSFER = BlindStratumSpec(
    stratum_id="h4_compiler_transfer_opt2",
    molecule="H4_chain",
    geometry_angstrom=1.0,
    basis="STO-3G",
    n_qubits=8,
    df_rank=12,
    ld=3,
    delta_time=0.02,
    optimization_level=2,
    holdout_master_seed=2026092502,
    operator_master_seed=2026092503,
    holdout_partition="holdout",
    operator_partition="equivalence",
    expected_hamiltonian_hash=(
        "56e4df83655aa2f2f8132126f2635996516dc2cfb1bf4cb8d95490184ad631e5"
    ),
    expected_partition_hash=(
        "9034397f5bde1f20cb712578d14245f4dac1236413d34642a75d83d867b52086"
    ),
    expected_preparation_hash=(
        "50c77f5455cc1a3c0710ed3c7588efc957b9cead7436bd0a6d851af61738a4bc"
    ),
    expected_event_stream_digest=FROZEN_H4_EVENT_STREAM_DIGEST,
)

STRATA = (H5_PHYSICAL_TRANSFER, H4_COMPILER_TRANSFER)


def task_key(stratum_id: str, sequence_length: int, sample_index: int) -> str:
    return f"{stratum_id}__L{sequence_length}__sample{sample_index}"


def probe_key(stratum_id: str, sequence_length: int) -> str:
    return f"{stratum_id}__L{sequence_length}__operator_probe"


def _physical_checks(
    hamiltonian: DFHamiltonian,
    preparation: DFPartialS2Preparation,
    spec: BlindStratumSpec,
) -> dict[str, bool]:
    return {
        "n_qubits_matches": hamiltonian.n_qubits == spec.n_qubits,
        "df_rank_matches": len(hamiltonian.lambdas) == spec.df_rank,
        "ld_matches": preparation.ld == spec.ld,
        "hamiltonian_hash_matches": (
            preparation.hamiltonian_hash == spec.expected_hamiltonian_hash
        ),
        "partition_hash_matches": (
            preparation.partition_hash == spec.expected_partition_hash
        ),
        "preparation_hash_matches": (
            preparation.preparation_hash == spec.expected_preparation_hash
        ),
    }


def build_expected_task_manifest(
    preparation: DFPartialS2Preparation,
    spec: BlindStratumSpec,
) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    for length in HOLDOUT_LENGTHS:
        for sample_index in range(HOLDOUT_SAMPLES_PER_LENGTH):
            seed = _derived_seed(
                spec.holdout_master_seed,
                spec.holdout_partition,
                length,
                sample_index,
            )
            _request, digest = _request_for_seed(
                preparation,
                delta_time=spec.delta_time,
                sequence_length=length,
                finite_taylor_order=FINITE_TAYLOR_ORDER,
                seed=seed,
            )
            tasks.append(
                {
                    "task_key": task_key(spec.stratum_id, length, sample_index),
                    "sequence_length": length,
                    "sample_index": sample_index,
                    "seed": seed,
                    "event_digest": digest,
                }
            )
    probes: list[dict[str, Any]] = []
    for length in HOLDOUT_LENGTHS:
        seed = _derived_seed(
            spec.operator_master_seed,
            spec.operator_partition,
            length,
        )
        _request, digest = _request_for_seed(
            preparation,
            delta_time=spec.delta_time,
            sequence_length=length,
            finite_taylor_order=FINITE_TAYLOR_ORDER,
            seed=seed,
        )
        probes.append(
            {
                "probe_key": probe_key(spec.stratum_id, length),
                "sequence_length": length,
                "seed": seed,
                "event_digest": digest,
            }
        )
    event_stream_digest = fingerprint(
        {"event_digests": [row["event_digest"] for row in tasks]}
    )
    payload: dict[str, Any] = {
        "schema_version": EXPECTED_MANIFEST_SCHEMA,
        "stratum": asdict(spec),
        "tasks": tasks,
        "operator_probes": probes,
        "event_stream_digest": event_stream_digest,
    }
    payload["content_fingerprint"] = fingerprint(payload)
    validate_expected_task_manifest(payload)
    return payload


def validate_expected_task_manifest(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != EXPECTED_MANIFEST_SCHEMA:
        raise ValueError("Unsupported P-A blind expected-manifest schema.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("P-A blind expected-manifest fingerprint mismatch.")
    tasks = payload.get("tasks", [])
    probes = payload.get("operator_probes", [])
    if len(tasks) != len(HOLDOUT_LENGTHS) * HOLDOUT_SAMPLES_PER_LENGTH:
        raise ValueError("P-A blind expected task grid is incomplete.")
    if len(probes) != len(HOLDOUT_LENGTHS):
        raise ValueError("P-A blind expected operator-probe grid is incomplete.")
    if len({row["task_key"] for row in tasks}) != len(tasks):
        raise ValueError("P-A blind expected task keys are not unique.")
    expected_stream = payload["stratum"].get("expected_event_stream_digest")
    if expected_stream is not None and payload["event_stream_digest"] != expected_stream:
        raise ValueError("P-A H4 event stream differs from the frozen pilot stream.")
    if payload["stratum"]["stratum_id"] == H4_COMPILER_TRANSFER.stratum_id:
        observed_probes = {
            int(row["sequence_length"]): str(row["event_digest"])
            for row in probes
        }
        if observed_probes != FROZEN_H4_PROBE_DIGESTS:
            raise ValueError("P-A H4 operator-probe stream differs from the pilot.")


def _compile_task(
    preparation: DFPartialS2Preparation,
    compiler: CompilerSettings,
    expected: Mapping[str, Any],
    *,
    support_definitions: Mapping[tuple[str, str], DFBasisDefinition],
    source_unitaries: Mapping[tuple[str, str], Any],
    union_cache: dict[tuple[str, str, tuple[int, ...]], DFBasisDefinition],
    builder: QiskitDFRTEEventCircuitBuilder,
) -> dict[str, Any]:
    request, digest = _request_for_seed(
        preparation,
        delta_time=float(expected["delta_time"]),
        sequence_length=int(expected["sequence_length"]),
        finite_taylor_order=FINITE_TAYLOR_ORDER,
        seed=int(expected["seed"]),
    )
    if digest != expected["event_digest"]:
        raise ValueError(f"Event digest changed for {expected['task_key']}.")
    search_started = time.perf_counter()
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
    search_elapsed = time.perf_counter() - search_started
    plans = {
        "full_basis_shared": None,
        "event_support_restricted": event_support_plan,
        "support_run_le_1": current_plan,
        "interval_union_dp": interval_plan,
    }
    costs: dict[str, dict[str, int]] = {}
    build_elapsed: dict[str, float] = {}
    compile_elapsed: dict[str, float] = {}
    for label in POLICY_LABELS:
        started = time.perf_counter()
        built = builder.build_sequence(request, basis_plan=plans[label])
        build_elapsed[label] = float(time.perf_counter() - started)
        started = time.perf_counter()
        costs[label] = _cost_record(built.circuit, compiler)
        compile_elapsed[label] = float(time.perf_counter() - started)
    oracle_label = min(
        POLICY_LABELS,
        key=lambda label: (
            costs[label]["rz_count"],
            costs[label]["cx_count"],
            costs[label]["total_depth"],
            POLICY_LABELS.index(label),
        ),
    )
    return {
        "task_key": expected["task_key"],
        "partition": expected["partition"],
        "sequence_length": int(expected["sequence_length"]),
        "sample_index": int(expected["sample_index"]),
        "seed": int(expected["seed"]),
        "event_digest": digest,
        "event_orders": [event.taylor_order for event in request.events],
        "costs": costs,
        "compiled_oracle_policy": oracle_label,
        "interval_metadata": interval_metadata,
        "current_plan_fingerprint": current_plan.plan_fingerprint,
        "interval_plan_fingerprint": interval_plan.plan_fingerprint,
        "interval_choice_differs_from_current": (
            tuple(choice.selected_basis_hash for choice in interval_plan.choices)
            != tuple(choice.selected_basis_hash for choice in current_plan.choices)
        ),
        "timing": {
            "policy_search_seconds": float(search_elapsed),
            "circuit_build_seconds": build_elapsed,
            "transpile_seconds": compile_elapsed,
        },
        "process_peak_rss_kib_after": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ),
    }


def compile_stratum_rows(
    hamiltonian: DFHamiltonian,
    preparation: DFPartialS2Preparation,
    compiler: CompilerSettings,
    spec: BlindStratumSpec,
    expected_manifest: Mapping[str, Any],
    *,
    existing_rows: Mapping[str, Mapping[str, Any]] | None = None,
    on_row: Callable[[Mapping[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    validate_expected_task_manifest(expected_manifest)
    if not all(_physical_checks(hamiltonian, preparation, spec).values()):
        raise ValueError(f"Physical input mismatch for {spec.stratum_id}.")
    if compiler.optimization_level != spec.optimization_level:
        raise ValueError(f"Compiler level mismatch for {spec.stratum_id}.")
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
    completed = dict(existing_rows or {})
    rows: list[dict[str, Any]] = []
    for entry in expected_manifest["tasks"]:
        expected = {
            **dict(entry),
            "delta_time": spec.delta_time,
            "partition": spec.holdout_partition,
        }
        key = str(entry["task_key"])
        if key in completed:
            row = dict(completed[key])
            if row.get("event_digest") != entry["event_digest"]:
                raise ValueError(f"Checkpoint event digest mismatch for {key}.")
        else:
            row = _compile_task(
                preparation,
                compiler,
                expected,
                support_definitions=support_definitions,
                source_unitaries=source_unitaries,
                union_cache=union_cache,
                builder=builder,
            )
            if on_row is not None:
                on_row(row)
        rows.append(row)
    return rows


def run_operator_probes(
    hamiltonian: DFHamiltonian,
    preparation: DFPartialS2Preparation,
    spec: BlindStratumSpec,
    expected_manifest: Mapping[str, Any],
    *,
    existing_probes: Mapping[str, Mapping[str, Any]] | None = None,
    on_probe: Callable[[Mapping[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    validate_expected_task_manifest(expected_manifest)
    source_unitaries = _source_unitaries(hamiltonian, preparation)
    union_cache: dict[
        tuple[str, str, tuple[int, ...]], DFBasisDefinition
    ] = {}
    builder = QiskitDFRTEEventCircuitBuilder(
        basis_registry=preparation.rte_preparation.basis_registry
    )
    completed = dict(existing_probes or {})
    probes: list[dict[str, Any]] = []
    for entry in expected_manifest["operator_probes"]:
        key = str(entry["probe_key"])
        if key in completed:
            row = dict(completed[key])
            if row.get("event_digest") != entry["event_digest"]:
                raise ValueError(f"Probe checkpoint digest mismatch for {key}.")
        else:
            request, digest = _request_for_seed(
                preparation,
                delta_time=spec.delta_time,
                sequence_length=int(entry["sequence_length"]),
                finite_taylor_order=FINITE_TAYLOR_ORDER,
                seed=int(entry["seed"]),
            )
            if digest != entry["event_digest"]:
                raise ValueError(f"Operator-probe digest changed for {key}.")
            plan, metadata = make_interval_union_basis_plan(
                request, preparation, source_unitaries, union_cache
            )
            full = builder.build_sequence(request)
            candidate = builder.build_sequence(request, basis_plan=plan)
            started = time.perf_counter()
            residual = maximum_operator_difference(
                full.circuit,
                candidate.circuit,
                allow_global_phase=False,
            )
            row = {
                "probe_key": key,
                "sequence_length": int(entry["sequence_length"]),
                "seed": int(entry["seed"]),
                "event_digest": digest,
                "operator_max_abs_difference": residual,
                "relative_ancilla_phase_matches": (
                    full.relative_ancilla_phase
                    == candidate.relative_ancilla_phase
                ),
                "operator_comparison_seconds": float(
                    time.perf_counter() - started
                ),
                "interval_metadata": metadata,
                "process_peak_rss_kib_after": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                ),
            }
            if on_probe is not None:
                on_probe(row)
        probes.append(row)
    return probes


def evaluate_preregistered_gates(
    rows: Sequence[Mapping[str, Any]],
    probes: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, bool], dict[str, Any]]:
    current_rz = [
        float(row["costs"]["support_run_le_1"]["rz_count"]) for row in rows
    ]
    candidate_rz = [
        float(row["costs"]["interval_union_dp"]["rz_count"]) for row in rows
    ]
    full_rz = [
        float(row["costs"]["full_basis_shared"]["rz_count"]) for row in rows
    ]
    oracle_rz = [
        min(float(row["costs"][label]["rz_count"]) for label in POLICY_LABELS)
        for row in rows
    ]
    pooled_change = math.fsum(candidate_rz) / math.fsum(current_rz) - 1.0
    increases = [
        0.0 if current == 0.0 else candidate / current - 1.0
        for current, candidate in zip(current_rz, candidate_rz)
    ]
    oracle_regret = (
        math.fsum(candidate_rz) - math.fsum(oracle_rz)
    ) / math.fsum(full_rz)
    changed_fraction = sum(
        int(row["interval_choice_differs_from_current"]) for row in rows
    ) / len(rows)
    multi_count = sum(
        int(
            row["interval_metadata"][
                "selected_multi_application_interval_count"
            ]
            > 0
        )
        for row in rows
    )
    union_count = sum(
        int(
            row["interval_metadata"][
                "selected_support_union_interval_count"
            ]
            > 0
        )
        for row in rows
    )
    maximum_residual = max(
        float(row["operator_max_abs_difference"]) for row in probes
    )
    gates = {
        "operator_equivalence_pass": maximum_residual <= EQUIVALENCE_ATOL
        and all(bool(row["relative_ancilla_phase_matches"]) for row in probes),
        "pooled_rz_improvement_over_current_at_least_2pct": pooled_change
        <= -MINIMUM_POOLED_RZ_IMPROVEMENT_OVER_CURRENT,
        "per_trajectory_rz_increase_over_current_at_most_5pct": max(increases)
        <= MAXIMUM_PER_TRAJECTORY_RZ_INCREASE_OVER_CURRENT,
        "compiled_oracle_regret_over_full_rz_at_most_1pct": oracle_regret
        <= MAXIMUM_ORACLE_REGRET_OVER_FULL_RZ,
        "interval_choice_changes_at_least_20pct_of_holdout": changed_fraction
        >= MINIMUM_CHANGED_TRAJECTORY_FRACTION,
        "multi_application_and_union_intervals_observed": multi_count > 0
        and union_count > 0,
    }
    summary = {
        "holdout_pooled_rz_relative_change_vs_current": pooled_change,
        "maximum_holdout_trajectory_rz_relative_increase_vs_current": max(
            increases
        ),
        "compiled_oracle_regret_over_full_rz": oracle_regret,
        "changed_holdout_trajectory_fraction": changed_fraction,
        "holdout_trajectories_with_multi_application_interval": multi_count,
        "holdout_trajectories_with_support_union_interval": union_count,
        "maximum_operator_equivalence_residual": maximum_residual,
        "all_preregistered_gates_pass": all(gates.values()),
    }
    return gates, summary


def summarize_stratum(
    hamiltonian: DFHamiltonian,
    preparation: DFPartialS2Preparation,
    compiler: CompilerSettings,
    spec: BlindStratumSpec,
    expected_manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    probes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    validate_expected_task_manifest(expected_manifest)
    physical_checks = _physical_checks(hamiltonian, preparation, spec)
    checks = {
        **physical_checks,
        "compiler_level_matches": (
            compiler.optimization_level == spec.optimization_level
        ),
        "holdout_grid_complete": (
            len(rows) == len(HOLDOUT_LENGTHS) * HOLDOUT_SAMPLES_PER_LENGTH
        ),
        "operator_probe_grid_complete": len(probes) == len(HOLDOUT_LENGTHS),
        "event_stream_digest_matches_manifest": (
            fingerprint({"event_digests": [row["event_digest"] for row in rows]})
            == expected_manifest["event_stream_digest"]
        ),
        "all_costs_nonnegative": all(
            value >= 0
            for row in rows
            for label in POLICY_LABELS
            for value in row["costs"][label].values()
        ),
    }
    gates, summary = evaluate_preregistered_gates(rows, probes)
    return {
        "configuration": {
            **asdict(spec),
            "holdout_lengths": list(HOLDOUT_LENGTHS),
            "holdout_samples_per_length": HOLDOUT_SAMPLES_PER_LENGTH,
            "finite_taylor_order": FINITE_TAYLOR_ORDER,
            "compiler": {
                "basis_gates": list(compiler.basis_gates),
                "optimization_level": compiler.optimization_level,
                "transpiler_seed": compiler.transpiler_seed,
                "coupling_map": compiler.coupling_map,
                "qiskit_version": compiler.qiskit_version,
            },
        },
        "physical_instance": {
            "hamiltonian_hash": preparation.hamiltonian_hash,
            "partition_hash": preparation.partition_hash,
            "preparation_hash": preparation.preparation_hash,
            "tail_hash": preparation.tail_extraction.tail_hash,
        },
        "expected_manifest_content_fingerprint": expected_manifest[
            "content_fingerprint"
        ],
        "holdout": _partition_summary(rows),
        "holdout_rows": [dict(row) for row in rows],
        "operator_equivalence_probes": [dict(row) for row in probes],
        "summary": summary,
        "preregistered_gates": gates,
        "checks": checks,
        "execution_valid": all(checks.values()),
    }


def finalize_blind_validation_artifact(
    strata: Mapping[str, Mapping[str, Any]],
    *,
    source_evidence: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    execution_checks = {
        "both_strata_present": set(strata)
        == {spec.stratum_id for spec in STRATA},
        "both_strata_execution_valid": all(
            bool(row.get("execution_valid")) for row in strata.values()
        ),
        "frozen_core_hashes_match": bool(
            source_evidence.get("frozen_core_hashes_match")
        ),
        "frozen_pilot_artifact_matches": bool(
            source_evidence.get("frozen_pilot_artifact_matches")
        ),
    }
    blind_pass = all(
        bool(row["summary"]["all_preregistered_gates_pass"])
        for row in strata.values()
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "stage": STAGE,
        "baseline_contract": {
            "policies": list(POLICY_LABELS),
            "candidate": "interval_union_dp_basis_operation_count_v1",
            "adjacent_basis_cancellation_enabled_for_all_policies": True,
            "compiled_oracle_scope": "minimum among the four listed policies only",
            "gates_changed_after_observing_results": False,
        },
        "strata": {key: dict(value) for key, value in strata.items()},
        "execution_checks": execution_checks,
        "overall_pass": all(execution_checks.values()),
        "decision": {
            "blind_validation_passed": blind_pass,
            "status": (
                "advance_pa_v1_to_formal_primary_theme_candidate"
                if blind_pass
                else "return_to_pc_after_pa_v1_blind_transfer_failure"
            ),
            "required_strata": [spec.stratum_id for spec in STRATA],
            "both_strata_must_pass": True,
        },
        "source_evidence": dict(source_evidence),
        "scope": {
            "pa_v2_evaluated": False,
            "coupling_map_or_noise_evaluated": False,
            "full_partial_s2_or_hadamard_wrapper_compiled": False,
            "rpe_or_final_total_cost_evaluated": False,
            "h12_evaluated": False,
            "scientific_superiority_claimed": False,
        },
        "provenance": dict(provenance),
    }
    payload["content_fingerprint"] = fingerprint(payload)
    validate_blind_validation_artifact(payload)
    return payload


def validate_blind_validation_artifact(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported P-A blind-validation schema.")
    if payload.get("method") != METHOD or payload.get("stage") != STAGE:
        raise ValueError("Unsupported P-A blind-validation method or stage.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("P-A blind-validation fingerprint mismatch.")
    checks = payload.get("execution_checks", {})
    if payload.get("overall_pass") != (bool(checks) and all(checks.values())):
        raise ValueError("P-A blind-validation overall status is inconsistent.")
    if set(payload.get("strata", {})) != {spec.stratum_id for spec in STRATA}:
        raise ValueError("P-A blind-validation strata are incomplete.")
    expected_decision = all(
        bool(row["summary"]["all_preregistered_gates_pass"])
        for row in payload["strata"].values()
    )
    if payload["decision"]["blind_validation_passed"] != expected_decision:
        raise ValueError("P-A blind-validation decision is inconsistent.")
    for key, value in payload.get("scope", {}).items():
        if value is not False:
            raise ValueError(f"P-A blind-validation overstates scope: {key}.")


def finalize_checkpoint(kind: str, row: Mapping[str, Any]) -> dict[str, Any]:
    if kind not in {"holdout", "operator_probe"}:
        raise ValueError("Unsupported P-A blind checkpoint kind.")
    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA,
        "kind": kind,
        "row": dict(row),
    }
    payload["content_fingerprint"] = fingerprint(payload)
    validate_checkpoint(payload)
    return payload


def validate_checkpoint(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("Unsupported P-A blind checkpoint schema.")
    if payload.get("kind") not in {"holdout", "operator_probe"}:
        raise ValueError("Unsupported P-A blind checkpoint kind.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("P-A blind checkpoint fingerprint mismatch.")
