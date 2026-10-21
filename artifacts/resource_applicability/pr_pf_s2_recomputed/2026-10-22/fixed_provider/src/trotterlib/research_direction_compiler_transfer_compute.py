"""Raw computation for the focused M06/L08 compiler-transfer validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from .df_hamiltonian import DFHamiltonian
from .df_partial_s2 import DFPartialS2Preparation
from .df_partial_s2_repeated import QiskitDFPartialS2RepeatedCircuitBuilder
from .df_partial_s2_repeated_cost import (
    make_exact_df_partial_s2_repeated_trajectory_stream,
)
from .research_direction_full_scope import (
    AXES,
    METRICS,
    fingerprint,
)
from .research_direction_full_scope_extension import _compile_randomized_delta
from .research_direction_full_scope_replication import validate_wp05br_artifact
from .research_direction_proxy_precision import (
    _compile_holdouts,
    validate_m08_artifact,
)
from .research_direction_round_dominance import validate_g08_artifact
from .research_direction_sequence_policy import (
    register_support_restricted_bases,
    validate_wp06b_artifact,
)
from .rpe_hadamard_compiled_cost_benchmark import (
    QiskitRPEHadamardBenchmarkCircuitBuilder,
)
from .rpe_hadamard_interrogation import RPEHadamardInterrogationRequest
from .rte import CompilerSettings
from .rte_compiled_cost import transpile_and_measure_cost


SCHEMA_VERSION = "research_direction_compiler_transfer_compute_v1"
METHOD = "m06_l08_opt2_same_trajectory_compute_v1"
Q_VALUES = (1, 2, 16, 32)


def _metric_record(cost: Any) -> dict[str, float]:
    return {metric: float(getattr(cost, metric)) for metric in METRICS}


def _compiler_record(compiler: CompilerSettings) -> dict[str, Any]:
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


def _compile_deterministic_point(
    preparation: DFPartialS2Preparation,
    compiler: CompilerSettings,
    *,
    delta_time: float,
    q_m: int,
    maximum_repetition_count: int,
) -> dict[str, Any]:
    stream = make_exact_df_partial_s2_repeated_trajectory_stream(
        preparation,
        delta_time,
        q_m,
        None,
        None,
        controlled=True,
        ancilla_qubit=preparation.num_system_qubits,
        construction_policy="boundary_optimized",
        maximum_trajectories=1,
    )
    records = tuple(stream.records)
    if len(records) != 1 or records[0][1] != 1.0:
        raise RuntimeError("Deterministic endpoint did not produce one exact circuit.")
    evolution = QiskitDFPartialS2RepeatedCircuitBuilder().build(
        records[0][0], construction_policy="boundary_optimized"
    )
    wrapper_builder = QiskitRPEHadamardBenchmarkCircuitBuilder(
        maximum_repetition_count=maximum_repetition_count
    )
    axes: dict[str, Any] = {}
    for axis in AXES:
        wrapper = wrapper_builder.build(
            RPEHadamardInterrogationRequest(
                evolution=evolution,
                axis=axis,
                include_measurement=True,
            )
        )
        values = _metric_record(
            transpile_and_measure_cost(
                wrapper.circuit,
                compiler,
                circuit_fingerprint=wrapper.compiler_independent_fingerprint,
                actual_circuit_fingerprint=wrapper.compiler_independent_fingerprint,
            )
        )
        axes[axis] = {
            metric: {
                "mean": value,
                "standard_error": 0.0,
                "minimum": value,
                "maximum": value,
            }
            for metric, value in values.items()
        }
    return {"sample_count": 1, "q_m": q_m, "axes": axes}


def compute_compiler_transfer_raw(
    hamiltonian: DFHamiltonian,
    ld3_preparation: DFPartialS2Preparation,
    ld12_preparation: DFPartialS2Preparation,
    compiler: CompilerSettings,
    wp05br: Mapping[str, Any],
    wp06b: Mapping[str, Any],
    g08: Mapping[str, Any],
    m08: Mapping[str, Any],
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Compile opt-level-2 raw points using the exact baseline trajectories."""
    validate_wp05br_artifact(wp05br)
    validate_wp06b_artifact(wp06b)
    validate_g08_artifact(g08)
    validate_m08_artifact(m08)
    if not all(
        bool(payload["overall_pass"])
        for payload in (wp05br, g08, m08)
    ):
        raise ValueError("Compiler transfer requires passing baseline artifacts.")
    if ld3_preparation.ld != 3 or ld12_preparation.ld != 12:
        raise ValueError("Compiler transfer requires L_D=3 and L_D=12.")
    if ld3_preparation.hamiltonian_hash != ld12_preparation.hamiltonian_hash:
        raise ValueError("Candidate preparations must share one Hamiltonian snapshot.")
    if compiler.optimization_level != 2 or compiler.transpiler_seed != 17:
        raise ValueError("Focused transfer changes only optimization level 1 to 2.")

    delta_time = 0.02
    rte_steps = 32
    training_fingerprint = str(
        wp06b["training_selection"]["training_fingerprint"]
    )
    support_definitions, proof_records = register_support_restricted_bases(
        hamiltonian, ld3_preparation
    )

    calibration, _probe = _compile_randomized_delta(
        ld3_preparation,
        support_definitions,
        compiler,
        delta_time=delta_time,
        q_values=(1, 2),
        schedule_rte_steps=(rte_steps,),
        finite_taylor_order=2,
        sample_count=32,
        master_seed=int(wp05br["configuration"]["master_seed"]),
        training_fingerprint=training_fingerprint,
        probe=False,
        progress=progress,
    )
    holdouts = _compile_holdouts(
        ld3_preparation,
        support_definitions,
        compiler,
        delta_time=delta_time,
        rte_steps=rte_steps,
        q_values=(16, 32),
        sample_count=8,
        master_seed=int(m08["configuration"]["master_seed"]),
        training_fingerprint=training_fingerprint,
        progress=progress,
    )
    ld3_points = {
        "1": calibration[str(rte_steps)]["points"]["1"],
        "2": calibration[str(rte_steps)]["points"]["2"],
        "16": holdouts[16],
        "32": holdouts[32],
    }

    ld12_points: dict[str, Any] = {}
    for q_m in Q_VALUES:
        if progress is not None:
            progress(f"compiler-transfer L_D=12 q={q_m}")
        ld12_points[str(q_m)] = _compile_deterministic_point(
            ld12_preparation,
            compiler,
            delta_time=delta_time,
            q_m=q_m,
            maximum_repetition_count=max(Q_VALUES),
        )

    proof_residual = max(
        float(row["preserved_columns_max_abs_residual"])
        for row in proof_records
    )
    return {
        "configuration": {
            "molecule": "H4_chain",
            "geometry_angstrom": 1.0,
            "basis": "STO-3G",
            "n_qubits": hamiltonian.n_qubits,
            "df_rank": len(hamiltonian.lambdas),
            "candidate_ld_values": [3, 12],
            "delta_time": delta_time,
            "finite_taylor_order": 2,
            "rte_steps": rte_steps,
            "q_values": list(Q_VALUES),
            "ld3_calibration_sample_count_per_q": 32,
            "ld3_holdout_sample_count_per_q": 8,
            "compiler": _compiler_record(compiler),
        },
        "trajectory_binding": {
            "q1_q2_matches_wp05br_master_seed": True,
            "q16_q32_matches_m08_master_seed": True,
            "wp05br_master_seed": int(wp05br["configuration"]["master_seed"]),
            "m08_master_seed": int(m08["configuration"]["master_seed"]),
        },
        "direct_points": {
            "3": ld3_points,
            "12": ld12_points,
        },
        "checks": {
            "only_optimization_level_changed_from_baseline_compiler": True,
            "both_candidates_compiled": True,
            "q1_q2_calibration_and_q16_q32_holdout_present": (
                set(ld3_points) == {"1", "2", "16", "32"}
                and set(ld12_points) == {"1", "2", "16", "32"}
            ),
            "support_basis_certificates_pass": proof_residual <= 1e-10,
            "scientific_decision_deferred_to_postprocessing": True,
        },
        "scope": {
            "raw_compiler_transfer_compute_only": True,
            "decision_reaggregation_performed": False,
            "q_above_32_directly_validated": False,
            "state_preparation_included": False,
            "backend_execution_included": False,
            "noise_included": False,
            "final_total_cost_evaluation_performed": False,
            "scientific_superiority_claimed": False,
        },
    }


def finalize_compiler_transfer_compute_artifact(
    body: Mapping[str, Any], *, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "stage": "M06-L08-compute",
        **dict(body),
        "provenance": dict(provenance),
    }
    payload["content_fingerprint"] = fingerprint(payload)
    validate_compiler_transfer_compute_artifact(payload)
    return payload


def validate_compiler_transfer_compute_artifact(
    payload: Mapping[str, Any],
) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported compiler-transfer compute schema.")
    if payload.get("method") != METHOD or payload.get("stage") != "M06-L08-compute":
        raise ValueError("Unsupported compiler-transfer compute method or stage.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("Compiler-transfer compute fingerprint mismatch.")
    checks = payload.get("checks", {})
    if not checks or not all(checks.values()):
        raise ValueError("Compiler-transfer compute completeness checks failed.")
    scope = payload.get("scope", {})
    if scope.get("decision_reaggregation_performed") is not False:
        raise ValueError("Raw compute artifact cannot claim decision reaggregation.")
    if scope.get("final_total_cost_evaluation_performed") is not False:
        raise ValueError("Raw compute artifact cannot claim final total cost.")
    if scope.get("scientific_superiority_claimed") is not False:
        raise ValueError("Raw compute artifact cannot claim scientific superiority.")


def write_compiler_transfer_compute_artifact(
    payload: Mapping[str, Any], path: str | Path
) -> None:
    validate_compiler_transfer_compute_artifact(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
