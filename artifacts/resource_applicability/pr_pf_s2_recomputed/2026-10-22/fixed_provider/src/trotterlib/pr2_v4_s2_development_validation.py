"""PR-2 V4 correctness and one development-only resource comparison.

The result-prior authorization is frozen at commit 33a1c0d.  This module
never loads the held-out NPZ and cannot authorize S3 or any automatic stage.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import qiskit
from qiskit.quantum_info import Operator, Pauli, Statevector
from scipy.linalg import expm

from .df_hamiltonian import DFHamiltonian
from .df_partial_s2 import (
    DFPartialS2Preparation,
    QiskitDFPartialS2CircuitBuilder,
    make_df_partial_s2_step_request,
    prepare_df_partial_s2,
)
from .df_partial_s2_repeated import (
    QiskitDFPartialS2RepeatedCircuitBuilder,
    make_df_partial_s2_repeated_request,
)
from .df_rte_tail import extraction_to_normalized_rte_tail
from .pr2_new_series_validation import (
    EXPECTED_DEVELOPMENT_FILE_SHA256,
    EXPECTED_HAMILTONIAN_HASH,
    EXPECTED_HELD_OUT_FILE_SHA256,
    HELD_OUT_RELATIVE_PATH,
    DEVELOPMENT_RELATIVE_PATH,
    PASS_STATUS as V1_V3_PASS_STATUS,
    SERIES_ID,
    _load_snapshot_once,
)
from .pr2_s0_s1_validation import (
    AXIS_ALPHA,
    AXIS_ERROR,
    COMPILER_BASIS,
    COMPILER_OPTIMIZATION_LEVEL,
    COMPILER_SEED,
    S1_K_VALUES,
    S1_R_VALUES,
    _complex_record,
    _primitive_structural_coverage,
    _to_qiskit_state,
    corrected_hoeffding_shots,
    file_sha256,
    generation_partition,
)
from .rpe_hadamard_compiled_cost_benchmark import (
    QiskitRPEHadamardBenchmarkCircuitBuilder,
    RPEHadamardCompiledCostBenchmarkRequest,
    generate_rpe_hadamard_compiled_cost_benchmark_dataset,
)
from .rpe_hadamard_interrogation import (
    RPE_HADAMARD_BIT_VALUE_MAPPING,
    RPEHadamardInterrogationRequest,
)
from .rpe_resource_accounting import RPE_COST_METRICS
from .rte import CompilerSettings, make_rte_config
from .rte_compiled_cost import TranspiledCircuitCostCache


AUTHORIZATION_COMMIT = "33a1c0d0a7f6880972f4e0ffdb5b6e7ffe0358e0"
AUTHORIZATION_SHA256 = (
    "ff1283be5c3315777dfd878ec305d76e6013077b3e0c7e04c6b25242092c22a9"
)
SPECIFICATION_SHA256 = (
    "a55d7878ab2c201b65cfaaf687806118c56df3a4135026d59811da8931056bcf"
)
V1_V3_RESULT_RELATIVE_PATH = (
    "artifacts/pr2_new_series_validation/2026-09-28/"
    "pr2_v1_v3_result_v1.json"
)
EXPECTED_V1_V3_SHA256 = (
    "0eb22c813eb838169eb455334146140467ebbc5636db78bd923b1e6bdaed46d8"
)
EXPECTED_V1_V3_FINGERPRINT = (
    "b210b394e9cd5a8eded947b0fd12cefe19ce9f27eb6b8140b3e863df73ea7961"
)

V4_SCHEMA_VERSION = "pr2_v4_correctness_result_v1"
S2_SCHEMA_VERSION = "pr2_s2_development_resource_result_v1"
V4_PASS_STATUS = "V4_CORRECTNESS_PASS_S2_DEVELOPMENT_AUTHORIZED"
V4_STATUSES = frozenset(
    {
        V4_PASS_STATUS,
        "BLOCKED_IMPLEMENTATION_INVALID",
        "STOP_ESTIMAND_OR_SCOPE_INVALID",
    }
)
S2_STATUSES = frozenset(
    {
        "COMPLETE_NEGATIVE_FULL_SCOPE_ERASES_GAIN",
        "S2_CONDITIONAL_RESOURCE_MAP_AWAITING_REVIEW",
        "S2_TRANSFER_CANDIDATE_AWAITING_REVIEW",
        "BLOCKED_IMPLEMENTATION_INVALID",
        "STOP_ESTIMAND_OR_SCOPE_INVALID",
    }
)
DELTA_TIME = 0.1
V4_Q = 1
S2_Q = 8
PRIMARY_RANK = 6
CONTROL_RANKS = (3, 9)
SENTINEL = (1, 2)
S2_INITIAL_SAMPLE_COUNT = 32
S2_EXTENSION_SAMPLE_COUNT = 96
S2_TOTAL_SAMPLE_COUNT = 128
S2_MASTER_SEED = 20260927102
S2_EXTENSION_MASTER_SEED = 20260927128
MATERIAL_RATIO = 0.9
RZ_RSE_TRIGGER = 0.02
SIGNAL_TOLERANCE = 1e-11
WRAPPER_TOLERANCE = 1e-10


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _cell_seed(*parts: Any) -> int:
    return int.from_bytes(
        hashlib.sha256(_canonical_json(list(parts))).digest()[:8],
        "big",
    ) % (2**63)


def _compiler() -> CompilerSettings:
    return CompilerSettings(
        basis_gates=COMPILER_BASIS,
        backend_name=None,
        coupling_map=None,
        optimization_level=COMPILER_OPTIMIZATION_LEVEL,
        layout_method=None,
        routing_method=None,
        transpiler_seed=COMPILER_SEED,
        qiskit_version=qiskit.__version__,
    )


def _prepare_random(
    hamiltonian: DFHamiltonian,
    method: str,
    rank: int,
) -> DFPartialS2Preparation:
    if method == "B2":
        partition = generation_partition(hamiltonian, rank)
        policy = "explicit_ordered_partition"
    elif method == "B3":
        partition = generation_partition(hamiltonian, 0)
        policy = "explicit_ordered_partition"
    else:
        raise ValueError(f"Unsupported random method: {method}")
    return prepare_df_partial_s2(
        hamiltonian,
        partition,
        identity_policy="extract_identity_phase",
        coefficient_atol=0.0,
        partition_policy=policy,
    )


def _prepare_deterministic(
    hamiltonian: DFHamiltonian,
    rank: int,
) -> DFPartialS2Preparation:
    truncated = hamiltonian.select_blocks(tuple(range(int(rank))))
    return prepare_df_partial_s2(
        truncated,
        generation_partition(truncated, truncated.n_blocks),
        identity_policy="extract_identity_phase",
        coefficient_atol=0.0,
        partition_policy="explicit_ordered_partition",
    )


def _load_inputs(
    root: Path,
    counters: dict[str, int],
) -> tuple[DFHamiltonian, np.ndarray, dict[str, Any]]:
    development = root / DEVELOPMENT_RELATIVE_PATH
    held_out = root / HELD_OUT_RELATIVE_PATH
    counters["development_raw_hash_checks"] += 1
    if file_sha256(development) != EXPECTED_DEVELOPMENT_FILE_SHA256:
        raise ValueError("Development snapshot SHA-256 differs from authorization.")
    counters["held_out_raw_hash_checks"] += 1
    if file_sha256(held_out) != EXPECTED_HELD_OUT_FILE_SHA256:
        raise ValueError("Held-out raw SHA-256 differs from authorization.")
    counters["development_npz_loads"] += 1
    hamiltonian, _sector, state, _sector_state, metadata, _layout = (
        _load_snapshot_once(development)
    )
    if metadata["hamiltonian_hash"] != EXPECTED_HAMILTONIAN_HASH:
        raise ValueError("Development Hamiltonian hash differs from authorization.")
    return hamiltonian, _to_qiskit_state(state, hamiltonian.n_qubits), metadata


def _new_counters() -> dict[str, int]:
    return {
        "development_raw_hash_checks": 0,
        "development_npz_loads": 0,
        "held_out_raw_hash_checks": 0,
        "held_out_npz_loads": 0,
        "molecular_calculations": 0,
        "signal_evaluations": 0,
        "random_trajectories_compiled": 0,
        "full_wrappers_compiled": 0,
        "quantum_shots": 0,
    }


def _signal(operator: np.ndarray, state: np.ndarray) -> complex:
    return complex(np.vdot(state, operator @ state))


def _target_signal(
    hamiltonian: DFHamiltonian,
    state: np.ndarray,
    total_time: float,
) -> tuple[float, complex]:
    # The frozen state is an eigenstate.  Reuse the validated dense DF helper
    # through the same conversion path as the V1--V3/S1 code.
    from .pr2_s0_s1_validation import _dense_df_operator_qiskit

    matrix = _dense_df_operator_qiskit(hamiltonian)
    energy_value = complex(np.vdot(state, matrix @ state))
    if abs(energy_value.imag) > 1e-11:
        raise ValueError("Frozen-state energy has a non-negligible imaginary part.")
    energy = float(energy_value.real)
    return energy, complex(np.exp(-1j * energy * total_time))



def _random_signal_point(
    preparation: DFPartialS2Preparation,
    state: np.ndarray,
    exact_target: complex,
    *,
    stage: str,
    method: str,
    rank: int,
    q: int,
    rte_steps: int,
    cutoff: int,
) -> dict[str, Any]:
    seed = _cell_seed(stage, "signal", method, rank, rte_steps, cutoff)
    request = make_df_partial_s2_step_request(
        preparation,
        step_time=DELTA_TIME,
        rte_steps=rte_steps,
        truncation_tolerance=1.0,
        finite_taylor_order=cutoff,
        seed=seed,
    )
    if request.rte_config is None:
        raise RuntimeError("Random signal point lost its RTE configuration.")
    parts = QiskitDFPartialS2CircuitBuilder().build_additive_circuits(request)
    forward = np.asarray(Operator(parts.forward_deterministic_half).data)
    reverse = np.asarray(Operator(parts.reverse_deterministic_half).data)
    normalized_tail = extraction_to_normalized_rte_tail(
        preparation.tail_extraction
    ).normalized_hamiltonian
    from .rte import finite_rte_operator_moments

    moments = finite_rte_operator_moments(normalized_tail, request.rte_config)
    corrected_step = reverse @ moments.corrected_operator @ forward
    raw_step = corrected_step / moments.normalization_product
    exact_tail = expm(
        -1j * DELTA_TIME * preparation.exact_rte_lambda_r * normalized_tail
    )
    pf_step = reverse @ exact_tail @ forward
    corrected = _signal(np.linalg.matrix_power(corrected_step, q), state)
    raw = _signal(np.linalg.matrix_power(raw_step, q), state)
    pf_signal = _signal(np.linalg.matrix_power(pf_step, q), state)
    multiplier = float(moments.normalization_product**q)
    attenuation = float(moments.attenuation_factor**q)
    reconstruction_error = abs(raw * multiplier - corrected)
    inverse_error = abs(multiplier * attenuation - 1.0)
    biases = {
        "real": abs(corrected.real - exact_target.real),
        "imag": abs(corrected.imag - exact_target.imag),
    }
    shots = {
        axis: corrected_hoeffding_shots(multiplier, bias)
        for axis, bias in biases.items()
    }
    return {
        "method": method,
        "rank": rank,
        "q": q,
        "r": rte_steps,
        "K": cutoff,
        "seed": seed,
        "raw_mean": _complex_record(raw),
        "corrected_mean": _complex_record(corrected),
        "pf_exact_tail_signal": _complex_record(pf_signal),
        "exact_target": _complex_record(exact_target),
        "normalization_multiplier": multiplier,
        "attenuation": attenuation,
        "corrected_raw_reconstruction_abs_error": reconstruction_error,
        "normalization_attenuation_inverse_error": inverse_error,
        "raw_bias_abs": abs(raw - exact_target),
        "corrected_bias_abs": abs(corrected - exact_target),
        "finite_truncation_bias_abs": abs(corrected - pf_signal),
        "outer_pf_bias_abs": abs(pf_signal - exact_target),
        "axis_bias": biases,
        "axis_shots": shots,
        "total_shots": (
            None if any(value is None for value in shots.values())
            else int(sum(int(value) for value in shots.values()))
        ),
        "accuracy_eligible": all(value is not None for value in shots.values()),
        "exact_rte_lambda_r": float(preparation.exact_rte_lambda_r),
        "component_count": len(preparation.tail_extraction.components),
        "probability_sum": math.fsum(
            component.probability
            for component in preparation.rte_preparation.symbolic_tail.components
        ),
    }


def _deterministic_signal_point(
    preparation: DFPartialS2Preparation,
    state: np.ndarray,
    exact_target: complex,
    *,
    method: str,
    rank: int,
    q: int,
) -> dict[str, Any]:
    request = make_df_partial_s2_step_request(preparation, step_time=DELTA_TIME)
    step = np.asarray(
        Operator(QiskitDFPartialS2CircuitBuilder().build_step(request).circuit).data
    )
    mean = _signal(np.linalg.matrix_power(step, q), state)
    biases = {
        "real": abs(mean.real - exact_target.real),
        "imag": abs(mean.imag - exact_target.imag),
    }
    shots = {
        axis: corrected_hoeffding_shots(1.0, bias)
        for axis, bias in biases.items()
    }
    return {
        "method": method,
        "rank": rank,
        "q": q,
        "mean": _complex_record(mean),
        "corrected_mean": _complex_record(mean),
        "raw_mean": _complex_record(mean),
        "exact_target": _complex_record(exact_target),
        "normalization_multiplier": 1.0,
        "attenuation": 1.0,
        "corrected_bias_abs": abs(mean - exact_target),
        "axis_bias": biases,
        "axis_shots": shots,
        "total_shots": (
            None if any(value is None for value in shots.values())
            else int(sum(int(value) for value in shots.values()))
        ),
        "accuracy_eligible": all(value is not None for value in shots.values()),
    }


def _statistics_record(statistics: Any) -> dict[str, float | None]:
    return {
        "mean": float(statistics.mean),
        "unbiased_sample_variance": (
            None
            if statistics.unbiased_sample_variance is None
            else float(statistics.unbiased_sample_variance)
        ),
        "standard_error": (
            None
            if statistics.standard_error is None
            else float(statistics.standard_error)
        ),
        "minimum": float(statistics.minimum),
        "maximum": float(statistics.maximum),
    }


def _compile_cost_batch(
    preparation: DFPartialS2Preparation,
    *,
    stage: str,
    stream: str,
    method: str,
    rank: int,
    q: int,
    rte_steps: int,
    cutoff: int,
    sample_count: int | None,
    master_seed: int,
    cache: TranspiledCircuitCostCache,
) -> dict[str, Any]:
    if preparation.is_deterministic_only:
        config = None
        distribution = None
        evaluation_method = "exact"
        request_seed = None
        request_r = 0
        request_k = 0
    else:
        request_seed = _cell_seed(
            master_seed, stage, stream, method, rank, rte_steps, cutoff
        )
        config, distribution = make_rte_config(
            preparation.rte_preparation.symbolic_tail,
            evolution_time=DELTA_TIME,
            rte_steps=rte_steps,
            truncation_tolerance=1.0,
            finite_taylor_order=cutoff,
            seed=request_seed,
        )
        evaluation_method = "monte_carlo"
        request_r = rte_steps
        request_k = cutoff
    maximum_samples = 1 if sample_count is None else sample_count
    request = RPEHadamardCompiledCostBenchmarkRequest(
        preparation=preparation,
        delta_time=DELTA_TIME,
        calibration_repetition_counts=(q,),
        holdout_repetition_counts=(),
        rte_steps_per_occurrence=request_r,
        finite_taylor_order=request_k,
        rte_config=config,
        rte_distribution=distribution,
        compiler=_compiler(),
        evaluation_method=evaluation_method,
        sample_count=sample_count,
        seed=request_seed,
        generation_id=(
            f"pr2-{stage.lower()}-{stream}-{method}-rank{rank}-"
            f"q{q}-r{rte_steps}-k{cutoff}"
        ),
        maximum_repetition_count=q,
        maximum_trajectories=1_000_000,
        maximum_samples=maximum_samples,
        maximum_untranspiled_circuit_size=10_000_000,
        maximum_retained_trajectory_records=4,
        maximum_build_requests=1_000_000,
        maximum_transpile_requests=1_000_000,
        maximum_planned_instruction_applications=2_000_000_000,
        construction_policy="boundary_optimized",
        cache=cache,
    )
    result = generate_rpe_hadamard_compiled_cost_benchmark_dataset(request)
    if not result.dataset.complete or len(result.dataset.records) != 2:
        raise RuntimeError(
            f"Compiled-cost dataset failed for {stage}/{method}/rank{rank}."
        )
    axes: dict[str, Any] = {}
    for point in result.dataset.records:
        if point.status != "complete":
            raise RuntimeError("A compiled full-wrapper point is incomplete.")
        axes[point.axis] = {
            "sample_count": point.sample_count,
            "metric_statistics": {
                name: _statistics_record(statistics)
                for name, statistics in point.metric_statistics
            },
            "measurement_included": point.measurement_included,
            "state_preparation_included": point.state_preparation_included,
            "wrapped_evolution_already_controlled": (
                point.wrapped_evolution_already_controlled
            ),
            "additional_control_applied": point.additional_control_applied,
            "backend_execution_included": point.backend_execution_included,
            "quantum_shots_executed": point.quantum_shots_executed,
            "sampling_provenance_fingerprint": (
                point.sampling_provenance_fingerprint
            ),
            "wrapper_circuit_semantics_digest": (
                point.wrapper_circuit_semantics_digest
            ),
            "actual_circuit_fingerprint_digest": (
                point.actual_circuit_fingerprint_digest
            ),
            "sampled_trajectory_seeds": (
                None
                if point.sampled_trajectory_seeds is None
                else list(point.sampled_trajectory_seeds)
            ),
            "trajectory_records_truncated": point.trajectory_records_truncated,
            "workload": {
                "planned_build_requests": point.planned_build_requests,
                "actual_build_requests": point.actual_build_requests,
                "planned_transpile_requests": point.planned_transpile_requests,
                "actual_transpile_requests": point.actual_transpile_requests,
                "planned_instruction_applications": (
                    point.planned_instruction_applications
                ),
                "actual_built_instruction_total": (
                    point.actual_built_instruction_total
                ),
            },
        }
    return {
        "stage": stage,
        "stream": stream,
        "method": method,
        "rank": rank,
        "q": q,
        "r": rte_steps,
        "K": cutoff,
        "evaluation_method": evaluation_method,
        "sample_count": sample_count,
        "master_seed": request_seed,
        "dataset_fingerprint": result.dataset.dataset_fingerprint,
        "compiler_settings_fingerprint": (
            result.dataset.compiler_settings_fingerprint
        ),
        "axes": axes,
    }


def _wrapper_semantics_probe(
    preparation: DFPartialS2Preparation,
    state: np.ndarray,
) -> dict[str, Any]:
    config, distribution = make_rte_config(
        preparation.rte_preparation.symbolic_tail,
        evolution_time=DELTA_TIME,
        rte_steps=1,
        truncation_tolerance=1.0,
        finite_taylor_order=2,
        seed=_cell_seed("V4", "wrapper-semantics"),
    )
    request = make_df_partial_s2_repeated_request(
        preparation,
        step_time=DELTA_TIME,
        repetition_count=1,
        rte_config=config,
        rte_distribution=distribution,
        seed=_cell_seed("V4", "wrapper-trajectory"),
        controlled=True,
        ancilla_qubit=preparation.num_system_qubits,
        construction_policy="boundary_optimized",
    )
    evolution = QiskitDFPartialS2RepeatedCircuitBuilder().build(request)
    controlled = np.asarray(Operator(evolution.circuit).data)
    dimension = state.size
    direct = complex(np.vdot(state, controlled[dimension:, dimension:] @ state))
    builder = QiskitRPEHadamardBenchmarkCircuitBuilder(maximum_repetition_count=1)
    initial = np.concatenate((state, np.zeros_like(state)))
    observable = Pauli("Z" + "I" * int(math.log2(dimension)))
    axes: dict[str, Any] = {}
    for axis, expected_component in (("cosine", direct.real), ("sine", direct.imag)):
        unmeasured = builder.build(
            RPEHadamardInterrogationRequest(
                evolution=evolution,
                axis=axis,
                include_measurement=False,
            )
        )
        measured = builder.build(
            RPEHadamardInterrogationRequest(
                evolution=evolution,
                axis=axis,
                include_measurement=True,
            )
        )
        statevector = Statevector(initial).evolve(unmeasured.circuit)
        observed = float(np.real(statevector.expectation_value(observable)))
        axes[axis] = {
            "signal_component": unmeasured.signal_component,
            "estimator_definition": unmeasured.estimator_definition,
            "bit_value_mapping": [list(item) for item in measured.bit_value_mapping],
            "expected_component": float(expected_component),
            "observed_z_expectation": observed,
            "absolute_error": abs(observed - expected_component),
            "measurement_included": measured.include_measurement,
            "state_preparation_included": measured.state_preparation_included,
            "wrapped_evolution_already_controlled": (
                measured.wrapped_evolution_already_controlled
            ),
            "additional_control_applied": measured.additional_control_applied,
            "constant_phase": float(measured.constant_phase),
            "extracted_identity_phase": float(
                measured.extracted_identity_phase
            ),
            "rte_relative_phase": float(measured.rte_relative_phase),
            "wrapper_fingerprint": measured.wrapper_fingerprint,
        }
    overall = bool(
        axes["cosine"]["signal_component"] == "real"
        and axes["sine"]["signal_component"] == "imaginary"
        and axes["cosine"]["bit_value_mapping"]
        == [list(item) for item in RPE_HADAMARD_BIT_VALUE_MAPPING]
        and axes["sine"]["bit_value_mapping"]
        == [list(item) for item in RPE_HADAMARD_BIT_VALUE_MAPPING]
        and all(item["absolute_error"] <= WRAPPER_TOLERANCE for item in axes.values())
        and all(item["measurement_included"] is True for item in axes.values())
        and all(item["state_preparation_included"] is False for item in axes.values())
        and all(
            item["wrapped_evolution_already_controlled"] is True
            and item["additional_control_applied"] is False
            for item in axes.values()
        )
    )
    return {
        "direct_controlled_signal": _complex_record(direct),
        "control_convention": "ordinary_controlled_diag_I_U",
        "axes": axes,
        "overall_pass": overall,
    }


def _pool_statistic_batches(
    batches: Sequence[tuple[int, Mapping[str, Any]]],
) -> dict[str, float]:
    if not batches:
        raise ValueError("At least one statistics batch is required.")
    total = sum(count for count, _stats in batches)
    mean = sum(count * float(stats["mean"]) for count, stats in batches) / total
    m2 = 0.0
    for count, stats in batches:
        variance = stats["unbiased_sample_variance"]
        if count > 1:
            if variance is None:
                raise ValueError("A multi-sample batch lacks sample variance.")
            m2 += (count - 1) * float(variance)
        m2 += count * (float(stats["mean"]) - mean) ** 2
    variance = 0.0 if total == 1 else m2 / (total - 1)
    return {
        "mean": float(mean),
        "unbiased_sample_variance": float(variance),
        "standard_error": float(math.sqrt(variance / total)),
        "minimum": min(float(stats["minimum"]) for _n, stats in batches),
        "maximum": max(float(stats["maximum"]) for _n, stats in batches),
    }


def _pool_cost_batches(batches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not batches:
        raise ValueError("At least one compiled-cost batch is required.")
    evaluation_method = batches[0]["evaluation_method"]
    if any(batch["evaluation_method"] != evaluation_method for batch in batches):
        raise ValueError("Cannot pool exact and Monte Carlo cost batches.")
    if evaluation_method == "exact":
        if len(batches) != 1:
            raise ValueError("An exact cost record must have one batch.")
        axes = batches[0]["axes"]
        return {
            "sample_count": None,
            "axes": {
                axis: {metric: dict(stats) for metric, stats in row["metric_statistics"].items()}
                for axis, row in axes.items()
            },
        }
    counts = [int(batch["sample_count"]) for batch in batches]
    axes: dict[str, Any] = {}
    for axis in ("cosine", "sine"):
        axes[axis] = {}
        for metric in RPE_COST_METRICS:
            axes[axis][metric] = _pool_statistic_batches(
                [
                    (count, batch["axes"][axis]["metric_statistics"][metric])
                    for count, batch in zip(counts, batches, strict=True)
                ]
            )
    return {"sample_count": sum(counts), "axes": axes}


def _resource_record(
    signal: Mapping[str, Any],
    pooled_cost: Mapping[str, Any],
) -> dict[str, Any]:
    if not signal["accuracy_eligible"]:
        return {
            "accuracy_eligible": False,
            "total_work_no_preparation": None,
            "work_interval": None,
            "rz_relative_standard_error_max": None,
        }
    work = 0.0
    lower = 0.0
    upper = 0.0
    maximum_rse = 0.0
    axis_rows: dict[str, Any] = {}
    for signal_axis, cost_axis in (("real", "cosine"), ("imag", "sine")):
        shots = int(signal["axis_shots"][signal_axis])
        stats = pooled_cost["axes"][cost_axis]["rz_count"]
        mean = float(stats["mean"])
        se = 0.0 if stats["standard_error"] is None else float(stats["standard_error"])
        interval = [max(0.0, mean - 2.0 * se), mean + 2.0 * se]
        work += shots * mean
        lower += shots * interval[0]
        upper += shots * interval[1]
        rse = 0.0 if mean == 0.0 else se / mean
        maximum_rse = max(maximum_rse, rse)
        axis_rows[signal_axis] = {
            "wrapper_axis": cost_axis,
            "shots": shots,
            "mean_rz_count": mean,
            "rz_standard_error": se,
            "rz_relative_standard_error": rse,
            "rz_engineering_interval": interval,
            "work": shots * mean,
            "work_interval": [shots * interval[0], shots * interval[1]],
        }
    return {
        "accuracy_eligible": True,
        "axis": axis_rows,
        "total_shots": int(signal["total_shots"]),
        "total_work_no_preparation": float(work),
        "work_interval": [float(lower), float(upper)],
        "rz_relative_standard_error_max": float(maximum_rse),
    }


def _candidate_key(candidate: Mapping[str, Any]) -> str:
    return (
        f"{candidate['method']}-rank{candidate['rank']}-"
        f"r{candidate.get('r', 0)}-k{candidate.get('K', 0)}"
    )


def _selected_candidate(
    candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    eligible = [item for item in candidates if item["resource"]["accuracy_eligible"]]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda item: (
            item["resource"]["total_work_no_preparation"],
            item.get("r", 0),
            item.get("K", 0),
        ),
    )


def _ratio_interval(
    numerator: Mapping[str, Any],
    denominator: Mapping[str, Any],
) -> list[float]:
    n_low, n_high = numerator["resource"]["work_interval"]
    d_low, d_high = denominator["resource"]["work_interval"]
    if d_low <= 0.0:
        raise ValueError("Resource ratio denominator interval reaches zero.")
    return [float(n_low / d_high), float(n_high / d_low)]


def _materially_cheaper(
    numerator: Mapping[str, Any],
    denominator: Mapping[str, Any],
) -> bool:
    return _ratio_interval(numerator, denominator)[1] < MATERIAL_RATIO


def _setting_uncertain(
    candidates: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Any] | None,
) -> tuple[bool, list[str]]:
    if selected is None:
        return False, []
    challengers = []
    for item in candidates:
        if item is selected or not item["resource"]["accuracy_eligible"]:
            continue
        if _ratio_interval(item, selected)[0] < MATERIAL_RATIO:
            challengers.append(_candidate_key(item))
    return bool(challengers), challengers


def _refresh_candidate(candidate: dict[str, Any]) -> None:
    candidate["pooled_cost"] = _pool_cost_batches(candidate["cost_batches"])
    candidate["resource"] = _resource_record(
        candidate["signal"], candidate["pooled_cost"]
    )


def _initial_expansion_keys(
    b2: Sequence[dict[str, Any]],
    b3: Sequence[dict[str, Any]],
    deterministic: Sequence[dict[str, Any]],
) -> tuple[set[str], dict[str, Any]]:
    keys = {
        _candidate_key(item)
        for item in (*b2, *b3)
        if item["resource"]["accuracy_eligible"]
        and item["resource"]["rz_relative_standard_error_max"] > RZ_RSE_TRIGGER
    }
    selected_b2 = _selected_candidate(b2)
    selected_b3 = _selected_candidate(b3)
    uncertain_b2, challengers_b2 = _setting_uncertain(b2, selected_b2)
    uncertain_b3, challengers_b3 = _setting_uncertain(b3, selected_b3)
    by_key = {_candidate_key(item): item for item in (*b2, *b3)}
    if uncertain_b2 and selected_b2 is not None:
        keys.add(_candidate_key(selected_b2))
        keys.update(challengers_b2)
    if uncertain_b3 and selected_b3 is not None:
        keys.add(_candidate_key(selected_b3))
        keys.update(challengers_b3)

    b0 = next(item for item in deterministic if item["method"] == "B0")
    b1 = next(item for item in deterministic if item["method"] == "B1")
    comparisons = []
    if selected_b2 is not None:
        for endpoint in (b0, b1, selected_b3):
            if endpoint is None or not endpoint["resource"]["accuracy_eligible"]:
                continue
            ratio = _ratio_interval(selected_b2, endpoint)
            reverse = _ratio_interval(endpoint, selected_b2)
            resolved = ratio[1] < MATERIAL_RATIO or reverse[1] < MATERIAL_RATIO
            comparisons.append(
                {
                    "numerator": _candidate_key(selected_b2),
                    "denominator": _candidate_key(endpoint),
                    "ratio_interval": ratio,
                    "reverse_ratio_interval": reverse,
                    "materiality_resolved": resolved,
                }
            )
            if not resolved:
                keys.add(_candidate_key(selected_b2))
                endpoint_key = _candidate_key(endpoint)
                if endpoint_key in by_key:
                    keys.add(endpoint_key)
    return keys, {
        "b2_setting_uncertain": uncertain_b2,
        "b2_challengers": challengers_b2,
        "b3_setting_uncertain": uncertain_b3,
        "b3_challengers": challengers_b3,
        "primary_comparisons": comparisons,
    }


def _break_even(
    primary: Mapping[str, Any],
    comparator: Mapping[str, Any],
) -> dict[str, Any]:
    primary_shots = int(primary["resource"]["total_shots"])
    comparator_shots = int(comparator["resource"]["total_shots"])
    denominator = primary_shots - comparator_shots
    numerator = (
        comparator["resource"]["total_work_no_preparation"]
        - primary["resource"]["total_work_no_preparation"]
    )
    point = None if denominator == 0 else float(numerator / denominator)
    return {
        "primary": _candidate_key(primary),
        "comparator": _candidate_key(comparator),
        "primary_total_shots": primary_shots,
        "comparator_total_shots": comparator_shots,
        "point_break_even_rz_equivalent_per_shot": (
            point if point is not None and point >= 0.0 else None
        ),
        "raw_equality_solution": point,
        "nonnegative_break_even_exists": point is not None and point >= 0.0,
    }


def _decision(
    b2: Sequence[dict[str, Any]],
    b3: Sequence[dict[str, Any]],
    deterministic: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    eligible_b2 = [item for item in b2 if item["resource"]["accuracy_eligible"]]
    selected_b2 = _selected_candidate(b2)
    selected_b3 = _selected_candidate(b3)
    b0 = next(item for item in deterministic if item["method"] == "B0")
    b1 = next(item for item in deterministic if item["method"] == "B1")
    setting_uncertain, challengers = _setting_uncertain(b2, selected_b2)
    endpoint_candidates = [
        item
        for item in (b0, b1, selected_b3)
        if item is not None and item["resource"]["accuracy_eligible"]
    ]
    dominating = [
        _candidate_key(endpoint)
        for endpoint in endpoint_candidates
        if eligible_b2
        and all(_materially_cheaper(endpoint, item) for item in eligible_b2)
    ]
    ratios = {}
    if selected_b2 is not None:
        for endpoint in endpoint_candidates:
            ratios[_candidate_key(endpoint)] = {
                "b2_over_endpoint": _ratio_interval(selected_b2, endpoint),
                "endpoint_over_b2": _ratio_interval(endpoint, selected_b2),
            }
    coherent = bool(
        selected_b2 is not None
        and math.isfinite(selected_b2["signal"]["normalization_multiplier"])
        and math.isfinite(selected_b2["signal"]["exact_rte_lambda_r"])
        and selected_b2["signal"]["component_count"] > 0
        and selected_b2["signal"]["corrected_raw_reconstruction_abs_error"]
        <= SIGNAL_TOLERANCE
    )
    transfer_conditions = {
        "at_least_one_b2_eligible": bool(eligible_b2),
        "b2_over_b1_ratio_upper_below_0p9": bool(
            selected_b2 is not None
            and b1["resource"]["accuracy_eligible"]
            and _ratio_interval(selected_b2, b1)[1] < MATERIAL_RATIO
        ),
        "not_materially_dominated_by_b0_or_b3": not any(
            key.startswith("B0-") or key.startswith("B3-") for key in dominating
        ),
        "setting_certain": not setting_uncertain,
        "component_breakdown_coherent": coherent,
    }
    if not eligible_b2:
        status = "COMPLETE_NEGATIVE_FULL_SCOPE_ERASES_GAIN"
        reason = "all_B2_candidates_accuracy_ineligible"
    elif dominating:
        status = "COMPLETE_NEGATIVE_FULL_SCOPE_ERASES_GAIN"
        reason = "endpoint_materially_dominates_all_eligible_B2"
    elif setting_uncertain:
        status = "COMPLETE_NEGATIVE_FULL_SCOPE_ERASES_GAIN"
        reason = "B2_setting_uncertain_after_maximum_sampling"
    elif all(transfer_conditions.values()):
        status = "S2_TRANSFER_CANDIDATE_AWAITING_REVIEW"
        reason = "all_preregistered_transfer_candidate_conditions_pass"
    else:
        status = "S2_CONDITIONAL_RESOURCE_MAP_AWAITING_REVIEW"
        reason = "B2_remains_interpretable_without_full_transfer_conditions"

    selected = [item for item in (b0, b1, selected_b2, selected_b3) if item]
    frontier = [
        _candidate_key(item)
        for item in selected
        if item["resource"]["accuracy_eligible"]
        and not any(
            other is not item
            and other["resource"]["accuracy_eligible"]
            and _materially_cheaper(other, item)
            for other in selected
        )
    ]
    break_even = []
    if selected_b2 is not None and selected_b2["resource"]["accuracy_eligible"]:
        break_even = [
            _break_even(selected_b2, endpoint)
            for endpoint in endpoint_candidates
        ]
    return {
        "status": status,
        "reason": reason,
        "selected_B2": None if selected_b2 is None else _candidate_key(selected_b2),
        "selected_B3": None if selected_b3 is None else _candidate_key(selected_b3),
        "B2_setting_uncertain": setting_uncertain,
        "B2_setting_challengers": challengers,
        "materially_dominating_endpoints": dominating,
        "selected_ratio_intervals": ratios,
        "material_frontier": frontier,
        "transfer_conditions": transfer_conditions,
        "state_preparation_break_even": break_even,
        "materiality_threshold": MATERIAL_RATIO,
        "engineering_interval_is_formal_confidence_interval": False,
    }


def run_v4(
    root: Path,
    *,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    counters = _new_counters()
    hamiltonian, state, metadata = _load_inputs(root, counters)
    energy, exact_target = _target_signal(
        hamiltonian, state, V4_Q * DELTA_TIME
    )
    cache = TranspiledCircuitCostCache(maximum_entries=4096)
    random_signals = []
    compile_smokes = []
    structural: list[tuple[str, int, DFPartialS2Preparation]] = []
    preparations = {
        "B2": _prepare_random(hamiltonian, "B2", PRIMARY_RANK),
        "B3": _prepare_random(hamiltonian, "B3", 0),
    }
    for method, preparation in preparations.items():
        rank = PRIMARY_RANK if method == "B2" else 0
        structural.append((method, rank, preparation))
        for rte_steps in S1_R_VALUES:
            for cutoff in S1_K_VALUES:
                random_signals.append(
                    _random_signal_point(
                        preparation,
                        state,
                        exact_target,
                        stage="V4",
                        method=method,
                        rank=rank,
                        q=V4_Q,
                        rte_steps=rte_steps,
                        cutoff=cutoff,
                    )
                )
                compile_smokes.append(
                    _compile_cost_batch(
                        preparation,
                        stage="V4",
                        stream="canonical",
                        method=method,
                        rank=rank,
                        q=V4_Q,
                        rte_steps=rte_steps,
                        cutoff=cutoff,
                        sample_count=1,
                        master_seed=20260927101,
                        cache=cache,
                    )
                )
    for rank in CONTROL_RANKS:
        preparation = _prepare_random(hamiltonian, "B2", rank)
        structural.append(("B2", rank, preparation))
        random_signals.append(
            _random_signal_point(
                preparation,
                state,
                exact_target,
                stage="V4-control",
                method="B2",
                rank=rank,
                q=V4_Q,
                rte_steps=SENTINEL[0],
                cutoff=SENTINEL[1],
            )
        )
        compile_smokes.append(
            _compile_cost_batch(
                preparation,
                stage="V4",
                stream="sentinel",
                method="B2",
                rank=rank,
                q=V4_Q,
                rte_steps=SENTINEL[0],
                cutoff=SENTINEL[1],
                sample_count=1,
                master_seed=20260927101,
                cache=cache,
            )
        )
    deterministic_signals = []
    for method, rank in (("B0", 3), ("B0", 6), ("B0", 9), ("B1", 12)):
        preparation = _prepare_deterministic(hamiltonian, rank)
        deterministic_signals.append(
            _deterministic_signal_point(
                preparation,
                state,
                exact_target,
                method=method,
                rank=rank,
                q=V4_Q,
            )
        )
        compile_smokes.append(
            _compile_cost_batch(
                preparation,
                stage="V4",
                stream="deterministic",
                method=method,
                rank=rank,
                q=V4_Q,
                rte_steps=0,
                cutoff=0,
                sample_count=None,
                master_seed=0,
                cache=cache,
            )
        )
    wrapper_probe = _wrapper_semantics_probe(preparations["B2"], state)
    primitive = _primitive_structural_coverage(structural)
    counters["signal_evaluations"] = len(random_signals) + len(
        deterministic_signals
    )
    counters["random_trajectories_compiled"] = sum(
        int(item["sample_count"] or 0) for item in compile_smokes
    )
    counters["full_wrappers_compiled"] = sum(
        2 * (1 if item["sample_count"] is None else int(item["sample_count"]))
        for item in compile_smokes
    )
    signal_pass = all(
        item["corrected_raw_reconstruction_abs_error"] <= SIGNAL_TOLERANCE
        and item["normalization_attenuation_inverse_error"] <= SIGNAL_TOLERANCE
        and abs(item["probability_sum"] - 1.0) <= 1e-12
        for item in random_signals
    )
    compile_pass = all(
        all(
            axis["measurement_included"] is True
            and axis["state_preparation_included"] is False
            and axis["wrapped_evolution_already_controlled"] is True
            and axis["additional_control_applied"] is False
            and axis["backend_execution_included"] is False
            and axis["quantum_shots_executed"] == 0
            for axis in item["axes"].values()
        )
        for item in compile_smokes
    )
    expected_wrapper_count = 60
    correctness_pass = bool(
        signal_pass
        and compile_pass
        and wrapper_probe["overall_pass"]
        and primitive["all_supports_and_orders_valid"]
        and counters["full_wrappers_compiled"] == expected_wrapper_count
    )
    status = (
        V4_PASS_STATUS if correctness_pass else "BLOCKED_IMPLEMENTATION_INVALID"
    )
    payload: dict[str, Any] = {
        "schema_version": V4_SCHEMA_VERSION,
        "series_id": SERIES_ID,
        "status": status,
        "authorization_commit": AUTHORIZATION_COMMIT,
        "authorization_sha256": AUTHORIZATION_SHA256,
        "specification_sha256": SPECIFICATION_SHA256,
        "V1_V3_result_fingerprint": EXPECTED_V1_V3_FINGERPRINT,
        "development_snapshot": {
            "path": DEVELOPMENT_RELATIVE_PATH,
            "raw_file_sha256": EXPECTED_DEVELOPMENT_FILE_SHA256,
            "hamiltonian_hash": metadata["hamiltonian_hash"],
        },
        "held_out": {
            "path": HELD_OUT_RELATIVE_PATH,
            "raw_file_sha256": EXPECTED_HELD_OUT_FILE_SHA256,
            "npz_loaded": False,
            "signal_cost_ranking_evaluated": False,
        },
        "task": {
            "T": V4_Q * DELTA_TIME,
            "delta": DELTA_TIME,
            "q": V4_Q,
            "energy_hartree": energy,
            "exact_target": _complex_record(exact_target),
        },
        "random_signal_points": random_signals,
        "deterministic_signal_points": deterministic_signals,
        "wrapper_semantics_probe": wrapper_probe,
        "compile_smoke_tests": compile_smokes,
        "primitive_structural_coverage": primitive,
        "summary": {
            "random_signal_point_count": len(random_signals),
            "deterministic_signal_point_count": len(deterministic_signals),
            "compiled_full_wrapper_count": counters["full_wrappers_compiled"],
            "expected_compiled_full_wrapper_count_after_identity_collapse": (
                expected_wrapper_count
            ),
            "corrected_raw_normalization_pass": signal_pass,
            "re_im_hadamard_semantics_pass": wrapper_probe["overall_pass"],
            "controlled_full_wrapper_compile_pass": compile_pass,
            "resource_winner_selected": False,
            "expected_cost_monte_carlo_32_or_128_performed": False,
        },
        "counters": counters,
        "provenance": dict(provenance),
        "deviations": [],
        "S2_development_authorized": status == V4_PASS_STATUS,
        "S3_authorized": False,
        "automatic_next_stage": None,
        "quantum_shots_executed": 0,
    }
    payload["result_fingerprint"] = _fingerprint(payload)
    return payload


def run_s2_development(
    root: Path,
    v4_payload: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    validate_v4_payload(v4_payload)
    if v4_payload["status"] != V4_PASS_STATUS or v4_payload["deviations"]:
        raise RuntimeError("S2 requires a deviation-free V4 PASS artifact.")
    counters = _new_counters()
    hamiltonian, state, metadata = _load_inputs(root, counters)
    energy, exact_target = _target_signal(
        hamiltonian, state, S2_Q * DELTA_TIME
    )
    cache = TranspiledCircuitCostCache(maximum_entries=8192)

    deterministic = []
    for method, rank in (("B0", PRIMARY_RANK), ("B1", 12)):
        preparation = _prepare_deterministic(hamiltonian, rank)
        signal = _deterministic_signal_point(
            preparation,
            state,
            exact_target,
            method=method,
            rank=rank,
            q=S2_Q,
        )
        batch = _compile_cost_batch(
            preparation,
            stage="S2",
            stream="deterministic",
            method=method,
            rank=rank,
            q=S2_Q,
            rte_steps=0,
            cutoff=0,
            sample_count=None,
            master_seed=0,
            cache=cache,
        )
        candidate = {
            "method": method,
            "rank": rank,
            "r": 0,
            "K": 0,
            "signal": signal,
            "cost_batches": [batch],
        }
        _refresh_candidate(candidate)
        deterministic.append(candidate)

    random_candidates: dict[str, list[dict[str, Any]]] = {"B2": [], "B3": []}
    random_preparations = {
        "B2": _prepare_random(hamiltonian, "B2", PRIMARY_RANK),
        "B3": _prepare_random(hamiltonian, "B3", 0),
    }
    for method, preparation in random_preparations.items():
        rank = PRIMARY_RANK if method == "B2" else 0
        for rte_steps in S1_R_VALUES:
            for cutoff in S1_K_VALUES:
                signal = _random_signal_point(
                    preparation,
                    state,
                    exact_target,
                    stage="S2",
                    method=method,
                    rank=rank,
                    q=S2_Q,
                    rte_steps=rte_steps,
                    cutoff=cutoff,
                )
                batch = _compile_cost_batch(
                    preparation,
                    stage="S2",
                    stream="initial32",
                    method=method,
                    rank=rank,
                    q=S2_Q,
                    rte_steps=rte_steps,
                    cutoff=cutoff,
                    sample_count=S2_INITIAL_SAMPLE_COUNT,
                    master_seed=S2_MASTER_SEED,
                    cache=cache,
                )
                candidate = {
                    "method": method,
                    "rank": rank,
                    "r": rte_steps,
                    "K": cutoff,
                    "signal": signal,
                    "cost_batches": [batch],
                }
                _refresh_candidate(candidate)
                random_candidates[method].append(candidate)

    expansion_keys, initial_diagnostics = _initial_expansion_keys(
        random_candidates["B2"], random_candidates["B3"], deterministic
    )
    expanded = []
    for method in ("B2", "B3"):
        preparation = random_preparations[method]
        for candidate in random_candidates[method]:
            key = _candidate_key(candidate)
            if key not in expansion_keys:
                continue
            extension = _compile_cost_batch(
                preparation,
                stage="S2",
                stream="extension96",
                method=method,
                rank=int(candidate["rank"]),
                q=S2_Q,
                rte_steps=int(candidate["r"]),
                cutoff=int(candidate["K"]),
                sample_count=S2_EXTENSION_SAMPLE_COUNT,
                master_seed=S2_EXTENSION_MASTER_SEED,
                cache=cache,
            )
            candidate["cost_batches"].append(extension)
            _refresh_candidate(candidate)
            if candidate["pooled_cost"]["sample_count"] != S2_TOTAL_SAMPLE_COUNT:
                raise RuntimeError("Expanded cost cell did not pool to 128 samples.")
            expanded.append(key)

    decision = _decision(
        random_candidates["B2"], random_candidates["B3"], deterministic
    )
    selected_b2 = _selected_candidate(random_candidates["B2"])
    controls = []
    if selected_b2 is not None:
        selected_r = int(selected_b2["r"])
        selected_k = int(selected_b2["K"])
        for rank in CONTROL_RANKS:
            preparation = _prepare_random(hamiltonian, "B2", rank)
            signal = _random_signal_point(
                preparation,
                state,
                exact_target,
                stage="S2-control",
                method="B2",
                rank=rank,
                q=S2_Q,
                rte_steps=selected_r,
                cutoff=selected_k,
            )
            batch = _compile_cost_batch(
                preparation,
                stage="S2-control",
                stream="fixed32",
                method="B2",
                rank=rank,
                q=S2_Q,
                rte_steps=selected_r,
                cutoff=selected_k,
                sample_count=S2_INITIAL_SAMPLE_COUNT,
                master_seed=S2_MASTER_SEED,
                cache=cache,
            )
            candidate = {
                "method": "B2",
                "rank": rank,
                "r": selected_r,
                "K": selected_k,
                "signal": signal,
                "cost_batches": [batch],
            }
            _refresh_candidate(candidate)
            controls.append(candidate)

    all_random = [*random_candidates["B2"], *random_candidates["B3"], *controls]
    counters["signal_evaluations"] = len(deterministic) + len(all_random)
    counters["random_trajectories_compiled"] = sum(
        int(batch["sample_count"])
        for item in all_random
        for batch in item["cost_batches"]
    )
    counters["full_wrappers_compiled"] = 2 * counters[
        "random_trajectories_compiled"
    ] + 2 * len(deterministic)
    payload: dict[str, Any] = {
        "schema_version": S2_SCHEMA_VERSION,
        "series_id": SERIES_ID,
        "status": decision["status"],
        "authorization_commit": AUTHORIZATION_COMMIT,
        "authorization_sha256": AUTHORIZATION_SHA256,
        "specification_sha256": SPECIFICATION_SHA256,
        "V4_result_fingerprint": v4_payload["result_fingerprint"],
        "development_snapshot": {
            "path": DEVELOPMENT_RELATIVE_PATH,
            "raw_file_sha256": EXPECTED_DEVELOPMENT_FILE_SHA256,
            "hamiltonian_hash": metadata["hamiltonian_hash"],
        },
        "held_out": {
            "path": HELD_OUT_RELATIVE_PATH,
            "raw_file_sha256": EXPECTED_HELD_OUT_FILE_SHA256,
            "npz_loaded": False,
            "signal_cost_ranking_evaluated": False,
        },
        "task": {
            "T": S2_Q * DELTA_TIME,
            "delta": DELTA_TIME,
            "q": S2_Q,
            "energy_hartree": energy,
            "exact_target": _complex_record(exact_target),
            "complex_signal_error": 0.05,
            "axis_error": AXIS_ERROR,
            "axis_alpha": AXIS_ALPHA,
        },
        "deterministic_candidates": deterministic,
        "B2_candidates": random_candidates["B2"],
        "B3_candidates": random_candidates["B3"],
        "rank3_rank9_controls": controls,
        "sampling": {
            "initial_sample_count": S2_INITIAL_SAMPLE_COUNT,
            "extension_sample_count": S2_EXTENSION_SAMPLE_COUNT,
            "maximum_total_sample_count": S2_TOTAL_SAMPLE_COUNT,
            "initial_master_seed": S2_MASTER_SEED,
            "extension_master_seed": S2_EXTENSION_MASTER_SEED,
            "rz_relative_standard_error_trigger": RZ_RSE_TRIGGER,
            "expansion_keys": sorted(expansion_keys),
            "expanded_keys": sorted(expanded),
            "initial_diagnostics": initial_diagnostics,
            "additional_sampling_authorized": False,
        },
        "decision": decision,
        "counters": counters,
        "provenance": dict(provenance),
        "deviations": [],
        "state_preparation_included_in_primary": False,
        "quantum_shots_executed": 0,
        "held_out_npz_loaded": False,
        "held_out_signal_cost_ranking_evaluated": False,
        "S3_authorized": False,
        "automatic_next_stage": None,
        "mandatory_stop_reached": True,
    }
    payload["result_fingerprint"] = _fingerprint(payload)
    return payload


def _validate_fingerprint(payload: Mapping[str, Any]) -> None:
    copied = dict(payload)
    observed = copied.pop("result_fingerprint", None)
    if observed != _fingerprint(copied):
        raise ValueError("PR-2 result fingerprint mismatch.")


def _validate_common_stop(payload: Mapping[str, Any]) -> None:
    if payload.get("S3_authorized") is not False:
        raise ValueError("S3 must remain unauthorized.")
    if payload.get("automatic_next_stage") is not None:
        raise ValueError("automatic_next_stage must remain null.")
    if payload.get("quantum_shots_executed") != 0:
        raise ValueError("The validation must not execute quantum shots.")
    held_out = payload.get("held_out", {})
    if held_out.get("npz_loaded") is not False:
        raise ValueError("The held-out NPZ must remain unopened.")
    if held_out.get("signal_cost_ranking_evaluated") is not False:
        raise ValueError("Held-out signal/cost/ranking is forbidden.")
    counters = payload.get("counters", {})
    if counters.get("held_out_npz_loads") != 0:
        raise ValueError("Held-out NPZ load counter is nonzero.")
    if counters.get("molecular_calculations") != 0:
        raise ValueError("Molecular regeneration is forbidden.")


def validate_v4_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != V4_SCHEMA_VERSION:
        raise ValueError("Unexpected V4 schema version.")
    if payload.get("series_id") != SERIES_ID:
        raise ValueError("Unexpected PR-2 series ID.")
    if payload.get("status") not in V4_STATUSES:
        raise ValueError("Unexpected V4 status.")
    if payload.get("authorization_commit") != AUTHORIZATION_COMMIT:
        raise ValueError("V4 authorization commit mismatch.")
    _validate_fingerprint(payload)
    _validate_common_stop(payload)
    summary = payload.get("summary", {})
    if summary.get("resource_winner_selected") is not False:
        raise ValueError("V4 must not select a resource winner.")
    if summary.get("expected_cost_monte_carlo_32_or_128_performed") is not False:
        raise ValueError("V4 must not perform 32/128 expected-cost MC.")
    if payload["status"] == V4_PASS_STATUS:
        if payload.get("S2_development_authorized") is not True:
            raise ValueError("Passing V4 must carry the frozen S2 authorization.")
        if payload.get("deviations") != []:
            raise ValueError("Passing V4 must be deviation-free.")
        for key in (
            "corrected_raw_normalization_pass",
            "re_im_hadamard_semantics_pass",
            "controlled_full_wrapper_compile_pass",
        ):
            if summary.get(key) is not True:
                raise ValueError(f"Passing V4 lacks {key}.")


def validate_s2_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != S2_SCHEMA_VERSION:
        raise ValueError("Unexpected S2 schema version.")
    if payload.get("series_id") != SERIES_ID:
        raise ValueError("Unexpected PR-2 series ID.")
    if payload.get("status") not in S2_STATUSES:
        raise ValueError("Unexpected S2 status.")
    if payload.get("authorization_commit") != AUTHORIZATION_COMMIT:
        raise ValueError("S2 authorization commit mismatch.")
    _validate_fingerprint(payload)
    _validate_common_stop(payload)
    if payload.get("mandatory_stop_reached") is not True:
        raise ValueError("S2 must reach the mandatory stop.")
    if payload.get("held_out_npz_loaded") is not False:
        raise ValueError("S2 must not load the held-out NPZ.")
    if payload.get("held_out_signal_cost_ranking_evaluated") is not False:
        raise ValueError("S2 must not evaluate held-out results.")
    sampling = payload.get("sampling", {})
    if sampling.get("additional_sampling_authorized") is not False:
        raise ValueError("S2 artifact must close additional sampling.")
    for item in (*payload.get("B2_candidates", []), *payload.get("B3_candidates", [])):
        count = item.get("pooled_cost", {}).get("sample_count")
        if count not in (S2_INITIAL_SAMPLE_COUNT, S2_TOTAL_SAMPLE_COUNT):
            raise ValueError("Random S2 candidate has an invalid sample count.")


def write_json_artifact(
    payload: Mapping[str, Any],
    path: Path,
    *,
    validator: Any,
) -> None:
    validator(payload)
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def validate_v1_v3_artifact(root: Path) -> dict[str, Any]:
    path = root / V1_V3_RESULT_RELATIVE_PATH
    if file_sha256(path) != EXPECTED_V1_V3_SHA256:
        raise ValueError("V1--V3 result artifact SHA-256 mismatch.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != V1_V3_PASS_STATUS:
        raise ValueError("V1--V3 artifact is not passing.")
    if payload.get("result_fingerprint") != EXPECTED_V1_V3_FINGERPRINT:
        raise ValueError("V1--V3 result fingerprint mismatch.")
    return payload
