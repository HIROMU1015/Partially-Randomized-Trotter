"""Frozen PR-2/PR-3 minimal-pilot calculations.

The scope is fixed by ``docs/research/pr2_pr3_minimal_pilot_preregistration.md``.
This module deliberately stops at one H4 structural compression screen and one
two-qubit coherent-signal extrapolation screen.  It is not an RPE total-cost
model and does not authorize a parameter sweep.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import qiskit
import scipy
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RZGate, RZZGate
from scipy.linalg import expm

from .df_hamiltonian import (
    DFHamiltonian,
    PhysicalSector,
    build_df_h_d_from_molecule,
    df_linear_operator,
    solve_df_ground_state,
)
from .df_partial_randomized_pf import (
    df_deterministic_step_rz_cost,
    df_hamiltonian_hash,
)
from .df_rte_tail import (
    DFTailExtraction,
    dense_extracted_df_tail,
    extract_df_tail_from_hamiltonian,
)


SCHEMA_VERSION = "pr2_pr3_minimal_pilot_v1"
PREREGISTRATION_SHA256 = (
    "d8562545981be6aff382ff7b01ff2f7f857dab1bd5622ffcdab61ee0f557e6e8"
)
PR2_COMPRESSION_RANKS = (3, 6, 9)
PR2_REFERENCE_RANK = 12
PR2_TIME = 0.1
PR2_TAIL_BIAS_BUDGET = 1e-2
PR3_LEVELS = (4, 8, 16)
PR3_TIME = 0.8
PR3_TARGET_RMSE = 0.05
_COST_BASIS = ("rz", "cx", "sx", "x")


def _json_fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _complex_record(value: complex) -> dict[str, float]:
    number = complex(value)
    return {"real": float(number.real), "imag": float(number.imag)}


def _finite_nonnegative(values: Sequence[float]) -> bool:
    return all(math.isfinite(float(value)) and float(value) >= 0.0 for value in values)


def _dense_df_hamiltonian(hamiltonian: DFHamiltonian) -> np.ndarray:
    """Return the dense DF operator in Qiskit's little-endian basis order."""
    dimension = 1 << int(hamiltonian.n_qubits)
    sector = PhysicalSector(
        n_qubits=int(hamiltonian.n_qubits),
        basis_indices=np.arange(dimension, dtype=np.int64),
    )
    operator, _counter = df_linear_operator(hamiltonian, sector, backend="python")
    basis = np.eye(dimension, dtype=np.complex128)
    dense = np.column_stack([operator @ basis[:, index] for index in range(dimension)])
    dense = 0.5 * (dense + dense.conj().T)
    # ``df_linear_operator`` follows OpenFermion's mode-0-most-significant
    # indexing, while the DF circuit/extraction layer follows Qiskit's
    # qubit-0-least-significant indexing.  Bit reversal is a basis permutation,
    # not a change to the represented Hamiltonian.
    width = int(hamiltonian.n_qubits)
    permutation = np.asarray(
        [int(f"{index:0{width}b}"[::-1], 2) for index in range(dimension)],
        dtype=np.int64,
    )
    return dense[np.ix_(permutation, permutation)]


def _prefix_difference(
    reference: DFHamiltonian,
    direct: DFHamiltonian,
) -> dict[str, float]:
    if direct.n_blocks > reference.n_blocks:
        raise ValueError("direct rank exceeds reference rank")
    count = int(direct.n_blocks)
    lambda_difference = float(
        np.max(
            np.abs(
                np.asarray(reference.lambdas[:count])
                - np.asarray(direct.lambdas)
            ),
            initial=0.0,
        )
    )
    block_difference = 0.0
    for left, right in zip(
        reference.g_matrices[:count], direct.g_matrices, strict=True
    ):
        block_difference = max(
            block_difference,
            float(np.max(np.abs(np.asarray(left) - np.asarray(right)), initial=0.0)),
        )
    return {
        "constant_abs_difference": abs(
            float(reference.constant) - float(direct.constant)
        ),
        "one_body_max_abs_difference": float(
            np.max(
                np.abs(np.asarray(reference.one_body) - np.asarray(direct.one_body)),
                initial=0.0,
            )
        ),
        "lambda_max_abs_difference": lambda_difference,
        "g_matrix_max_abs_difference": block_difference,
        "overall_max_abs_difference": max(
            abs(float(reference.constant) - float(direct.constant)),
            float(
                np.max(
                    np.abs(
                        np.asarray(reference.one_body) - np.asarray(direct.one_body)
                    ),
                    initial=0.0,
                )
            ),
            lambda_difference,
            block_difference,
        ),
    }


def _append_basis(
    circuit: QuantumCircuit,
    extraction: DFTailExtraction,
    basis_id: str,
    *,
    inverse: bool,
) -> None:
    operations = list(extraction.basis_definition(basis_id).runtime_operations)
    if inverse:
        operations.reverse()
    for gate, qubits in operations:
        circuit.append(gate.inverse() if inverse else gate, list(qubits))


def _compiled_component_rz_count(
    extraction: DFTailExtraction,
    component_index: int,
    *,
    angle: float,
) -> dict[str, Any]:
    component = extraction.components[int(component_index)]
    if component.is_identity:
        raise ValueError("identity components must be extracted before cost evaluation")
    circuit = QuantumCircuit(int(extraction.num_system_qubits))
    _append_basis(circuit, extraction, component.basis_id, inverse=False)
    signed_angle = float(angle) * int(component.coefficient_sign)
    support = tuple(int(qubit) for qubit in component.diagonal_pauli_support)
    if len(support) == 1:
        circuit.append(RZGate(2.0 * signed_angle), list(support))
    elif len(support) == 2:
        circuit.append(RZZGate(2.0 * signed_angle), list(support))
    else:
        raise ValueError("DF residual components must have Z or ZZ support")
    _append_basis(circuit, extraction, component.basis_id, inverse=True)
    compiled = transpile(
        circuit,
        basis_gates=list(_COST_BASIS),
        optimization_level=0,
    )
    counts = compiled.count_ops()
    return {
        "component_id": component.component_id,
        "probability": float(component.coefficient_abs / extraction.rte_lambda_r),
        "support_size": len(support),
        "basis_operation_count": len(component.basis_change_operations),
        "rz_count": int(counts.get("rz", 0)),
        "compiled_size": int(compiled.size()),
        "compiled_depth": int(compiled.depth()),
    }


def _residual_sample_cost(
    extraction: DFTailExtraction,
    *,
    angle: float,
) -> dict[str, Any]:
    records = tuple(
        _compiled_component_rz_count(extraction, index, angle=angle)
        for index in range(len(extraction.components))
    )
    probability_sum = math.fsum(record["probability"] for record in records)
    expected_rz = math.fsum(
        record["probability"] * record["rz_count"] for record in records
    )
    expected_size = math.fsum(
        record["probability"] * record["compiled_size"] for record in records
    )
    return {
        "component_count": len(records),
        "probability_sum": float(probability_sum),
        "expected_rz_count": float(expected_rz),
        "expected_compiled_size": float(expected_size),
        "minimum_rz_count": min(record["rz_count"] for record in records),
        "maximum_rz_count": max(record["rz_count"] for record in records),
        "support_histogram": {
            str(size): sum(record["support_size"] == size for record in records)
            for size in (1, 2)
        },
        "components": list(records),
    }


def run_pr2_pilot() -> dict[str, Any]:
    """Execute the frozen H4 rank-compression structural screen."""
    reference, sector = build_df_h_d_from_molecule(
        4,
        distance=1.0,
        basis="sto-3g",
        df_rank=PR2_REFERENCE_RANK,
    )
    if reference.n_blocks != PR2_REFERENCE_RANK:
        raise ValueError("H4 reference did not produce the preregistered rank 12")
    reference_energy = solve_df_ground_state(reference, sector).energy
    reference_cost = df_deterministic_step_rz_cost(
        reference,
        "2nd",
        time=PR2_TIME,
        optimization_level=0,
    )
    reference_dense = _dense_df_hamiltonian(reference)
    reference_norm = float(np.linalg.norm(reference_dense, ord=2))
    points: list[dict[str, Any]] = []

    for rank in PR2_COMPRESSION_RANKS:
        direct, direct_sector = build_df_h_d_from_molecule(
            4,
            distance=1.0,
            basis="sto-3g",
            df_rank=int(rank),
        )
        if not np.array_equal(direct_sector.basis_indices, sector.basis_indices):
            raise ValueError("Physical sector changed across compression ranks")
        prefix_difference = _prefix_difference(reference, direct)
        compressed = reference.select_blocks(tuple(range(int(rank))))
        compressed_energy = solve_df_ground_state(compressed, sector).energy
        compressed_cost = df_deterministic_step_rz_cost(
            compressed,
            "2nd",
            time=PR2_TIME,
            optimization_level=0,
        )
        residual_indices = tuple(range(int(rank), PR2_REFERENCE_RANK))
        extraction = extract_df_tail_from_hamiltonian(
            f"pr2-h4-rank12-minus-rank{rank}",
            reference,
            residual_indices,
            identity_policy="extract_identity_phase",
            coefficient_atol=0.0,
        )
        if extraction.rte_lambda_r <= 0.0 or not extraction.components:
            raise ValueError("Preregistered PR-2 residual unexpectedly became empty")
        qdrift_steps = max(
            1,
            math.ceil(
                2.0
                * extraction.rte_lambda_r**2
                * PR2_TIME**2
                / PR2_TAIL_BIAS_BUDGET
            ),
        )
        qdrift_angle = extraction.rte_lambda_r * PR2_TIME / qdrift_steps
        sample_cost = _residual_sample_cost(extraction, angle=qdrift_angle)
        compressed_dense = _dense_df_hamiltonian(compressed)
        residual_dense = dense_extracted_df_tail(extraction, max_dense_qubits=8)
        reconstruction_residual = reference_dense - compressed_dense - residual_dense
        relative_reconstruction_error = float(
            np.linalg.norm(reconstruction_residual, ord=2)
            / max(1.0, reference_norm)
        )
        deterministic_rz = int(compressed_cost["total_ref_rz_count"])
        reference_rz = int(reference_cost["total_ref_rz_count"])
        hybrid_work = float(
            deterministic_rz + qdrift_steps * sample_cost["expected_rz_count"]
        )
        work_ratio = hybrid_work / reference_rz
        deterministic_saving = 1.0 - deterministic_rz / reference_rz
        correctness = {
            "prefix_max_abs_difference": prefix_difference[
                "overall_max_abs_difference"
            ],
            "prefix_matches_tolerance": bool(
                prefix_difference["overall_max_abs_difference"] <= 1e-10
            ),
            "relative_residual_reconstruction_error": (
                relative_reconstruction_error
            ),
            "residual_reconstruction_pass": bool(
                relative_reconstruction_error <= 1e-10
            ),
            "probability_sum_error": abs(sample_cost["probability_sum"] - 1.0),
            "probability_normalization_pass": bool(
                abs(sample_cost["probability_sum"] - 1.0) <= 1e-12
            ),
            "finite_nonnegative_work_pass": _finite_nonnegative(
                (
                    deterministic_rz,
                    reference_rz,
                    extraction.rte_lambda_r,
                    qdrift_steps,
                    sample_cost["expected_rz_count"],
                    sample_cost["expected_compiled_size"],
                    hybrid_work,
                    work_ratio,
                    deterministic_saving,
                    abs(float(compressed_energy) - float(reference_energy)),
                )
            ),
        }
        points.append(
            {
                "compression_rank": int(rank),
                "reference_rank": PR2_REFERENCE_RANK,
                "prefix_difference": prefix_difference,
                "compressed_energy": float(compressed_energy),
                "reference_energy": float(reference_energy),
                "discard_energy_bias": abs(
                    float(compressed_energy) - float(reference_energy)
                ),
                "deterministic_cost": compressed_cost,
                "deterministic_rz_saving_fraction": float(deterministic_saving),
                "residual_block_indices": list(residual_indices),
                "residual_tail_hash": extraction.tail_hash,
                "residual_lambda_r": float(extraction.rte_lambda_r),
                "residual_component_count": len(extraction.components),
                "residual_identity_coefficient": float(
                    extraction.deterministic_identity_coefficient
                ),
                "qdrift_screening_steps": int(qdrift_steps),
                "qdrift_sample_cost": sample_cost,
                "hybrid_screening_rz_work": hybrid_work,
                "hybrid_to_reference_work_ratio": float(work_ratio),
                "correctness": correctness,
            }
        )

    implementation_valid = all(
        point["correctness"]["prefix_matches_tolerance"]
        and point["correctness"]["residual_reconstruction_pass"]
        and point["correctness"]["probability_normalization_pass"]
        and point["correctness"]["finite_nonnegative_work_pass"]
        for point in points
    )
    candidates = [
        point
        for point in points
        if point["discard_energy_bias"] > 1e-3
        and point["deterministic_rz_saving_fraction"] >= 0.20
    ]
    if not implementation_valid:
        decision = "IMPLEMENTATION_INVALID"
    elif any(point["hybrid_to_reference_work_ratio"] < 1.0 for point in candidates):
        decision = "GO_PR2"
    elif any(point["hybrid_to_reference_work_ratio"] < 2.0 for point in candidates):
        decision = "CONDITIONAL_PR2"
    else:
        decision = "STOP_PR2_NO_COMPETITIVE_TRADEOFF"

    payload = {
        "pilot_id": "pr2_h4_rank_compression_random_residual",
        "scope": "one_outer_step_structural_screen_not_rpe_total_cost",
        "input": {
            "molecule": "H4_linear_chain",
            "distance_angstrom": 1.0,
            "basis": "sto-3g",
            "num_qubits": int(reference.n_qubits),
            "reference_rank": PR2_REFERENCE_RANK,
            "compression_ranks": list(PR2_COMPRESSION_RANKS),
            "time": PR2_TIME,
            "tail_bias_budget": PR2_TAIL_BIAS_BUDGET,
            "reference_hamiltonian_hash": df_hamiltonian_hash(reference),
        },
        "reference": {
            "ground_energy": float(reference_energy),
            "deterministic_cost": reference_cost,
            "spectral_norm": reference_norm,
        },
        "points": points,
        "correctness": {"implementation_valid": implementation_valid},
        "decision": decision,
        "limitations": [
            "The rank-12 DF Hamiltonian, not the exact molecular Hamiltonian, "
            "is the reference.",
            "The qDRIFT diamond-bound step count is a conservative screen.",
            "The work comparison is one uncontrolled outer step and excludes "
            "shots, state preparation, and RPE.",
        ],
    }
    payload["pilot_fingerprint"] = _json_fingerprint(payload)
    return payload


_PAULI_SINGLE = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    "Y": np.asarray([[0.0, -1j], [1j, 0.0]], dtype=np.complex128),
    "Z": np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
}


def _pauli_matrix(num_qubits: int, labels: Mapping[int, str]) -> np.ndarray:
    result = np.asarray([[1.0]], dtype=np.complex128)
    for qubit in reversed(range(int(num_qubits))):
        result = np.kron(result, _PAULI_SINGLE[labels.get(qubit, "I")])
    return result


def _rotation(pauli: np.ndarray, angle: float) -> np.ndarray:
    dimension = int(pauli.shape[0])
    return (
        math.cos(float(angle)) * np.eye(dimension, dtype=np.complex128)
        - 1j * math.sin(float(angle)) * pauli
    )


def _sequential_unitary(operations: Sequence[np.ndarray]) -> np.ndarray:
    if not operations:
        raise ValueError("operations must not be empty")
    result = np.eye(operations[0].shape[0], dtype=np.complex128)
    for operation in operations:
        result = np.asarray(operation) @ result
    return result


def _pr3_input_state() -> np.ndarray:
    ry = np.asarray(
        [
            [math.cos(0.73 / 2.0), -math.sin(0.73 / 2.0)],
            [math.sin(0.73 / 2.0), math.cos(0.73 / 2.0)],
        ],
        dtype=np.complex128,
    )
    rx = np.asarray(
        [
            [math.cos(-0.41 / 2.0), -1j * math.sin(-0.41 / 2.0)],
            [-1j * math.sin(-0.41 / 2.0), math.cos(-0.41 / 2.0)],
        ],
        dtype=np.complex128,
    )
    local = np.kron(rx, ry)
    cnot = np.zeros((4, 4), dtype=np.complex128)
    for source in range(4):
        target = source ^ 0b10 if source & 0b01 else source
        cnot[target, source] = 1.0
    state = cnot @ local @ np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    return state / np.linalg.norm(state)


def _qdrift_mean_operator(
    terms: Sequence[tuple[float, np.ndarray]],
    *,
    time: float,
    steps: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if steps <= 0:
        raise ValueError("qDRIFT steps must be positive")
    lambda_value = math.fsum(abs(float(coefficient)) for coefficient, _ in terms)
    if lambda_value <= 0.0:
        raise ValueError("qDRIFT terms must have positive total weight")
    dimension = terms[0][1].shape[0]
    mean_step = np.zeros((dimension, dimension), dtype=np.complex128)
    probabilities: list[float] = []
    angle = lambda_value * float(time) / int(steps)
    for coefficient, pauli in terms:
        probability = abs(float(coefficient)) / lambda_value
        signed_pauli = math.copysign(1.0, float(coefficient)) * np.asarray(pauli)
        probabilities.append(probability)
        mean_step += probability * _rotation(signed_pauli, angle)
    return np.linalg.matrix_power(mean_step, int(steps)), {
        "lambda": float(lambda_value),
        "probabilities": probabilities,
        "probability_sum": float(math.fsum(probabilities)),
        "angle": float(angle),
    }


def _partial_s2_mean_operator(
    deterministic_terms: Sequence[tuple[float, np.ndarray]],
    tail_mean: np.ndarray,
    *,
    time: float,
) -> np.ndarray:
    forward = [
        _rotation(pauli, float(coefficient) * float(time) / 2.0)
        for coefficient, pauli in deterministic_terms
    ]
    reverse = [
        _rotation(pauli, float(coefficient) * float(time) / 2.0)
        for coefficient, pauli in reversed(deterministic_terms)
    ]
    return _sequential_unitary([*forward, np.asarray(tail_mean), *reverse])


def _linear_estimator_cost(
    means: Sequence[complex],
    weights: Sequence[float],
    shot_costs: Sequence[float],
    *,
    target: complex,
    target_rmse: float,
) -> dict[str, Any]:
    if not (len(means) == len(weights) == len(shot_costs)):
        raise ValueError("means, weights, and shot_costs must align")
    estimate = sum(
        float(weight) * complex(mean) for weight, mean in zip(weights, means)
    )
    systematic_bias = abs(estimate - complex(target))
    if systematic_bias >= float(target_rmse):
        return {
            "eligible": False,
            "estimate": _complex_record(estimate),
            "systematic_bias": float(systematic_bias),
            "target_rmse": float(target_rmse),
            "total_work": None,
            "axes": {},
        }
    total_variance_budget = float(target_rmse) ** 2 - systematic_bias**2
    axis_variance_budget = total_variance_budget / 2.0
    axes: dict[str, Any] = {}
    total_work = 0.0
    for axis in ("real", "imag"):
        axis_means = [
            float(complex(mean).real if axis == "real" else complex(mean).imag)
            for mean in means
        ]
        variances = [max(0.0, 1.0 - value**2) for value in axis_means]
        scale = math.fsum(
            abs(float(weight)) * math.sqrt(variance * float(cost))
            for weight, variance, cost in zip(weights, variances, shot_costs)
        )
        shots: list[int] = []
        for weight, variance, cost in zip(weights, variances, shot_costs):
            if variance == 0.0 or weight == 0.0:
                shots.append(1)
                continue
            continuous = (
                abs(float(weight))
                * math.sqrt(variance / float(cost))
                * scale
                / axis_variance_budget
            )
            shots.append(max(1, math.ceil(continuous)))
        achieved_variance = math.fsum(
            float(weight) ** 2 * variance / shot_count
            for weight, variance, shot_count in zip(weights, variances, shots)
        )
        axis_work = math.fsum(
            shot_count * float(cost) for shot_count, cost in zip(shots, shot_costs)
        )
        total_work += axis_work
        axes[axis] = {
            "means": axis_means,
            "single_shot_variances": variances,
            "shots": shots,
            "variance_budget": float(axis_variance_budget),
            "achieved_variance": float(achieved_variance),
            "work": float(axis_work),
        }
    return {
        "eligible": True,
        "estimate": _complex_record(estimate),
        "systematic_bias": float(systematic_bias),
        "target_rmse": float(target_rmse),
        "remaining_variance_budget": float(total_variance_budget),
        "total_work": float(total_work),
        "axes": axes,
    }


def run_pr3_pilot() -> dict[str, Any]:
    """Execute the frozen two-qubit tail-only Richardson screen."""
    z0 = _pauli_matrix(2, {0: "Z"})
    x0x1 = _pauli_matrix(2, {0: "X", 1: "X"})
    x0 = _pauli_matrix(2, {0: "X"})
    z0z1 = _pauli_matrix(2, {0: "Z", 1: "Z"})
    y1 = _pauli_matrix(2, {1: "Y"})
    deterministic_terms = ((0.9, z0), (0.7, x0x1))
    randomized_terms = ((0.31, x0), (-0.27, z0z1), (0.19, y1))
    all_terms = (*deterministic_terms, *randomized_terms)
    state = _pr3_input_state()
    h_d = sum(coefficient * pauli for coefficient, pauli in deterministic_terms)
    h_r = sum(coefficient * pauli for coefficient, pauli in randomized_terms)
    h_full = h_d + h_r
    exact_unitary = expm(-1j * PR3_TIME * h_full)
    exact_tail = expm(-1j * PR3_TIME * h_r)
    tail_exact_unitary = _partial_s2_mean_operator(
        deterministic_terms,
        exact_tail,
        time=PR3_TIME,
    )
    deterministic_endpoint = _partial_s2_mean_operator(
        all_terms[:-1],
        _rotation(all_terms[-1][1], all_terms[-1][0] * PR3_TIME),
        time=PR3_TIME,
    )
    z_exact = complex(np.vdot(state, exact_unitary @ state))
    z_tail_exact = complex(np.vdot(state, tail_exact_unitary @ state))
    z_deterministic = complex(np.vdot(state, deterministic_endpoint @ state))

    partial_levels: list[dict[str, Any]] = []
    full_levels: list[dict[str, Any]] = []
    partial_values: list[complex] = []
    full_values: list[complex] = []
    probability_errors: list[float] = []
    for steps in PR3_LEVELS:
        tail_mean, tail_meta = _qdrift_mean_operator(
            randomized_terms,
            time=PR3_TIME,
            steps=steps,
        )
        partial_operator = _partial_s2_mean_operator(
            deterministic_terms,
            tail_mean,
            time=PR3_TIME,
        )
        partial_value = complex(np.vdot(state, partial_operator @ state))
        partial_values.append(partial_value)
        probability_errors.append(abs(tail_meta["probability_sum"] - 1.0))
        partial_levels.append(
            {
                "steps": int(steps),
                "mean": _complex_record(partial_value),
                "tail_exact_bias": abs(partial_value - z_tail_exact),
                "exact_h_bias": abs(partial_value - z_exact),
                "one_shot_work": int(4 + steps),
                "qdrift": tail_meta,
            }
        )

        full_mean, full_meta = _qdrift_mean_operator(
            all_terms,
            time=PR3_TIME,
            steps=steps,
        )
        full_value = complex(np.vdot(state, full_mean @ state))
        full_values.append(full_value)
        probability_errors.append(abs(full_meta["probability_sum"] - 1.0))
        full_levels.append(
            {
                "steps": int(steps),
                "mean": _complex_record(full_value),
                "exact_h_bias": abs(full_value - z_exact),
                "one_shot_work": int(steps),
                "qdrift": full_meta,
            }
        )

    ordinary_estimators = [
        {
            "steps": int(level["steps"]),
            **_linear_estimator_cost(
                [value],
                [1.0],
                [float(level["one_shot_work"])],
                target=z_tail_exact,
                target_rmse=PR3_TARGET_RMSE,
            ),
        }
        for value, level in zip(partial_values, partial_levels, strict=True)
    ]
    extrapolated_value = 2.0 * partial_values[1] - partial_values[0]
    extrapolated = _linear_estimator_cost(
        partial_values[:2],
        [-1.0, 2.0],
        [
            float(partial_levels[0]["one_shot_work"]),
            float(partial_levels[1]["one_shot_work"]),
        ],
        target=z_tail_exact,
        target_rmse=PR3_TARGET_RMSE,
    )
    extrapolated["weights"] = [-1.0, 2.0]
    extrapolated["levels"] = [4, 8]
    full_extrapolated = _linear_estimator_cost(
        full_values[:2],
        [-1.0, 2.0],
        [4.0, 8.0],
        target=z_exact,
        target_rmse=PR3_TARGET_RMSE,
    )
    full_extrapolated["weights"] = [-1.0, 2.0]
    full_extrapolated["levels"] = [4, 8]
    deterministic_estimator = _linear_estimator_cost(
        [z_deterministic],
        [1.0],
        [9.0],
        target=z_exact,
        target_rmse=PR3_TARGET_RMSE,
    )

    eligible_ordinary = [item for item in ordinary_estimators if item["eligible"]]
    best_ordinary = (
        min(eligible_ordinary, key=lambda item: float(item["total_work"]))
        if eligible_ordinary
        else None
    )
    bias_8 = abs(partial_values[1] - z_tail_exact)
    extrapolated_bias = abs(extrapolated_value - z_tail_exact)
    bias_condition = extrapolated_bias <= 0.75 * bias_8
    work_ratio = (
        float(extrapolated["total_work"]) / float(best_ordinary["total_work"])
        if extrapolated["eligible"] and best_ordinary is not None
        else None
    )
    paulis = [pauli for _coefficient, pauli in all_terms]
    maximum_hermiticity_defect = max(
        float(np.linalg.norm(pauli - pauli.conj().T, ord=2)) for pauli in paulis
    )
    maximum_involution_defect = max(
        float(np.linalg.norm(pauli @ pauli - np.eye(4), ord=2)) for pauli in paulis
    )
    unitary_defects = [
        float(np.linalg.norm(unitary.conj().T @ unitary - np.eye(4), ord=2))
        for unitary in (exact_unitary, tail_exact_unitary, deterministic_endpoint)
    ]
    monotone_window = bool(
        partial_levels[1]["tail_exact_bias"] < partial_levels[0]["tail_exact_bias"]
        or partial_levels[2]["tail_exact_bias"] < partial_levels[1]["tail_exact_bias"]
    )
    correctness = {
        "state_norm_error": abs(float(np.linalg.norm(state)) - 1.0),
        "maximum_pauli_hermiticity_defect": maximum_hermiticity_defect,
        "maximum_pauli_involution_defect": maximum_involution_defect,
        "maximum_exact_unitary_defect": max(unitary_defects),
        "maximum_probability_sum_error": max(probability_errors),
        "monotone_window": monotone_window,
        "finite_nonnegative_statistics_pass": bool(
            all(
                math.isfinite(value.real) and math.isfinite(value.imag)
                for value in (
                    z_exact,
                    z_tail_exact,
                    z_deterministic,
                    *partial_values,
                    *full_values,
                )
            )
            and _finite_nonnegative(
                (
                    *(level["tail_exact_bias"] for level in partial_levels),
                    *(level["exact_h_bias"] for level in partial_levels),
                    *(level["exact_h_bias"] for level in full_levels),
                    *(
                        item["total_work"]
                        for item in ordinary_estimators
                        if item["eligible"]
                    ),
                    *(
                        [extrapolated["total_work"]]
                        if extrapolated["eligible"]
                        else []
                    ),
                    *(
                        [full_extrapolated["total_work"]]
                        if full_extrapolated["eligible"]
                        else []
                    ),
                    *(
                        [deterministic_estimator["total_work"]]
                        if deterministic_estimator["eligible"]
                        else []
                    ),
                )
            )
        ),
    }
    implementation_valid = bool(
        correctness["state_norm_error"] <= 1e-12
        and maximum_hermiticity_defect <= 1e-12
        and maximum_involution_defect <= 1e-12
        and max(unitary_defects) <= 1e-12
        and max(probability_errors) <= 1e-14
        and correctness["finite_nonnegative_statistics_pass"]
    )
    if not implementation_valid:
        decision = "IMPLEMENTATION_INVALID"
    elif not monotone_window:
        decision = "MODEL_NOT_IN_ASYMPTOTIC_WINDOW"
    elif not bias_condition or work_ratio is None or work_ratio >= 2.0:
        decision = "STOP_PR3_VARIANCE_BACKBONE_DOMINATES"
    elif work_ratio < 1.0:
        decision = "GO_PR3"
    else:
        decision = "CONDITIONAL_PR3"

    input_record = {
        "num_qubits": 2,
        "time": PR3_TIME,
        "deterministic_terms": ["0.9 Z0", "0.7 X0X1"],
        "randomized_terms": ["0.31 X0", "-0.27 Z0Z1", "0.19 Y1"],
        "state_preparation": "Ry(0.73) q0; Rx(-0.41) q1; CNOT q0->q1",
        "levels": list(PR3_LEVELS),
        "target_complex_rmse": PR3_TARGET_RMSE,
    }
    input_record["input_fingerprint"] = _json_fingerprint(input_record)
    payload = {
        "pilot_id": "pr3_two_qubit_tail_only_richardson",
        "scope": "fixed_time_complex_signal_not_phase_or_rpe_total_cost",
        "input": input_record,
        "references": {
            "z_exact_h": _complex_record(z_exact),
            "z_tail_exact": _complex_record(z_tail_exact),
            "outer_pf_bias": abs(z_tail_exact - z_exact),
            "z_deterministic_endpoint": _complex_record(z_deterministic),
        },
        "partial_levels": partial_levels,
        "ordinary_estimators": ordinary_estimators,
        "best_ordinary_estimator": best_ordinary,
        "partial_tail_extrapolated": extrapolated,
        "partial_tail_extrapolated_bias": float(extrapolated_bias),
        "bias_8": float(bias_8),
        "bias_condition_pass": bool(bias_condition),
        "work_ratio_to_best_ordinary": work_ratio,
        "full_random_levels": full_levels,
        "full_random_extrapolated": full_extrapolated,
        "deterministic_endpoint": deterministic_estimator,
        "correctness": {**correctness, "implementation_valid": implementation_valid},
        "decision": decision,
        "limitations": [
            "This is one preregistered two-qubit condition.",
            "The exact random mean is evaluated analytically; no hardware shots "
            "are executed.",
            "Primitive-rotation work excludes state preparation and common "
            "ancilla wrappers.",
            "No qFLO asymptotic depth guarantee is transferred to the "
            "partial-tail circuit.",
        ],
    }
    payload["pilot_fingerprint"] = _json_fingerprint(payload)
    return payload


def _theme_scores(pr2: Mapping[str, Any], pr3: Mapping[str, Any]) -> dict[str, Any]:
    pr2_viable = pr2["decision"] in {"GO_PR2", "CONDITIONAL_PR2"}
    pr3_viable = pr3["decision"] in {"GO_PR3", "CONDITIONAL_PR3"}
    # Literature-difference and completion scores are fixed by the audit; the
    # three result-sensitive entries depend only on the preregistered decisions.
    pr2_scores = {
        "scoped_prior_art_difference": 1,
        "partial_randomization_is_essential": 2 if pr2_viable else 1,
        "one_sentence_result": 2,
        "negative_result_reuse": 2,
        "completion_feasibility": 1 if pr2_viable else 0,
    }
    pr3_scores = {
        "scoped_prior_art_difference": 1,
        "partial_randomization_is_essential": 2 if pr3_viable else 1,
        "one_sentence_result": 2,
        "negative_result_reuse": 2,
        "completion_feasibility": 2 if pr3_viable else 0,
    }
    pr2_total = sum(pr2_scores.values())
    pr3_total = sum(pr3_scores.values())
    if pr2_viable and not pr3_viable:
        selection = "SELECT_PR2_PRIMARY_CANDIDATE"
    elif pr3_viable and not pr2_viable:
        selection = "SELECT_PR3_PRIMARY_CANDIDATE"
    elif not pr2_viable and not pr3_viable:
        selection = "RETURN_TO_PR4_ESTIMATOR_COST_AUDIT"
    elif pr2_total >= pr3_total + 2:
        selection = "SELECT_PR2_PRIMARY_CANDIDATE"
    elif pr3_total >= pr2_total + 2:
        selection = "SELECT_PR3_PRIMARY_CANDIDATE"
    else:
        selection = "NO_PRIMARY_ONE_DOCUMENT_AUDIT_ONLY"
    return {
        "pr2": {"items": pr2_scores, "total": pr2_total},
        "pr3": {"items": pr3_scores, "total": pr3_total},
        "selection": selection,
        "mandatory_stop": "STOP_AFTER_PR2_PR3_MINIMAL_PILOTS",
        "additional_numerics_authorized": False,
    }


def run_pr2_pr3_minimal_pilots(
    *,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    pr2 = run_pr2_pilot()
    pr3 = run_pr3_pilot()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "qiskit": qiskit.__version__,
            "scipy": scipy.__version__,
        },
        "provenance": dict(provenance or {}),
        "pr2": pr2,
        "pr3": pr3,
        "theme_selection": _theme_scores(pr2, pr3),
        "evidence_status": "local_preregistered_pilot_not_external_ci",
    }
    payload["result_fingerprint"] = _json_fingerprint(payload)
    validate_pr2_pr3_payload(payload)
    return payload


def validate_pr2_pr3_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unexpected PR-2/PR-3 pilot schema")
    if payload.get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise ValueError("Preregistration fingerprint mismatch")
    if payload.get("theme_selection", {}).get("mandatory_stop") != (
        "STOP_AFTER_PR2_PR3_MINIMAL_PILOTS"
    ):
        raise ValueError("Mandatory post-pilot stop is missing")
    if (
        payload.get("theme_selection", {}).get("additional_numerics_authorized")
        is not False
    ):
        raise ValueError("Pilot payload must not authorize additional numerics")
    for key in ("pr2", "pr3"):
        section = payload.get(key)
        if not isinstance(section, Mapping) or not section.get("pilot_fingerprint"):
            raise ValueError(f"Missing {key} pilot record")
    expected = dict(payload)
    observed_fingerprint = str(expected.pop("result_fingerprint", ""))
    if observed_fingerprint != _json_fingerprint(expected):
        raise ValueError("Result fingerprint mismatch")


def write_pr2_pr3_payload(payload: Mapping[str, Any], path: Path) -> None:
    validate_pr2_pr3_payload(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
