"""Result-prior PR-2 S0/S1 correctness validation.

The authorization is frozen in
``docs/research/pr2_s0_s1_execution_amendment_v3.md``.  S0 freezes inputs,
checks prefix identity and implementation semantics.  S1 evaluates only
correctness and one canonical compiled wrapper per fixed cell.  This module
cannot run S2 or S3 and never performs quantum shots.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import qiskit
import scipy
from qiskit.quantum_info import Operator
from scipy.linalg import expm

from .df_hamiltonian import (
    DFHamiltonian,
    PhysicalSector,
    build_df_h_d_from_molecule,
    df_linear_operator,
    solve_df_ground_state,
)
from .df_partial_randomized_pf import (
    DFFragmentPartition,
    RankedDFFragment,
    df_fragment_weight,
    df_hamiltonian_hash,
    rank_df_fragments,
    split_df_hamiltonian_by_ld,
)
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
from .df_rte_tail import (
    dense_extracted_df_tail,
    extraction_to_normalized_rte_tail,
)
from .rpe_hadamard_compiled_cost_benchmark import (
    QiskitRPEHadamardBenchmarkCircuitBuilder,
    RPEHadamardCompiledCostBenchmarkRequest,
    generate_rpe_hadamard_compiled_cost_benchmark_dataset,
)
from .rpe_hadamard_interrogation import RPEHadamardInterrogationRequest
from .rte import (
    CompilerSettings,
    finite_rte_operator_moments,
    make_rte_config,
)


SCHEMA_S0 = "pr2_s0_validation_v1"
SCHEMA_S1 = "pr2_s1_correctness_summary_v1"
S0_STATUSES = frozenset(
    {
        "S0_PASS_S1_AUTHORIZED",
        "BLOCKED_IMPLEMENTATION_INVALID",
        "STOP_INPUT_REPRODUCTION_MISMATCH",
        "STOP_ENVIRONMENT_MISMATCH_NO_EXECUTION",
        "STOP_ESTIMAND_OR_SCOPE_INVALID",
    }
)
S1_STATUSES = frozenset(
    {
        "S1_CORRECTNESS_PASS_AWAITING_EXTERNAL_REVIEW",
        "BLOCKED_IMPLEMENTATION_INVALID",
        "STOP_ESTIMAND_OR_SCOPE_INVALID",
    }
)
AUTHORIZATION_COMMIT = "e9bffb85f9ed57712bb83150172a6a4662cecf7f"
AUTHORIZATION_MANIFEST_SHA256 = (
    "09b3760a890b3f7e15bf35be04ebaa936a68f4a2e322a14dc1e97a5fd1433f9c"
)
AMENDMENT_V3_SHA256 = (
    "a64b977034d6371095558c384d0d114c5cdb17a575dfb0bdbfe6c45604726344"
)
EXPECTED_DEVELOPMENT_HAMILTONIAN_HASH = (
    "d8b4aaf21afcc3935d5b5aa4d0805b358c5ec670d8104d25807c7cd0620a3dc3"
)
REFERENCE_RANK = 12
PRIMARY_RANK = 6
CONTROL_RANKS = (3, 9)
S1_R_VALUES = (1, 2, 4, 8, 16, 32)
S1_K_VALUES = (2, 4)
S1_SENTINEL = (1, 2)
DELTA_TIME = 0.1
TARGET_COMPLEX_ERROR = 0.05
AXIS_ERROR = TARGET_COMPLEX_ERROR / math.sqrt(2.0)
AXIS_ALPHA = 0.025
COMPILER_BASIS = ("rz", "sx", "x", "cx")
COMPILER_OPTIMIZATION_LEVEL = 1
COMPILER_SEED = 17
S1_MASTER_SEED = 20260927101
WEIGHT_RULE = "lambda_frobenius_squared"


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_hash(array: np.ndarray) -> str:
    normalized = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(normalized.dtype.str.encode())
    digest.update(_canonical_json(list(normalized.shape)))
    digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def _complex_record(value: complex) -> dict[str, float]:
    value = complex(value)
    return {"real": float(value.real), "imag": float(value.imag)}


def _canonicalize_state_phase(
    state: np.ndarray,
    sector_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    full = np.asarray(state, dtype=np.complex128).copy()
    sector = np.asarray(sector_state, dtype=np.complex128).copy()
    pivot = int(np.argmax(np.abs(sector)))
    amplitude = sector[pivot]
    if abs(amplitude) == 0.0:
        raise ValueError("Ground state has no nonzero phase pivot.")
    phase = np.exp(-1j * np.angle(amplitude))
    full *= phase
    sector *= phase
    if sector[pivot].real < 0.0:
        full *= -1.0
        sector *= -1.0
    return full, sector


def _bit_reverse(value: int, width: int) -> int:
    return int(f"{int(value):0{int(width)}b}"[::-1], 2)


def _to_qiskit_state(state: np.ndarray, num_qubits: int) -> np.ndarray:
    source = np.asarray(state, dtype=np.complex128)
    target = np.zeros_like(source)
    for index, amplitude in enumerate(source):
        target[_bit_reverse(index, num_qubits)] = amplitude
    return target


def _sector_hash(sector: PhysicalSector) -> str:
    payload = {
        "n_qubits": int(sector.n_qubits),
        "basis_indices_hash": _array_hash(
            np.asarray(sector.basis_indices, dtype="<i8")
        ),
        "n_electrons": sector.n_electrons,
        "nelec_alpha": sector.nelec_alpha,
        "nelec_beta": sector.nelec_beta,
        "sz_value": sector.sz_value,
    }
    return _fingerprint(payload)


def _state_hash(state: np.ndarray, sector_state: np.ndarray) -> str:
    return _fingerprint(
        {
            "state_vector_hash": _array_hash(
                np.asarray(state, dtype=np.complex128)
            ),
            "sector_state_vector_hash": _array_hash(
                np.asarray(sector_state, dtype=np.complex128)
            ),
            "global_phase_policy": "largest_sector_amplitude_real_positive_v1",
        }
    )


def _dense_df_operator_qiskit(hamiltonian: DFHamiltonian) -> np.ndarray:
    dimension = 1 << int(hamiltonian.n_qubits)
    sector = PhysicalSector(
        n_qubits=int(hamiltonian.n_qubits),
        basis_indices=np.arange(dimension, dtype=np.int64),
    )
    operator, _counter = df_linear_operator(hamiltonian, sector, backend="python")
    identity = np.eye(dimension, dtype=np.complex128)
    dense = np.column_stack(
        [operator @ identity[:, column] for column in range(dimension)]
    )
    dense = 0.5 * (dense + dense.conj().T)
    permutation = np.asarray(
        [_bit_reverse(index, hamiltonian.n_qubits) for index in range(dimension)],
        dtype=np.int64,
    )
    return dense[np.ix_(permutation, permutation)]


def _package_version(name: str) -> str:
    return importlib.metadata.version(name)


def environment_record() -> dict[str, Any]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "qiskit": qiskit.__version__,
        "openfermion": _package_version("openfermion"),
        "openfermionpyscf": _package_version("openfermionpyscf"),
        "pyscf": _package_version("pyscf"),
    }
    expected = {
        "python": "3.11.0rc1",
        "numpy": "1.26.4",
        "scipy": "1.14.1",
        "qiskit": "1.3.0",
        "openfermion": "1.6.1",
        "openfermionpyscf": "0.5",
        "pyscf": "2.7.0",
    }
    return {
        "versions": versions,
        "expected_versions": expected,
        "exact_match": versions == expected,
        "platform": platform.platform(),
    }


def generation_ranked_fragments(
    hamiltonian: DFHamiltonian,
) -> tuple[RankedDFFragment, ...]:
    return tuple(
        RankedDFFragment(
            rank=index,
            original_index=index,
            lam=float(hamiltonian.lambdas[index]),
            weight=df_fragment_weight(
                hamiltonian,
                index,
                weight_rule=WEIGHT_RULE,
            ),
            weight_rule=WEIGHT_RULE,
        )
        for index in range(hamiltonian.n_blocks)
    )


def generation_partition(
    hamiltonian: DFHamiltonian,
    rank: int,
) -> DFFragmentPartition:
    return split_df_hamiltonian_by_ld(
        hamiltonian,
        int(rank),
        ranked_fragments=generation_ranked_fragments(hamiltonian),
        weight_rule=WEIGHT_RULE,
    )


def _fragment_record(hamiltonian: DFHamiltonian, index: int) -> dict[str, Any]:
    return {
        "original_index": int(index),
        "lambda": float(hamiltonian.lambdas[index]),
        "weight": df_fragment_weight(
            hamiltonian,
            index,
            weight_rule=WEIGHT_RULE,
        ),
        "g_matrix_hash": _array_hash(hamiltonian.g_matrices[index]),
        "fragment_hash": _fingerprint(
            {
                "original_index": int(index),
                "lambda_hex": float(hamiltonian.lambdas[index]).hex(),
                "g_matrix_hash": _array_hash(hamiltonian.g_matrices[index]),
            }
        ),
    }


def _snapshot_metadata(
    hamiltonian: DFHamiltonian,
    sector: PhysicalSector,
    state: np.ndarray,
    sector_state: np.ndarray,
    *,
    distance: float,
    role: str,
    source_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "pr2_h4_rank12_snapshot_v1",
        "role": role,
        "model": "linear_H4",
        "distance_angstrom": float(distance),
        "basis": "sto-3g",
        "reference_rank": REFERENCE_RANK,
        "generation_call": (
            "build_df_h_d_from_molecule(4,distance=d,basis='sto-3g',df_rank=12)"
        ),
        "hamiltonian_metadata": _json_ready(hamiltonian.metadata),
        "hamiltonian_hash": df_hamiltonian_hash(hamiltonian),
        "constant_hex": float(hamiltonian.constant).hex(),
        "one_body_hash": _array_hash(hamiltonian.one_body),
        "lambdas_hash": _array_hash(hamiltonian.lambdas),
        "g_matrix_hashes": [
            _array_hash(matrix) for matrix in hamiltonian.g_matrices
        ],
        "sector_hash": _sector_hash(sector),
        "sector": {
            "n_qubits": int(sector.n_qubits),
            "n_electrons": sector.n_electrons,
            "nelec_alpha": sector.nelec_alpha,
            "nelec_beta": sector.nelec_beta,
            "sz_value": sector.sz_value,
            "basis_indices_hash": _array_hash(
                np.asarray(sector.basis_indices, dtype="<i8")
            ),
        },
        "state_hash": _state_hash(state, sector_state),
        "state_vector_hash": _array_hash(
            np.asarray(state, dtype=np.complex128)
        ),
        "sector_state_vector_hash": _array_hash(
            np.asarray(sector_state, dtype=np.complex128)
        ),
        "fragment_order": list(range(hamiltonian.n_blocks)),
        "fragment_hashes": [
            _fragment_record(hamiltonian, index)["fragment_hash"]
            for index in range(hamiltonian.n_blocks)
        ],
        "environment": environment_record()["versions"],
        "source_hashes": dict(source_hashes or {}),
        "global_phase_policy": "largest_sector_amplitude_real_positive_v1",
        "held_out_signal_cost_ranking_evaluated": False,
    }


def write_snapshot(
    path: Path,
    hamiltonian: DFHamiltonian,
    sector: PhysicalSector,
    state: np.ndarray,
    sector_state: np.ndarray,
    *,
    distance: float,
    role: str,
    source_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite snapshot: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = _snapshot_metadata(
        hamiltonian,
        sector,
        state,
        sector_state,
        distance=distance,
        role=role,
        source_hashes=source_hashes,
    )
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            constant=np.asarray(hamiltonian.constant, dtype=np.float64),
            one_body=np.asarray(hamiltonian.one_body, dtype=np.complex128),
            lambdas=np.asarray(hamiltonian.lambdas, dtype=np.float64),
            g_matrices=np.asarray(hamiltonian.g_matrices, dtype=np.complex128),
            sector_basis_indices=np.asarray(sector.basis_indices, dtype=np.int64),
            state_vector=np.asarray(state, dtype=np.complex128),
            sector_state_vector=np.asarray(sector_state, dtype=np.complex128),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    temporary.replace(target)
    return {**metadata, "path": str(target), "file_sha256": file_sha256(target)}


def load_snapshot(
    path: str | Path,
) -> tuple[DFHamiltonian, PhysicalSector, np.ndarray, dict[str, Any]]:
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        g_matrices = tuple(
            np.asarray(matrix, dtype=np.complex128)
            for matrix in np.asarray(payload["g_matrices"])
        )
        hamiltonian = DFHamiltonian(
            constant=float(payload["constant"]),
            one_body=np.asarray(payload["one_body"], dtype=np.complex128),
            lambdas=np.asarray(payload["lambdas"], dtype=np.float64),
            g_matrices=g_matrices,
            metadata=dict(metadata["hamiltonian_metadata"]),
        )
        sector_record = metadata["sector"]
        stored_indices = np.asarray(payload["sector_basis_indices"], dtype=np.int64)
        sector = PhysicalSector(
            n_qubits=int(sector_record["n_qubits"]),
            basis_indices=stored_indices,
            n_electrons=sector_record["n_electrons"],
            nelec_alpha=sector_record["nelec_alpha"],
            nelec_beta=sector_record["nelec_beta"],
            sz_value=sector_record["sz_value"],
        )
        state = np.asarray(payload["state_vector"], dtype=np.complex128)
        sector_state = np.asarray(payload["sector_state_vector"], dtype=np.complex128)
    if float(hamiltonian.constant).hex() != metadata["constant_hex"]:
        raise ValueError("Snapshot constant mismatch.")
    if _array_hash(hamiltonian.one_body) != metadata["one_body_hash"]:
        raise ValueError("Snapshot one-body hash mismatch.")
    if _array_hash(hamiltonian.lambdas) != metadata["lambdas_hash"]:
        raise ValueError("Snapshot lambda hash mismatch.")
    if [
        _array_hash(matrix) for matrix in hamiltonian.g_matrices
    ] != metadata["g_matrix_hashes"]:
        raise ValueError("Snapshot DF-fragment hash mismatch.")
    if df_hamiltonian_hash(hamiltonian) != metadata["hamiltonian_hash"]:
        raise ValueError("Snapshot canonical Hamiltonian hash mismatch.")
    if _sector_hash(sector) != metadata["sector_hash"]:
        raise ValueError("Snapshot sector hash mismatch.")
    if _array_hash(np.asarray(sector.basis_indices, dtype="<i8")) != metadata[
        "sector"
    ]["basis_indices_hash"]:
        raise ValueError("Snapshot sector-basis hash mismatch.")
    if _array_hash(state) != metadata["state_vector_hash"]:
        raise ValueError("Snapshot state-vector hash mismatch.")
    if _array_hash(sector_state) != metadata["sector_state_vector_hash"]:
        raise ValueError("Snapshot sector-state-vector hash mismatch.")
    if _state_hash(state, sector_state) != metadata["state_hash"]:
        raise ValueError("Snapshot state hash mismatch.")
    return hamiltonian, sector, state, metadata


def prefix_identity_record(hamiltonian: DFHamiltonian) -> dict[str, Any]:
    full = _dense_df_operator_qiskit(hamiltonian)
    full_norm = float(np.linalg.norm(full, ord=2))
    generation = generation_ranked_fragments(hamiltonian)
    weighted = rank_df_fragments(hamiltonian, weight_rule=WEIGHT_RULE)
    rows: list[dict[str, Any]] = []
    all_reconstruction_pass = True
    all_ordered_same = True
    for rank in (*CONTROL_RANKS[:1], PRIMARY_RANK, *CONTROL_RANKS[1:]):
        partitions = {
            "B2-G": split_df_hamiltonian_by_ld(
                hamiltonian,
                rank,
                ranked_fragments=generation,
                weight_rule=WEIGHT_RULE,
            ),
            "B2-W": split_df_hamiltonian_by_ld(
                hamiltonian,
                rank,
                ranked_fragments=weighted,
                weight_rule=WEIGHT_RULE,
            ),
        }
        method_records: dict[str, Any] = {}
        dense_parts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for method, partition in partitions.items():
            preparation = prepare_df_partial_s2(
                hamiltonian,
                partition,
                identity_policy="extract_identity_phase",
                coefficient_atol=0.0,
                partition_policy=(
                    "explicit_ordered_partition"
                    if method == "B2-G"
                    else "weight_ranked_prefix"
                ),
            )
            deterministic = _dense_df_operator_qiskit(
                hamiltonian.select_blocks(partition.deterministic_block_indices)
            )
            residual = dense_extracted_df_tail(
                preparation.tail_extraction,
                max_dense_qubits=8,
            )
            exact_residual = full - deterministic
            error = float(np.linalg.norm(residual - exact_residual, ord=2))
            relative_error = error / max(1.0, full_norm)
            probability_sum = math.fsum(
                component.probability
                for component in preparation.rte_preparation.symbolic_tail.components
            )
            reconstruction_pass = relative_error <= 1e-10
            all_reconstruction_pass &= reconstruction_pass
            dense_parts[method] = (deterministic, residual)
            method_records[method] = {
                "ordered_indices": list(partition.deterministic_block_indices),
                "unordered_indices": sorted(partition.deterministic_block_indices),
                "randomized_indices": list(partition.randomized_block_indices),
                "deterministic_fragments": [
                    _fragment_record(hamiltonian, index)
                    for index in partition.deterministic_block_indices
                ],
                "tail_hash": preparation.tail_extraction.tail_hash,
                "partition_hash": preparation.partition_hash,
                "preparation_hash": preparation.preparation_hash,
                "exact_rte_lambda_r": preparation.exact_rte_lambda_r,
                "probability_sum": float(probability_sum),
                "probability_sum_error": abs(float(probability_sum) - 1.0),
                "residual_reconstruction_relative_spectral_error": relative_error,
                "residual_reconstruction_pass": reconstruction_pass,
            }
        ordered_same = (
            partitions["B2-G"].deterministic_block_indices
            == partitions["B2-W"].deterministic_block_indices
        )
        set_same = set(partitions["B2-G"].deterministic_block_indices) == set(
            partitions["B2-W"].deterministic_block_indices
        )
        all_ordered_same &= ordered_same
        hd_norm = float(
            np.linalg.norm(
                dense_parts["B2-G"][0] - dense_parts["B2-W"][0],
                ord=2,
            )
        )
        hr_norm = float(
            np.linalg.norm(
                dense_parts["B2-G"][1] - dense_parts["B2-W"][1],
                ord=2,
            )
        )
        rows.append(
            {
                "rank": rank,
                "methods": method_records,
                "ordered_indices_identical": ordered_same,
                "unordered_sets_identical": set_same,
                "hd_difference_spectral_norm": hd_norm,
                "hr_difference_spectral_norm": hr_norm,
                "hamiltonian_identity_tolerance": 1e-10 * max(1.0, full_norm),
                "order_only_difference_is_method_delta": False,
            }
        )
    return {
        "ranks": rows,
        "all_ordered_indices_identical": all_ordered_same,
        "all_residual_reconstructions_pass": all_reconstruction_pass,
        "collapse_B2_G_and_B2_W": all_ordered_same,
        "explicit_generation_prefix_adapter_available": True,
    }


def corrected_hoeffding_shots(
    normalization_multiplier: float,
    axis_bias: float,
) -> int | None:
    multiplier = float(normalization_multiplier)
    bias = float(axis_bias)
    allowance = AXIS_ERROR - bias
    if allowance <= 0.0:
        return None
    return int(
        math.ceil(
            2.0
            * multiplier**2
            / allowance**2
            * math.log(2.0 / AXIS_ALPHA)
        )
    )


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


def _cell_seed(*parts: Any) -> int:
    return int.from_bytes(
        hashlib.sha256(_canonical_json(list(parts))).digest()[:8],
        "big",
    ) % (2**63)


def _prepare(
    hamiltonian: DFHamiltonian,
    method: str,
    rank: int,
) -> DFPartialS2Preparation:
    if method == "B2-G":
        partition = generation_partition(hamiltonian, rank)
    elif method == "B2-W":
        partition = split_df_hamiltonian_by_ld(
            hamiltonian,
            rank,
            weight_rule=WEIGHT_RULE,
        )
    elif method == "B3":
        partition = split_df_hamiltonian_by_ld(
            hamiltonian,
            0,
            weight_rule=WEIGHT_RULE,
        )
    else:
        raise ValueError(f"Unsupported randomized method: {method}")
    return prepare_df_partial_s2(
        hamiltonian,
        partition,
        identity_policy="extract_identity_phase",
        coefficient_atol=0.0,
        partition_policy=(
            "explicit_ordered_partition"
            if method == "B2-G"
            else "weight_ranked_prefix"
        ),
    )


def _deterministic_preparation(
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


def _actual_wrapper_probe(preparation: DFPartialS2Preparation) -> dict[str, Any]:
    config, distribution = make_rte_config(
        preparation.rte_preparation.symbolic_tail,
        evolution_time=DELTA_TIME,
        rte_steps=1,
        truncation_tolerance=1.0,
        finite_taylor_order=2,
        seed=S1_MASTER_SEED,
    )
    rows: list[dict[str, Any]] = []
    for q_value in (1, 8):
        repeated_request = make_df_partial_s2_repeated_request(
            preparation,
            step_time=DELTA_TIME,
            repetition_count=q_value,
            rte_config=config,
            rte_distribution=distribution,
            seed=_cell_seed("S0", "wrapper", q_value),
            controlled=True,
            ancilla_qubit=preparation.num_system_qubits,
            construction_policy="boundary_optimized",
        )
        evolution = QiskitDFPartialS2RepeatedCircuitBuilder().build(repeated_request)
        builder = QiskitRPEHadamardBenchmarkCircuitBuilder(
            maximum_repetition_count=8
        )
        axes: dict[str, Any] = {}
        for axis in ("cosine", "sine"):
            wrapper = builder.build(
                RPEHadamardInterrogationRequest(
                    evolution=evolution,
                    axis=axis,
                    include_measurement=True,
                )
            )
            axes[axis] = {
                "signal_component": wrapper.signal_component,
                "bit_value_mapping": [list(item) for item in wrapper.bit_value_mapping],
                "measurement_included": wrapper.include_measurement,
                "state_preparation_included": wrapper.state_preparation_included,
                "quantum_shots_executed": wrapper.quantum_shots_executed,
                "wrapper_fingerprint": wrapper.wrapper_fingerprint,
                "untranspiled_size": wrapper.circuit.size(),
                "untranspiled_depth": wrapper.circuit.depth(),
            }
        rows.append(
            {
                "q": q_value,
                "trajectory_fingerprint": evolution.trajectory_fingerprint,
                "repeated_circuit_size": evolution.untranspiled_circuit_size,
                "attenuation": asdict(evolution.attenuation),
                "axes": axes,
            }
        )
    return {"q_values": rows, "q1_q8_wrapper_build_pass": True}


def _corrected_estimator_probe(
    preparation: DFPartialS2Preparation,
) -> dict[str, Any]:
    normalized_tail = extraction_to_normalized_rte_tail(
        preparation.tail_extraction
    ).normalized_hamiltonian
    checks: list[dict[str, Any]] = []
    previous_bound: float | None = None
    for cutoff in (2, 4):
        config, distribution = make_rte_config(
            preparation.rte_preparation.symbolic_tail,
            evolution_time=DELTA_TIME,
            rte_steps=1,
            truncation_tolerance=1.0,
            finite_taylor_order=cutoff,
            seed=S1_MASTER_SEED,
        )
        moments = finite_rte_operator_moments(normalized_tail, config)
        reconstruction_error = float(
            np.linalg.norm(
                moments.normalization_product
                * moments.attenuated_event_mean_operator
                - moments.corrected_operator,
                ord=2,
            )
        )
        inverse_error = abs(
            moments.normalization_product * moments.attenuation_factor - 1.0
        )
        bound = float(distribution.step_truncation_residual_bound)
        checks.append(
            {
                "K": cutoff,
                "normalization_multiplier": moments.normalization_product,
                "attenuation": moments.attenuation_factor,
                "corrected_raw_reconstruction_spectral_error": reconstruction_error,
                "normalization_attenuation_inverse_error": inverse_error,
                "truncation_residual_bound": bound,
            }
        )
        if previous_bound is not None and bound > previous_bound + 1e-15:
            raise ValueError("Finite-RTE truncation bound increased with K.")
        previous_bound = bound
    q8 = checks[0]["normalization_multiplier"] ** 8
    q8_attenuation = checks[0]["attenuation"] ** 8
    return {
        "points": checks,
        "deterministic_multiplier": 1.0,
        "q8_total_multiplier": q8,
        "q8_total_attenuation": q8_attenuation,
        "q8_inverse_error": abs(q8 * q8_attenuation - 1.0),
        "eligible_example_shots": corrected_hoeffding_shots(1.2, 0.01),
        "ineligible_example_shots": corrected_hoeffding_shots(1.0, AXIS_ERROR),
        "overall_pass": all(
            row["corrected_raw_reconstruction_spectral_error"] <= 1e-12
            and row["normalization_attenuation_inverse_error"] <= 1e-12
            for row in checks
        )
        and abs(q8 * q8_attenuation - 1.0) <= 1e-12
        and corrected_hoeffding_shots(1.0, AXIS_ERROR) is None,
    }


def run_s0(
    output_directory: Path,
    *,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    environment = environment_record()
    if not environment["exact_match"]:
        payload = {
            "schema_version": SCHEMA_S0,
            "status": "STOP_ENVIRONMENT_MISMATCH_NO_EXECUTION",
            "authorization_commit": AUTHORIZATION_COMMIT,
            "authorization_manifest_sha256": AUTHORIZATION_MANIFEST_SHA256,
            "amendment_v3_sha256": AMENDMENT_V3_SHA256,
            "environment": environment,
            "provenance": dict(provenance),
            "automatic_next_stage": None,
            "S1_authorized": False,
            "molecular_calculations_executed": 0,
            "signal_evaluations_executed": 0,
            "circuits_compiled": 0,
            "trajectory_samples_drawn": 0,
            "quantum_shots_executed": 0,
        }
        payload["result_fingerprint"] = _fingerprint(payload)
        return payload

    generated: dict[str, Any] = {}
    snapshots: dict[str, Any] = {}
    for role, distance in (("development", 1.0), ("held_out_geometry", 1.3)):
        hamiltonian, sector = build_df_h_d_from_molecule(
            4,
            distance=distance,
            basis="sto-3g",
            df_rank=REFERENCE_RANK,
        )
        ground = solve_df_ground_state(
            hamiltonian,
            sector,
            matrix_free_backend="python",
            tol=1e-12,
        )
        if not ground.converged:
            raise RuntimeError(f"{role} ground-state solve did not converge.")
        state, sector_state = _canonicalize_state_phase(
            ground.state_vector,
            ground.sector_state_vector,
        )
        filename = (
            "h4_1p00_rank12_development_v1.npz"
            if role == "development"
            else "h4_1p30_rank12_held_out_v1.npz"
        )
        snapshots[role] = write_snapshot(
            output / filename,
            hamiltonian,
            sector,
            state,
            sector_state,
            distance=distance,
            role=role,
            source_hashes=provenance.get("source_hashes", {}),
        )
        generated[role] = (hamiltonian, sector, state)

    development_hash = snapshots["development"]["hamiltonian_hash"]
    if development_hash != EXPECTED_DEVELOPMENT_HAMILTONIAN_HASH:
        status = "STOP_INPUT_REPRODUCTION_MISMATCH"
        identity = None
        estimator = None
        wrappers = None
    else:
        development = generated["development"][0]
        identity = prefix_identity_record(development)
        rank6 = _prepare(development, "B2-G", PRIMARY_RANK)
        estimator = _corrected_estimator_probe(rank6)
        wrappers = _actual_wrapper_probe(rank6)
        if not identity["all_residual_reconstructions_pass"]:
            status = "STOP_ESTIMAND_OR_SCOPE_INVALID"
        elif not estimator["overall_pass"] or not wrappers["q1_q8_wrapper_build_pass"]:
            status = "BLOCKED_IMPLEMENTATION_INVALID"
        else:
            status = "S0_PASS_S1_AUTHORIZED"

    payload = {
        "schema_version": SCHEMA_S0,
        "status": status,
        "authorization_commit": AUTHORIZATION_COMMIT,
        "authorization_manifest_sha256": AUTHORIZATION_MANIFEST_SHA256,
        "amendment_v3_sha256": AMENDMENT_V3_SHA256,
        "environment": environment,
        "snapshots": snapshots,
        "prefix_identity": identity,
        "corrected_estimator_probe": estimator,
        "wrapper_probe": wrappers,
        "provenance": dict(provenance),
        "automatic_next_stage": None,
        "S1_authorized": status == "S0_PASS_S1_AUTHORIZED",
        "held_out_signal_cost_ranking_evaluated": False,
        "molecular_calculations_executed": 2,
        "signal_evaluations_executed": 0,
        "circuits_compiled": 0,
        "trajectory_samples_drawn": 0,
        "quantum_shots_executed": 0,
    }
    payload["result_fingerprint"] = _fingerprint(payload)
    return payload


def validate_s0_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_S0:
        raise ValueError("Unexpected PR-2 S0 schema.")
    expected = dict(payload)
    observed = str(expected.pop("result_fingerprint", ""))
    if observed != _fingerprint(expected):
        raise ValueError("PR-2 S0 fingerprint mismatch.")
    if payload.get("status") not in S0_STATUSES:
        raise ValueError("Unexpected PR-2 S0 terminal status.")
    if payload.get("authorization_commit") != AUTHORIZATION_COMMIT:
        raise ValueError("S0 authorization commit mismatch.")
    if payload.get("authorization_manifest_sha256") != (
        AUTHORIZATION_MANIFEST_SHA256
    ):
        raise ValueError("S0 authorization manifest hash mismatch.")
    if payload.get("amendment_v3_sha256") != AMENDMENT_V3_SHA256:
        raise ValueError("S0 amendment hash mismatch.")
    if payload.get("automatic_next_stage") is not None:
        raise ValueError("S0 must not contain an automatic next stage.")
    if payload.get("S1_authorized") is not (
        payload.get("status") == "S0_PASS_S1_AUTHORIZED"
    ):
        raise ValueError("S0 status and S1 authorization disagree.")
    if payload.get("quantum_shots_executed") != 0:
        raise ValueError("S0 must not execute quantum shots.")
    if payload.get("signal_evaluations_executed") != 0:
        raise ValueError("S0 must not evaluate candidate signals.")
    if payload.get("circuits_compiled") != 0:
        raise ValueError("S0 must not compile circuits.")
    if payload.get("trajectory_samples_drawn") != 0:
        raise ValueError("S0 must not draw trajectory samples.")
    if payload.get("held_out_signal_cost_ranking_evaluated", False) is not False:
        raise ValueError("S0 must not open held-out signal/cost/ranking.")
    if payload.get("status") == "S0_PASS_S1_AUTHORIZED":
        if set(payload.get("snapshots", {})) != {
            "development",
            "held_out_geometry",
        }:
            raise ValueError("Passing S0 must bind both frozen snapshots.")
        if not payload.get("prefix_identity", {}).get(
            "all_residual_reconstructions_pass", False
        ):
            raise ValueError("Passing S0 requires exact residual reconstruction.")
        if not payload.get("corrected_estimator_probe", {}).get(
            "overall_pass", False
        ):
            raise ValueError("Passing S0 requires the corrected-estimator gate.")
        if not payload.get("wrapper_probe", {}).get(
            "q1_q8_wrapper_build_pass", False
        ):
            raise ValueError("Passing S0 requires the q=1/q=8 wrapper gate.")


def write_json_artifact(
    payload: Mapping[str, Any],
    path: Path,
    *,
    validator,
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


def _signal(value: np.ndarray, state: np.ndarray) -> complex:
    return complex(np.vdot(state, value @ state))


def _random_signal_point(
    preparation: DFPartialS2Preparation,
    state: np.ndarray,
    exact_target: complex,
    *,
    method: str,
    rank: int,
    rte_steps: int,
    cutoff: int,
) -> dict[str, Any]:
    seed = _cell_seed("S1", method, rank, rte_steps, cutoff)
    request = make_df_partial_s2_step_request(
        preparation,
        step_time=DELTA_TIME,
        rte_steps=rte_steps,
        truncation_tolerance=1.0,
        finite_taylor_order=cutoff,
        seed=seed,
    )
    if request.rte_config is None or request.rte_distribution is None:
        raise RuntimeError("Random signal point lost its finite-RTE configuration.")
    parts = QiskitDFPartialS2CircuitBuilder().build_additive_circuits(request)
    forward = np.asarray(Operator(parts.forward_deterministic_half).data)
    reverse = np.asarray(Operator(parts.reverse_deterministic_half).data)
    normalized_tail = extraction_to_normalized_rte_tail(
        preparation.tail_extraction
    ).normalized_hamiltonian
    moments = finite_rte_operator_moments(normalized_tail, request.rte_config)
    corrected_operator = reverse @ moments.corrected_operator @ forward
    raw_operator = corrected_operator / moments.normalization_product
    exact_tail = expm(
        -1j * DELTA_TIME * preparation.exact_rte_lambda_r * normalized_tail
    )
    pf_operator = reverse @ exact_tail @ forward
    corrected = _signal(corrected_operator, state)
    raw = _signal(raw_operator, state)
    pf_signal = _signal(pf_operator, state)
    biases = {
        "real": abs(corrected.real - exact_target.real),
        "imag": abs(corrected.imag - exact_target.imag),
    }
    shots = {
        axis: corrected_hoeffding_shots(
            moments.normalization_product,
            bias,
        )
        for axis, bias in biases.items()
    }
    return {
        "method": method,
        "rank": rank,
        "r": rte_steps,
        "K": cutoff,
        "seed": seed,
        "raw_mean": _complex_record(raw),
        "corrected_mean": _complex_record(corrected),
        "pf_exact_tail_signal": _complex_record(pf_signal),
        "exact_target": _complex_record(exact_target),
        "normalization_multiplier": moments.normalization_product,
        "log_normalization_multiplier": math.log(moments.normalization_product),
        "attenuation": moments.attenuation_factor,
        "raw_bias_abs": abs(raw - exact_target),
        "corrected_bias_abs": abs(corrected - exact_target),
        "finite_truncation_bias_abs": abs(corrected - pf_signal),
        "outer_pf_bias_abs": abs(pf_signal - exact_target),
        "axis_bias": biases,
        "axis_shots": shots,
        "accuracy_eligible": all(value is not None for value in shots.values()),
        "exact_rte_lambda_r": preparation.exact_rte_lambda_r,
        "component_count": len(preparation.tail_extraction.components),
        "probability_sum": math.fsum(
            component.probability
            for component in preparation.rte_preparation.symbolic_tail.components
        ),
    }


def _compile_cell(
    preparation: DFPartialS2Preparation,
    *,
    method: str,
    rank: int,
    rte_steps: int,
    cutoff: int,
    seed: int,
) -> dict[str, Any]:
    if preparation.is_deterministic_only:
        config = None
        distribution = None
        evaluation_method = "exact"
        sample_count = None
        request_seed = None
        request_r = 0
        request_k = 0
    else:
        config, distribution = make_rte_config(
            preparation.rte_preparation.symbolic_tail,
            evolution_time=DELTA_TIME,
            rte_steps=rte_steps,
            truncation_tolerance=1.0,
            finite_taylor_order=cutoff,
            seed=seed,
        )
        evaluation_method = "monte_carlo"
        sample_count = 1
        request_seed = seed
        request_r = rte_steps
        request_k = cutoff
    request = RPEHadamardCompiledCostBenchmarkRequest(
        preparation=preparation,
        delta_time=DELTA_TIME,
        calibration_repetition_counts=(1,),
        holdout_repetition_counts=(),
        rte_steps_per_occurrence=request_r,
        finite_taylor_order=request_k,
        rte_config=config,
        rte_distribution=distribution,
        compiler=_compiler(),
        evaluation_method=evaluation_method,
        sample_count=sample_count,
        seed=request_seed,
        generation_id=f"pr2-s1-{method}-rank{rank}-r{rte_steps}-k{cutoff}",
        maximum_repetition_count=1,
        maximum_trajectories=100_000,
        maximum_samples=1,
        maximum_untranspiled_circuit_size=1_000_000,
        maximum_retained_trajectory_records=2,
        maximum_build_requests=4,
        maximum_transpile_requests=4,
        maximum_planned_instruction_applications=10_000_000,
        construction_policy="boundary_optimized",
    )
    result = generate_rpe_hadamard_compiled_cost_benchmark_dataset(request)
    records = [record.to_dict() for record in result.dataset.records]
    if len(records) != 2 or any(record["status"] != "complete" for record in records):
        raise RuntimeError(f"Full-wrapper compile failed for {method} rank {rank}.")
    return {
        "method": method,
        "rank": rank,
        "r": rte_steps,
        "K": cutoff,
        "canonical_trajectory_count": 1,
        "dataset_fingerprint": result.dataset.dataset_fingerprint,
        "records": records,
    }


def _primitive_structural_coverage(
    preparations: Sequence[tuple[str, int, DFPartialS2Preparation]],
) -> dict[str, Any]:
    records: set[tuple[str, tuple[int, ...], str, int]] = set()
    for method, rank, preparation in preparations:
        del method, rank
        specs = {
            spec.component_id: spec
            for spec in preparation.rte_preparation.component_specs
        }
        for cutoff in S1_K_VALUES:
            for order in range(0, cutoff + 1, 2):
                for slot in range(order + 1):
                    role = f"order_{order}_slot_{slot}"
                    for component in preparation.tail_extraction.components:
                        spec = specs[component.component_id]
                        records.add(
                            (
                                component.basis_id,
                                tuple(spec.diagonal_pauli_support),
                                role,
                                cutoff,
                            )
                        )
    rows = [
        {
            "basis_id": basis_id,
            "support": list(support),
            "event_role": role,
            "finite_taylor_order": cutoff,
        }
        for basis_id, support, role, cutoff in sorted(records)
    ]
    valid = all(
        len(row["support"]) in (1, 2)
        and row["finite_taylor_order"] in S1_K_VALUES
        for row in rows
    )
    return {
        "distinct_primitive_count": len(rows),
        "records": rows,
        "all_supports_and_orders_valid": valid,
    }


def run_s1(
    s0_payload: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    validate_s0_payload(s0_payload)
    if s0_payload["status"] != "S0_PASS_S1_AUTHORIZED":
        raise RuntimeError("S1 is not authorized by the supplied S0 artifact.")
    development_path = Path(s0_payload["snapshots"]["development"]["path"])
    hamiltonian, _sector, state_openfermion, snapshot_meta = load_snapshot(
        development_path
    )
    if file_sha256(development_path) != s0_payload["snapshots"]["development"][
        "file_sha256"
    ]:
        raise ValueError("Development snapshot file hash mismatch.")
    state = _to_qiskit_state(state_openfermion, hamiltonian.n_qubits)
    dense = _dense_df_operator_qiskit(hamiltonian)
    energy = float(np.real(np.vdot(state, dense @ state)))
    exact_target = complex(np.exp(-1j * energy * DELTA_TIME))

    collapse = bool(
        s0_payload["prefix_identity"]["collapse_B2_G_and_B2_W"]
    )
    methods = ("B2-G", "B3") if collapse else ("B2-G", "B2-W", "B3")
    preparations = {
        method: _prepare(
            hamiltonian,
            method,
            PRIMARY_RANK if method != "B3" else 0,
        )
        for method in methods
    }
    signal_points: list[dict[str, Any]] = []
    compile_records: list[dict[str, Any]] = []
    structural_preparations: list[tuple[str, int, DFPartialS2Preparation]] = []
    for method, preparation in preparations.items():
        rank = PRIMARY_RANK if method != "B3" else 0
        structural_preparations.append((method, rank, preparation))
        for rte_steps in S1_R_VALUES:
            for cutoff in S1_K_VALUES:
                signal_points.append(
                    _random_signal_point(
                        preparation,
                        state,
                        exact_target,
                        method=method,
                        rank=rank,
                        rte_steps=rte_steps,
                        cutoff=cutoff,
                    )
                )
                compile_records.append(
                    _compile_cell(
                        preparation,
                        method=method,
                        rank=rank,
                        rte_steps=rte_steps,
                        cutoff=cutoff,
                        seed=_cell_seed(
                            "S1", "compile", method, rank, rte_steps, cutoff
                        ),
                    )
                )

    sentinel_r, sentinel_k = S1_SENTINEL
    control_methods = ("B2-G",) if collapse else ("B2-G", "B2-W")
    for rank in CONTROL_RANKS:
        for method in control_methods:
            preparation = _prepare(hamiltonian, method, rank)
            structural_preparations.append((method, rank, preparation))
            compile_records.append(
                _compile_cell(
                    preparation,
                    method=method,
                    rank=rank,
                    rte_steps=sentinel_r,
                    cutoff=sentinel_k,
                    seed=_cell_seed(
                        "S1", "sentinel", method, rank, sentinel_r, sentinel_k
                    ),
                )
            )

    deterministic_signals: list[dict[str, Any]] = []
    for method, rank in (("B0", 3), ("B0", 6), ("B0", 9), ("B1", 12)):
        preparation = _deterministic_preparation(hamiltonian, rank)
        request = make_df_partial_s2_step_request(
            preparation,
            step_time=DELTA_TIME,
        )
        operator = np.asarray(
            Operator(QiskitDFPartialS2CircuitBuilder().build_step(request).circuit).data
        )
        mean = _signal(operator, state)
        deterministic_signals.append(
            {
                "method": method,
                "rank": rank,
                "mean": _complex_record(mean),
                "exact_target": _complex_record(exact_target),
                "bias_abs": abs(mean - exact_target),
                "normalization_multiplier": 1.0,
            }
        )
        compile_records.append(
            _compile_cell(
                preparation,
                method=method,
                rank=rank,
                rte_steps=0,
                cutoff=0,
                seed=0,
            )
        )

    primitive_coverage = _primitive_structural_coverage(structural_preparations)
    compiled_wrapper_count = sum(
        len(record["records"]) for record in compile_records
    )
    expected_upper = 88
    all_compile_complete = all(
        point["status"] == "complete"
        for record in compile_records
        for point in record["records"]
    )
    all_probability_pass = all(
        abs(point["probability_sum"] - 1.0) <= 1e-12
        for point in signal_points
    )
    all_finite = all(
        math.isfinite(point["normalization_multiplier"])
        and point["normalization_multiplier"] >= 1.0
        and math.isfinite(point["corrected_bias_abs"])
        for point in signal_points
    )
    correctness_pass = bool(
        all_compile_complete
        and compiled_wrapper_count <= expected_upper
        and primitive_coverage["all_supports_and_orders_valid"]
        and all_probability_pass
        and all_finite
    )
    status = (
        "S1_CORRECTNESS_PASS_AWAITING_EXTERNAL_REVIEW"
        if correctness_pass
        else "BLOCKED_IMPLEMENTATION_INVALID"
    )
    eligible_count = sum(point["accuracy_eligible"] for point in signal_points)
    payload = {
        "schema_version": SCHEMA_S1,
        "status": status,
        "authorization_commit": AUTHORIZATION_COMMIT,
        "authorization_manifest_sha256": AUTHORIZATION_MANIFEST_SHA256,
        "amendment_v3_sha256": AMENDMENT_V3_SHA256,
        "S0_result_fingerprint": s0_payload["result_fingerprint"],
        "development_snapshot": {
            "path": str(development_path),
            "file_sha256": file_sha256(development_path),
            "hamiltonian_hash": snapshot_meta["hamiltonian_hash"],
            "sector_hash": snapshot_meta["sector_hash"],
            "state_hash": snapshot_meta["state_hash"],
        },
        "prefix_identity": s0_payload["prefix_identity"],
        "B2_G_B2_W_collapsed": collapse,
        "exact_target": _complex_record(exact_target),
        "random_signal_points": signal_points,
        "deterministic_signal_points": deterministic_signals,
        "compile_smoke_tests": compile_records,
        "primitive_structural_coverage": primitive_coverage,
        "summary": {
            "random_signal_point_count": len(signal_points),
            "accuracy_eligible_count": eligible_count,
            "compiled_full_wrapper_count": compiled_wrapper_count,
            "compiled_full_wrapper_upper_bound_before_identity_collapse": 88,
            "all_compile_complete": all_compile_complete,
            "all_probability_normalization_pass": all_probability_pass,
            "all_numeric_fields_finite": all_finite,
            "resource_winner_selected": False,
            "expected_cost_monte_carlo_32_or_128_performed": False,
            "materiality_decision_performed": False,
            "state_preparation_break_even_performed": False,
            "held_out_signal_cost_ranking_evaluated": False,
        },
        "provenance": dict(provenance),
        "deviations": [],
        "automatic_next_stage": None,
        "S2_authorized": False,
        "molecular_calculations_executed": 0,
        "signal_evaluations_executed": len(signal_points)
        + len(deterministic_signals),
        "circuits_compiled": compiled_wrapper_count,
        "trajectory_samples_drawn": sum(
            1 for record in compile_records if record["method"] not in {"B0", "B1"}
        ),
        "quantum_shots_executed": 0,
    }
    payload["result_fingerprint"] = _fingerprint(payload)
    return payload


def validate_s1_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_S1:
        raise ValueError("Unexpected PR-2 S1 schema.")
    expected = dict(payload)
    observed = str(expected.pop("result_fingerprint", ""))
    if observed != _fingerprint(expected):
        raise ValueError("PR-2 S1 fingerprint mismatch.")
    if payload.get("status") not in S1_STATUSES:
        raise ValueError("Unexpected PR-2 S1 terminal status.")
    if payload.get("authorization_commit") != AUTHORIZATION_COMMIT:
        raise ValueError("S1 authorization commit mismatch.")
    if payload.get("authorization_manifest_sha256") != (
        AUTHORIZATION_MANIFEST_SHA256
    ):
        raise ValueError("S1 authorization manifest hash mismatch.")
    if payload.get("amendment_v3_sha256") != AMENDMENT_V3_SHA256:
        raise ValueError("S1 amendment hash mismatch.")
    if payload.get("automatic_next_stage") is not None:
        raise ValueError("S1 must stop without an automatic next stage.")
    if payload.get("S2_authorized") is not False:
        raise ValueError("S1 must not authorize S2.")
    if payload.get("quantum_shots_executed") != 0:
        raise ValueError("S1 must not execute quantum shots.")
    summary = payload.get("summary", {})
    if summary.get("resource_winner_selected") is not False:
        raise ValueError("S1 must not select a resource winner.")
    if summary.get("expected_cost_monte_carlo_32_or_128_performed") is not False:
        raise ValueError("S1 must not run expected-cost Monte Carlo.")
    for forbidden_flag in (
        "materiality_decision_performed",
        "state_preparation_break_even_performed",
        "held_out_signal_cost_ranking_evaluated",
    ):
        if summary.get(forbidden_flag) is not False:
            raise ValueError(f"S1 forbidden flag is not false: {forbidden_flag}")
    if payload.get("molecular_calculations_executed") != 0:
        raise ValueError("S1 must use the frozen snapshot without molecular builds.")


def s1_markdown_summary(payload: Mapping[str, Any]) -> str:
    validate_s1_payload(payload)
    summary = payload["summary"]
    identity = payload["prefix_identity"]
    return "\n".join(
        (
            "# PR-2 S1 correctness summary",
            "",
            f"- status: `{payload['status']}`",
            f"- result fingerprint: `{payload['result_fingerprint']}`",
            f"- source commit: `{payload['provenance'].get('git_commit')}`",
            f"- B2-G/B2-W collapsed: `{payload['B2_G_B2_W_collapsed']}`",
            "- prefix residual reconstruction pass: "
            f"`{identity['all_residual_reconstructions_pass']}`",
            f"- random signal points: `{summary['random_signal_point_count']}`",
            f"- accuracy-eligible points: `{summary['accuracy_eligible_count']}`",
            f"- compiled full wrappers: `{summary['compiled_full_wrapper_count']}`",
            f"- quantum shots executed: `{payload['quantum_shots_executed']}`",
            "- resource winner selected: `false`",
            "- 32/128 expected-cost Monte Carlo: `false`",
            "- held-out signal/cost/ranking opened: `false`",
            "- automatic next stage: `null`",
            "",
            "S1 correctness終了後のmandatory STOPである。S2/S3は未承認。",
            "",
        )
    )
