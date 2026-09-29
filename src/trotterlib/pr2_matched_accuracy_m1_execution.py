"""Authorized M1-A signal evaluation for the PR-2 matched-accuracy study.

The M1-A path is deliberately compile-free.  It reads the frozen development
snapshot once, evaluates the preregistered candidate ledger with dense
small-system reference actions, runs the frozen selector, and crosses the hard
precompile barrier.  It never resolves, stats, hashes, or loads the held-out
snapshot and never builds a Qiskit circuit.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import resource
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy

from .df_hamiltonian import DFHamiltonian
from .df_partial_s2 import (
    DFDeterministicFragmentSpec,
    DFDeterministicOneBodySpec,
    DFPartialS2Preparation,
    prepare_df_partial_s2,
)
from .df_rte_tail import extraction_to_normalized_rte_tail
from .pr2_matched_accuracy_m1_contract import (
    DEVELOPMENT_HAMILTONIAN_HASH,
    DEVELOPMENT_SNAPSHOT_SHA256,
    DEVELOPMENT_STATE_HASH,
    DEVELOPMENT_STATE_VECTOR_HASH,
    MAXIMUM_BOUNDARY_CANDIDATES,
    SERIES_ID,
    TOTAL_TIME,
    boundary_requests,
    canonical_json,
    enumerate_base_candidates,
    file_sha256,
    fingerprint,
    select_random_compile_cells,
)
from .pr2_matched_accuracy_m1_precompile_barrier import (
    evaluate_precompile_barrier,
)
from .pr2_new_series_validation import _load_snapshot_once
from .pr2_s0_s1_validation import (
    AXIS_ALPHA,
    AXIS_ERROR,
    _complex_record,
    _dense_df_operator_qiskit,
    _to_qiskit_state,
    corrected_hoeffding_shots,
    generation_partition,
)
from .rte import make_rte_config


SCHEMA_VERSION = "pr2_matched_accuracy_m1_a_result_v2"
DEVELOPMENT_RELATIVE_PATH = (
    "artifacts/pr2_s0_s1_validation/2026-09-28/"
    "h4_1p00_rank12_development_v1.npz"
)
MAXIMUM_SIGNAL_CANDIDATES = 212
M1_A_STATUSES = frozenset(
    {
        "M1_A_COMPLETE_M1_B_ELIGIBLE",
        "SELECTION_LIMITED",
        "IMPLEMENTATION_GATE_FAILED",
    }
)
RECONSTRUCTION_TOLERANCE = 1e-10
NORMALIZATION_TOLERANCE = 1e-12


def _package_version(name: str) -> str:
    return importlib.metadata.version(name)


def environment_record() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "qiskit": _package_version("qiskit"),
        "process_policy": "single_process_blas_threads_one",
        "gpu_queries": 0,
        "gpu_allocations": 0,
        "gpu_kernels": 0,
    }


def _new_counters() -> dict[str, int]:
    return {
        "development_raw_hash_checks": 0,
        "development_npz_loads": 0,
        "held_out_path_stats": 0,
        "held_out_raw_hash_checks": 0,
        "held_out_npz_loads": 0,
        "molecular_calculations": 0,
        "signal_evaluations": 0,
        "circuits_built": 0,
        "circuit_compilations": 0,
        "random_trajectories_sampled": 0,
        "random_trajectories_compiled": 0,
        "full_wrappers_compiled": 0,
        "quantum_shots_executed": 0,
    }


def _load_development_only(
    root: Path,
    counters: dict[str, int],
) -> tuple[DFHamiltonian, np.ndarray, dict[str, Any]]:
    path = root / DEVELOPMENT_RELATIVE_PATH
    counters["development_raw_hash_checks"] += 1
    observed_sha = file_sha256(path)
    if observed_sha != DEVELOPMENT_SNAPSHOT_SHA256:
        raise ValueError("Development snapshot SHA-256 differs from authorization.")
    counters["development_npz_loads"] += 1
    hamiltonian, _sector, state, _sector_state, metadata, _layout = (
        _load_snapshot_once(path)
    )
    if metadata["hamiltonian_hash"] != DEVELOPMENT_HAMILTONIAN_HASH:
        raise ValueError("Development Hamiltonian hash differs from authorization.")
    if metadata["state_hash"] != DEVELOPMENT_STATE_HASH:
        raise ValueError("Development state hash differs from authorization.")
    if metadata["state_vector_hash"] != DEVELOPMENT_STATE_VECTOR_HASH:
        raise ValueError("Development state-vector hash differs from authorization.")
    qiskit_state = _to_qiskit_state(state, hamiltonian.n_qubits)
    return hamiltonian, qiskit_state, metadata


def _prepare(
    hamiltonian: DFHamiltonian,
    rank: int,
) -> DFPartialS2Preparation:
    partition = generation_partition(hamiltonian, rank)
    return prepare_df_partial_s2(
        hamiltonian,
        partition,
        identity_policy="extract_identity_phase",
        coefficient_atol=0.0,
        partition_policy="explicit_ordered_partition",
    )


def _prepare_discard(
    hamiltonian: DFHamiltonian,
    rank: int,
) -> DFPartialS2Preparation:
    truncated = hamiltonian.select_blocks(tuple(range(int(rank))))
    partition = generation_partition(truncated, truncated.n_blocks)
    return prepare_df_partial_s2(
        truncated,
        partition,
        identity_policy="extract_identity_phase",
        coefficient_atol=0.0,
        partition_policy="explicit_ordered_partition",
    )


def _dense_block_operators(
    hamiltonian: DFHamiltonian,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], dict[str, float]]:
    """Return one-body and individual DF blocks in Qiskit basis order."""

    dimension = 1 << hamiltonian.n_qubits
    identity = np.eye(dimension, dtype=np.complex128)
    base = _dense_df_operator_qiskit(hamiltonian.select_blocks(()))
    one_body = base - float(hamiltonian.constant) * identity
    blocks = []
    for index in range(hamiltonian.n_blocks):
        selected = _dense_df_operator_qiskit(hamiltonian.select_blocks((index,)))
        blocks.append(selected - base)
    full = _dense_df_operator_qiskit(hamiltonian)
    reconstructed = (
        float(hamiltonian.constant) * identity
        + one_body
        + sum(blocks, np.zeros_like(one_body))
    )
    absolute = float(np.linalg.norm(full - reconstructed, ord=2))
    relative = absolute / max(1.0, float(np.linalg.norm(full, ord=2)))
    if relative > RECONSTRUCTION_TOLERANCE:
        raise ValueError("Dense deterministic-block reconstruction failed.")
    return one_body, tuple(blocks), {
        "absolute_operator_norm_error": absolute,
        "relative_operator_norm_error": relative,
    }


def _eigendecomposition(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hermitian = 0.5 * (matrix + matrix.conj().T)
    values, vectors = np.linalg.eigh(hermitian)
    residual = float(
        np.linalg.norm(hermitian @ vectors - vectors * values[None, :], ord=2)
    )
    if residual > 1e-9:
        raise ValueError("Dense block eigendecomposition residual is too large.")
    return values, vectors


def _apply_exponential(
    eigensystem: tuple[np.ndarray, np.ndarray],
    vector: np.ndarray,
    time_value: float,
) -> np.ndarray:
    values, vectors = eigensystem
    coefficients = vectors.conj().T @ vector
    return vectors @ (np.exp(-1j * float(time_value) * values) * coefficients)


def _apply_spectral_values(
    eigensystem: tuple[np.ndarray, np.ndarray],
    vector: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    _eigenvalues, vectors = eigensystem
    return vectors @ (values * (vectors.conj().T @ vector))


def _target_signal(
    hamiltonian: DFHamiltonian,
    state: np.ndarray,
) -> tuple[float, complex, float]:
    matrix = _dense_df_operator_qiskit(hamiltonian)
    energy_value = complex(np.vdot(state, matrix @ state))
    if abs(energy_value.imag) > 1e-11:
        raise ValueError("Frozen-state energy has a non-negligible imaginary part.")
    energy = float(energy_value.real)
    residual = float(np.linalg.norm(matrix @ state - energy * state))
    if residual > 1e-9:
        raise ValueError("Frozen-state Rayleigh residual exceeds the M1 gate.")
    return energy, complex(np.exp(-1j * energy * TOTAL_TIME)), residual


def _preparation_eigensystems(
    preparation: DFPartialS2Preparation,
    one_body: np.ndarray,
    fragments: Sequence[np.ndarray],
    block_cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> list[tuple[str, tuple[np.ndarray, np.ndarray]]]:
    result = []
    for block in preparation.deterministic_blocks:
        if isinstance(block, DFDeterministicOneBodySpec):
            key = "one_body"
            matrix = one_body
        elif isinstance(block, DFDeterministicFragmentSpec):
            key = f"fragment_{block.original_fragment_index}"
            matrix = fragments[int(block.original_fragment_index)]
        else:  # pragma: no cover - closed public union
            raise TypeError("Unknown deterministic block specification.")
        if key not in block_cache:
            block_cache[key] = _eigendecomposition(matrix)
        result.append((key, block_cache[key]))
    return result


def _apply_outer_step(
    vector: np.ndarray,
    deterministic: Sequence[tuple[str, tuple[np.ndarray, np.ndarray]]],
    *,
    delta: float,
    phase: complex,
    tail_action: Any | None,
) -> np.ndarray:
    current = phase * vector
    for _key, eigensystem in deterministic:
        current = _apply_exponential(eigensystem, current, delta / 2.0)
    if tail_action is not None:
        current = tail_action(current)
    for _key, eigensystem in reversed(deterministic):
        current = _apply_exponential(eigensystem, current, delta / 2.0)
    return current


def _axis_record(
    corrected: complex,
    exact_target: complex,
    normalization: float,
) -> tuple[dict[str, float], dict[str, int | None], int | None]:
    biases = {
        "real": abs(corrected.real - exact_target.real),
        "imag": abs(corrected.imag - exact_target.imag),
    }
    shots = {
        axis: corrected_hoeffding_shots(normalization, bias)
        for axis, bias in biases.items()
    }
    total = (
        None
        if any(value is None for value in shots.values())
        else int(sum(int(value) for value in shots.values()))
    )
    return biases, shots, total


def _fixed_action_count(preparation: DFPartialS2Preparation, q: int) -> int:
    one_body = sum(
        isinstance(block, DFDeterministicOneBodySpec)
        for block in preparation.deterministic_blocks
    )
    phase_occurrences = int(preparation.constant_coefficient != 0.0) + int(
        preparation.extracted_identity_coefficient != 0.0
    )
    # One full measured wrapper: ancilla preparation/readout, two one-body
    # halves per outer step, and explicit scalar phases per outer step.
    return int(2 + 2 * q * one_body + q * phase_occurrences)


def _deterministic_fragment_count(
    preparation: DFPartialS2Preparation,
    q: int,
) -> int:
    fragments = sum(
        isinstance(block, DFDeterministicFragmentSpec)
        for block in preparation.deterministic_blocks
    )
    return int(2 * q * fragments)


def _base_record(
    candidate: Mapping[str, Any],
    corrected: complex,
    raw: complex,
    exact_target: complex,
    normalization: float,
) -> dict[str, Any]:
    biases, shots, total_shots = _axis_record(
        corrected, exact_target, normalization
    )
    return {
        "candidate": dict(candidate),
        "candidate_id": candidate["candidate_id"],
        "candidate_fingerprint": candidate["candidate_fingerprint"],
        "exact_target": _complex_record(exact_target),
        "raw_mean": _complex_record(raw),
        "corrected_mean": _complex_record(corrected),
        "normalization_multiplier": float(normalization),
        "axis_bias": biases,
        "axis_allowance": {
            axis: float(AXIS_ERROR - bias) for axis, bias in biases.items()
        },
        "axis_shots": shots,
        "total_shots": total_shots,
        "accuracy_eligible": total_shots is not None,
        "ineligibility_reason": (
            None if total_shots is not None else "nonpositive_axis_allowance"
        ),
        "corrected_bias_abs": abs(corrected - exact_target),
    }


def _deterministic_signal_record(
    candidate: Mapping[str, Any],
    preparation: DFPartialS2Preparation,
    deterministic: Sequence[tuple[str, tuple[np.ndarray, np.ndarray]]],
    state: np.ndarray,
    exact_target: complex,
) -> dict[str, Any]:
    q = int(candidate["q"])
    delta = float(candidate["delta"])
    phase = np.exp(
        -1j
        * delta
        * (
            preparation.constant_coefficient
            + preparation.extracted_identity_coefficient
        )
    )
    evolved = state.copy()
    for _outer_step in range(q):
        evolved = _apply_outer_step(
            evolved,
            deterministic,
            delta=delta,
            phase=phase,
            tail_action=None,
        )
    mean = complex(np.vdot(state, evolved))
    record = _base_record(candidate, mean, mean, exact_target, 1.0)
    record.update(
        {
            "pf_exact_tail_signal": _complex_record(mean),
            "outer_pf_bias_abs": abs(mean - exact_target),
            "finite_truncation_bias_abs": 0.0,
            "normalization_log": 0.0,
            "normalization_direct_log_abs_difference": 0.0,
            "corrected_raw_reconstruction_abs_error": 0.0,
            "n_det": _deterministic_fragment_count(preparation, q),
            "n_rand": 0,
            "n_fixed": _fixed_action_count(preparation, q),
        }
    )
    return record


def _random_signal_record(
    candidate: Mapping[str, Any],
    preparation: DFPartialS2Preparation,
    deterministic: Sequence[tuple[str, tuple[np.ndarray, np.ndarray]]],
    tail_eigensystem: tuple[np.ndarray, np.ndarray],
    state: np.ndarray,
    exact_target: complex,
) -> dict[str, Any]:
    q = int(candidate["q"])
    delta = float(candidate["delta"])
    rte_steps = int(candidate["r"])
    cutoff = int(candidate["K"])
    config, distribution = make_rte_config(
        preparation.rte_preparation.symbolic_tail,
        evolution_time=delta,
        rte_steps=rte_steps,
        truncation_tolerance=1.0,
        finite_taylor_order=cutoff,
        seed=0,
    )
    tail_values, _tail_vectors = tail_eigensystem
    tau = float(config.dimensionless_step_time)
    polynomial = np.ones_like(tail_values, dtype=np.complex128)
    term = np.ones_like(tail_values, dtype=np.complex128)
    for degree in range(1, cutoff + 2):
        term = term * ((-1j * tau / degree) * tail_values)
        polynomial = polynomial + term
    corrected_occurrence_values = polynomial**rte_steps
    occurrence_log_normalization = rte_steps * math.log(
        distribution.exact_finite_distribution
    )
    total_log_normalization = q * occurrence_log_normalization
    normalization = float(math.exp(total_log_normalization))
    direct_normalization = float(
        distribution.exact_finite_distribution ** (q * rte_steps)
    )
    normalization_error = abs(normalization - direct_normalization)
    if normalization_error > NORMALIZATION_TOLERANCE * max(1.0, normalization):
        raise ValueError("Direct and log-domain normalization disagree.")
    raw_occurrence_values = corrected_occurrence_values / math.exp(
        occurrence_log_normalization
    )
    exact_occurrence_values = np.exp(
        -1j * delta * preparation.exact_rte_lambda_r * tail_values
    )
    phase = np.exp(
        -1j
        * delta
        * (
            preparation.constant_coefficient
            + preparation.extracted_identity_coefficient
        )
    )

    def apply_values(values: np.ndarray, vector: np.ndarray) -> np.ndarray:
        return _apply_spectral_values(tail_eigensystem, vector, values)

    corrected_state = state.copy()
    raw_state = state.copy()
    exact_tail_state = state.copy()
    for _outer_step in range(q):
        corrected_state = _apply_outer_step(
            corrected_state,
            deterministic,
            delta=delta,
            phase=phase,
            tail_action=lambda value: apply_values(
                corrected_occurrence_values, value
            ),
        )
        raw_state = _apply_outer_step(
            raw_state,
            deterministic,
            delta=delta,
            phase=phase,
            tail_action=lambda value: apply_values(raw_occurrence_values, value),
        )
        exact_tail_state = _apply_outer_step(
            exact_tail_state,
            deterministic,
            delta=delta,
            phase=phase,
            tail_action=lambda value: apply_values(
                exact_occurrence_values, value
            ),
        )
    corrected = complex(np.vdot(state, corrected_state))
    raw = complex(np.vdot(state, raw_state))
    exact_tail = complex(np.vdot(state, exact_tail_state))
    reconstruction_error = abs(raw * normalization - corrected)
    if reconstruction_error > 1e-10:
        raise ValueError("Raw/corrected signal reconstruction failed.")

    expected_applications = math.fsum(
        probability * (order + 1)
        for order, probability in zip(
            distribution.orders,
            distribution.order_probabilities,
            strict=True,
        )
    )
    n_det = _deterministic_fragment_count(preparation, q)
    n_rand_exact = q * rte_steps * expected_applications
    n_rand = int(math.ceil(n_rand_exact - 1e-15))
    n_fixed = _fixed_action_count(preparation, q)
    record = _base_record(
        candidate, corrected, raw, exact_target, normalization
    )
    record.update(
        {
            "pf_exact_tail_signal": _complex_record(exact_tail),
            "outer_pf_bias_abs": abs(exact_tail - exact_target),
            "finite_truncation_bias_abs": abs(corrected - exact_tail),
            "normalization_log": total_log_normalization,
            "normalization_direct_log_abs_difference": normalization_error,
            "corrected_raw_reconstruction_abs_error": reconstruction_error,
            "attenuation": 1.0 / normalization,
            "exact_rte_lambda_r": float(preparation.exact_rte_lambda_r),
            "finite_distribution": distribution.to_dict(),
            "expected_random_applications_exact": float(n_rand_exact),
            "random_action_integer_policy": "ceil_expected_applications_v1",
            "n_det": n_det,
            "n_rand": n_rand,
            "n_fixed": n_fixed,
            "W_action": (
                None
                if record["total_shots"] is None
                else int(record["total_shots"])
                * (n_det + n_rand + n_fixed)
            ),
            "W_tail": (
                None
                if record["total_shots"] is None
                else int(record["total_shots"]) * n_rand
            ),
        }
    )
    return record


def _validate_authorization(
    root: Path,
    authorization_path: Path,
) -> tuple[dict[str, Any], str]:
    absolute = authorization_path
    if not absolute.is_absolute():
        absolute = root / absolute
    authorization_sha = file_sha256(absolute)
    authorization = json.loads(absolute.read_text(encoding="utf-8"))
    if authorization.get("schema_version") != (
        "pr2_matched_accuracy_m1_execution_authorization_v1"
    ):
        raise ValueError("Unexpected M1 execution authorization schema.")
    permissions = authorization.get("permissions", {})
    required_true = (
        "m1_a_scientific_execution_authorized",
        "development_npz_load_authorized",
        "signal_evaluation_authorized",
    )
    if any(permissions.get(name) is not True for name in required_true):
        raise ValueError("M1-A permissions are incomplete.")
    required_false = (
        "held_out_access_authorized",
        "m1_b_direct_compile_before_clear_barrier_authorized",
        "quantum_shots_authorized",
        "s3_authorized",
    )
    if any(permissions.get(name) is not False for name in required_false):
        raise ValueError("M1 authorization violates a mandatory prohibition.")
    for relative, expected in authorization["required_source_hashes"].items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"Authorized source hash differs: {relative}")
    return authorization, authorization_sha


def run_m1_a(
    project_root: Path,
    *,
    authorization_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    root = project_root.resolve()
    authorization, authorization_sha = _validate_authorization(
        root, authorization_path
    )
    counters = _new_counters()
    hamiltonian, state, metadata = _load_development_only(root, counters)
    energy, exact_target, rayleigh_residual = _target_signal(hamiltonian, state)
    one_body, fragment_matrices, reconstruction = _dense_block_operators(
        hamiltonian
    )
    preparations = {
        rank: _prepare(hamiltonian, rank) for rank in (0, 3, 6, 9, 12)
    }
    discard_preparations = {
        rank: _prepare_discard(hamiltonian, rank) for rank in (3, 6, 9)
    }
    block_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    deterministic_by_rank = {
        rank: _preparation_eigensystems(
            preparation, one_body, fragment_matrices, block_cache
        )
        for rank, preparation in preparations.items()
    }
    discard_deterministic_by_rank = {
        rank: _preparation_eigensystems(
            preparation, one_body, fragment_matrices, block_cache
        )
        for rank, preparation in discard_preparations.items()
    }
    tail_eigensystems: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    tail_reconstruction = {}
    full_dense = _dense_df_operator_qiskit(hamiltonian)
    for rank in (0, 3, 6, 9):
        preparation = preparations[rank]
        normalized = extraction_to_normalized_rte_tail(
            preparation.tail_extraction,
            max_dense_qubits=8,
        ).normalized_hamiltonian
        tail_eigensystems[rank] = _eigendecomposition(normalized)
        prefix_dense = _dense_df_operator_qiskit(
            hamiltonian.select_blocks(
                preparation.deterministic_fragment_indices
            )
        )
        extracted = (
            preparation.extracted_identity_coefficient
            * np.eye(full_dense.shape[0], dtype=np.complex128)
            + preparation.exact_rte_lambda_r * normalized
        )
        error = float(np.linalg.norm(full_dense - prefix_dense - extracted, ord=2))
        relative = error / max(1.0, float(np.linalg.norm(full_dense, ord=2)))
        if relative > RECONSTRUCTION_TOLERANCE:
            raise ValueError(f"Rank-{rank} tail reconstruction failed.")
        tail_reconstruction[str(rank)] = {
            "absolute_operator_norm_error": error,
            "relative_operator_norm_error": relative,
        }

    candidates = enumerate_base_candidates()
    records: list[dict[str, Any]] = []

    def evaluate(candidate: Mapping[str, Any]) -> dict[str, Any]:
        rank = int(candidate["rank"])
        if candidate["method"] == "B0":
            preparation = discard_preparations[rank]
            deterministic = discard_deterministic_by_rank[rank]
        else:
            preparation = preparations[rank]
            deterministic = deterministic_by_rank[rank]
        if candidate["method"] in {"B0", "B1"}:
            result = _deterministic_signal_record(
                candidate,
                preparation,
                deterministic,
                state,
                exact_target,
            )
        else:
            result = _random_signal_record(
                candidate,
                preparation,
                deterministic,
                tail_eigensystems[rank],
                state,
                exact_target,
            )
        counters["signal_evaluations"] += 1
        return result

    for candidate in candidates:
        records.append(evaluate(candidate))
    random_base = [
        record for record in records if record["candidate"]["method"] in {"B2", "B3"}
    ]
    requests = boundary_requests(random_base)
    if len(requests) > MAXIMUM_BOUNDARY_CANDIDATES:
        raise ValueError("Boundary candidate cap exceeded.")
    boundary_fingerprints = []
    for request in requests:
        candidate = request["candidate"]
        candidates.append(candidate)
        records.append(evaluate(candidate))
        boundary_fingerprints.append(candidate["candidate_fingerprint"])

    random_records = [
        record for record in records if record["candidate"]["method"] in {"B2", "B3"}
    ]
    selection = select_random_compile_cells(
        random_records,
        boundary_fingerprints=boundary_fingerprints,
    )
    barrier = evaluate_precompile_barrier(selection)
    status = str(barrier["status"])
    if status not in M1_A_STATUSES:
        raise ValueError("Unexpected M1-A status.")
    elapsed = time.perf_counter() - started
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "series_id": SERIES_ID,
        "status": status,
        "execution_authorization_sha256": authorization_sha,
        "development_snapshot": {
            "path": DEVELOPMENT_RELATIVE_PATH,
            "raw_file_sha256": DEVELOPMENT_SNAPSHOT_SHA256,
            "hamiltonian_hash": metadata["hamiltonian_hash"],
            "state_hash": metadata["state_hash"],
            "state_vector_hash": metadata["state_vector_hash"],
            "energy_hartree": energy,
            "rayleigh_residual_l2": rayleigh_residual,
            "dense_block_reconstruction": reconstruction,
            "tail_reconstruction_by_rank": tail_reconstruction,
        },
        "candidate_ledger": candidates,
        "signal_records": records,
        "compile_selection": selection,
        "precompile_barrier": barrier,
        "compile_records": [],
        "counters": counters,
        "decision": {
            "status": status,
            "selection_limited": bool(selection["selection_limited"]),
            "selection_limited_reasons": selection[
                "selection_limited_reasons"
            ],
            "winner_claim_permitted": False,
            "held_out_candidate_selection_permitted": False,
            "m1_b_next_action": barrier["next_action"],
            "execution": {
                "authorization_schema": authorization["schema_version"],
                "environment": environment_record(),
                "wall_time_s": float(elapsed),
                "peak_rss_kib": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                ),
                "axis_error": AXIS_ERROR,
                "axis_alpha": AXIS_ALPHA,
                "total_time": TOTAL_TIME,
            },
        },
        "held_out": {
            "path_resolved_or_statted": False,
            "raw_sha256_read": False,
            "npz_loaded": False,
            "signal_cost_ranking_evaluated": False,
        },
        "S3_authorized": False,
        "automatic_next_stage": None,
    }
    payload["result_fingerprint"] = fingerprint(payload)
    validate_m1_a_result(payload)
    return payload


def validate_m1_a_result(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unexpected M1-A result schema.")
    if payload.get("status") not in M1_A_STATUSES:
        raise ValueError("Unexpected M1-A status.")
    ledger = payload.get("candidate_ledger")
    records = payload.get("signal_records")
    if not isinstance(ledger, list) or not (208 <= len(ledger) <= 212):
        raise ValueError("M1-A candidate ledger is outside the frozen bounds.")
    if not isinstance(records, list) or len(records) != len(ledger):
        raise ValueError("M1-A signal-record count differs from the ledger.")
    counters = payload.get("counters", {})
    required_zero = (
        "held_out_path_stats",
        "held_out_raw_hash_checks",
        "held_out_npz_loads",
        "molecular_calculations",
        "circuits_built",
        "circuit_compilations",
        "random_trajectories_sampled",
        "random_trajectories_compiled",
        "full_wrappers_compiled",
        "quantum_shots_executed",
    )
    if any(counters.get(name) != 0 for name in required_zero):
        raise ValueError("A prohibited M1-A counter is nonzero.")
    if counters.get("development_npz_loads") != 1:
        raise ValueError("M1-A must load the development snapshot exactly once.")
    if counters.get("signal_evaluations") != len(records):
        raise ValueError("M1-A signal-evaluation counter is inconsistent.")
    if payload.get("compile_records") != []:
        raise ValueError("M1-A compile records must be empty.")
    barrier = payload.get("precompile_barrier", {})
    limited = bool(payload.get("compile_selection", {}).get("selection_limited"))
    expected_status = "SELECTION_LIMITED" if limited else (
        "M1_A_COMPLETE_M1_B_ELIGIBLE"
    )
    if payload.get("status") != expected_status or barrier.get("status") != expected_status:
        raise ValueError("M1-A result and barrier statuses disagree.")
    if barrier.get("compile_jobs_materialized_at_barrier") != 0:
        raise ValueError("M1-A materialized compile jobs before the barrier.")
    stored = payload.get("result_fingerprint")
    unsigned = dict(payload)
    unsigned.pop("result_fingerprint", None)
    if stored != fingerprint(unsigned):
        raise ValueError("M1-A result fingerprint mismatch.")


def write_m1_a_result(path: Path, payload: Mapping[str, Any]) -> None:
    validate_m1_a_result(payload)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite M1-A output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(payload) + b"\n")
