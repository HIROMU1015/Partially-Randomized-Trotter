"""Result-prior V1--V3 validation for the rebased PR-2 data series.

This module consumes one already-frozen development snapshot.  It cannot
rebuild a molecule, evaluate a coherent signal, sample an RTE trajectory,
compile a circuit, or advance to V4/S1'.  The old S0 result remains immutable.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .df_hamiltonian import DFHamiltonian, PhysicalSector, df_linear_operator
from .df_partial_randomized_pf import (
    df_hamiltonian_hash,
    rank_df_fragments,
    split_df_hamiltonian_by_ld,
)
from .df_partial_s2 import prepare_df_partial_s2
from .df_rte_tail import dense_extracted_df_tail
from .pr2_s0_s1_validation import (
    WEIGHT_RULE,
    _array_hash,
    _dense_df_operator_qiskit,
    _fragment_record,
    _sector_hash,
    _state_hash,
    file_sha256,
    generation_ranked_fragments,
)


SCHEMA_VERSION = "pr2_new_series_v1_v3_result_v1"
SERIES_ID = "pr2-rebaseline-de7a5492-v1"
SPECIFICATION_COMMIT = "30ea857ed4960d3e189bc0877282f11a9846635b"
AMENDMENT_SHA256 = (
    "5522cc2b45d617c9be91cc5715828b1cc63baf5c73d8ea2efd51bd36cfb22b9b"
)
AUTHORIZATION_MANIFEST_SHA256 = (
    "8040a734487dba44c68f763e2d6d9577232f16ec8ad5ba9cc8d768e1f255bc86"
)
DEVELOPMENT_RELATIVE_PATH = (
    "artifacts/pr2_s0_s1_validation/2026-09-28/"
    "h4_1p00_rank12_development_v1.npz"
)
HELD_OUT_RELATIVE_PATH = (
    "artifacts/pr2_s0_s1_validation/2026-09-28/"
    "h4_1p30_rank12_held_out_v1.npz"
)
EXPECTED_DEVELOPMENT_FILE_SHA256 = (
    "3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a"
)
EXPECTED_HELD_OUT_FILE_SHA256 = (
    "ad7e3e7165c55dbaa395eef7a1dd74db89e1f7ab29a69ac64333f4aebf8b3e37"
)
EXPECTED_HAMILTONIAN_HASH = (
    "de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424"
)
RANKS = (3, 6, 9)
HERMITICITY_TOLERANCE = 1e-12
STATE_TOLERANCE = 1e-12
RAYLEIGH_RESIDUAL_TOLERANCE = 1e-9
PROBABILITY_TOLERANCE = 1e-12
RECONSTRUCTION_TOLERANCE = 1e-10
EXPECTED_LAYOUT = {
    "constant": ((), "float64"),
    "one_body": ((8, 8), "complex128"),
    "lambdas": ((12,), "float64"),
    "g_matrices": ((12, 8, 8), "complex128"),
    "sector_basis_indices": ((36,), "int64"),
    "state_vector": ((256,), "complex128"),
    "sector_state_vector": ((36,), "complex128"),
}
EXPECTED_KEYS = frozenset((*EXPECTED_LAYOUT, "metadata_json"))
PASS_STATUS = "S0_PRIME_PASS_V4_REVIEW_REQUIRED"
STATUSES = frozenset(
    {
        PASS_STATUS,
        "STOP_V1_SNAPSHOT_INTEGRITY",
        "STOP_V1_MODEL_VALIDATION",
        "STOP_V2_PARTIAL_STRUCTURE",
    }
)


class SnapshotIntegrityError(ValueError):
    """The frozen bytes or their internally recorded digests are invalid."""


class ModelValidationError(ValueError):
    """The frozen model or state fails a preregistered numerical check."""


class PartialStructureError(ValueError):
    """A preregistered partition, tail, or reconstruction check failed."""


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _new_counters() -> dict[str, int]:
    return {
        "input_files_read": 0,
        "snapshot_loads": 0,
        "development_snapshot_loads": 0,
        "held_out_raw_hash_checks": 0,
        "held_out_npz_loads": 0,
        "molecular_calculations": 0,
        "operator_reconstructions": 0,
        "signal_evaluations": 0,
        "wrapper_probe_trajectories": 0,
        "candidate_trajectories": 0,
        "circuits_compiled": 0,
        "quantum_shots": 0,
    }


def _layout_record(payload: Any) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "shape": list(np.asarray(payload[key]).shape),
            "dtype": str(np.asarray(payload[key]).dtype),
        }
        for key in EXPECTED_LAYOUT
    }


def _load_snapshot_once(
    path: Path,
) -> tuple[
    DFHamiltonian,
    PhysicalSector,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
    dict[str, dict[str, Any]],
]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            keys = frozenset(payload.files)
            if keys != EXPECTED_KEYS:
                raise SnapshotIntegrityError(
                    f"Snapshot keys differ: observed={sorted(keys)}"
                )
            layout = _layout_record(payload)
            for key, (shape, dtype) in EXPECTED_LAYOUT.items():
                if tuple(layout[key]["shape"]) != shape:
                    raise SnapshotIntegrityError(
                        f"Snapshot shape mismatch for {key}: {layout[key]['shape']}"
                    )
                if layout[key]["dtype"] != dtype:
                    raise SnapshotIntegrityError(
                        f"Snapshot dtype mismatch for {key}: {layout[key]['dtype']}"
                    )
            metadata = json.loads(str(payload["metadata_json"].item()))
            constant = float(payload["constant"])
            one_body = np.array(payload["one_body"], dtype=np.complex128, copy=True)
            lambdas = np.array(payload["lambdas"], dtype=np.float64, copy=True)
            g_stack = np.array(payload["g_matrices"], dtype=np.complex128, copy=True)
            basis_indices = np.array(
                payload["sector_basis_indices"], dtype=np.int64, copy=True
            )
            state = np.array(payload["state_vector"], dtype=np.complex128, copy=True)
            sector_state = np.array(
                payload["sector_state_vector"], dtype=np.complex128, copy=True
            )
    except SnapshotIntegrityError:
        raise
    except Exception as exc:
        raise SnapshotIntegrityError(f"Snapshot load failed: {exc}") from exc

    try:
        hamiltonian = DFHamiltonian(
            constant=constant,
            one_body=one_body,
            lambdas=lambdas,
            g_matrices=tuple(np.array(item, copy=True) for item in g_stack),
            metadata=dict(metadata["hamiltonian_metadata"]),
        )
        sector_data = metadata["sector"]
        sector = PhysicalSector(
            n_qubits=int(sector_data["n_qubits"]),
            basis_indices=basis_indices,
            n_electrons=sector_data["n_electrons"],
            nelec_alpha=sector_data["nelec_alpha"],
            nelec_beta=sector_data["nelec_beta"],
            sz_value=sector_data["sz_value"],
        )
        if constant.hex() != metadata["constant_hex"]:
            raise SnapshotIntegrityError("Snapshot constant digest mismatch.")
        if _array_hash(one_body) != metadata["one_body_hash"]:
            raise SnapshotIntegrityError("Snapshot one-body digest mismatch.")
        if _array_hash(lambdas) != metadata["lambdas_hash"]:
            raise SnapshotIntegrityError("Snapshot lambda digest mismatch.")
        if [_array_hash(item) for item in hamiltonian.g_matrices] != metadata[
            "g_matrix_hashes"
        ]:
            raise SnapshotIntegrityError("Snapshot DF-fragment digest mismatch.")
        if df_hamiltonian_hash(hamiltonian) != metadata["hamiltonian_hash"]:
            raise SnapshotIntegrityError("Snapshot Hamiltonian digest mismatch.")
        if _sector_hash(sector) != metadata["sector_hash"]:
            raise SnapshotIntegrityError("Snapshot sector digest mismatch.")
        if _array_hash(np.asarray(basis_indices, dtype="<i8")) != metadata["sector"][
            "basis_indices_hash"
        ]:
            raise SnapshotIntegrityError("Snapshot sector-basis digest mismatch.")
        if _array_hash(state) != metadata["state_vector_hash"]:
            raise SnapshotIntegrityError("Snapshot state-vector digest mismatch.")
        if _array_hash(sector_state) != metadata["sector_state_vector_hash"]:
            raise SnapshotIntegrityError("Snapshot sector-state digest mismatch.")
        if _state_hash(state, sector_state) != metadata["state_hash"]:
            raise SnapshotIntegrityError("Snapshot combined state digest mismatch.")
    except SnapshotIntegrityError:
        raise
    except Exception as exc:
        raise SnapshotIntegrityError(f"Snapshot metadata is invalid: {exc}") from exc
    return hamiltonian, sector, state, sector_state, metadata, layout


def _digest_record(
    hamiltonian: DFHamiltonian,
    sector: PhysicalSector,
    state: np.ndarray,
    sector_state: np.ndarray,
) -> dict[str, Any]:
    return {
        "hamiltonian_hash": df_hamiltonian_hash(hamiltonian),
        "constant_hex": float(hamiltonian.constant).hex(),
        "one_body_hash": _array_hash(hamiltonian.one_body),
        "lambdas_hash": _array_hash(hamiltonian.lambdas),
        "g_matrix_hashes": [_array_hash(item) for item in hamiltonian.g_matrices],
        "sector_hash": _sector_hash(sector),
        "state_hash": _state_hash(state, sector_state),
        "state_vector_hash": _array_hash(state),
        "sector_state_vector_hash": _array_hash(sector_state),
    }


def _relative_hermiticity_residual(array: np.ndarray) -> float:
    matrix = np.asarray(array, dtype=np.complex128)
    return float(
        np.linalg.norm(matrix - matrix.conj().T, ord="fro")
        / max(1.0, float(np.linalg.norm(matrix, ord="fro")))
    )


def _require_model(condition: bool, message: str) -> None:
    if not condition:
        raise ModelValidationError(message)


def run_v1(
    development_path: Path,
    held_out_path: Path,
    counters: dict[str, int],
) -> tuple[dict[str, Any], DFHamiltonian]:
    counters["input_files_read"] += 1
    observed_development_sha = file_sha256(development_path)
    if observed_development_sha != EXPECTED_DEVELOPMENT_FILE_SHA256:
        raise SnapshotIntegrityError(
            "Development raw file SHA-256 differs from the frozen authorization."
        )

    counters["input_files_read"] += 1
    counters["held_out_raw_hash_checks"] += 1
    observed_held_out_sha = file_sha256(held_out_path)
    if observed_held_out_sha != EXPECTED_HELD_OUT_FILE_SHA256:
        raise SnapshotIntegrityError(
            "Held-out raw file SHA-256 differs from the frozen authorization."
        )

    loads = []
    for _ in range(2):
        counters["input_files_read"] += 1
        counters["snapshot_loads"] += 1
        counters["development_snapshot_loads"] += 1
        loads.append(_load_snapshot_once(development_path))
    first, second = loads
    hamiltonian, sector, state, sector_state, metadata, layout = first
    digest_first = _digest_record(hamiltonian, sector, state, sector_state)
    digest_second = _digest_record(second[0], second[1], second[2], second[3])
    if digest_first != digest_second:
        raise SnapshotIntegrityError("Two independent snapshot loads differ.")
    if digest_first["hamiltonian_hash"] != EXPECTED_HAMILTONIAN_HASH:
        raise SnapshotIntegrityError("Development Hamiltonian hash differs.")

    arrays = [
        np.asarray(hamiltonian.constant),
        hamiltonian.one_body,
        hamiltonian.lambdas,
        *hamiltonian.g_matrices,
        sector.basis_indices,
        state,
        sector_state,
    ]
    finite = all(
        bool(np.all(np.isfinite(np.asarray(array).real)))
        and bool(np.all(np.isfinite(np.asarray(array).imag)))
        for array in arrays
    )
    _require_model(finite, "Development snapshot contains non-finite values.")

    one_body_residual = _relative_hermiticity_residual(hamiltonian.one_body)
    g_residuals = [
        _relative_hermiticity_residual(item) for item in hamiltonian.g_matrices
    ]
    _require_model(
        one_body_residual <= HERMITICITY_TOLERANCE,
        "One-body Hermiticity residual exceeds tolerance.",
    )
    _require_model(
        max(g_residuals, default=0.0) <= HERMITICITY_TOLERANCE,
        "A DF-fragment Hermiticity residual exceeds tolerance.",
    )

    full_norm_error = abs(float(np.linalg.norm(state)) - 1.0)
    sector_norm_error = abs(float(np.linalg.norm(sector_state)) - 1.0)
    _require_model(full_norm_error <= STATE_TOLERANCE, "Full state is not normalized.")
    _require_model(
        sector_norm_error <= STATE_TOLERANCE, "Sector state is not normalized."
    )
    outside = np.ones(state.shape[0], dtype=bool)
    outside[np.asarray(sector.basis_indices, dtype=np.int64)] = False
    outside_max = float(np.max(np.abs(state[outside]))) if np.any(outside) else 0.0
    sector_max_difference = float(
        np.max(np.abs(state[sector.basis_indices] - sector_state))
    )
    _require_model(
        outside_max <= STATE_TOLERANCE,
        "Full state has amplitude outside the frozen sector.",
    )
    _require_model(
        sector_max_difference <= STATE_TOLERANCE,
        "Full and sector state vectors are inconsistent.",
    )

    expected_metadata = {
        "model": "linear_H4",
        "distance_angstrom": 1.0,
        "basis": "sto-3g",
        "reference_rank": 12,
    }
    for key, expected in expected_metadata.items():
        _require_model(metadata.get(key) == expected, f"Metadata mismatch for {key}.")
    expected_sector = {
        "n_qubits": 8,
        "n_electrons": 4,
        "nelec_alpha": 2,
        "nelec_beta": 2,
        "sz_value": 0.0,
    }
    for key, expected in expected_sector.items():
        _require_model(
            metadata["sector"].get(key) == expected,
            f"Sector metadata mismatch for {key}.",
        )
    _require_model(
        hamiltonian.metadata.get("df_rank_actual") == 12,
        "Hamiltonian metadata rank is not 12.",
    )

    linear_operator, matvec_counter = df_linear_operator(
        hamiltonian,
        sector,
        backend="python",
    )
    h_state = linear_operator @ sector_state
    counters["operator_reconstructions"] += 1
    rayleigh_energy_complex = complex(np.vdot(sector_state, h_state))
    _require_model(
        abs(rayleigh_energy_complex.imag) <= 1e-12,
        "Rayleigh energy has a non-negligible imaginary part.",
    )
    rayleigh_energy = float(rayleigh_energy_complex.real)
    rayleigh_residual = float(
        np.linalg.norm(h_state - rayleigh_energy * sector_state)
    )
    _require_model(
        rayleigh_residual <= RAYLEIGH_RESIDUAL_TOLERANCE,
        "Rayleigh residual exceeds the frozen tolerance.",
    )

    return (
        {
            "status": "V1_PASS",
            "development": {
                "path": DEVELOPMENT_RELATIVE_PATH,
                "raw_file_sha256": observed_development_sha,
                "layout": layout,
                "digests": digest_first,
                "two_load_digest_match": True,
                "all_values_finite": finite,
                "one_body_hermiticity_relative_frobenius": one_body_residual,
                "g_matrix_hermiticity_relative_frobenius": g_residuals,
                "max_g_matrix_hermiticity_relative_frobenius": max(
                    g_residuals, default=0.0
                ),
                "full_state_norm_error": full_norm_error,
                "sector_state_norm_error": sector_norm_error,
                "sector_outside_max_abs": outside_max,
                "sector_state_max_abs_difference": sector_max_difference,
                "rayleigh_energy_hartree": rayleigh_energy,
                "rayleigh_residual_l2": rayleigh_residual,
                "rayleigh_matvec_count": int(matvec_counter["count"]),
                "sector_claim": "N=4,N_alpha=2,N_beta=2,S_z=0",
                "singlet_proven": False,
            },
            "held_out": {
                "path": HELD_OUT_RELATIVE_PATH,
                "raw_file_sha256": observed_held_out_sha,
                "npz_loaded": False,
                "signal_cost_ranking_evaluated": False,
            },
            "thresholds": {
                "hermiticity_relative_frobenius": HERMITICITY_TOLERANCE,
                "state_and_sector": STATE_TOLERANCE,
                "rayleigh_residual_l2": RAYLEIGH_RESIDUAL_TOLERANCE,
            },
        },
        hamiltonian,
    )


def _require_structure(condition: bool, message: str) -> None:
    if not condition:
        raise PartialStructureError(message)


def _method_record(
    hamiltonian: DFHamiltonian,
    full: np.ndarray,
    full_norm: float,
    rank: int,
    label: str,
    partition: Any,
    counters: dict[str, int],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    policy = (
        "explicit_ordered_partition" if label == "B2-G" else "weight_ranked_prefix"
    )
    preparation = prepare_df_partial_s2(
        hamiltonian,
        partition,
        identity_policy="extract_identity_phase",
        coefficient_atol=0.0,
        partition_policy=policy,
    )
    repeated = prepare_df_partial_s2(
        hamiltonian,
        partition,
        identity_policy="extract_identity_phase",
        coefficient_atol=0.0,
        partition_policy=policy,
    )
    deterministic_indices = tuple(partition.deterministic_block_indices)
    randomized_indices = tuple(partition.randomized_block_indices)
    _require_structure(
        len(deterministic_indices) == rank,
        f"{label} rank {rank}: deterministic count mismatch.",
    )
    _require_structure(
        len(randomized_indices) == hamiltonian.n_blocks - rank,
        f"{label} rank {rank}: randomized count mismatch.",
    )
    _require_structure(
        not set(deterministic_indices).intersection(randomized_indices),
        f"{label} rank {rank}: partition overlap.",
    )
    _require_structure(
        set((*deterministic_indices, *randomized_indices))
        == set(range(hamiltonian.n_blocks)),
        f"{label} rank {rank}: partition is not an exact cover.",
    )

    deterministic = _dense_df_operator_qiskit(
        hamiltonian.select_blocks(deterministic_indices)
    )
    residual = dense_extracted_df_tail(
        preparation.tail_extraction,
        max_dense_qubits=8,
    )
    counters["operator_reconstructions"] += 2
    exact_residual = full - deterministic
    relative_error = float(
        np.linalg.norm(residual - exact_residual, ord=2) / max(1.0, full_norm)
    )
    _require_structure(
        relative_error <= RECONSTRUCTION_TOLERANCE,
        f"{label} rank {rank}: H_D + H_R reconstruction failed.",
    )

    exact_components = preparation.tail_extraction.components
    symbolic_components = preparation.rte_preparation.symbolic_tail.components
    _require_structure(
        len(exact_components) == len(symbolic_components),
        f"{label} rank {rank}: component count mismatch.",
    )
    component_rows = []
    all_signs_match = True
    all_probabilities_valid = True
    for exact, symbolic in zip(exact_components, symbolic_components, strict=True):
        expected_sign = 1 if exact.coefficient >= 0.0 else -1
        sign_match = (
            exact.coefficient_sign == expected_sign
            and symbolic.coefficient_sign == expected_sign
        )
        probability_valid = bool(
            math.isfinite(symbolic.probability) and symbolic.probability >= 0.0
        )
        all_signs_match &= sign_match
        all_probabilities_valid &= probability_valid
        component_rows.append(
            {
                "component_id": exact.component_id,
                "df_fragment_id": exact.df_fragment_id,
                "coefficient": float(exact.coefficient),
                "coefficient_abs": float(exact.coefficient_abs),
                "coefficient_sign": int(exact.coefficient_sign),
                "probability": float(symbolic.probability),
                "source_sign_match": sign_match,
                "diagonal_pauli_support": list(exact.diagonal_pauli_support),
                "is_identity": bool(exact.is_identity),
            }
        )
    probability_sum = math.fsum(item.probability for item in symbolic_components)
    probability_error = abs(float(probability_sum) - 1.0)
    _require_structure(
        all_signs_match,
        f"{label} rank {rank}: sampling coefficient sign mismatch.",
    )
    _require_structure(
        all_probabilities_valid,
        f"{label} rank {rank}: invalid component probability.",
    )
    _require_structure(
        probability_error <= PROBABILITY_TOLERANCE,
        f"{label} rank {rank}: component probabilities do not sum to one.",
    )
    repeat_match = bool(
        repeated.partition_hash == preparation.partition_hash
        and repeated.preparation_hash == preparation.preparation_hash
        and repeated.tail_extraction.tail_hash == preparation.tail_extraction.tail_hash
        and float(repeated.extracted_identity_coefficient).hex()
        == float(preparation.extracted_identity_coefficient).hex()
    )
    _require_structure(
        repeat_match,
        f"{label} rank {rank}: repeated preparation differs.",
    )

    return (
        {
            "partition_policy": policy,
            "ordered_indices": list(deterministic_indices),
            "unordered_indices": sorted(deterministic_indices),
            "randomized_indices": list(randomized_indices),
            "exact_cover": True,
            "deterministic_fragments": [
                _fragment_record(hamiltonian, index)
                for index in deterministic_indices
            ],
            "partition_hash": preparation.partition_hash,
            "preparation_hash": preparation.preparation_hash,
            "tail_hash": preparation.tail_extraction.tail_hash,
            "exact_rte_lambda_r": float(preparation.exact_rte_lambda_r),
            "tail_identity_coefficient": float(
                preparation.extracted_identity_coefficient
            ),
            "tail_identity_phase_convention": "exp(-i*t*coefficient)",
            "sampling_components": component_rows,
            "component_count": len(component_rows),
            "probability_sum": float(probability_sum),
            "probability_sum_error": probability_error,
            "all_sampling_coefficient_signs_match": all_signs_match,
            "all_probabilities_finite_nonnegative": all_probabilities_valid,
            "repeated_preparation_identical": repeat_match,
            "reconstruction_relative_spectral_error": relative_error,
            "reconstruction_pass": True,
        },
        deterministic,
        residual,
    )


def run_v2(
    hamiltonian: DFHamiltonian,
    counters: dict[str, int],
) -> dict[str, Any]:
    full = _dense_df_operator_qiskit(hamiltonian)
    counters["operator_reconstructions"] += 1
    full_norm = float(np.linalg.norm(full, ord=2))
    generation = generation_ranked_fragments(hamiltonian)
    weighted = rank_df_fragments(hamiltonian, weight_rule=WEIGHT_RULE)
    rows = []
    all_ordered_same = True
    all_sets_same = True
    for rank in RANKS:
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
        methods = {}
        dense_parts = {}
        for label in ("B2-G", "B2-W"):
            record, deterministic, residual = _method_record(
                hamiltonian,
                full,
                full_norm,
                rank,
                label,
                partitions[label],
                counters,
            )
            methods[label] = record
            dense_parts[label] = (deterministic, residual)
        ordered_same = bool(
            partitions["B2-G"].deterministic_block_indices
            == partitions["B2-W"].deterministic_block_indices
        )
        sets_same = bool(
            set(partitions["B2-G"].deterministic_block_indices)
            == set(partitions["B2-W"].deterministic_block_indices)
        )
        all_ordered_same &= ordered_same
        all_sets_same &= sets_same
        rows.append(
            {
                "rank": rank,
                "methods": methods,
                "ordered_indices_identical": ordered_same,
                "unordered_sets_identical": sets_same,
                "hd_difference_spectral_norm": float(
                    np.linalg.norm(
                        dense_parts["B2-G"][0] - dense_parts["B2-W"][0],
                        ord=2,
                    )
                ),
                "hr_difference_spectral_norm": float(
                    np.linalg.norm(
                        dense_parts["B2-G"][1] - dense_parts["B2-W"][1],
                        ord=2,
                    )
                ),
            }
        )
    return {
        "status": "V2_PASS",
        "ranks": rows,
        "all_ordered_indices_identical": all_ordered_same,
        "all_unordered_sets_identical": all_sets_same,
        "collapse_B2_G_and_B2_W": all_ordered_same,
        "method_difference_is_failure": False,
        "resource_comparison_performed": False,
        "thresholds": {
            "probability_sum_error": PROBABILITY_TOLERANCE,
            "reconstruction_relative_spectral": RECONSTRUCTION_TOLERANCE,
        },
    }


def run_v1_v3(
    root: Path,
    *,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    counters = _new_counters()
    stages: dict[str, Any] = {
        "V0": {
            "status": "OLD_INPUT_UNRECOVERABLE_USE_SEPARATE_NEW_SERIES",
            "numerical_execution": False,
        }
    }
    status = PASS_STATUS
    failure: dict[str, Any] | None = None
    try:
        v1, hamiltonian = run_v1(
            root / DEVELOPMENT_RELATIVE_PATH,
            root / HELD_OUT_RELATIVE_PATH,
            counters,
        )
        stages["V1"] = v1
        stages["V2"] = run_v2(hamiltonian, counters)
    except SnapshotIntegrityError as exc:
        status = "STOP_V1_SNAPSHOT_INTEGRITY"
        failure = {"type": type(exc).__name__, "message": str(exc)}
    except ModelValidationError as exc:
        status = "STOP_V1_MODEL_VALIDATION"
        failure = {"type": type(exc).__name__, "message": str(exc)}
    except PartialStructureError as exc:
        status = "STOP_V2_PARTIAL_STRUCTURE"
        failure = {"type": type(exc).__name__, "message": str(exc)}

    stages["V3"] = {
        "status": "V3_PASS",
        "dedicated_test_log": provenance.get("dedicated_test_log"),
        "old_s0_module_modified": False,
        "v4_guard_present": True,
        "counter_schema": list(counters),
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "series_id": SERIES_ID,
        "status": status,
        "specification_commit": SPECIFICATION_COMMIT,
        "amendment_sha256": AMENDMENT_SHA256,
        "authorization_manifest_sha256": AUTHORIZATION_MANIFEST_SHA256,
        "provenance": dict(provenance),
        "old_series": {
            "status": "STOP_INPUT_REPRODUCTION_MISMATCH",
            "s1_authorized": False,
            "superseded": False,
        },
        "stages": stages,
        "failure": failure,
        "counters": counters,
        "deviations": [],
        "V4_authorized": False,
        "S1_prime_authorized": False,
        "automatic_next_stage": None,
        "held_out_signal_cost_ranking_evaluated": False,
        "S2_executed": False,
        "S3_executed": False,
        "resource_winner_determined": False,
    }
    payload["result_fingerprint"] = _fingerprint(payload)
    return payload


def validate_result_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unexpected PR-2 new-series schema version.")
    if payload.get("series_id") != SERIES_ID:
        raise ValueError("Unexpected PR-2 new-series ID.")
    if payload.get("status") not in STATUSES:
        raise ValueError("Unexpected PR-2 new-series status.")
    fingerprint_payload = dict(payload)
    observed_fingerprint = fingerprint_payload.pop("result_fingerprint", None)
    if observed_fingerprint != _fingerprint(fingerprint_payload):
        raise ValueError("PR-2 new-series result fingerprint mismatch.")
    for key in (
        "V4_authorized",
        "S1_prime_authorized",
        "held_out_signal_cost_ranking_evaluated",
        "S2_executed",
        "S3_executed",
        "resource_winner_determined",
    ):
        if payload.get(key) is not False:
            raise ValueError(f"Forbidden authorization or result flag: {key}")
    if payload.get("automatic_next_stage") is not None:
        raise ValueError("automatic_next_stage must remain null.")
    counters = payload.get("counters")
    if not isinstance(counters, Mapping):
        raise ValueError("Missing numerical-operation counters.")
    for key in (
        "held_out_npz_loads",
        "molecular_calculations",
        "signal_evaluations",
        "wrapper_probe_trajectories",
        "candidate_trajectories",
        "circuits_compiled",
        "quantum_shots",
    ):
        if counters.get(key) != 0:
            raise ValueError(f"Forbidden operation counter is nonzero: {key}")
    if payload["status"] == PASS_STATUS:
        if payload["stages"].get("V1", {}).get("status") != "V1_PASS":
            raise ValueError("PASS result lacks V1_PASS.")
        if payload["stages"].get("V2", {}).get("status") != "V2_PASS":
            raise ValueError("PASS result lacks V2_PASS.")
        if payload["stages"].get("V3", {}).get("status") != "V3_PASS":
            raise ValueError("PASS result lacks V3_PASS.")
        expected = {
            "input_files_read": 4,
            "snapshot_loads": 2,
            "development_snapshot_loads": 2,
            "held_out_raw_hash_checks": 1,
        }
        for key, value in expected.items():
            if counters.get(key) != value:
                raise ValueError(f"Unexpected PASS counter {key}.")


def write_json_artifact(payload: Mapping[str, Any], path: Path) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {target}")
    validate_result_payload(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def run_v4(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError(
        "V4/S1' is not authorized by the PR-2 new-series V1--V3 amendment."
    )

