"""Preregistered P-C geometry tracking and signed-error breakdown validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from pyscf import gto
from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment

from .chemistry_hamiltonian import geo
from .df_hamiltonian import (
    DFHamiltonian,
    PhysicalSector,
    df_hamiltonian_from_integrals,
)
from .df_partial_randomized_pf import split_df_hamiltonian_by_ld
from .df_partial_s2 import (
    QiskitDFPartialS2CircuitBuilder,
    make_df_partial_s2_step_request,
    prepare_df_partial_s2,
)
from .finite_rte_signal_validation import (
    _circuit_operator_in_openfermion_sector,
    _explicit_cutoff_tolerance,
    _normalized_symbolic_tail_in_sector,
    _qiskit_to_openfermion_sector_permutation,
    dense_df_operator_in_sector,
)
from .parallel_validation_executor import atomic_write_json
from .pf_delta_validation import _qpe_spectral_energy_distribution
from .research_direction_full_scope import fingerprint
from .research_direction_geometry_energy_difference_pilot import (
    validate_geometry_energy_difference_pilot_artifact,
)


EXPECTED_SCHEMA = "research_direction_geometry_tracking_breakdown_expected_v1"
RESULT_SCHEMA = "research_direction_geometry_tracking_breakdown_v1"
METHOD = "pc_h4_orbital_fragment_tracking_signed_error_breakdown_v1"
GEOMETRIES = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6)
TRAINING_GEOMETRIES = (0.8, 1.0, 1.2)
BLIND_GEOMETRIES = (0.7, 0.9, 1.1, 1.4, 1.6)
ANCHOR_GEOMETRY = 1.0
LEFT_PATH = (1.0, 0.9, 0.8, 0.7)
RIGHT_PATH = (1.0, 1.1, 1.2, 1.4, 1.6)
FIT_DELTAS = (0.025, 0.05, 0.1)
DELTA_HOLDOUT = 0.2
ALL_DELTAS = (*FIT_DELTAS, DELTA_HOLDOUT)
POLICIES = ("independent", "tracked")
PAIR_SPECS = (
    ("interpolation", 0.9, 1.1),
    ("compression_boundary", 0.7, 0.9),
    ("stretched_region", 1.4, 1.6),
    ("cross_region", 0.9, 1.4),
)
DF_RANK = 12
LD = 3
N_QUBITS = 8
N_ELECTRONS = 4
COEFFICIENT_ATOL = 1.0e-12
NUMERICAL_ATOL = 1.0e-10
QPE_CLUSTER_ENERGY_TOLERANCE = 1.0e-8
SEED = 20260818

GATE_THRESHOLDS = {
    "maximum_operator_reorder_difference": 1.0e-10,
    "maximum_training_ground_energy_absolute_difference_hartree": 1.0e-8,
    "maximum_training_coefficient_relative_difference": 0.01,
    "maximum_delta_holdout_relative_error": 0.10,
    "maximum_blind_coefficient_relative_error": 0.15,
    "minimum_blind_coefficient_pass_count": 4,
    "maximum_pair_prediction_normalized_error": 0.15,
    "minimum_pair_prediction_pass_count": 3,
    "minimum_nontrivial_exact_energy_difference_hartree": 0.02,
    "maximum_nontrivial_cancellation_ratio": 0.50,
    "smooth_coefficient_prediction_relative_error": 0.10,
    "breakdown_coefficient_prediction_relative_error": 0.20,
    "minimum_clear_diagnostic_count": 4,
    "minimum_diagnostic_accuracy": 0.80,
    "minimum_orbital_singular_value": 0.80,
    "minimum_ground_state_overlap": 0.95,
    "minimum_ground_state_gap_hartree": 0.02,
    "minimum_fragment_similarity": 0.90,
    "minimum_changed_blind_prefixes": 2,
    "minimum_tracking_median_error_reduction": 0.10,
    "maximum_pair_error_degradation": 0.05,
}


@dataclass
class GeometryBundle:
    geometry: float
    hamiltonian: DFHamiltonian
    sector: PhysicalSector
    canonical_orbitals: np.ndarray
    pyscf_molecule: Any
    dense_operator: np.ndarray
    eigenvalues: np.ndarray
    ground_state: np.ndarray


def _relative_error(predicted: float, actual: float) -> float:
    return float(abs(float(predicted) - float(actual)) / max(abs(float(actual)), 1.0e-14))


def expected_task_manifest_body() -> dict[str, Any]:
    tasks = [
        {
            "task_id": f"R{int(round(100 * geometry)):03d}_{policy}",
            "geometry_angstrom": geometry,
            "role": (
                "coefficient_training"
                if geometry in TRAINING_GEOMETRIES
                else "blind_holdout"
            ),
            "policy": policy,
            "fit_deltas": list(FIT_DELTAS),
            "delta_holdout": DELTA_HOLDOUT,
            "df_rank": DF_RANK,
            "ld": LD,
        }
        for geometry in GEOMETRIES
        for policy in POLICIES
    ]
    return {
        "schema_version": EXPECTED_SCHEMA,
        "method": METHOD,
        "configuration": {
            "molecule": "H4 linear chain",
            "basis": "STO-3G",
            "n_qubits": N_QUBITS,
            "n_electrons": N_ELECTRONS,
            "df_rank": DF_RANK,
            "ld": LD,
            "product_formula": "second_order_partial_S2_exact_tail_reference",
            "identity_policy": "extract_identity_phase",
            "geometries_angstrom": list(GEOMETRIES),
            "training_geometries_angstrom": list(TRAINING_GEOMETRIES),
            "blind_geometries_angstrom": list(BLIND_GEOMETRIES),
            "anchor_geometry_angstrom": ANCHOR_GEOMETRY,
            "left_tracking_path_angstrom": list(LEFT_PATH),
            "right_tracking_path_angstrom": list(RIGHT_PATH),
            "policies": list(POLICIES),
            "fit_deltas": list(FIT_DELTAS),
            "delta_holdout": DELTA_HOLDOUT,
            "pairs": [
                {"label": label, "left": left, "right": right}
                for label, left, right in PAIR_SPECS
            ],
            "gate_thresholds": dict(GATE_THRESHOLDS),
        },
        "tasks": tasks,
        "task_count": len(tasks),
        "tracking_edges": [
            {"left": left, "right": right}
            for path in (LEFT_PATH, RIGHT_PATH)
            for left, right in zip(path, path[1:])
        ],
        "decision_rules": [
            "advance_pc_tracking_and_breakdown_design",
            "pc_signed_error_remains_smooth_only_require_new_condition",
            "pc_geometry_response_unexplained_do_not_advance",
            "stop_pc_current_h4_family_as_primary",
        ],
        "scope": {
            "potential_energy_surface_claimed": False,
            "force_or_reaction_barrier_claimed": False,
            "cross_system_transfer_claimed": False,
            "rpe_rte_sampling_evaluated": False,
            "circuit_resources_evaluated": False,
            "h12_evaluated": False,
            "final_total_cost_evaluated": False,
            "scientific_superiority_claimed": False,
        },
    }


def finalize_expected_task_manifest(
    body: Mapping[str, Any], *, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {**dict(body), "provenance": dict(provenance)}
    payload["content_fingerprint"] = fingerprint(payload)
    validate_expected_task_manifest(payload)
    return payload


def validate_expected_task_manifest(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != EXPECTED_SCHEMA:
        raise ValueError("Unexpected P-C tracking expected-task schema.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("P-C tracking expected-task fingerprint mismatch.")
    config = payload.get("configuration", {})
    if tuple(config.get("geometries_angstrom", ())) != GEOMETRIES:
        raise ValueError("P-C tracking geometry grid changed.")
    if tuple(config.get("training_geometries_angstrom", ())) != TRAINING_GEOMETRIES:
        raise ValueError("P-C tracking training geometries changed.")
    if tuple(config.get("blind_geometries_angstrom", ())) != BLIND_GEOMETRIES:
        raise ValueError("P-C tracking blind geometries changed.")
    if config.get("gate_thresholds") != GATE_THRESHOLDS:
        raise ValueError("P-C tracking gate thresholds changed.")
    if int(payload.get("task_count", -1)) != len(GEOMETRIES) * len(POLICIES):
        raise ValueError("P-C tracking expected task count changed.")


def _build_geometry_bundle(geometry: float, scratch: Path) -> GeometryBundle:
    atoms, multiplicity, charge = geo(4, geometry)
    molecule = MolecularData(
        atoms,
        "sto-3g",
        multiplicity,
        charge,
        f"pc_tracking_d{int(round(100 * geometry)):03d}",
        filename=str(scratch / f"pc_tracking_d{int(round(100 * geometry)):03d}"),
    )
    molecule = run_pyscf(molecule, run_scf=1, run_fci=0)
    interaction = molecule.get_molecular_hamiltonian()
    hamiltonian = df_hamiltonian_from_integrals(
        constant=float(interaction.constant),
        one_body=np.asarray(interaction.one_body_tensor, dtype=np.complex128),
        two_body=np.asarray(interaction.two_body_tensor, dtype=np.complex128),
        df_rank=DF_RANK,
        metadata={
            "molecule_type": 4,
            "distance": float(geometry),
            "basis": "sto-3g",
            "multiplicity": int(multiplicity),
            "charge": int(charge),
            "hf_energy": float(molecule.hf_energy),
            "df_rank_source": "explicit",
            "tracking_validation": True,
        },
    )
    if hamiltonian.n_qubits != N_QUBITS or hamiltonian.n_blocks != DF_RANK:
        raise ValueError("P-C tracking H4 DF dimensions changed.")
    sector = PhysicalSector.number_sector(
        n_qubits=N_QUBITS,
        n_electrons=N_ELECTRONS,
    )
    dense = dense_df_operator_in_sector(
        hamiltonian,
        sector,
        matrix_free_backend="python",
    )
    eigenvalues, eigenvectors = np.linalg.eigh(dense)
    pyscf_molecule = gto.M(
        atom=atoms,
        basis="sto-3g",
        spin=int(multiplicity) - 1,
        charge=int(charge),
        unit="Angstrom",
        verbose=0,
    )
    return GeometryBundle(
        geometry=float(geometry),
        hamiltonian=hamiltonian,
        sector=sector,
        canonical_orbitals=np.asarray(
            molecule.canonical_orbitals,
            dtype=np.complex128,
        ),
        pyscf_molecule=pyscf_molecule,
        dense_operator=dense,
        eigenvalues=np.asarray(eigenvalues, dtype=float),
        ground_state=np.asarray(eigenvectors[:, 0], dtype=np.complex128),
    )


def build_geometry_bundles(scratch: Path) -> dict[float, GeometryBundle]:
    return {
        geometry: _build_geometry_bundle(geometry, scratch)
        for geometry in GEOMETRIES
    }


def _spin_orbital_alignment(
    left: GeometryBundle,
    right: GeometryBundle,
) -> tuple[np.ndarray, np.ndarray]:
    ao_cross = gto.intor_cross(
        "int1e_ovlp",
        left.pyscf_molecule,
        right.pyscf_molecule,
    )
    mo_cross = (
        left.canonical_orbitals.conj().T
        @ np.asarray(ao_cross, dtype=np.complex128)
        @ right.canonical_orbitals
    )
    u, singular_values, vh = np.linalg.svd(mo_cross)
    spatial = u @ vh
    spin = np.zeros((N_QUBITS, N_QUBITS), dtype=np.complex128)
    spin[np.ix_(range(0, N_QUBITS, 2), range(0, N_QUBITS, 2))] = spatial
    spin[np.ix_(range(1, N_QUBITS, 2), range(1, N_QUBITS, 2))] = spatial
    return spin, np.asarray(singular_values, dtype=float)


def _occupied_modes(index: int, n_qubits: int) -> tuple[int, ...]:
    return tuple(
        mode
        for mode in range(n_qubits)
        if int(index) & (1 << (n_qubits - 1 - mode))
    )


def _fock_sector_transform(
    orbital_alignment: np.ndarray,
    sector: PhysicalSector,
) -> np.ndarray:
    occupations = [
        _occupied_modes(int(index), sector.n_qubits)
        for index in sector.basis_indices
    ]
    transform = np.empty(
        (sector.dimension, sector.dimension),
        dtype=np.complex128,
    )
    for left_index, left_modes in enumerate(occupations):
        for right_index, right_modes in enumerate(occupations):
            transform[left_index, right_index] = np.linalg.det(
                orbital_alignment[np.ix_(left_modes, right_modes)]
            )
    return transform


def _fragment_scores(
    previous_selected: Sequence[np.ndarray],
    current: DFHamiltonian,
    orbital_alignment: np.ndarray,
) -> np.ndarray:
    aligned = [
        orbital_alignment
        @ np.asarray(matrix, dtype=np.complex128)
        @ orbital_alignment.conj().T
        for matrix in current.g_matrices
    ]
    scores = np.empty((LD, current.n_blocks), dtype=float)
    for row, previous in enumerate(previous_selected):
        previous_norm = float(np.linalg.norm(previous, ord="fro"))
        for column, candidate in enumerate(aligned):
            denominator = previous_norm * float(np.linalg.norm(candidate, ord="fro"))
            scores[row, column] = (
                0.0
                if denominator == 0.0
                else float(
                    abs(np.vdot(previous, candidate))
                    / denominator
                )
            )
    return scores


def _reordered_hamiltonian(
    hamiltonian: DFHamiltonian,
    selected_original_indices: Sequence[int],
) -> tuple[DFHamiltonian, tuple[int, ...]]:
    selected = tuple(int(index) for index in selected_original_indices)
    if len(selected) != LD or len(set(selected)) != LD:
        raise ValueError("Tracked P-C prefix must contain three unique fragments.")
    remaining = tuple(
        index for index in range(hamiltonian.n_blocks) if index not in selected
    )
    order = (*selected, *remaining)
    return hamiltonian.select_blocks(order), order


def track_geometry_fragments(
    bundles: Mapping[float, GeometryBundle],
) -> tuple[dict[float, tuple[int, ...]], list[dict[str, Any]]]:
    tracked: dict[float, tuple[int, ...]] = {
        ANCHOR_GEOMETRY: tuple(range(LD))
    }
    diagnostics: list[dict[str, Any]] = []
    for path in (LEFT_PATH, RIGHT_PATH):
        previous_geometry = path[0]
        previous_indices = tracked[previous_geometry]
        previous_selected = [
            np.asarray(
                bundles[previous_geometry].hamiltonian.g_matrices[index],
                dtype=np.complex128,
            )
            for index in previous_indices
        ]
        for current_geometry in path[1:]:
            previous = bundles[previous_geometry]
            current = bundles[current_geometry]
            alignment, singular_values = _spin_orbital_alignment(
                previous,
                current,
            )
            scores = _fragment_scores(
                previous_selected,
                current.hamiltonian,
                alignment,
            )
            row_indices, column_indices = linear_sum_assignment(-scores)
            assignments = {
                int(row): int(column)
                for row, column in zip(row_indices, column_indices)
            }
            selected = tuple(assignments[row] for row in range(LD))
            tracked[current_geometry] = selected
            assigned_scores = [
                float(scores[row, selected[row]]) for row in range(LD)
            ]
            margins = []
            for row in range(LD):
                alternatives = np.delete(scores[row], selected[row])
                margins.append(
                    float(assigned_scores[row] - np.max(alternatives))
                )
            fock_transform = _fock_sector_transform(
                alignment,
                current.sector,
            )
            state_overlap = float(
                abs(
                    np.vdot(
                        previous.ground_state,
                        fock_transform @ current.ground_state,
                    )
                )
            )
            gap = float(current.eigenvalues[1] - current.eigenvalues[0])
            minimum_singular_value = float(np.min(singular_values))
            minimum_fragment_similarity = float(min(assigned_scores))
            predicted_breakdown = bool(
                minimum_singular_value
                < GATE_THRESHOLDS["minimum_orbital_singular_value"]
                or state_overlap
                < GATE_THRESHOLDS["minimum_ground_state_overlap"]
                or gap
                < GATE_THRESHOLDS["minimum_ground_state_gap_hartree"]
                or minimum_fragment_similarity
                < GATE_THRESHOLDS["minimum_fragment_similarity"]
            )
            diagnostics.append(
                {
                    "left_geometry_angstrom": previous_geometry,
                    "right_geometry_angstrom": current_geometry,
                    "spatial_orbital_cross_overlap_singular_values": (
                        singular_values.tolist()
                    ),
                    "minimum_orbital_singular_value": minimum_singular_value,
                    "fock_transform_unitary_defect": float(
                        np.linalg.norm(
                            fock_transform.conj().T @ fock_transform
                            - np.eye(current.sector.dimension),
                            ord=2,
                        )
                    ),
                    "ground_state_overlap": state_overlap,
                    "ground_state_gap_hartree": gap,
                    "assigned_fragment_indices": list(selected),
                    "assigned_fragment_similarities": assigned_scores,
                    "minimum_fragment_similarity": minimum_fragment_similarity,
                    "assignment_margins": margins,
                    "minimum_assignment_margin": float(min(margins)),
                    "independent_prefix_indices": list(range(LD)),
                    "prefix_differs_from_independent": (
                        selected != tuple(range(LD))
                    ),
                    "predicted_breakdown": predicted_breakdown,
                }
            )
            previous_geometry = current_geometry
            previous_indices = selected
            previous_selected = [
                np.asarray(
                    current.hamiltonian.g_matrices[index],
                    dtype=np.complex128,
                )
                for index in previous_indices
            ]
    if set(tracked) != set(GEOMETRIES):
        raise ValueError("P-C tracking did not cover the fixed geometry grid.")
    return tracked, diagnostics


def _fit_coefficient(biases: Mapping[float, float]) -> float:
    numerator = sum(delta**2 * float(biases[delta]) for delta in FIT_DELTAS)
    denominator = sum(delta**4 for delta in FIT_DELTAS)
    return float(numerator / denominator)


def _evaluate_pf_policy(
    bundle: GeometryBundle,
    hamiltonian: DFHamiltonian,
    *,
    policy: str,
    selected_original_indices: Sequence[int],
) -> dict[str, Any]:
    partition = split_df_hamiltonian_by_ld(hamiltonian, LD)
    preparation = prepare_df_partial_s2(
        hamiltonian,
        partition,
        identity_policy="extract_identity_phase",
        coefficient_atol=COEFFICIENT_ATOL,
    )
    permutation = _qiskit_to_openfermion_sector_permutation(bundle.sector)
    normalized_tail = _normalized_symbolic_tail_in_sector(
        preparation,
        bundle.sector,
        permutation,
        max_dense_qubits=N_QUBITS,
    )
    if normalized_tail.leakage_frobenius_norm > NUMERICAL_ATOL:
        raise ValueError("P-C tracked exact tail leaks outside the number sector.")
    builder = QiskitDFPartialS2CircuitBuilder()
    bias_by_delta: dict[float, float] = {}
    weights: dict[float, float] = {}
    maximum_unitary_defect = 0.0
    for delta in ALL_DELTAS:
        request = make_df_partial_s2_step_request(
            preparation,
            step_time=delta,
            rte_steps=1,
            truncation_tolerance=_explicit_cutoff_tolerance(
                preparation.exact_rte_lambda_r * delta,
                0,
            ),
            finite_taylor_order=0,
            seed=SEED,
        )
        parts = builder.build_additive_circuits(request)
        forward = _circuit_operator_in_openfermion_sector(
            parts.forward_deterministic_half,
            bundle.sector,
            permutation,
        )
        reverse = _circuit_operator_in_openfermion_sector(
            parts.reverse_deterministic_half,
            bundle.sector,
            permutation,
        )
        if max(
            forward.leakage_frobenius_norm,
            reverse.leakage_frobenius_norm,
        ) > NUMERICAL_ATOL:
            raise ValueError("P-C tracked deterministic circuit leaks outside sector.")
        exact_tail = expm(
            -1j
            * delta
            * preparation.exact_rte_lambda_r
            * normalized_tail.matrix
        )
        unitary = reverse.matrix @ exact_tail @ forward.matrix
        spectrum = _qpe_spectral_energy_distribution(
            unitary=unitary,
            state=bundle.ground_state,
            target_energy=float(bundle.eigenvalues[0]),
            delta_time=delta,
            cluster_energy_tolerance=QPE_CLUSTER_ENERGY_TOLERANCE,
            numerical_atol=NUMERICAL_ATOL,
        )
        if not spectrum["numerical_consistency_pass"]:
            raise ValueError("P-C tracked QPE spectrum failed numerical checks.")
        bias_by_delta[delta] = float(
            spectrum["dominant_phase_cluster_signed_energy_bias"]
        )
        weights[delta] = float(spectrum["dominant_phase_cluster_weight"])
        maximum_unitary_defect = max(
            maximum_unitary_defect,
            float(spectrum["unitary_defect_spectral_norm"]),
        )
    coefficient = _fit_coefficient(bias_by_delta)
    holdout_actual = bias_by_delta[DELTA_HOLDOUT]
    holdout_predicted = coefficient * DELTA_HOLDOUT**2
    return {
        "geometry_angstrom": bundle.geometry,
        "policy": policy,
        "role": (
            "coefficient_training"
            if bundle.geometry in TRAINING_GEOMETRIES
            else "blind_holdout"
        ),
        "selected_original_fragment_indices": [
            int(index) for index in selected_original_indices
        ],
        "exact_df_rank12_ground_energy_hartree": float(bundle.eigenvalues[0]),
        "ground_state_gap_hartree": float(
            bundle.eigenvalues[1] - bundle.eigenvalues[0]
        ),
        "signed_bias_by_delta_hartree": {
            str(delta): bias_by_delta[delta] for delta in ALL_DELTAS
        },
        "dominant_cluster_weight_by_delta": {
            str(delta): weights[delta] for delta in ALL_DELTAS
        },
        "signed_second_order_coefficient_hartree": coefficient,
        "delta_holdout": {
            "delta": DELTA_HOLDOUT,
            "actual_signed_bias_hartree": holdout_actual,
            "predicted_signed_bias_hartree": holdout_predicted,
            "relative_error": _relative_error(
                holdout_predicted,
                holdout_actual,
            ),
        },
        "maximum_unitary_defect": maximum_unitary_defect,
        "hamiltonian_hash": preparation.hamiltonian_hash,
        "partition_hash": preparation.partition_hash,
        "preparation_hash": preparation.preparation_hash,
    }


def _predict_coefficient(
    geometry: float,
    training: Mapping[float, float],
) -> float:
    ordered = sorted(training)
    if geometry in training:
        return float(training[geometry])
    if geometry < ordered[0]:
        left, right = ordered[0], ordered[1]
    elif geometry > ordered[-1]:
        left, right = ordered[-2], ordered[-1]
    else:
        for left, right in zip(ordered, ordered[1:]):
            if left < geometry < right:
                break
    fraction = (geometry - left) / (right - left)
    return float(training[left] + fraction * (training[right] - training[left]))


def _pair_result(
    rows: Mapping[float, Mapping[str, Any]],
    predictions: Mapping[float, float],
    *,
    label: str,
    left: float,
    right: float,
) -> dict[str, Any]:
    left_row = rows[left]
    right_row = rows[right]
    left_bias = float(
        left_row["signed_bias_by_delta_hartree"][str(DELTA_HOLDOUT)]
    )
    right_bias = float(
        right_row["signed_bias_by_delta_hartree"][str(DELTA_HOLDOUT)]
    )
    actual = right_bias - left_bias
    predicted = DELTA_HOLDOUT**2 * (
        predictions[right] - predictions[left]
    )
    endpoint_max = max(abs(left_bias), abs(right_bias), 1.0e-14)
    endpoint_sum = max(abs(left_bias) + abs(right_bias), 1.0e-14)
    exact_difference = float(
        right_row["exact_df_rank12_ground_energy_hartree"]
        - left_row["exact_df_rank12_ground_energy_hartree"]
    )
    return {
        "label": label,
        "left_geometry_angstrom": left,
        "right_geometry_angstrom": right,
        "delta": DELTA_HOLDOUT,
        "exact_df_rank12_energy_difference_hartree": exact_difference,
        "actual_signed_pf_difference_error_hartree": actual,
        "predicted_signed_pf_difference_error_hartree": predicted,
        "prediction_absolute_error_hartree": abs(predicted - actual),
        "prediction_error_normalized_by_larger_endpoint_bias": float(
            abs(predicted - actual) / endpoint_max
        ),
        "cancellation_ratio": float(abs(actual) / endpoint_sum),
    }


def evaluate_geometry_tracking_breakdown(
    bundles: Mapping[float, GeometryBundle],
    prior_pc_artifact: Mapping[str, Any],
    expected_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    validate_geometry_energy_difference_pilot_artifact(prior_pc_artifact)
    validate_expected_task_manifest(expected_manifest)
    if set(bundles) != set(GEOMETRIES):
        raise ValueError("P-C tracking bundle geometry grid changed.")
    tracked_indices, edge_diagnostics = track_geometry_fragments(bundles)
    diagnostics_by_right = {
        float(row["right_geometry_angstrom"]): row
        for row in edge_diagnostics
    }
    rows: list[dict[str, Any]] = []
    maximum_reorder_difference = 0.0
    for geometry in GEOMETRIES:
        bundle = bundles[geometry]
        independent = bundle.hamiltonian
        tracked_hamiltonian, tracked_order = _reordered_hamiltonian(
            bundle.hamiltonian,
            tracked_indices[geometry],
        )
        tracked_dense = dense_df_operator_in_sector(
            tracked_hamiltonian,
            bundle.sector,
            matrix_free_backend="python",
        )
        reorder_difference = float(
            np.linalg.norm(tracked_dense - bundle.dense_operator, ord=2)
        )
        maximum_reorder_difference = max(
            maximum_reorder_difference,
            reorder_difference,
        )
        independent_row = _evaluate_pf_policy(
            bundle,
            independent,
            policy="independent",
            selected_original_indices=tuple(range(LD)),
        )
        independent_row["full_operator_reorder_difference"] = 0.0
        rows.append(independent_row)
        if tracked_indices[geometry] == tuple(range(LD)):
            tracked_row = json.loads(json.dumps(independent_row))
            tracked_row["policy"] = "tracked"
            tracked_row["reused_identical_independent_policy"] = True
        else:
            tracked_row = _evaluate_pf_policy(
                bundle,
                tracked_hamiltonian,
                policy="tracked",
                selected_original_indices=tracked_indices[geometry],
            )
            tracked_row["reused_identical_independent_policy"] = False
        tracked_row["full_fragment_order"] = list(tracked_order)
        tracked_row["full_operator_reorder_difference"] = reorder_difference
        rows.append(tracked_row)

    by_policy = {
        policy: {
            float(row["geometry_angstrom"]): row
            for row in rows
            if row["policy"] == policy
        }
        for policy in POLICIES
    }
    predictions: dict[str, dict[float, float]] = {}
    blind_results: dict[str, list[dict[str, Any]]] = {}
    pair_results: dict[str, list[dict[str, Any]]] = {}
    for policy in POLICIES:
        training = {
            geometry: float(
                by_policy[policy][geometry][
                    "signed_second_order_coefficient_hartree"
                ]
            )
            for geometry in TRAINING_GEOMETRIES
        }
        policy_predictions = {
            geometry: _predict_coefficient(geometry, training)
            for geometry in GEOMETRIES
        }
        predictions[policy] = policy_predictions
        blind_results[policy] = []
        for geometry in BLIND_GEOMETRIES:
            actual = float(
                by_policy[policy][geometry][
                    "signed_second_order_coefficient_hartree"
                ]
            )
            predicted = policy_predictions[geometry]
            error = _relative_error(predicted, actual)
            blind_results[policy].append(
                {
                    "geometry_angstrom": geometry,
                    "actual_coefficient_hartree": actual,
                    "predicted_coefficient_hartree": predicted,
                    "relative_error": error,
                }
            )
        pair_results[policy] = [
            _pair_result(
                by_policy[policy],
                policy_predictions,
                label=label,
                left=left,
                right=right,
            )
            for label, left, right in PAIR_SPECS
        ]

    tracked_blind = {
        float(row["geometry_angstrom"]): row
        for row in blind_results["tracked"]
    }
    diagnostic_rows = []
    for geometry in BLIND_GEOMETRIES:
        error = float(tracked_blind[geometry]["relative_error"])
        actual_class = (
            "smooth"
            if error
            <= GATE_THRESHOLDS[
                "smooth_coefficient_prediction_relative_error"
            ]
            else "breakdown"
            if error
            >= GATE_THRESHOLDS[
                "breakdown_coefficient_prediction_relative_error"
            ]
            else "inconclusive"
        )
        predicted_breakdown = bool(
            diagnostics_by_right[geometry]["predicted_breakdown"]
        )
        correct = (
            None
            if actual_class == "inconclusive"
            else predicted_breakdown == (actual_class == "breakdown")
        )
        diagnostic_rows.append(
            {
                "geometry_angstrom": geometry,
                "coefficient_prediction_relative_error": error,
                "actual_class": actual_class,
                "predicted_breakdown": predicted_breakdown,
                "diagnostic_correct": correct,
                "incoming_edge": diagnostics_by_right[geometry],
            }
        )

    prior_rows = {
        float(row["geometry_angstrom"]): row
        for row in prior_pc_artifact["geometry_rows"]
    }
    training_energy_differences = {
        str(geometry): abs(
            float(
                by_policy["independent"][geometry][
                    "exact_df_rank12_ground_energy_hartree"
                ]
            )
            - float(
                prior_rows[geometry][
                    "exact_df_rank12_ground_energy_hartree"
                ]
            )
        )
        for geometry in TRAINING_GEOMETRIES
    }
    training_coefficient_differences = {
        str(geometry): _relative_error(
            float(
                by_policy["independent"][geometry][
                    "signed_second_order_coefficient_hartree"
                ]
            ),
            float(
                prior_rows[geometry][
                    "signed_second_order_coefficient_hartree"
                ]
            ),
        )
        for geometry in TRAINING_GEOMETRIES
    }
    maximum_delta_holdout_error = max(
        float(row["delta_holdout"]["relative_error"]) for row in rows
    )
    tracked_blind_pass_count = sum(
        float(row["relative_error"])
        <= GATE_THRESHOLDS["maximum_blind_coefficient_relative_error"]
        for row in blind_results["tracked"]
    )
    tracked_pair_pass_count = sum(
        float(row["prediction_error_normalized_by_larger_endpoint_bias"])
        <= GATE_THRESHOLDS["maximum_pair_prediction_normalized_error"]
        for row in pair_results["tracked"]
    )
    nontrivial_cancellation_count = sum(
        abs(float(row["exact_df_rank12_energy_difference_hartree"]))
        >= GATE_THRESHOLDS[
            "minimum_nontrivial_exact_energy_difference_hartree"
        ]
        and float(row["cancellation_ratio"])
        <= GATE_THRESHOLDS["maximum_nontrivial_cancellation_ratio"]
        for row in pair_results["tracked"]
    )
    clear_diagnostics = [
        row for row in diagnostic_rows if row["diagnostic_correct"] is not None
    ]
    diagnostic_accuracy = (
        0.0
        if not clear_diagnostics
        else float(
            sum(bool(row["diagnostic_correct"]) for row in clear_diagnostics)
            / len(clear_diagnostics)
        )
    )
    changed_blind_prefixes = sum(
        tracked_indices[geometry] != tuple(range(LD))
        for geometry in BLIND_GEOMETRIES
    )
    independent_median = float(
        np.median(
            [row["relative_error"] for row in blind_results["independent"]]
        )
    )
    tracked_median = float(
        np.median([row["relative_error"] for row in blind_results["tracked"]])
    )
    tracking_reduction = float(
        (independent_median - tracked_median)
        / max(independent_median, 1.0e-14)
    )
    maximum_pair_degradation = max(
        float(tracked["prediction_error_normalized_by_larger_endpoint_bias"])
        - float(independent["prediction_error_normalized_by_larger_endpoint_bias"])
        for independent, tracked in zip(
            pair_results["independent"],
            pair_results["tracked"],
        )
    )
    tracking_mechanism_pass = bool(
        changed_blind_prefixes
        >= GATE_THRESHOLDS["minimum_changed_blind_prefixes"]
        and tracking_reduction
        >= GATE_THRESHOLDS["minimum_tracking_median_error_reduction"]
        and maximum_pair_degradation
        <= GATE_THRESHOLDS["maximum_pair_error_degradation"]
    )
    breakdown_rows = [
        row for row in diagnostic_rows if row["actual_class"] == "breakdown"
    ]
    breakdown_mechanism_pass = bool(
        breakdown_rows
        and all(bool(row["predicted_breakdown"]) for row in breakdown_rows)
    )
    gates = {
        "representation_integrity_pass": bool(
            maximum_reorder_difference
            <= GATE_THRESHOLDS["maximum_operator_reorder_difference"]
            and max(training_energy_differences.values())
            <= GATE_THRESHOLDS[
                "maximum_training_ground_energy_absolute_difference_hartree"
            ]
            and max(training_coefficient_differences.values())
            <= GATE_THRESHOLDS[
                "maximum_training_coefficient_relative_difference"
            ]
        ),
        "delta_holdout_pass": bool(
            maximum_delta_holdout_error
            <= GATE_THRESHOLDS["maximum_delta_holdout_relative_error"]
        ),
        "blind_coefficient_prediction_pass": bool(
            tracked_blind_pass_count
            >= GATE_THRESHOLDS["minimum_blind_coefficient_pass_count"]
        ),
        "pair_prediction_pass": bool(
            tracked_pair_pass_count
            >= GATE_THRESHOLDS["minimum_pair_prediction_pass_count"]
        ),
        "nontrivial_cancellation_pass": bool(
            nontrivial_cancellation_count >= 1
        ),
        "diagnostic_transfer_pass": bool(
            len(clear_diagnostics)
            >= GATE_THRESHOLDS["minimum_clear_diagnostic_count"]
            and diagnostic_accuracy
            >= GATE_THRESHOLDS["minimum_diagnostic_accuracy"]
        ),
        "mechanism_discrimination_pass": bool(
            tracking_mechanism_pass or breakdown_mechanism_pass
        ),
    }
    first_five = all(list(gates.values())[:5])
    if all(gates.values()):
        status = "advance_pc_tracking_and_breakdown_design"
    elif first_five and gates["diagnostic_transfer_pass"]:
        status = "pc_signed_error_remains_smooth_only_require_new_condition"
    elif first_five:
        status = "pc_geometry_response_unexplained_do_not_advance"
    else:
        status = "stop_pc_current_h4_family_as_primary"
    return {
        "schema_version": RESULT_SCHEMA,
        "method": METHOD,
        "expected_task_fingerprint": expected_manifest["content_fingerprint"],
        "configuration": expected_manifest["configuration"],
        "geometry_rows": rows,
        "tracked_original_fragment_indices": {
            str(geometry): list(tracked_indices[geometry])
            for geometry in GEOMETRIES
        },
        "tracking_edge_diagnostics": edge_diagnostics,
        "blind_coefficient_predictions": blind_results,
        "pair_predictions": pair_results,
        "diagnostic_classification": diagnostic_rows,
        "gates": gates,
        "overall_pass": all(gates.values()),
        "summary": {
            "maximum_full_operator_reorder_difference": (
                maximum_reorder_difference
            ),
            "training_ground_energy_absolute_differences_hartree": (
                training_energy_differences
            ),
            "training_coefficient_relative_differences": (
                training_coefficient_differences
            ),
            "maximum_delta_holdout_relative_error": (
                maximum_delta_holdout_error
            ),
            "tracked_blind_coefficient_pass_count": tracked_blind_pass_count,
            "tracked_pair_prediction_pass_count": tracked_pair_pass_count,
            "nontrivial_cancellation_count": nontrivial_cancellation_count,
            "clear_diagnostic_count": len(clear_diagnostics),
            "diagnostic_accuracy": diagnostic_accuracy,
            "changed_blind_prefix_count": changed_blind_prefixes,
            "independent_blind_coefficient_error_median": independent_median,
            "tracked_blind_coefficient_error_median": tracked_median,
            "tracking_median_error_reduction": tracking_reduction,
            "maximum_pair_error_degradation": maximum_pair_degradation,
            "actual_breakdown_count": len(breakdown_rows),
            "tracking_mechanism_pass": tracking_mechanism_pass,
            "breakdown_mechanism_pass": breakdown_mechanism_pass,
        },
        "decision": {
            "status": status,
            "thresholds_changed_after_results": False,
            "current_primary_theme": (
                "P-C_geometry_energy_difference"
                if status == "advance_pc_tracking_and_breakdown_design"
                else "none_confirmed"
            ),
            "next_action": (
                "compare_difference_targeted_pf_selection_on_new_blind_pairs"
                if status == "advance_pc_tracking_and_breakdown_design"
                else "preregister_a_distinct_condition_or_stop_pc"
                if status
                == "pc_signed_error_remains_smooth_only_require_new_condition"
                else "do_not_advance_pc_from_this_h4_family"
            ),
        },
        "scope": {
            **expected_manifest["scope"],
            "exact_energy_means_df_rank12_sector_reference": True,
            "signed_bias_convention": "E_PF_minus_E_exact",
            "natural_quantum_measurement_covariance_claimed": False,
        },
        "checks": {
            "expected_manifest_validated": True,
            "fixed_geometry_grid_complete": len(bundles) == len(GEOMETRIES),
            "all_tasks_complete": len(rows)
            == len(GEOMETRIES) * len(POLICIES),
            "all_values_finite": all(
                math.isfinite(
                    float(row["signed_second_order_coefficient_hartree"])
                )
                for row in rows
            ),
        },
    }


def finalize_geometry_tracking_breakdown_artifact(
    body: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
    source_evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = {
        **dict(body),
        "source_evidence": [dict(item) for item in source_evidence],
        "provenance": dict(provenance),
    }
    payload["content_fingerprint"] = fingerprint(payload)
    validate_geometry_tracking_breakdown_artifact(payload)
    return payload


def validate_geometry_tracking_breakdown_artifact(
    payload: Mapping[str, Any],
) -> None:
    if payload.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("Unexpected P-C tracking result schema.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("P-C tracking result fingerprint mismatch.")
    configuration = payload.get("configuration", {})
    if tuple(configuration.get("geometries_angstrom", ())) != GEOMETRIES:
        raise ValueError("P-C tracking result geometry grid changed.")
    if configuration.get("gate_thresholds") != GATE_THRESHOLDS:
        raise ValueError("P-C tracking result thresholds changed.")
    if len(payload.get("geometry_rows", ())) != len(GEOMETRIES) * len(POLICIES):
        raise ValueError("P-C tracking result is incomplete.")
    if payload.get("decision", {}).get("thresholds_changed_after_results") is not False:
        raise ValueError("P-C tracking thresholds changed after results.")
    allowed = {
        "advance_pc_tracking_and_breakdown_design",
        "pc_signed_error_remains_smooth_only_require_new_condition",
        "pc_geometry_response_unexplained_do_not_advance",
        "stop_pc_current_h4_family_as_primary",
    }
    if payload.get("decision", {}).get("status") not in allowed:
        raise ValueError("P-C tracking decision is invalid.")
    scope = payload.get("scope", {})
    forbidden = (
        "potential_energy_surface_claimed",
        "force_or_reaction_barrier_claimed",
        "cross_system_transfer_claimed",
        "rpe_rte_sampling_evaluated",
        "circuit_resources_evaluated",
        "h12_evaluated",
        "final_total_cost_evaluated",
        "scientific_superiority_claimed",
        "natural_quantum_measurement_covariance_claimed",
    )
    if any(scope.get(key) is not False for key in forbidden):
        raise ValueError("P-C tracking artifact overstates scope.")
    if scope.get("signed_bias_convention") != "E_PF_minus_E_exact":
        raise ValueError("P-C tracking signed-bias convention changed.")


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return payload


def write_json_nonoverwriting(payload: Mapping[str, Any], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    atomic_write_json(path, dict(payload))
