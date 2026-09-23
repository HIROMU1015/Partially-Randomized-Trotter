"""WP06-a circuit-structure pilot for representative DF Z/ZZ events.

The pilot compares the established full Gaussian basis construction with a
support-restricted unitary completion, explicit equal-basis fusion, whole-event
versus diagonal-only control, and scalar-phase handling.  It is deliberately a
small-circuit routing decision, not a production circuit-policy replacement or
a final resource estimate.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PhaseGate, RZGate, RZZGate
from qiskit.quantum_info import Operator

from .df_hamiltonian import DFHamiltonian
from .df_partial_s2 import DFPartialS2Preparation
from .df_rte_circuit import DFRTEComponentCircuitSpec
from .df_rte_tail import describe_basis_change_operations
from .df_trotter.decompose import diag_hermitian
from .df_trotter.ops import U_to_qiskit_ops_jw
from .rte import CompilerSettings
from .rte_compiled_cost import transpile_and_measure_cost


SCHEMA_VERSION = "research_direction_structure_pilot_v1"
METHOD = "wp06a_df_support_basis_control_phase_pilot_v1"
METRICS = (
    "rz_count",
    "rz_depth",
    "cx_count",
    "cx_depth",
    "total_depth",
    "circuit_size",
)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def support_restricted_unitary_completion(
    full_unitary: np.ndarray,
    support: Sequence[int],
    *,
    atol: float = 1e-10,
) -> np.ndarray:
    """Complete only the columns needed by a conjugated Z/ZZ operator.

    ``B Z_i B^dagger`` depends only on column ``i`` of ``B`` and
    ``B Z_i Z_j B^dagger`` only on columns ``i,j``.  The remaining columns are
    filled deterministically by twice-reorthogonalized projected standard
    basis vectors.  This changes the Gaussian implementation but not the
    conjugated central Pauli.
    """
    unitary = np.asarray(full_unitary)
    if unitary.ndim != 2 or unitary.shape[0] != unitary.shape[1]:
        raise ValueError("full_unitary must be square.")
    n = unitary.shape[0]
    normalized_support = tuple(int(index) for index in support)
    if len(normalized_support) not in (1, 2):
        raise ValueError("support must contain one or two columns.")
    if len(set(normalized_support)) != len(normalized_support):
        raise ValueError("support columns must be distinct.")
    if any(index < 0 or index >= n for index in normalized_support):
        raise ValueError("support column is outside the unitary.")
    if np.max(np.abs(unitary.conj().T @ unitary - np.eye(n))) > atol:
        raise ValueError("full_unitary is not unitary within tolerance.")

    dtype = np.result_type(unitary.dtype, np.float64)
    completed = np.zeros((n, n), dtype=dtype)
    filled: list[int] = []
    for index in normalized_support:
        completed[:, index] = unitary[:, index]
        filled.append(index)

    for target in range(n):
        if target in normalized_support:
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
            raise ValueError("Could not complete the support-restricted unitary.")

    residual = np.max(
        np.abs(completed.conj().T @ completed - np.eye(n))
    )
    if residual > 10.0 * atol:
        raise ValueError("Completed support-restricted basis is not unitary.")
    return completed


def _append_basis(
    circuit: QuantumCircuit,
    operations: Sequence[tuple[Any, tuple[int, ...]]],
    *,
    inverse: bool,
) -> None:
    ordered = list(operations)
    if inverse:
        ordered.reverse()
    for gate, qubits in ordered:
        circuit.append(gate.inverse() if inverse else gate, list(qubits))


def _append_rotation(
    circuit: QuantumCircuit,
    support: Sequence[int],
    angle: float,
    *,
    controlled: bool,
    ancilla_qubit: int | None,
) -> None:
    gate = RZGate(float(angle)) if len(support) == 1 else RZZGate(float(angle))
    if controlled:
        if ancilla_qubit is None:
            raise ValueError("Controlled rotation requires an ancilla.")
        circuit.append(gate.control(1), [ancilla_qubit, *support])
    else:
        circuit.append(gate, list(support))


def _build_from_operations(
    operations: Sequence[tuple[Any, tuple[int, ...]]],
    *,
    num_system_qubits: int,
    support: Sequence[int],
    angle: float,
    controlled: bool,
    scalar_phase: float,
    control_policy: str,
) -> QuantumCircuit:
    if control_policy not in ("diagonal_only", "whole_event"):
        raise ValueError("Unsupported control policy.")
    if controlled and control_policy == "whole_event":
        base = _build_from_operations(
            operations,
            num_system_qubits=num_system_qubits,
            support=support,
            angle=angle,
            controlled=False,
            scalar_phase=scalar_phase,
            control_policy="diagonal_only",
        )
        circuit = QuantumCircuit(num_system_qubits + 1)
        ancilla = num_system_qubits
        circuit.append(
            base.to_gate().control(1),
            [ancilla, *range(num_system_qubits)],
        )
        return circuit

    circuit = QuantumCircuit(num_system_qubits + int(controlled))
    ancilla = num_system_qubits if controlled else None
    _append_basis(circuit, operations, inverse=True)
    _append_rotation(
        circuit,
        support,
        angle,
        controlled=controlled,
        ancilla_qubit=ancilla,
    )
    _append_basis(circuit, operations, inverse=False)
    if scalar_phase != 0.0:
        if controlled:
            circuit.append(PhaseGate(float(scalar_phase)), [ancilla])
        else:
            circuit.global_phase += float(scalar_phase)
    return circuit


def build_conjugated_pauli_rotation(
    basis_unitary: np.ndarray,
    *,
    support: Sequence[int],
    angle: float,
    controlled: bool,
    scalar_phase: float = 0.0,
    control_policy: str = "diagonal_only",
) -> QuantumCircuit:
    """Build a full-basis or support-completed conjugated Z/ZZ rotation."""
    unitary = np.asarray(basis_unitary)
    operations = tuple(U_to_qiskit_ops_jw(unitary))
    return _build_from_operations(
        operations,
        num_system_qubits=unitary.shape[0],
        support=tuple(int(index) for index in support),
        angle=float(angle),
        controlled=bool(controlled),
        scalar_phase=float(scalar_phase),
        control_policy=control_policy,
    )


def build_conjugated_pauli_sequence(
    basis_unitaries: Sequence[np.ndarray],
    *,
    supports: Sequence[Sequence[int]],
    angles: Sequence[float],
    controlled: bool,
    scalar_phase: float = 0.0,
    fuse_shared_basis: bool,
) -> QuantumCircuit:
    """Build a short sequence with either one shared or per-event bases."""
    if not basis_unitaries or not (
        len(basis_unitaries) == len(supports) == len(angles)
    ):
        raise ValueError("Sequence bases, supports, and angles must align.")
    sizes = {np.asarray(unitary).shape for unitary in basis_unitaries}
    if len(sizes) != 1:
        raise ValueError("Every sequence basis must have the same size.")
    shape = next(iter(sizes))
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("Every sequence basis must be square.")
    n = shape[0]
    circuit = QuantumCircuit(n + int(controlled))
    ancilla = n if controlled else None
    operation_sets = tuple(
        tuple(U_to_qiskit_ops_jw(np.asarray(unitary)))
        for unitary in basis_unitaries
    )
    if fuse_shared_basis:
        reference = np.asarray(basis_unitaries[0])
        if any(
            not np.allclose(reference, np.asarray(unitary), atol=1e-12, rtol=0.0)
            for unitary in basis_unitaries[1:]
        ):
            raise ValueError("Shared-basis fusion requires identical bases.")
        _append_basis(circuit, operation_sets[0], inverse=True)
        for support, angle in zip(supports, angles, strict=True):
            _append_rotation(
                circuit,
                support,
                float(angle),
                controlled=controlled,
                ancilla_qubit=ancilla,
            )
        _append_basis(circuit, operation_sets[0], inverse=False)
    else:
        for operations, support, angle in zip(
            operation_sets, supports, angles, strict=True
        ):
            _append_basis(circuit, operations, inverse=True)
            _append_rotation(
                circuit,
                support,
                float(angle),
                controlled=controlled,
                ancilla_qubit=ancilla,
            )
            _append_basis(circuit, operations, inverse=False)
    if scalar_phase != 0.0:
        if controlled:
            circuit.append(PhaseGate(float(scalar_phase)), [ancilla])
        else:
            circuit.global_phase += float(scalar_phase)
    return circuit


def maximum_operator_difference(
    reference: QuantumCircuit,
    candidate: QuantumCircuit,
    *,
    allow_global_phase: bool,
) -> float:
    """Return the maximum entrywise unitary difference."""
    left = np.asarray(Operator(reference).data)
    right = np.asarray(Operator(candidate).data)
    if left.shape != right.shape:
        raise ValueError("Circuit unitary dimensions differ.")
    if allow_global_phase:
        overlap = np.vdot(right.reshape(-1), left.reshape(-1))
        if abs(overlap) > 0.0:
            right = right * overlap / abs(overlap)
    return float(np.max(np.abs(left - right)))


def _cost_record(
    circuit: QuantumCircuit,
    compiler: CompilerSettings,
) -> dict[str, Any]:
    cost = transpile_and_measure_cost(circuit, compiler)
    return {
        "untranspiled_size": int(circuit.size()),
        "untranspiled_depth": int(circuit.depth() or 0),
        "pretranspile_gate_counts": [list(row) for row in cost.pretranspile_gate_counts],
        "posttranspile_gate_counts": [list(row) for row in cost.posttranspile_gate_counts],
        **{metric: int(getattr(cost, metric)) for metric in METRICS},
    }


def _cost_comparison(
    baseline: QuantumCircuit,
    candidate: QuantumCircuit,
    compiler: CompilerSettings,
) -> dict[str, Any]:
    baseline_cost = _cost_record(baseline, compiler)
    candidate_cost = _cost_record(candidate, compiler)
    changes = {}
    for metric in ("untranspiled_size", "untranspiled_depth", *METRICS):
        denominator = float(baseline_cost[metric])
        changes[metric] = (
            None
            if denominator == 0.0
            else float(candidate_cost[metric]) / denominator - 1.0
        )
    return {
        "baseline": baseline_cost,
        "candidate": candidate_cost,
        "candidate_relative_change": changes,
    }


def _fragment_index(spec: DFRTEComponentCircuitSpec) -> int:
    prefix = "df-fragment-"
    if not spec.df_fragment_id.startswith(prefix):
        raise ValueError("Representative component lacks a DF fragment index.")
    return int(spec.df_fragment_id[len(prefix) :])


def _representative_spec(
    preparation: DFPartialS2Preparation,
    support_size: int,
) -> DFRTEComponentCircuitSpec:
    candidates = [
        spec
        for spec in preparation.rte_preparation.component_specs
        if isinstance(spec, DFRTEComponentCircuitSpec)
        and len(spec.diagonal_pauli_support) == support_size
    ]
    if not candidates:
        raise ValueError(
            f"No support-size-{support_size} component is available."
        )
    return max(candidates, key=lambda spec: spec.coefficient_abs)


def _basis_unitary(
    hamiltonian: DFHamiltonian,
    spec: DFRTEComponentCircuitSpec,
    *,
    diagonal_sort: str,
) -> np.ndarray:
    unitary, _eigenvalues = diag_hermitian(
        hamiltonian.g_matrices[_fragment_index(spec)],
        sort=diagonal_sort,
        assume_hermitian=True,
    )
    operations = tuple(U_to_qiskit_ops_jw(unitary))
    if describe_basis_change_operations(operations) != spec.basis_change_operations:
        raise ValueError("Rebuilt Gaussian basis differs from the preparation.")
    return np.asarray(unitary)


def _basis_residuals(
    full: np.ndarray,
    restricted: np.ndarray,
    support: Sequence[int],
) -> dict[str, float]:
    n = full.shape[0]
    return {
        "full_unitarity_max_abs": float(
            np.max(np.abs(full.conj().T @ full - np.eye(n)))
        ),
        "restricted_unitarity_max_abs": float(
            np.max(np.abs(restricted.conj().T @ restricted - np.eye(n)))
        ),
        "preserved_support_columns_max_abs": float(
            np.max(np.abs(full[:, support] - restricted[:, support]))
        ),
    }


def _event_comparison(
    *,
    spec: DFRTEComponentCircuitSpec,
    full_unitary: np.ndarray,
    angle: float,
    scalar_phase: float,
    compiler: CompilerSettings,
) -> dict[str, Any]:
    support = spec.diagonal_pauli_support
    restricted = support_restricted_unitary_completion(full_unitary, support)
    full_ops = tuple(U_to_qiskit_ops_jw(full_unitary))
    restricted_ops = tuple(U_to_qiskit_ops_jw(restricted))
    circuit_rows = {}
    equivalence = {}
    for controlled in (False, True):
        label = "controlled" if controlled else "uncontrolled"
        baseline = _build_from_operations(
            full_ops,
            num_system_qubits=full_unitary.shape[0],
            support=support,
            angle=angle,
            controlled=controlled,
            scalar_phase=scalar_phase,
            control_policy="diagonal_only",
        )
        candidate = _build_from_operations(
            restricted_ops,
            num_system_qubits=full_unitary.shape[0],
            support=support,
            angle=angle,
            controlled=controlled,
            scalar_phase=scalar_phase,
            control_policy="diagonal_only",
        )
        equivalence[label] = maximum_operator_difference(
            baseline,
            candidate,
            allow_global_phase=not controlled,
        )
        circuit_rows[label] = _cost_comparison(
            baseline,
            candidate,
            compiler,
        )
    return {
        "component_id": spec.component_id,
        "coefficient_abs": float(spec.coefficient_abs),
        "df_fragment_id": spec.df_fragment_id,
        "support": list(support),
        "support_size": len(support),
        "full_basis_operation_count": len(full_ops),
        "support_restricted_basis_operation_count": len(restricted_ops),
        "basis_residuals": _basis_residuals(full_unitary, restricted, support),
        "operator_max_abs_difference": equivalence,
        "compiled_cost": circuit_rows,
    }


def _orthogonal_matrix(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q_matrix, r_matrix = np.linalg.qr(rng.normal(size=(size, size)))
    signs = np.where(np.diag(r_matrix) < 0.0, -1.0, 1.0)
    return (q_matrix @ np.diag(signs)).astype(np.complex128)


def evaluate_wp06a_structure_pilot(
    hamiltonian: DFHamiltonian,
    preparation: DFPartialS2Preparation,
    compiler: CompilerSettings,
    *,
    delta_time: float = 0.02,
    eta_decision_relative_rz: float = 0.05,
    equivalence_atol: float = 1e-10,
    size_grid: Sequence[int] = (4, 6, 8),
    size_seed: int = 20260922,
) -> dict[str, Any]:
    """Run the physical representative and structural size diagnostics."""
    if preparation.num_system_qubits != hamiltonian.n_qubits:
        raise ValueError("Preparation and Hamiltonian system sizes differ.")
    if preparation.ld != 3:
        raise ValueError("The preregistered WP06-a physical pilot uses L_D=3.")
    if eta_decision_relative_rz <= 0.0:
        raise ValueError("eta_decision_relative_rz must be positive.")

    scalar_components = {
        "constant_phase": float(-delta_time * preparation.constant_coefficient),
        "extracted_identity_phase": float(
            -delta_time * preparation.extracted_identity_coefficient
        ),
    }
    total_scalar_phase = math.fsum(scalar_components.values())
    dimensionless_time = preparation.exact_rte_lambda_r * delta_time
    rotation_angles = {
        "order_0_rz_or_rzz_angle": float(2.0 * math.atan(dimensionless_time)),
        "order_2_rz_or_rzz_angle": float(
            2.0 * math.atan(dimensionless_time / 3.0)
        ),
    }

    z_spec = _representative_spec(preparation, 1)
    zz_spec = _representative_spec(preparation, 2)
    z_unitary = _basis_unitary(
        hamiltonian,
        z_spec,
        diagonal_sort=preparation.diagonal_sort,
    )
    zz_unitary = _basis_unitary(
        hamiltonian,
        zz_spec,
        diagonal_sort=preparation.diagonal_sort,
    )
    event_results = [
        _event_comparison(
            spec=z_spec,
            full_unitary=z_unitary,
            angle=rotation_angles["order_0_rz_or_rzz_angle"],
            scalar_phase=total_scalar_phase,
            compiler=compiler,
        ),
        _event_comparison(
            spec=zz_spec,
            full_unitary=zz_unitary,
            angle=rotation_angles["order_0_rz_or_rzz_angle"],
            scalar_phase=total_scalar_phase,
            compiler=compiler,
        ),
    ]

    same_basis_specs = sorted(
        (
            spec
            for spec in preparation.rte_preparation.component_specs
            if isinstance(spec, DFRTEComponentCircuitSpec)
            and len(spec.diagonal_pauli_support) == 2
            and spec.basis_id == zz_spec.basis_id
        ),
        key=lambda spec: spec.coefficient_abs,
        reverse=True,
    )
    distinct_support_specs: list[DFRTEComponentCircuitSpec] = []
    seen_supports: set[tuple[int, ...]] = set()
    for spec in same_basis_specs:
        if spec.diagonal_pauli_support in seen_supports:
            continue
        seen_supports.add(spec.diagonal_pauli_support)
        distinct_support_specs.append(spec)
        if len(distinct_support_specs) == 3:
            break
    if len(distinct_support_specs) < 3:
        raise ValueError("WP06-a requires three distinct same-basis ZZ supports.")
    supports = tuple(spec.diagonal_pauli_support for spec in distinct_support_specs)
    angles = (
        rotation_angles["order_0_rz_or_rzz_angle"],
        rotation_angles["order_2_rz_or_rzz_angle"],
        -rotation_angles["order_0_rz_or_rzz_angle"],
    )
    sequence_results = []
    sequence_circuits: dict[int, tuple[QuantumCircuit, QuantumCircuit]] = {}
    for length in (1, 2, 3):
        selected_supports = supports[:length]
        selected_angles = angles[:length]
        baseline = build_conjugated_pauli_sequence(
            (zz_unitary,) * length,
            supports=selected_supports,
            angles=selected_angles,
            controlled=True,
            scalar_phase=total_scalar_phase,
            fuse_shared_basis=True,
        )
        restricted_bases = tuple(
            support_restricted_unitary_completion(zz_unitary, support)
            for support in selected_supports
        )
        candidate = build_conjugated_pauli_sequence(
            restricted_bases,
            supports=selected_supports,
            angles=selected_angles,
            controlled=True,
            scalar_phase=total_scalar_phase,
            fuse_shared_basis=False,
        )
        difference = maximum_operator_difference(
            baseline,
            candidate,
            allow_global_phase=False,
        )
        comparison = _cost_comparison(baseline, candidate, compiler)
        rz_change = comparison["candidate_relative_change"]["rz_count"]
        sequence_results.append(
            {
                "sequence_length": length,
                "component_ids": [
                    spec.component_id for spec in distinct_support_specs[:length]
                ],
                "supports": [list(support) for support in selected_supports],
                "operator_max_abs_difference": difference,
                "compiled_cost": comparison,
                "rz_preferred_structure": (
                    "support_restricted_per_event"
                    if rz_change < 0.0
                    else "full_basis_fused"
                ),
            }
        )
        sequence_circuits[length] = (baseline, candidate)

    unfused_full = build_conjugated_pauli_sequence(
        (zz_unitary, zz_unitary),
        supports=supports[:2],
        angles=angles[:2],
        controlled=True,
        scalar_phase=total_scalar_phase,
        fuse_shared_basis=False,
    )
    fused_full = sequence_circuits[2][0]
    basis_fusion = {
        "sequence_length": 2,
        "operator_max_abs_difference": maximum_operator_difference(
            unfused_full,
            fused_full,
            allow_global_phase=False,
        ),
        "compiled_cost": _cost_comparison(
            unfused_full,
            fused_full,
            compiler,
        ),
        "interpretation": (
            "explicit_fusion_reduces_untranspiled_work;_this_compiler_also_"
            "cancels_the_adjacent_inverse_pair"
        ),
    }

    zz_ops = tuple(U_to_qiskit_ops_jw(zz_unitary))
    whole_control = _build_from_operations(
        zz_ops,
        num_system_qubits=hamiltonian.n_qubits,
        support=zz_spec.diagonal_pauli_support,
        angle=rotation_angles["order_0_rz_or_rzz_angle"],
        controlled=True,
        scalar_phase=total_scalar_phase,
        control_policy="whole_event",
    )
    diagonal_control = _build_from_operations(
        zz_ops,
        num_system_qubits=hamiltonian.n_qubits,
        support=zz_spec.diagonal_pauli_support,
        angle=rotation_angles["order_0_rz_or_rzz_angle"],
        controlled=True,
        scalar_phase=total_scalar_phase,
        control_policy="diagonal_only",
    )
    control_optimization = {
        "operator_max_abs_difference": maximum_operator_difference(
            whole_control,
            diagonal_control,
            allow_global_phase=False,
        ),
        "compiled_cost": _cost_comparison(
            whole_control,
            diagonal_control,
            compiler,
        ),
        "current_builder_policy": "diagonal_only",
    }

    no_scalar = _build_from_operations(
        zz_ops,
        num_system_qubits=hamiltonian.n_qubits,
        support=zz_spec.diagonal_pauli_support,
        angle=rotation_angles["order_0_rz_or_rzz_angle"],
        controlled=True,
        scalar_phase=0.0,
        control_policy="diagonal_only",
    )
    separate_scalar = _build_from_operations(
        zz_ops,
        num_system_qubits=hamiltonian.n_qubits,
        support=zz_spec.diagonal_pauli_support,
        angle=rotation_angles["order_0_rz_or_rzz_angle"],
        controlled=True,
        scalar_phase=0.0,
        control_policy="diagonal_only",
    )
    ancilla = hamiltonian.n_qubits
    for phase in scalar_components.values():
        if phase != 0.0:
            separate_scalar.append(PhaseGate(phase), [ancilla])
    scalar_phase_handling = {
        "components": scalar_components,
        "aggregated_phase": total_scalar_phase,
        "compensated_vs_whole_control_max_abs_difference": (
            maximum_operator_difference(
                whole_control,
                diagonal_control,
                allow_global_phase=False,
            )
        ),
        "omitted_compensation_vs_whole_control_max_abs_difference": (
            maximum_operator_difference(
                whole_control,
                no_scalar,
                allow_global_phase=False,
            )
        ),
        "separate_vs_aggregated_max_abs_difference": maximum_operator_difference(
            separate_scalar,
            diagonal_control,
            allow_global_phase=False,
        ),
        "separate_to_aggregated_cost": _cost_comparison(
            separate_scalar,
            diagonal_control,
            compiler,
        ),
        "controlled_relative_phase_is_required": total_scalar_phase != 0.0,
        "current_builder_preserves_relative_phase": True,
    }

    size_results = []
    for size in size_grid:
        n = int(size)
        if n < 2:
            raise ValueError("The structural size grid requires at least two modes.")
        full = _orthogonal_matrix(n, size_seed + n)
        support = (0, 1)
        restricted = support_restricted_unitary_completion(full, support)
        baseline = build_conjugated_pauli_rotation(
            full,
            support=support,
            angle=rotation_angles["order_0_rz_or_rzz_angle"],
            controlled=True,
        )
        candidate = build_conjugated_pauli_rotation(
            restricted,
            support=support,
            angle=rotation_angles["order_0_rz_or_rzz_angle"],
            controlled=True,
        )
        size_results.append(
            {
                "num_system_qubits": n,
                "synthetic_basis_seed": size_seed + n,
                "full_basis_operation_count": len(U_to_qiskit_ops_jw(full)),
                "support_restricted_basis_operation_count": len(
                    U_to_qiskit_ops_jw(restricted)
                ),
                "operator_max_abs_difference": maximum_operator_difference(
                    baseline,
                    candidate,
                    allow_global_phase=False,
                ),
                "compiled_cost": _cost_comparison(
                    baseline,
                    candidate,
                    compiler,
                ),
            }
        )

    equivalence_values = [
        value
        for event in event_results
        for value in event["operator_max_abs_difference"].values()
    ]
    equivalence_values.extend(
        row["operator_max_abs_difference"] for row in sequence_results
    )
    equivalence_values.extend(
        row["operator_max_abs_difference"] for row in size_results
    )
    equivalence_values.extend(
        (
            basis_fusion["operator_max_abs_difference"],
            control_optimization["operator_max_abs_difference"],
            scalar_phase_handling[
                "compensated_vs_whole_control_max_abs_difference"
            ],
            scalar_phase_handling["separate_vs_aggregated_max_abs_difference"],
        )
    )
    physical_rz_changes = [
        event["compiled_cost"]["controlled"]["candidate_relative_change"][
            "rz_count"
        ]
        for event in event_results
    ]
    physical_rz_changes.extend(
        row["compiled_cost"]["candidate_relative_change"]["rz_count"]
        for row in sequence_results
    )
    eta_trigger = any(
        abs(float(value)) >= eta_decision_relative_rz
        for value in physical_rz_changes
    )
    sequence_preferences = [
        row["rz_preferred_structure"] for row in sequence_results
    ]
    sequence_dependent = len(set(sequence_preferences)) > 1
    maximum_equivalence_residual = max(float(value) for value in equivalence_values)
    omitted_phase_difference = float(
        scalar_phase_handling[
            "omitted_compensation_vs_whole_control_max_abs_difference"
        ]
    )

    checks = {
        "all_required_equivalences_pass": maximum_equivalence_residual
        <= equivalence_atol,
        "all_support_completions_preserve_columns": all(
            event["basis_residuals"]["preserved_support_columns_max_abs"]
            <= equivalence_atol
            for event in event_results
        ),
        "controlled_relative_phase_omission_detected": omitted_phase_difference
        > equivalence_atol,
        "current_relative_phase_compensation_passes": scalar_phase_handling[
            "compensated_vs_whole_control_max_abs_difference"
        ]
        <= equivalence_atol,
        "current_diagonal_only_control_passes": control_optimization[
            "operator_max_abs_difference"
        ]
        <= equivalence_atol,
        "eta_decision_rule_evaluated": isinstance(eta_trigger, bool),
        "sequence_structure_ranking_evaluated": len(sequence_results) == 3,
        "final_total_cost_evaluation_not_claimed": True,
    }
    decision = {
        "eta_decision_relative_rz": float(eta_decision_relative_rz),
        "eta_trigger_fired": eta_trigger,
        "physical_controlled_rz_relative_changes": physical_rz_changes,
        "support_strategy_is_sequence_dependent": sequence_dependent,
        "algorithm_candidate_ranking_reversal_evaluated": False,
        "rpe_q_slope_change_evaluated": False,
        "controlled_relative_phase_missing_from_current_structure": False,
        "current_proxy_recalibration_required_before_wp05": eta_trigger,
        "universal_support_restricted_replacement_adopted": False,
        "current_control_and_phase_policies_retained": True,
        "next_action": (
            "WP06b_sequence_aware_full_vs_support_basis_policy_and_recalibration"
            if eta_trigger
            else "WP05_full_controlled_interrogation_connection"
        ),
        "wp05_status": (
            "wait_for_triggered_structure_integration_and_recalibration"
            if eta_trigger
            else "ready"
        ),
        "direction_update": {
            "T4": "continue_high_priority",
            "T6": "continue_triggered_focused_followup",
        },
        "reason": (
            "Support-restricted bases materially change representative RZ cost, "
            "but the preferred basis strategy changes with short-sequence length; "
            "a sequence-aware policy must precede WP05."
        ),
    }

    return {
        "configuration": {
            "molecule": "H4_chain",
            "geometry_angstrom": 1.0,
            "basis": "STO-3G",
            "n_qubits": hamiltonian.n_qubits,
            "df_rank": len(hamiltonian.lambdas),
            "ld": preparation.ld,
            "delta_time": float(delta_time),
            "eta_decision_relative_rz": float(eta_decision_relative_rz),
            "equivalence_atol": float(equivalence_atol),
            "size_grid": [int(value) for value in size_grid],
            "size_seed": int(size_seed),
            "compiler": {
                "basis_gates": list(compiler.basis_gates),
                "backend_name": compiler.backend_name,
                "coupling_map": compiler.coupling_map,
                "optimization_level": compiler.optimization_level,
                "layout_method": compiler.layout_method,
                "routing_method": compiler.routing_method,
                "transpiler_seed": compiler.transpiler_seed,
                "qiskit_version": compiler.qiskit_version,
            },
        },
        "physical_instance": {
            "hamiltonian_hash": preparation.hamiltonian_hash,
            "partition_hash": preparation.partition_hash,
            "preparation_hash": preparation.preparation_hash,
            "tail_hash": preparation.tail_extraction.tail_hash,
            "exact_rte_lambda_r": float(preparation.exact_rte_lambda_r),
            "rotation_angles": rotation_angles,
            "scalar_phase": scalar_phase_handling,
        },
        "representative_events": event_results,
        "short_sequence_comparison": sequence_results,
        "basis_fusion_ablation": basis_fusion,
        "control_optimization_ablation": control_optimization,
        "synthetic_size_diagnostic": size_results,
        "decision": decision,
        "scope": {
            "physical_event_components": 2,
            "physical_short_sequence_lengths": [1, 2, 3],
            "state_preparation_included": False,
            "full_partial_s2_step_included": False,
            "full_hadamard_interrogation_included": False,
            "algorithm_candidate_ranking_evaluated": False,
            "backend_execution_included": False,
            "synthetic_size_grid_is_chemistry_evidence": False,
            "decision_grade": False,
            "final_total_cost_evaluation_performed": False,
        },
        "limitations": [
            "The physical pilot uses one H4 snapshot, L_D=3, two representative components, and one topology-free compiler context.",
            "Support-restricted completion is a validation candidate and is not yet integrated into the production event builder or cost proxy.",
            "The length-1/2/3 sequence comparison is not an expectation over the finite-RTE event distribution.",
            "The synthetic n=4/6/8 diagnostic measures circuit-structure size dependence and is not molecular or system-size chemistry evidence.",
            "No L_D=3 versus L_D=12 ranking, RPE q-slope, full partial-S2, Hadamard wrapper, state preparation, backend, noise, or final total cost is evaluated.",
        ],
        "checks": checks,
        "overall_pass": all(checks.values()),
        "summary": {
            "status": "WP06a_circuit_structure_pilot_complete",
            "maximum_operator_equivalence_residual": maximum_equivalence_residual,
            "eta_trigger_fired": eta_trigger,
            "support_strategy_is_sequence_dependent": sequence_dependent,
            "current_control_and_phase_policies_validated": True,
            "universal_support_restricted_replacement_adopted": False,
            "current_proxy_recalibration_required_before_wp05": eta_trigger,
            "next_action": decision["next_action"],
        },
    }


def finalize_wp06a_artifact(
    body: Mapping[str, Any], *, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "stage": "WP06-a",
        **dict(body),
        "provenance": dict(provenance),
    }
    payload["content_fingerprint"] = fingerprint(payload)
    validate_wp06a_artifact(payload)
    return payload


def validate_wp06a_artifact(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported WP06-a structure-pilot schema.")
    if payload.get("method") != METHOD or payload.get("stage") != "WP06-a":
        raise ValueError("Unsupported WP06-a method or stage.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("WP06-a content_fingerprint mismatch.")
    checks = payload.get("checks", {})
    if payload.get("overall_pass") != (bool(checks) and all(checks.values())):
        raise ValueError("WP06-a overall status does not match its checks.")
    scope = payload.get("scope", {})
    if scope.get("final_total_cost_evaluation_performed") is not False:
        raise ValueError("WP06-a cannot claim a final total-cost evaluation.")
    if scope.get("decision_grade") is not False:
        raise ValueError("WP06-a must remain non-decision-grade.")
    decision = payload.get("decision", {})
    if decision.get("universal_support_restricted_replacement_adopted") is not False:
        raise ValueError("WP06-a cannot adopt a universal support-only policy.")
    if decision.get("eta_trigger_fired") is not True:
        raise ValueError("The recorded WP06-a physical pilot must preserve its trigger.")


def write_wp06a_artifact(payload: Mapping[str, Any], path: str | Path) -> None:
    validate_wp06a_artifact(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
