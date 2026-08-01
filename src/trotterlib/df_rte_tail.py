"""Exact DF diagonal-block extraction into signed I/Z/ZZ RTE components."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Literal, Sequence, TypeAlias

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from .df_trotter.model import DFBlock
from .rte import (
    BasisChangeOperation,
    InvolutoryTailTerm,
    NormalizedRTETail,
    TailNormalizationMetadata,
    normalize_involutory_tail,
)

if TYPE_CHECKING:
    from .df_rte_circuit import DFRTECircuitSpec


IdentityPolicy: TypeAlias = Literal[
    "faithful_identity_in_tail",
    "extract_identity_phase",
]


@dataclass(frozen=True)
class DFDiagonalPauliComponent:
    """One exact signed central Pauli with its source basis provenance."""

    component_id: str
    coefficient: float
    coefficient_abs: float
    coefficient_sign: int
    df_fragment_id: str
    basis_id: str
    diagonal_pauli_support: tuple[int, ...]
    basis_change_operations: tuple[BasisChangeOperation, ...]

    @property
    def is_identity(self) -> bool:
        return not self.diagonal_pauli_support


@dataclass(frozen=True)
class DFTailExtraction:
    """Canonical exact expansion of selected squared DF fragments."""

    tail_id: str
    tail_hash: str
    identity_policy: IdentityPolicy
    components: tuple[DFDiagonalPauliComponent, ...]
    identity_coefficient: float
    deterministic_identity_coefficient: float
    rte_lambda_r: float
    ranking_proxy_lambda_r: float | None
    normalization_metadata: TailNormalizationMetadata
    num_system_qubits: int
    basis_unitaries: tuple[tuple[str, np.ndarray], ...] = field(
        repr=False, compare=False
    )

    def basis_unitary(self, basis_id: str) -> np.ndarray:
        for candidate_id, unitary in self.basis_unitaries:
            if candidate_id == basis_id:
                return unitary
        raise KeyError(f"Unknown basis_id: {basis_id}")


def _matrix_sha256(matrix: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(np.asarray(matrix, dtype=np.complex128))
    return hashlib.sha256(contiguous.view(np.uint8).tobytes()).hexdigest()


def describe_basis_change_operations(
    u_ops: Sequence[tuple[object, tuple[int, ...]]],
) -> tuple[BasisChangeOperation, ...]:
    """Convert established DF ``U_ops`` to stable, serializable metadata."""
    described: list[BasisChangeOperation] = []
    for gate, qubits in u_ops:
        parameters = tuple(str(parameter) for parameter in getattr(gate, "params", ()))
        try:
            matrix_hash = _matrix_sha256(Operator(gate).data)
        except Exception:
            matrix_hash = None
        described.append(
            BasisChangeOperation(
                name=str(getattr(gate, "name", type(gate).__name__)),
                qubits=tuple(int(qubit) for qubit in qubits),
                parameters=parameters,
                matrix_sha256=matrix_hash,
            )
        )
    return tuple(described)


def basis_change_unitary(block: DFBlock) -> np.ndarray:
    """Materialize the forward ``U_ops`` unitary used by ``apply_df_block``."""
    num_qubits = len(np.asarray(block.eta))
    circuit = QuantumCircuit(num_qubits)
    for gate, qubits in block.U_ops:
        circuit.append(gate, list(qubits))
    return np.asarray(Operator(circuit).data, dtype=np.complex128)


def diagonal_pauli_matrix(
    num_qubits: int,
    support: Sequence[int],
) -> np.ndarray:
    """Return little-endian Qiskit I/Z/ZZ matrix for ``support``."""
    support_tuple = tuple(sorted(set(int(qubit) for qubit in support)))
    if len(support_tuple) != len(tuple(support)) or len(support_tuple) > 2:
        raise ValueError("support must contain zero, one, or two unique qubits.")
    if any(qubit < 0 or qubit >= num_qubits for qubit in support_tuple):
        raise ValueError("support qubit is outside the system register.")
    dimension = 1 << int(num_qubits)
    diagonal = np.ones(dimension, dtype=np.complex128)
    for basis_state in range(dimension):
        parity = sum((basis_state >> qubit) & 1 for qubit in support_tuple) % 2
        if parity:
            diagonal[basis_state] = -1.0
    return np.diag(diagonal)


def exact_df_diagonal_coefficients(
    eta: Sequence[float],
    lam: float,
) -> tuple[tuple[tuple[int, ...], float], ...]:
    """Expand ``lam * (sum_k eta_k n_k)^2`` exactly into I, Z, and ZZ."""
    eta_array = np.asarray(eta)
    eta_real = np.real_if_close(eta_array, tol=1000)
    if np.iscomplexobj(eta_real):
        raise ValueError("DF eta values must be real.")
    eta_real = np.asarray(eta_real, dtype=float)
    lam_real = float(lam)
    if not math.isfinite(lam_real) or not np.all(np.isfinite(eta_real)):
        raise ValueError("DF diagonal coefficients must be finite.")

    aggregate: dict[tuple[int, ...], float] = {(): 0.0}
    aggregate[()] += 0.5 * lam_real * float(np.dot(eta_real, eta_real))
    eta_sum = float(np.sum(eta_real))
    for qubit, eta_value in enumerate(eta_real):
        aggregate[(qubit,)] = aggregate.get((qubit,), 0.0) - (
            0.5 * lam_real * float(eta_value) * eta_sum
        )
    for left in range(len(eta_real)):
        for right in range(left + 1, len(eta_real)):
            pair = 0.5 * lam_real * float(eta_real[left]) * float(eta_real[right])
            aggregate[()] += pair
            support = (left, right)
            aggregate[support] = aggregate.get(support, 0.0) + pair
    return tuple(sorted(aggregate.items(), key=lambda item: (len(item[0]), item[0])))


def _support_label(support: tuple[int, ...]) -> str:
    return "I" if not support else "".join(f"Z{qubit}" for qubit in support)


def extract_df_diagonal_tail(
    tail_id: str,
    blocks: Sequence[DFBlock],
    *,
    fragment_ids: Sequence[str] | None = None,
    basis_ids: Sequence[str] | None = None,
    identity_policy: IdentityPolicy = "faithful_identity_in_tail",
    coefficient_atol: float = 0.0,
    ranking_proxy_lambda_r: float | None = None,
) -> DFTailExtraction:
    """Extract selected DF blocks without aggregating equal support across bases."""
    if not tail_id or not blocks:
        raise ValueError("tail_id and at least one DF block are required.")
    if identity_policy not in (
        "faithful_identity_in_tail",
        "extract_identity_phase",
    ):
        raise ValueError(f"Unsupported identity policy: {identity_policy}")
    if not math.isfinite(coefficient_atol) or coefficient_atol < 0.0:
        raise ValueError("coefficient_atol must be finite and non-negative.")
    if fragment_ids is None:
        fragment_ids = tuple(f"df-fragment-{index}" for index in range(len(blocks)))
    if basis_ids is None:
        basis_ids = tuple(f"{fragment_id}:basis" for fragment_id in fragment_ids)
    if len(fragment_ids) != len(blocks) or len(basis_ids) != len(blocks):
        raise ValueError("fragment_ids and basis_ids must match the number of blocks.")
    if len(set(fragment_ids)) != len(fragment_ids):
        raise ValueError("fragment_ids must be unique.")
    if len(set(basis_ids)) != len(basis_ids):
        raise ValueError("basis_ids must be unique per DF fragment.")

    pre_threshold: list[DFDiagonalPauliComponent] = []
    basis_unitaries: list[tuple[str, np.ndarray]] = []
    for fragment_id, basis_id, block in zip(
        fragment_ids, basis_ids, blocks, strict=True
    ):
        num_qubits = len(np.asarray(block.eta))
        if num_qubits != len(np.asarray(blocks[0].eta)):
            raise ValueError("All DF blocks must act on the same number of qubits.")
        operations = describe_basis_change_operations(block.U_ops)
        basis_unitaries.append((str(basis_id), basis_change_unitary(block)))
        # Aggregation is intentionally local to this fragment/basis pair.
        for support, coefficient in exact_df_diagonal_coefficients(block.eta, block.lam):
            sign = 1 if coefficient >= 0.0 else -1
            pre_threshold.append(
                DFDiagonalPauliComponent(
                    component_id=f"{fragment_id}:{_support_label(support)}",
                    coefficient=float(coefficient),
                    coefficient_abs=abs(float(coefficient)),
                    coefficient_sign=sign,
                    df_fragment_id=str(fragment_id),
                    basis_id=str(basis_id),
                    diagonal_pauli_support=support,
                    basis_change_operations=operations,
                )
            )

    pre_threshold.sort(
        key=lambda component: (
            component.df_fragment_id,
            len(component.diagonal_pauli_support),
            component.diagonal_pauli_support,
        )
    )
    retained = tuple(
        component
        for component in pre_threshold
        if component.coefficient_abs > coefficient_atol
    )
    dropped = tuple(
        component
        for component in pre_threshold
        if component.coefficient_abs <= coefficient_atol
    )
    identity_coefficient = math.fsum(
        component.coefficient for component in retained if component.is_identity
    )
    if identity_policy == "extract_identity_phase":
        randomized_components = tuple(
            component for component in retained if not component.is_identity
        )
        deterministic_identity = float(identity_coefficient)
    else:
        randomized_components = retained
        deterministic_identity = 0.0
    rte_lambda_r = float(
        math.fsum(component.coefficient_abs for component in randomized_components)
    )
    dropped_l1 = float(math.fsum(component.coefficient_abs for component in dropped))
    policy = "drop_abs_coefficient_lte_atol_then_apply_identity_policy"
    payload = {
        "tail_id": tail_id,
        "pre_threshold_components": [asdict(component) for component in pre_threshold],
        "coefficient_atol": float(coefficient_atol),
        "normalization_policy": policy,
        "identity_policy": identity_policy,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return DFTailExtraction(
        tail_id=tail_id,
        tail_hash=hashlib.sha256(encoded).hexdigest(),
        identity_policy=identity_policy,
        components=randomized_components,
        identity_coefficient=float(identity_coefficient),
        deterministic_identity_coefficient=deterministic_identity,
        rte_lambda_r=rte_lambda_r,
        ranking_proxy_lambda_r=(
            None if ranking_proxy_lambda_r is None else float(ranking_proxy_lambda_r)
        ),
        normalization_metadata=TailNormalizationMetadata(
            coefficient_atol=float(coefficient_atol),
            input_component_count=len(pre_threshold),
            retained_component_count=len(retained),
            dropped_component_count=len(dropped),
            dropped_coefficient_l1=dropped_l1,
            operator_error_bound=dropped_l1,
            normalization_policy=policy,
        ),
        num_system_qubits=len(np.asarray(blocks[0].eta)),
        basis_unitaries=tuple(basis_unitaries),
    )


def extract_df_tail_from_hamiltonian(
    tail_id: str,
    hamiltonian: object,
    block_indices: Sequence[int],
    *,
    identity_policy: IdentityPolicy = "faithful_identity_in_tail",
    coefficient_atol: float = 0.0,
    diagonal_sort: str = "descending_abs",
    ranking_weight_rule: str = "lambda_frobenius_squared",
) -> DFTailExtraction:
    """Build and extract selected native DF fragments from ``DFHamiltonian``."""
    from .df_partial_randomized_pf import df_fragment_weight, df_hamiltonian_to_model
    from .df_trotter.ops import build_df_blocks

    selected = hamiltonian.select_blocks(tuple(int(index) for index in block_indices))
    blocks = build_df_blocks(
        df_hamiltonian_to_model(selected),
        sort=diagonal_sort,
    )
    proxy = math.fsum(
        df_fragment_weight(
            hamiltonian,
            int(index),
            weight_rule=ranking_weight_rule,
        )
        for index in block_indices
    )
    fragment_ids = tuple(f"df-fragment-{int(index)}" for index in block_indices)
    return extract_df_diagonal_tail(
        tail_id,
        blocks,
        fragment_ids=fragment_ids,
        identity_policy=identity_policy,
        coefficient_atol=coefficient_atol,
        ranking_proxy_lambda_r=float(proxy),
    )


def component_dense_operator(
    extraction: DFTailExtraction,
    component: DFDiagonalPauliComponent,
) -> np.ndarray:
    """Return unsigned ``U P U^dagger`` matching established DF circuit order."""
    diagonal = diagonal_pauli_matrix(
        extraction.num_system_qubits,
        component.diagonal_pauli_support,
    )
    basis = extraction.basis_unitary(component.basis_id)
    return basis @ diagonal @ basis.conj().T


def dense_extracted_df_tail(extraction: DFTailExtraction) -> np.ndarray:
    """Reconstruct the retained physical tail, including extracted identity."""
    dimension = 1 << extraction.num_system_qubits
    result = (
        extraction.deterministic_identity_coefficient
        * np.eye(dimension, dtype=np.complex128)
    )
    for component in extraction.components:
        result += component.coefficient * component_dense_operator(
            extraction, component
        )
    return result


def dense_df_block_hamiltonian(block: DFBlock) -> np.ndarray:
    """Direct number-basis Hamiltonian conjugated in established circuit order."""
    eta = np.asarray(np.real_if_close(block.eta), dtype=float)
    dimension = 1 << len(eta)
    diagonal = np.empty(dimension, dtype=np.complex128)
    for basis_state in range(dimension):
        occupation_sum = math.fsum(
            float(eta[qubit]) * ((basis_state >> qubit) & 1)
            for qubit in range(len(eta))
        )
        diagonal[basis_state] = float(block.lam) * occupation_sum**2
    basis = basis_change_unitary(block)
    return basis @ np.diag(diagonal) @ basis.conj().T


def extraction_to_normalized_rte_tail(
    extraction: DFTailExtraction,
) -> NormalizedRTETail:
    """Materialize a small-system extracted DF tail as generic finite-RTE input."""
    terms = tuple(
        InvolutoryTailTerm(
            component_id=component.component_id,
            coefficient=component.coefficient,
            operator=component_dense_operator(extraction, component),
            df_fragment_id=component.df_fragment_id,
            basis_id=component.basis_id,
            diagonal_pauli_support=component.diagonal_pauli_support,
            basis_change_operations=component.basis_change_operations,
        )
        for component in extraction.components
    )
    tail = normalize_involutory_tail(
        extraction.tail_id,
        terms,
        atol=0.0,
    )
    if not math.isclose(tail.lambda_r, extraction.rte_lambda_r, abs_tol=1e-14):
        raise ValueError("Extracted symbolic and dense RTE lambda values disagree.")
    return replace(
        tail,
        tail_hash=extraction.tail_hash,
        normalization_metadata=extraction.normalization_metadata,
    )


def extraction_component_circuit_specs(
    extraction: DFTailExtraction,
) -> tuple[DFRTECircuitSpec, ...]:
    """Translate extracted components to the future builder's typed specs."""
    from .df_rte_circuit import (
        DFRTEComponentCircuitSpec,
        DFRTEIdentityCircuitSpec,
    )

    specs: list[DFRTECircuitSpec] = []
    for component in extraction.components:
        if component.is_identity:
            specs.append(
                DFRTEIdentityCircuitSpec(
                    component_id=component.component_id,
                    coefficient_abs=component.coefficient_abs,
                    coefficient_sign=component.coefficient_sign,
                    num_system_qubits=extraction.num_system_qubits,
                    df_fragment_id=component.df_fragment_id,
                    basis_id=component.basis_id,
                )
            )
        else:
            specs.append(
                DFRTEComponentCircuitSpec(
                    component_id=component.component_id,
                    coefficient_abs=component.coefficient_abs,
                    coefficient_sign=component.coefficient_sign,
                    df_fragment_id=component.df_fragment_id,
                    basis_id=component.basis_id,
                    diagonal_pauli_support=component.diagonal_pauli_support,
                    basis_change_operations=component.basis_change_operations,
                    num_system_qubits=extraction.num_system_qubits,
                )
            )
    return tuple(specs)


def uncontrolled_identity_evolution_operator(
    coefficient: float,
    evolution_time: float,
    system_dimension: int,
) -> np.ndarray:
    """Return the uncontrolled global-phase operator ``exp(-it c) I``."""
    return np.exp(-1j * float(evolution_time) * float(coefficient)) * np.eye(
        int(system_dimension), dtype=np.complex128
    )


def controlled_identity_evolution_operator(
    coefficient: float,
    evolution_time: float,
    system_dimension: int,
) -> np.ndarray:
    """Return ancilla-relative identity evolution, control qubit as MSB."""
    identity = np.eye(int(system_dimension), dtype=np.complex128)
    phase = np.exp(-1j * float(evolution_time) * float(coefficient))
    return np.block(
        [
            [identity, np.zeros_like(identity)],
            [np.zeros_like(identity), phase * identity],
        ]
    )
