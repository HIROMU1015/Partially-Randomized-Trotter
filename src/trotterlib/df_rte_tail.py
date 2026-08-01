"""Symbolic DF diagonal-tail extraction and guarded small-system references."""

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
    DeterministicOnlyRTETailError,
    InvolutoryTailTerm,
    NormalizedRTETail,
    RTEComponent,
    TailNormalizationMetadata,
    normalize_involutory_tail,
    require_integer_count,
)

if TYPE_CHECKING:
    from .df_rte_circuit import DFRTECircuitSpec, DFRTEEventPreparation


IdentityPolicy: TypeAlias = Literal[
    "faithful_identity_in_tail",
    "extract_identity_phase",
]

DEFAULT_MAX_DENSE_QUBITS = 8
_MAX_LOCAL_GATE_HASH_QUBITS = 4


class BasisFingerprintError(ValueError):
    """Raised when a basis operation lacks a safe canonical fingerprint."""


@dataclass(frozen=True)
class BasisChangeMetadata:
    """Serializable canonical metadata for one executable DF basis."""

    basis_id: str
    basis_hash: str
    num_system_qubits: int
    operations: tuple[BasisChangeOperation, ...]


@dataclass(frozen=True)
class DFBasisDefinition:
    """Serializable basis metadata plus runtime Qiskit operations."""

    metadata: BasisChangeMetadata
    runtime_operations: tuple[tuple[object, tuple[int, ...]], ...] = field(
        repr=False,
        compare=False,
    )

    @property
    def basis_id(self) -> str:
        return self.metadata.basis_id

    @property
    def basis_hash(self) -> str:
        return self.metadata.basis_hash

    @property
    def num_system_qubits(self) -> int:
        return self.metadata.num_system_qubits


class DFBasisRegistry:
    """Runtime registry that rejects conflicting definitions for one basis ID."""

    def __init__(self) -> None:
        self._definitions: dict[str, DFBasisDefinition] = {}

    def register(
        self,
        u_ops: Sequence[tuple[object, tuple[int, ...]]],
        *,
        num_system_qubits: int,
        basis_id: str | None = None,
    ) -> DFBasisDefinition:
        num_system_qubits = require_integer_count(
            num_system_qubits,
            name="num_system_qubits",
            minimum=1,
        )
        operations = tuple(
            (
                gate,
                tuple(
                    require_integer_count(qubit, name="basis operation qubit")
                    for qubit in qubits
                ),
            )
            for gate, qubits in u_ops
        )
        if any(
            qubit < 0 or qubit >= num_system_qubits
            for _gate, qubits in operations
            for qubit in qubits
        ):
            raise ValueError("Basis operation qubit is outside the system register.")
        described = describe_basis_change_operations(operations)
        basis_hash = canonical_basis_hash(num_system_qubits, described)
        if basis_id is not None and not str(basis_id):
            raise ValueError("basis_id must not be empty.")
        identifier = str(basis_id) if basis_id is not None else f"df-basis-{basis_hash}"
        metadata = BasisChangeMetadata(
            basis_id=str(identifier),
            basis_hash=basis_hash,
            num_system_qubits=int(num_system_qubits),
            operations=described,
        )
        existing = self._definitions.get(str(identifier))
        if existing is not None:
            if existing.metadata != metadata:
                raise ValueError(
                    f"basis_id {identifier!r} is already registered with a "
                    "different operation sequence."
                )
            return existing
        definition = DFBasisDefinition(
            metadata=metadata,
            runtime_operations=operations,
        )
        self._definitions[str(identifier)] = definition
        return definition

    def definition(self, basis_id: str) -> DFBasisDefinition:
        try:
            return self._definitions[str(basis_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown basis_id: {basis_id}") from exc

    def operations(
        self, basis_id: str
    ) -> tuple[tuple[object, tuple[int, ...]], ...]:
        return self.definition(basis_id).runtime_operations

    def metadata(self) -> tuple[BasisChangeMetadata, ...]:
        return tuple(
            self._definitions[identifier].metadata
            for identifier in sorted(self._definitions)
        )


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
    basis_hash: str | None = None

    @property
    def is_identity(self) -> bool:
        return not self.diagonal_pauli_support


@dataclass(frozen=True)
class DFTailExtractionMetadata:
    """Audit counts separating threshold approximation from identity movement."""

    coefficient_atol: float
    threshold_input_component_count: int
    threshold_retained_component_count: int
    threshold_dropped_component_count: int
    retained_identity_component_count: int
    extracted_identity_component_count: int
    randomized_component_count: int
    threshold_dropped_coefficient_l1: float
    extracted_identity_coefficient: float
    randomized_coefficient_l1: float
    threshold_operator_error_bound: float
    normalization_policy: str

    def __post_init__(self) -> None:
        counts = (
            self.threshold_input_component_count,
            self.threshold_retained_component_count,
            self.threshold_dropped_component_count,
            self.retained_identity_component_count,
            self.extracted_identity_component_count,
            self.randomized_component_count,
        )
        if any(count < 0 for count in counts):
            raise ValueError("DF extraction component counts must be non-negative.")
        if self.threshold_input_component_count != (
            self.threshold_retained_component_count
            + self.threshold_dropped_component_count
        ):
            raise ValueError("Threshold input count must equal retained plus dropped.")
        if (
            self.extracted_identity_component_count
            > self.retained_identity_component_count
        ):
            raise ValueError("Extracted identity count exceeds retained identities.")
        if self.randomized_component_count != (
            self.threshold_retained_component_count
            - self.extracted_identity_component_count
        ):
            raise ValueError(
                "Randomized count is inconsistent with identity extraction."
            )


@dataclass(frozen=True)
class SymbolicRTETail:
    """Normalized RTE components with no dense many-body operators."""

    tail_id: str
    tail_hash: str
    lambda_r: float
    components: tuple[RTEComponent, ...]
    extraction_metadata: DFTailExtractionMetadata
    normalization_metadata: TailNormalizationMetadata
    identity_policy: IdentityPolicy
    deterministic_identity_coefficient: float
    num_system_qubits: int
    referenced_basis_ids: tuple[str, ...]
    basis_definitions: tuple[BasisChangeMetadata, ...]

    def __post_init__(self) -> None:
        if not self.tail_id or not self.tail_hash:
            raise ValueError("Symbolic RTE tail identifiers must not be empty.")
        if not math.isfinite(self.lambda_r) or self.lambda_r < 0.0:
            raise ValueError("lambda_r must be finite and non-negative.")
        if len(self.components) != self.extraction_metadata.randomized_component_count:
            raise ValueError(
                "Symbolic component count does not match extraction metadata."
            )
        if tuple(item.basis_id for item in self.basis_definitions) != (
            self.referenced_basis_ids
        ):
            raise ValueError("Symbolic basis definitions do not match referenced IDs.")
        if (
            self.normalization_metadata.input_component_count
            != len(self.components)
            or self.normalization_metadata.retained_component_count
            != len(self.components)
            or self.normalization_metadata.dropped_component_count != 0
        ):
            raise ValueError(
                "Symbolic normalization metadata must describe the physical "
                "randomized component set."
            )
        component_l1 = math.fsum(
            component.coefficient_abs for component in self.components
        )
        if not math.isclose(component_l1, self.lambda_r, abs_tol=1e-14):
            raise ValueError("Symbolic component L1 must equal lambda_r.")
        if self.lambda_r == 0.0:
            if self.components:
                raise ValueError("A zero-lambda symbolic tail must have no components.")
            return
        if not self.components:
            raise ValueError("A positive-lambda symbolic tail requires components.")
        probability_sum = math.fsum(
            component.probability for component in self.components
        )
        if not math.isclose(probability_sum, 1.0, abs_tol=1e-14):
            raise ValueError("Symbolic RTE component probabilities must sum to one.")
        for component in self.components:
            expected = component.coefficient_abs / self.lambda_r
            if not math.isclose(component.probability, expected, abs_tol=1e-14):
                raise ValueError("Symbolic RTE component probability is inconsistent.")

    @property
    def is_deterministic_only(self) -> bool:
        return self.lambda_r == 0.0


@dataclass(frozen=True)
class DFTailExtraction:
    """Canonical symbolic expansion of selected squared DF fragments.

    This normal path stores no many-body dense matrix. Runtime basis operations
    remain available through ``basis_registry`` for a future circuit builder.
    """

    tail_id: str
    tail_hash: str
    identity_policy: IdentityPolicy
    components: tuple[DFDiagonalPauliComponent, ...]
    identity_coefficient: float
    deterministic_identity_coefficient: float
    rte_lambda_r: float
    ranking_proxy_lambda_r: float | None
    extraction_metadata: DFTailExtractionMetadata
    normalization_metadata: TailNormalizationMetadata
    num_system_qubits: int
    referenced_basis_ids: tuple[str, ...]
    basis_registry: DFBasisRegistry = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.components) != self.extraction_metadata.randomized_component_count:
            raise ValueError("Randomized component count does not match components.")
        component_l1 = math.fsum(
            abs(component.coefficient) for component in self.components
        )
        if not math.isclose(component_l1, self.rte_lambda_r, abs_tol=1e-14):
            raise ValueError("rte_lambda_r must equal randomized component L1.")
        if not math.isclose(
            self.extraction_metadata.randomized_coefficient_l1,
            self.rte_lambda_r,
            abs_tol=1e-14,
        ):
            raise ValueError("Extraction metadata randomized L1 is inconsistent.")

    @property
    def basis_definitions(self) -> tuple[BasisChangeMetadata, ...]:
        return tuple(
            self.basis_registry.definition(basis_id).metadata
            for basis_id in self.referenced_basis_ids
        )

    def basis_definition(self, basis_id: str) -> DFBasisDefinition:
        return self.basis_registry.definition(basis_id)

    def basis_operations(
        self, basis_id: str
    ) -> tuple[tuple[object, tuple[int, ...]], ...]:
        return self.basis_registry.operations(basis_id)

    def basis_unitary(
        self,
        basis_id: str,
        *,
        max_dense_qubits: int = DEFAULT_MAX_DENSE_QUBITS,
    ) -> np.ndarray:
        """Backward-compatible guarded small-system dense reference."""
        return basis_change_unitary(
            self.basis_definition(basis_id),
            max_dense_qubits=max_dense_qubits,
        )

    @property
    def basis_unitaries(self) -> tuple[tuple[str, np.ndarray], ...]:
        """Legacy, lazily materialized small-system dense basis references."""
        return tuple(
            (basis_id, self.basis_unitary(basis_id))
            for basis_id in self.referenced_basis_ids
        )


def _matrix_sha256(matrix: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(np.asarray(matrix, dtype=np.complex128))
    return hashlib.sha256(contiguous.view(np.uint8).tobytes()).hexdigest()


def describe_basis_change_operations(
    u_ops: Sequence[tuple[object, tuple[int, ...]]],
) -> tuple[BasisChangeOperation, ...]:
    """Fingerprint supported local operations without a many-body unitary.

    Standard registry operations must act on at most four qubits and expose a
    Qiskit ``Operator`` matrix. Wider or opaque operations are rejected instead
    of being identified only by potentially ambiguous display metadata.
    """
    described: list[BasisChangeOperation] = []
    for gate, qubits in u_ops:
        normalized_qubits = tuple(
            require_integer_count(qubit, name="basis operation qubit")
            for qubit in qubits
        )
        if not normalized_qubits:
            raise BasisFingerprintError(
                "Basis operations must act on at least one qubit."
            )
        if len(set(normalized_qubits)) != len(normalized_qubits):
            raise BasisFingerprintError(
                "Basis operation qubit support must not contain duplicates."
            )
        parameters = tuple(str(parameter) for parameter in getattr(gate, "params", ()))
        if len(normalized_qubits) > _MAX_LOCAL_GATE_HASH_QUBITS:
            raise BasisFingerprintError(
                "The standard basis registry rejects operations wider than "
                "four qubits."
            )
        try:
            local_matrix = np.asarray(Operator(gate).data, dtype=np.complex128)
        except Exception as exc:
            raise BasisFingerprintError(
                "Basis operation cannot be converted to a stable local matrix "
                "fingerprint."
            ) from exc
        expected_dimension = 1 << len(normalized_qubits)
        if local_matrix.shape != (expected_dimension, expected_dimension):
            raise BasisFingerprintError(
                "Basis operation matrix shape does not match its qubit support."
            )
        described.append(
            BasisChangeOperation(
                name=str(getattr(gate, "name", type(gate).__name__)),
                qubits=normalized_qubits,
                parameters=parameters,
                matrix_sha256=_matrix_sha256(local_matrix),
            )
        )
    return tuple(described)


def canonical_basis_hash(
    num_system_qubits: int,
    operations: Sequence[BasisChangeOperation],
) -> str:
    """Hash a basis from system size and its ordered local operation metadata."""
    num_system_qubits = require_integer_count(
        num_system_qubits,
        name="num_system_qubits",
        minimum=1,
    )
    payload = {
        "num_system_qubits": num_system_qubits,
        "operations": [asdict(operation) for operation in operations],
        "hash_policy": "ordered_local_basis_operations_v1",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_small_dense_reference(
    num_system_qubits: int,
    max_dense_qubits: int,
    function_name: str,
) -> None:
    num_system_qubits = require_integer_count(
        num_system_qubits,
        name="num_system_qubits",
    )
    max_dense_qubits = require_integer_count(
        max_dense_qubits,
        name="max_dense_qubits",
    )
    if num_system_qubits > max_dense_qubits:
        raise ValueError(
            f"{function_name} is a small-system dense reference and refuses "
            f"{num_system_qubits} qubits above max_dense_qubits={max_dense_qubits}."
        )


def basis_change_unitary(
    basis: DFBlock | DFBasisDefinition,
    *,
    max_dense_qubits: int = DEFAULT_MAX_DENSE_QUBITS,
) -> np.ndarray:
    """Materialize a basis unitary only for an explicitly bounded small system."""
    if isinstance(basis, DFBlock):
        num_qubits = len(np.asarray(basis.eta))
        operations = basis.U_ops
    else:
        num_qubits = basis.num_system_qubits
        operations = basis.runtime_operations
    _require_small_dense_reference(num_qubits, max_dense_qubits, "basis_change_unitary")
    circuit = QuantumCircuit(num_qubits)
    for gate, qubits in operations:
        circuit.append(gate, list(qubits))
    return np.asarray(Operator(circuit).data, dtype=np.complex128)


def diagonal_pauli_matrix(
    num_qubits: int,
    support: Sequence[int],
    *,
    max_dense_qubits: int = DEFAULT_MAX_DENSE_QUBITS,
) -> np.ndarray:
    """Return a guarded small-system little-endian I/Z/ZZ reference matrix."""
    _require_small_dense_reference(
        num_qubits,
        max_dense_qubits,
        "diagonal_pauli_matrix",
    )
    normalized_support = tuple(
        require_integer_count(qubit, name="Pauli support qubit")
        for qubit in support
    )
    support_tuple = tuple(sorted(set(normalized_support)))
    if len(support_tuple) != len(normalized_support) or len(support_tuple) > 2:
        raise ValueError("support must contain zero, one, or two unique qubits.")
    if any(qubit < 0 or qubit >= num_qubits for qubit in support_tuple):
        raise ValueError("support qubit is outside the system register.")
    dimension = 1 << num_qubits
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
    """Expand ``lam * (sum_k eta_k n_k)^2`` symbolically into I, Z, and ZZ."""
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
    basis_registry: DFBasisRegistry | None = None,
) -> DFTailExtraction:
    """Extract a symbolic DF tail without constructing any many-body matrix."""
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
    if len(fragment_ids) != len(blocks):
        raise ValueError("fragment_ids must match the number of blocks.")
    if len(set(fragment_ids)) != len(fragment_ids):
        raise ValueError("fragment_ids must be unique.")
    if basis_ids is not None and len(basis_ids) != len(blocks):
        raise ValueError("basis_ids must match the number of blocks.")

    registry = basis_registry or DFBasisRegistry()
    num_system_qubits = len(np.asarray(blocks[0].eta))
    pre_threshold: list[DFDiagonalPauliComponent] = []
    for index, (fragment_id, block) in enumerate(
        zip(fragment_ids, blocks, strict=True)
    ):
        num_qubits = len(np.asarray(block.eta))
        if num_qubits != num_system_qubits:
            raise ValueError("All DF blocks must act on the same number of qubits.")
        requested_basis_id = None if basis_ids is None else str(basis_ids[index])
        definition = registry.register(
            block.U_ops,
            num_system_qubits=num_qubits,
            basis_id=requested_basis_id,
        )
        operations = definition.metadata.operations
        for support, coefficient in exact_df_diagonal_coefficients(
            block.eta,
            block.lam,
        ):
            sign = 1 if coefficient >= 0.0 else -1
            pre_threshold.append(
                DFDiagonalPauliComponent(
                    component_id=f"{fragment_id}:{_support_label(support)}",
                    coefficient=float(coefficient),
                    coefficient_abs=abs(float(coefficient)),
                    coefficient_sign=sign,
                    df_fragment_id=str(fragment_id),
                    basis_id=definition.basis_id,
                    basis_hash=definition.basis_hash,
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
    threshold_dropped = tuple(
        component
        for component in pre_threshold
        if component.coefficient_abs <= coefficient_atol
    )
    retained_identities = tuple(
        component for component in retained if component.is_identity
    )
    identity_coefficient = float(
        math.fsum(component.coefficient for component in retained_identities)
    )
    if identity_policy == "extract_identity_phase":
        randomized_components = tuple(
            component for component in retained if not component.is_identity
        )
        deterministic_identity = identity_coefficient
        extracted_identity_count = len(retained_identities)
    else:
        randomized_components = retained
        deterministic_identity = 0.0
        extracted_identity_count = 0
    rte_lambda_r = float(
        math.fsum(component.coefficient_abs for component in randomized_components)
    )
    dropped_l1 = float(
        math.fsum(component.coefficient_abs for component in threshold_dropped)
    )
    policy = "threshold_then_exact_identity_policy_v2"
    referenced_basis_id_set = {component.basis_id for component in pre_threshold}
    referenced_basis_ids = tuple(sorted(referenced_basis_id_set))
    basis_metadata = tuple(
        metadata
        for metadata in registry.metadata()
        if metadata.basis_id in referenced_basis_id_set
    )
    payload = {
        "tail_id": tail_id,
        "pre_threshold_components": [
            {
                "component_id": component.component_id,
                "coefficient": component.coefficient,
                "df_fragment_id": component.df_fragment_id,
                "basis_id": component.basis_id,
                "basis_hash": component.basis_hash,
                "diagonal_pauli_support": component.diagonal_pauli_support,
            }
            for component in pre_threshold
        ],
        "basis_definitions": [asdict(metadata) for metadata in basis_metadata],
        "coefficient_atol": float(coefficient_atol),
        "normalization_policy": policy,
        "identity_policy": identity_policy,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    extraction_metadata = DFTailExtractionMetadata(
        coefficient_atol=float(coefficient_atol),
        threshold_input_component_count=len(pre_threshold),
        threshold_retained_component_count=len(retained),
        threshold_dropped_component_count=len(threshold_dropped),
        retained_identity_component_count=len(retained_identities),
        extracted_identity_component_count=extracted_identity_count,
        randomized_component_count=len(randomized_components),
        threshold_dropped_coefficient_l1=dropped_l1,
        extracted_identity_coefficient=deterministic_identity,
        randomized_coefficient_l1=rte_lambda_r,
        threshold_operator_error_bound=dropped_l1,
        normalization_policy=policy,
    )
    randomized_metadata = TailNormalizationMetadata(
        coefficient_atol=0.0,
        input_component_count=len(randomized_components),
        retained_component_count=len(randomized_components),
        dropped_component_count=0,
        dropped_coefficient_l1=0.0,
        operator_error_bound=0.0,
        normalization_policy="already_thresholded_symbolic_df_randomized_tail",
    )
    return DFTailExtraction(
        tail_id=tail_id,
        tail_hash=hashlib.sha256(encoded).hexdigest(),
        identity_policy=identity_policy,
        components=randomized_components,
        identity_coefficient=identity_coefficient,
        deterministic_identity_coefficient=deterministic_identity,
        rte_lambda_r=rte_lambda_r,
        ranking_proxy_lambda_r=(
            None if ranking_proxy_lambda_r is None else float(ranking_proxy_lambda_r)
        ),
        extraction_metadata=extraction_metadata,
        normalization_metadata=randomized_metadata,
        num_system_qubits=num_system_qubits,
        referenced_basis_ids=referenced_basis_ids,
        basis_registry=registry,
    )


def empty_df_tail_extraction(
    tail_id: str,
    *,
    num_system_qubits: int,
    identity_policy: IdentityPolicy = "faithful_identity_in_tail",
    coefficient_atol: float = 0.0,
    ranking_proxy_lambda_r: float = 0.0,
) -> DFTailExtraction:
    """Create the canonical zero-randomized-tail extraction."""
    if not tail_id:
        raise ValueError("tail_id must not be empty.")
    num_system_qubits = require_integer_count(
        num_system_qubits,
        name="num_system_qubits",
        minimum=1,
    )
    if identity_policy not in (
        "faithful_identity_in_tail",
        "extract_identity_phase",
    ):
        raise ValueError(f"Unsupported identity policy: {identity_policy}")
    if not math.isfinite(coefficient_atol) or coefficient_atol < 0.0:
        raise ValueError("coefficient_atol must be finite and non-negative.")
    payload = {
        "tail_id": tail_id,
        "pre_threshold_components": [],
        "basis_definitions": [],
        "coefficient_atol": float(coefficient_atol),
        "normalization_policy": "threshold_then_exact_identity_policy_v2",
        "identity_policy": identity_policy,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    extraction_metadata = DFTailExtractionMetadata(
        coefficient_atol=float(coefficient_atol),
        threshold_input_component_count=0,
        threshold_retained_component_count=0,
        threshold_dropped_component_count=0,
        retained_identity_component_count=0,
        extracted_identity_component_count=0,
        randomized_component_count=0,
        threshold_dropped_coefficient_l1=0.0,
        extracted_identity_coefficient=0.0,
        randomized_coefficient_l1=0.0,
        threshold_operator_error_bound=0.0,
        normalization_policy="threshold_then_exact_identity_policy_v2",
    )
    normalization_metadata = TailNormalizationMetadata(
        coefficient_atol=0.0,
        input_component_count=0,
        retained_component_count=0,
        dropped_component_count=0,
        dropped_coefficient_l1=0.0,
        operator_error_bound=0.0,
        normalization_policy="already_thresholded_symbolic_df_randomized_tail",
    )
    return DFTailExtraction(
        tail_id=tail_id,
        tail_hash=hashlib.sha256(encoded).hexdigest(),
        identity_policy=identity_policy,
        components=(),
        identity_coefficient=0.0,
        deterministic_identity_coefficient=0.0,
        rte_lambda_r=0.0,
        ranking_proxy_lambda_r=float(ranking_proxy_lambda_r),
        extraction_metadata=extraction_metadata,
        normalization_metadata=normalization_metadata,
        num_system_qubits=num_system_qubits,
        referenced_basis_ids=(),
        basis_registry=DFBasisRegistry(),
    )


def extraction_to_symbolic_rte_tail(
    extraction: DFTailExtraction,
) -> SymbolicRTETail:
    """Normalize a DF extraction for RTE event generation without dense data."""
    component_l1 = math.fsum(
        component.coefficient_abs for component in extraction.components
    )
    if not math.isclose(component_l1, extraction.rte_lambda_r, abs_tol=1e-14):
        raise ValueError("Extraction component L1 does not match rte_lambda_r.")
    if extraction.rte_lambda_r == 0.0:
        normalized_components: tuple[RTEComponent, ...] = ()
    else:
        normalized_components = tuple(
            RTEComponent(
                component_id=component.component_id,
                probability=component.coefficient_abs / extraction.rte_lambda_r,
                coefficient_abs=component.coefficient_abs,
                coefficient_sign=component.coefficient_sign,
                df_fragment_id=component.df_fragment_id,
                basis_id=component.basis_id,
                basis_hash=component.basis_hash,
                is_identity=component.is_identity,
                diagonal_pauli_support=component.diagonal_pauli_support,
                basis_change_operations=component.basis_change_operations,
            )
            for component in extraction.components
        )
    return SymbolicRTETail(
        tail_id=extraction.tail_id,
        tail_hash=extraction.tail_hash,
        lambda_r=extraction.rte_lambda_r,
        components=normalized_components,
        extraction_metadata=extraction.extraction_metadata,
        normalization_metadata=extraction.normalization_metadata,
        identity_policy=extraction.identity_policy,
        deterministic_identity_coefficient=(
            extraction.deterministic_identity_coefficient
        ),
        num_system_qubits=extraction.num_system_qubits,
        referenced_basis_ids=extraction.referenced_basis_ids,
        basis_definitions=extraction.basis_definitions,
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
    """Build and symbolically extract selected native DF fragments."""
    from .df_partial_randomized_pf import df_fragment_weight, df_hamiltonian_to_model
    from .df_trotter.ops import build_df_blocks

    normalized_indices = tuple(
        require_integer_count(index, name="tail block index")
        for index in block_indices
    )
    if not normalized_indices:
        return empty_df_tail_extraction(
            tail_id,
            num_system_qubits=hamiltonian.n_qubits,
            identity_policy=identity_policy,
            coefficient_atol=coefficient_atol,
            ranking_proxy_lambda_r=0.0,
        )
    selected = hamiltonian.select_blocks(normalized_indices)
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
        for index in normalized_indices
    )
    fragment_ids = tuple(f"df-fragment-{index}" for index in normalized_indices)
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
    *,
    max_dense_qubits: int = DEFAULT_MAX_DENSE_QUBITS,
) -> np.ndarray:
    """Build one conjugated component only as a guarded small-system reference."""
    _require_small_dense_reference(
        extraction.num_system_qubits,
        max_dense_qubits,
        "component_dense_operator",
    )
    diagonal = diagonal_pauli_matrix(
        extraction.num_system_qubits,
        component.diagonal_pauli_support,
        max_dense_qubits=max_dense_qubits,
    )
    basis = basis_change_unitary(
        extraction.basis_definition(component.basis_id),
        max_dense_qubits=max_dense_qubits,
    )
    return basis @ diagonal @ basis.conj().T


def dense_extracted_df_tail(
    extraction: DFTailExtraction,
    *,
    max_dense_qubits: int = DEFAULT_MAX_DENSE_QUBITS,
) -> np.ndarray:
    """Reconstruct a tail only as a guarded small-system dense reference."""
    _require_small_dense_reference(
        extraction.num_system_qubits,
        max_dense_qubits,
        "dense_extracted_df_tail",
    )
    dimension = 1 << extraction.num_system_qubits
    result = (
        extraction.deterministic_identity_coefficient
        * np.eye(dimension, dtype=np.complex128)
    )
    for component in extraction.components:
        result += component.coefficient * component_dense_operator(
            extraction,
            component,
            max_dense_qubits=max_dense_qubits,
        )
    return result


def dense_df_block_hamiltonian(
    block: DFBlock,
    *,
    max_dense_qubits: int = DEFAULT_MAX_DENSE_QUBITS,
) -> np.ndarray:
    """Build a DF block only as a guarded small-system dense reference."""
    eta = np.asarray(np.real_if_close(block.eta), dtype=float)
    _require_small_dense_reference(
        len(eta),
        max_dense_qubits,
        "dense_df_block_hamiltonian",
    )
    dimension = 1 << len(eta)
    diagonal = np.empty(dimension, dtype=np.complex128)
    for basis_state in range(dimension):
        occupation_sum = math.fsum(
            float(eta[qubit]) * ((basis_state >> qubit) & 1)
            for qubit in range(len(eta))
        )
        diagonal[basis_state] = float(block.lam) * occupation_sum**2
    basis = basis_change_unitary(block, max_dense_qubits=max_dense_qubits)
    return basis @ np.diag(diagonal) @ basis.conj().T


def extraction_to_normalized_rte_tail(
    extraction: DFTailExtraction,
    *,
    max_dense_qubits: int = DEFAULT_MAX_DENSE_QUBITS,
) -> NormalizedRTETail:
    """Materialize generic RTE operators only for a guarded small-system check."""
    if extraction.rte_lambda_r == 0.0:
        raise DeterministicOnlyRTETailError(
            "The extraction has no randomized components; dense RTE normalization "
            "is not required."
        )
    _require_small_dense_reference(
        extraction.num_system_qubits,
        max_dense_qubits,
        "extraction_to_normalized_rte_tail",
    )
    terms = tuple(
        InvolutoryTailTerm(
            component_id=component.component_id,
            coefficient=component.coefficient,
            operator=component_dense_operator(
                extraction,
                component,
                max_dense_qubits=max_dense_qubits,
            ),
            df_fragment_id=component.df_fragment_id,
            basis_id=component.basis_id,
            basis_hash=component.basis_hash,
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
    normalized_by_id = {
        component.component_id: (component, operator)
        for component, operator in zip(
            tail.components,
            tail.operators,
            strict=True,
        )
    }
    ordered_pairs = tuple(
        normalized_by_id[component.component_id]
        for component in extraction.components
    )
    return replace(
        tail,
        tail_hash=extraction.tail_hash,
        components=tuple(component for component, _operator in ordered_pairs),
        operators=tuple(operator for _component, operator in ordered_pairs),
        normalization_metadata=extraction.normalization_metadata,
    )


def extraction_component_circuit_specs(
    extraction: DFTailExtraction,
) -> tuple[DFRTECircuitSpec, ...]:
    """Translate symbolic components to the future builder's typed specs."""
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
                    basis_hash=component.basis_hash,
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
                    basis_hash=component.basis_hash,
                    diagonal_pauli_support=component.diagonal_pauli_support,
                    basis_change_operations=component.basis_change_operations,
                    num_system_qubits=extraction.num_system_qubits,
                )
            )
    return tuple(specs)


def prepare_df_rte_event_inputs(
    extraction: DFTailExtraction,
    basis_registry: DFBasisRegistry | None = None,
) -> DFRTEEventPreparation:
    """Bundle a symbolic tail, circuit specs, and executable basis registry."""
    from .df_rte_circuit import DFRTEEventPreparation

    return DFRTEEventPreparation(
        symbolic_tail=extraction_to_symbolic_rte_tail(extraction),
        component_specs=extraction_component_circuit_specs(extraction),
        basis_registry=(
            extraction.basis_registry if basis_registry is None else basis_registry
        ),
    )


def uncontrolled_identity_evolution_operator(
    coefficient: float,
    evolution_time: float,
    system_dimension: int,
) -> np.ndarray:
    """Return the small-system uncontrolled global-phase reference."""
    return np.exp(-1j * float(evolution_time) * float(coefficient)) * np.eye(
        int(system_dimension), dtype=np.complex128
    )


def controlled_identity_evolution_operator(
    coefficient: float,
    evolution_time: float,
    system_dimension: int,
) -> np.ndarray:
    """Return the small-system ancilla-relative identity reference."""
    identity = np.eye(int(system_dimension), dtype=np.complex128)
    phase = np.exp(-1j * float(evolution_time) * float(coefficient))
    return np.block(
        [
            [identity, np.zeros_like(identity)],
            [np.zeros_like(identity), phase * identity],
        ]
    )
