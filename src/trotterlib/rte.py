"""Finite Randomized Taylor Expansion (RTE) distributions and references.

The implementation follows arXiv:2503.05647v2, Appendix A, Eqs. (A18)--(A40).
The proof requires a normalized decomposition into Hermitian involutions.  It
is deliberately not a sampler over complete, generally non-involutory DF
fragments; exact DF provenance can instead be attached to each involution.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import operator
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Protocol, Sequence, TypeAlias

import numpy as np


MeasurementAxis: TypeAlias = Literal["X", "Y"]
TaylorDistributionKind: TypeAlias = Literal["rte_even_taylor_paired"]
FidelityLevel: TypeAlias = Literal[0, 1, 2, 3, 4, 5, 6]
TruncationAllocationPolicy: TypeAlias = Literal[
    "equal_log_budget_per_short_step",
    "user_selected_orders",
]


def require_integer_count(
    value: object,
    *,
    name: str,
    minimum: int = 0,
) -> int:
    """Return a Python integer count without silently truncating other types."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer count, not bool.")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer count.") from exc
    result = int(normalized)
    if result < minimum:
        qualifier = "positive" if minimum == 1 else f"at least {minimum}"
        raise ValueError(f"{name} must be {qualifier}.")
    return result


class DeterministicOnlyRTETailError(ValueError):
    """Raised when an RTE-only API receives no randomized tail components."""


class RTETailLike(Protocol):
    """Dense-free fields required to configure and sample finite RTE events."""

    tail_id: str
    tail_hash: str
    lambda_r: float
    components: tuple["RTEComponent", ...]


@dataclass(frozen=True)
class BasisChangeOperation:
    """Serializable description of one operation in a DF orbital basis change."""

    name: str
    qubits: tuple[int, ...]
    parameters: tuple[str, ...] = ()
    matrix_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Basis-change operation name must not be empty.")
        if not self.qubits:
            raise ValueError("Basis-change operation qubits must not be empty.")
        normalized_qubits = tuple(
            require_integer_count(qubit, name="basis operation qubit")
            for qubit in self.qubits
        )
        object.__setattr__(self, "qubits", normalized_qubits)


@dataclass(frozen=True)
class RTEComponent:
    """One normalized, sign-absorbed Hermitian involution in ``H/lambda``."""

    component_id: str
    probability: float
    coefficient_abs: float
    coefficient_sign: int
    df_fragment_id: str | None = None
    basis_id: str | None = None
    basis_hash: str | None = None
    is_identity: bool = False
    diagonal_pauli_support: tuple[int, ...] | None = None
    basis_change_operations: tuple[BasisChangeOperation, ...] = ()

    def __post_init__(self) -> None:
        if not self.component_id:
            raise ValueError("component_id must not be empty.")
        if not math.isfinite(self.probability) or self.probability <= 0.0:
            raise ValueError("component probability must be finite and positive.")
        if self.coefficient_sign not in (-1, 1):
            raise ValueError("coefficient_sign must be -1 or +1.")
        if not math.isfinite(self.coefficient_abs) or self.coefficient_abs < 0.0:
            raise ValueError("coefficient_abs must be finite and non-negative.")
        if self.diagonal_pauli_support is not None:
            support = self.diagonal_pauli_support
            if tuple(sorted(set(support))) != support or len(support) > 2:
                raise ValueError("diagonal_pauli_support must be a sorted I/Z/ZZ support.")


@dataclass(frozen=True)
class InvolutoryTailTerm:
    """Raw tail term ``coefficient * operator`` before sign absorption."""

    component_id: str
    coefficient: float
    operator: np.ndarray = field(repr=False, compare=False)
    df_fragment_id: str | None = None
    basis_id: str | None = None
    basis_hash: str | None = None
    diagonal_pauli_support: tuple[int, ...] | None = None
    basis_change_operations: tuple[BasisChangeOperation, ...] = ()


@dataclass(frozen=True)
class TailNormalizationMetadata:
    """Audit record for exact or thresholded involutory normalization."""

    coefficient_atol: float
    input_component_count: int
    retained_component_count: int
    dropped_component_count: int
    dropped_coefficient_l1: float
    operator_error_bound: float
    normalization_policy: str


@dataclass(frozen=True)
class NormalizedRTETail:
    """Validated decomposition ``H_R = lambda_r * sum_l p_l P_l``."""

    tail_id: str
    tail_hash: str
    lambda_r: float
    components: tuple[RTEComponent, ...]
    operators: tuple[np.ndarray, ...] = field(repr=False, compare=False)
    normalization_metadata: TailNormalizationMetadata

    @property
    def dense_hamiltonian(self) -> np.ndarray:
        result = np.zeros_like(self.operators[0], dtype=np.complex128)
        for component, operator in zip(self.components, self.operators, strict=True):
            result += component.probability * operator
        return self.lambda_r * result

    @property
    def normalized_hamiltonian(self) -> np.ndarray:
        return self.dense_hamiltonian / self.lambda_r


@dataclass(frozen=True)
class RTEFiniteDistribution:
    """Exact finite order distribution and separately labeled paper bound."""

    dimensionless_step_time: float
    finite_taylor_order: int
    orders: tuple[int, ...]
    unnormalized_order_weights: tuple[float, ...]
    order_probabilities: tuple[float, ...]
    exact_finite_distribution: float
    truncation_residual_bound: float
    paper_upper_bound: float

    def __post_init__(self) -> None:
        cutoff = require_integer_count(
            self.finite_taylor_order,
            name="finite_taylor_order",
        )
        object.__setattr__(self, "finite_taylor_order", cutoff)
        if cutoff % 2:
            raise ValueError("finite_taylor_order must be a non-negative even integer.")
        if self.orders != tuple(range(0, self.finite_taylor_order + 1, 2)):
            raise ValueError("orders must contain every even order through the cutoff.")
        if len(self.orders) != len(self.unnormalized_order_weights):
            raise ValueError("order weights length mismatch.")
        if len(self.orders) != len(self.order_probabilities):
            raise ValueError("order probabilities length mismatch.")
        if not math.isclose(sum(self.order_probabilities), 1.0, abs_tol=1e-14):
            raise ValueError("finite Taylor order probabilities must sum to one.")

    @property
    def step_truncation_residual_bound(self) -> float:
        """Explicit name for the legacy one-short-step residual field."""
        return self.truncation_residual_bound


@dataclass(frozen=True)
class RTEConfig:
    """Configuration for finite RTE of one tail evolution interval."""

    tail_id: str
    tail_hash: str
    lambda_r: float
    evolution_time: float
    rte_steps: int
    step_time: float
    dimensionless_step_time: float
    taylor_distribution: TaylorDistributionKind
    finite_taylor_order: int
    truncation_tolerance: float
    distribution_normalization: float
    truncation_residual_bound: float
    seed: int

    def __post_init__(self) -> None:
        rte_steps = require_integer_count(
            self.rte_steps,
            name="rte_steps",
            minimum=1,
        )
        cutoff = require_integer_count(
            self.finite_taylor_order,
            name="finite_taylor_order",
        )
        object.__setattr__(self, "rte_steps", rte_steps)
        object.__setattr__(self, "finite_taylor_order", cutoff)
        if self.lambda_r <= 0.0 or not math.isfinite(self.lambda_r):
            raise ValueError("lambda_r must be finite and positive.")
        if cutoff % 2:
            raise ValueError("finite_taylor_order must be a non-negative even integer.")
        if self.truncation_tolerance <= 0.0:
            raise ValueError("truncation_tolerance must be positive.")
        if self.distribution_normalization <= 0.0:
            raise ValueError("distribution_normalization must be positive.")
        expected_step = self.evolution_time / self.rte_steps
        if not math.isclose(self.step_time, expected_step, rel_tol=1e-14, abs_tol=1e-15):
            raise ValueError("step_time must equal evolution_time / rte_steps.")
        if not math.isclose(
            self.dimensionless_step_time,
            self.lambda_r * self.step_time,
            rel_tol=1e-14,
            abs_tol=1e-15,
        ):
            raise ValueError("dimensionless_step_time must equal lambda_r * step_time.")
        if self.step_truncation_residual_bound > self.truncation_tolerance:
            raise ValueError("finite Taylor cutoff does not meet truncation_tolerance.")
        if self.taylor_distribution != "rte_even_taylor_paired":
            raise ValueError("Unsupported Taylor distribution.")
        finite = finite_rte_distribution(
            self.dimensionless_step_time, self.finite_taylor_order
        )
        if not math.isclose(
            self.distribution_normalization,
            finite.exact_finite_distribution,
            rel_tol=1e-14,
            abs_tol=1e-15,
        ):
            raise ValueError("distribution_normalization does not match the cutoff.")
        if not math.isclose(
            self.step_truncation_residual_bound,
            finite.step_truncation_residual_bound,
            rel_tol=1e-14,
            abs_tol=1e-15,
        ):
            raise ValueError("truncation_residual_bound does not match the cutoff.")

    @property
    def step_truncation_residual_bound(self) -> float:
        """Explicit name for the legacy one-short-step residual field."""
        return self.truncation_residual_bound

    @property
    def step_truncation_tolerance(self) -> float:
        """Clarify that the legacy ``truncation_tolerance`` is per short step."""
        return self.truncation_tolerance

    @property
    def occurrence_truncation_residual_bound(self) -> float:
        """Compose the short-step bound across this occurrence's RTE steps."""
        return occurrence_truncation_residual_bound(
            self.step_truncation_residual_bound,
            self.rte_steps,
        )


@dataclass(frozen=True)
class BasisReuseInterval:
    """Maximal adjacent interval sharing a known DF basis in circuit order."""

    start: int
    stop: int
    basis_id: str
    basis_hash: str | None = None


@dataclass(frozen=True)
class RTEEventApplication:
    """One product/rotation occurrence in physical circuit application order."""

    application_index: int
    role: Literal["product", "rotation"]
    component_id: str
    coefficient_abs: float
    coefficient_sign: int
    is_identity: bool
    df_fragment_id: str | None
    basis_id: str | None
    diagonal_pauli_support: tuple[int, ...] | None
    basis_change_operations: tuple[BasisChangeOperation, ...]
    basis_hash: str | None = None


@dataclass(frozen=True)
class RTEEvent:
    """One v2 Eq. (A23) event, with probability and LCU coefficient separated."""

    taylor_order: int
    rotation_component_id: str
    product_component_ids: tuple[str, ...]
    selected_component_ids: tuple[str, ...]
    df_fragment_ids: tuple[str | None, ...]
    event_probability: float
    event_coefficient: float
    phase: complex
    rotation_angle: float
    event_normalization: float
    basis_id: str | None
    basis_reuse_intervals: tuple[BasisReuseInterval, ...]
    application_sequence: tuple[RTEEventApplication, ...]
    basis_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["phase"] = {"real": float(self.phase.real), "imag": float(self.phase.imag)}
        return data

    @property
    def product_sign_phase(self) -> int:
        """Phase from implementing signed product involutions via unsigned Paulis."""
        return math.prod(
            application.coefficient_sign
            for application in self.application_sequence
            if application.role == "product"
        )

    @property
    def unsigned_rotation_angle(self) -> float:
        """Angle for the unsigned Pauli support of the rotation occurrence."""
        rotation = self.application_sequence[-1]
        if rotation.role != "rotation":
            raise ValueError("RTE event application sequence must end in rotation.")
        return rotation.coefficient_sign * self.rotation_angle


@dataclass(frozen=True)
class EventOperatorSampleEstimate:
    """Unweighted Monte Carlo estimate of an RTE event-operator mean."""

    operator_mean: np.ndarray = field(repr=False, compare=False)
    entrywise_standard_error: np.ndarray = field(repr=False, compare=False)
    frobenius_standard_error: float
    sample_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sample_count",
            require_integer_count(self.sample_count, name="sample_count", minimum=1),
        )


@dataclass(frozen=True)
class RTEOperatorMoments:
    """Normalization-corrected polynomial and its attenuated event mean."""

    corrected_operator: np.ndarray = field(repr=False, compare=False)
    attenuated_event_mean_operator: np.ndarray = field(repr=False, compare=False)
    normalization_product: float
    attenuation_factor: float


@dataclass(frozen=True)
class RTEOccurrenceParameters:
    """Dense-free inputs needed to choose one occurrence's Taylor cutoff."""

    occurrence_id: str
    tail_id: str
    tail_hash: str
    lambda_r: float
    evolution_time: float
    rte_steps: int
    round_occurrence_count: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rte_steps",
            require_integer_count(self.rte_steps, name="rte_steps", minimum=1),
        )
        object.__setattr__(
            self,
            "round_occurrence_count",
            require_integer_count(
                self.round_occurrence_count,
                name="round_occurrence_count",
                minimum=1,
            ),
        )
        if not self.occurrence_id or not self.tail_id or not self.tail_hash:
            raise ValueError("Occurrence and tail identifiers must not be empty.")
        if not math.isfinite(self.lambda_r) or self.lambda_r <= 0.0:
            raise ValueError("lambda_r must be finite and positive.")
        if not math.isfinite(self.evolution_time):
            raise ValueError("evolution_time must be finite.")


@dataclass(frozen=True)
class RTEOccurrenceTruncation:
    """Finite Taylor truncation accounting for one tail occurrence kind."""

    occurrence_id: str
    tail_id: str
    tail_hash: str
    lambda_r: float
    evolution_time: float
    rte_steps: int
    finite_taylor_order: int
    dimensionless_step_time: float
    step_truncation_residual_bound: float
    occurrence_truncation_residual_bound: float
    round_occurrence_count: int
    round_contribution_residual_bound: float
    allocated_step_error_bound: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rte_steps",
            require_integer_count(self.rte_steps, name="rte_steps", minimum=1),
        )
        object.__setattr__(
            self,
            "round_occurrence_count",
            require_integer_count(
                self.round_occurrence_count,
                name="round_occurrence_count",
                minimum=1,
            ),
        )
        cutoff = require_integer_count(
            self.finite_taylor_order,
            name="finite_taylor_order",
        )
        object.__setattr__(self, "finite_taylor_order", cutoff)
        if cutoff % 2:
            raise ValueError("finite_taylor_order must be a non-negative even integer.")
        bounds = (
            self.step_truncation_residual_bound,
            self.occurrence_truncation_residual_bound,
            self.round_contribution_residual_bound,
        )
        if any(math.isnan(bound) or bound < 0.0 for bound in bounds):
            raise ValueError("Truncation residual bounds must be non-negative.")


@dataclass(frozen=True)
class RPERoundTruncationBudget:
    """Baseline allocation policy and target for one RPE round."""

    round_index: int
    target_round_truncation_error: float
    allocation_policy: TruncationAllocationPolicy
    total_short_step_count: int
    allocated_log_error_per_short_step: float | None
    allocated_step_error_bound: float | None
    partial_s2_repetitions: int | None = None
    expected_tail_evolutions: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "round_index",
            require_integer_count(self.round_index, name="round_index"),
        )
        object.__setattr__(
            self,
            "total_short_step_count",
            require_integer_count(
                self.total_short_step_count,
                name="total_short_step_count",
                minimum=1,
            ),
        )
        if self.partial_s2_repetitions is not None:
            object.__setattr__(
                self,
                "partial_s2_repetitions",
                require_integer_count(
                    self.partial_s2_repetitions,
                    name="partial_s2_repetitions",
                ),
            )
        if self.expected_tail_evolutions is not None:
            object.__setattr__(
                self,
                "expected_tail_evolutions",
                require_integer_count(
                    self.expected_tail_evolutions,
                    name="expected_tail_evolutions",
                ),
            )
        if (
            not math.isfinite(self.target_round_truncation_error)
            or self.target_round_truncation_error < 0.0
        ):
            raise ValueError("Round truncation target must be finite and non-negative.")


@dataclass(frozen=True)
class RPETruncationSummary:
    """Finite Taylor error only; attenuation and other errors remain separate."""

    budget: RPERoundTruncationBudget
    occurrences: tuple[RTEOccurrenceTruncation, ...]
    round_truncation_residual_bound: float
    meets_round_budget: bool


@dataclass(frozen=True)
class RPERound:
    """One measurement axis of one finite robust phase-estimation round."""

    round_index: int
    effective_evolution_time: float
    partial_s2_repetitions: int
    tail_evolutions: int
    rte_total_steps: int
    attenuation_factor: float
    measurement_axis: MeasurementAxis
    required_shots: int

    def __post_init__(self) -> None:
        for name in (
            "round_index",
            "partial_s2_repetitions",
            "tail_evolutions",
            "rte_total_steps",
        ):
            object.__setattr__(
                self,
                name,
                require_integer_count(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "required_shots",
            require_integer_count(
                self.required_shots,
                name="required_shots",
                minimum=1,
            ),
        )


@dataclass(frozen=True)
class CompilerSettings:
    basis_gates: tuple[str, ...]
    backend_name: str | None
    coupling_map: tuple[tuple[int, int], ...] | None
    optimization_level: int
    layout_method: str | None
    routing_method: str | None
    transpiler_seed: int
    qiskit_version: str

    def __post_init__(self) -> None:
        if not self.basis_gates or any(
            not isinstance(name, str) or not name.strip()
            for name in self.basis_gates
        ):
            raise ValueError("basis_gates must contain at least one gate name.")
        object.__setattr__(
            self,
            "basis_gates",
            tuple(name.strip().lower() for name in self.basis_gates),
        )
        optimization_level = require_integer_count(
            self.optimization_level,
            name="optimization_level",
        )
        if optimization_level > 3:
            raise ValueError("optimization_level must be between 0 and 3.")
        object.__setattr__(self, "optimization_level", optimization_level)
        object.__setattr__(
            self,
            "transpiler_seed",
            require_integer_count(self.transpiler_seed, name="transpiler_seed"),
        )
        if not self.qiskit_version:
            raise ValueError("qiskit_version must not be empty.")
        if self.coupling_map is not None:
            normalized_edges = tuple(
                (
                    require_integer_count(left, name="coupling_map qubit"),
                    require_integer_count(right, name="coupling_map qubit"),
                )
                for left, right in self.coupling_map
            )
            object.__setattr__(self, "coupling_map", normalized_edges)


@dataclass(frozen=True)
class CircuitCost:
    """Compiled cost record; RZ count is the primary metric."""

    rz_count: float
    rz_depth: float
    cx_count: float
    cx_depth: float
    total_depth: float
    circuit_size: float
    compiler: CompilerSettings
    fidelity_level: FidelityLevel
    estimate_kind: Literal[
        "paper_upper_bound",
        "exact_finite_distribution",
        "empirical_compiled_estimate",
        "exact_compiled_expectation",
        "monte_carlo_compiled_expectation",
        "monte_carlo_compiled_sequence_expectation",
        "compiled_cost_standard_error",
        "compiled_sequence_nonadditive_difference",
        "exact_compiled_partial_s2_expectation",
        "monte_carlo_compiled_partial_s2_expectation",
        "compiled_partial_s2_additive_expectation",
        "compiled_partial_s2_nonadditive_difference",
        "exact_compiled_repeated_partial_s2_expectation",
        "monte_carlo_compiled_repeated_partial_s2_expectation",
        "compiled_repeated_partial_s2_raw_concatenation",
        "compiled_repeated_partial_s2_boundary_optimized",
        "compiled_repeated_partial_s2_boundary_difference",
        "compiled_repeated_partial_s2_matched_step_sum",
        "compiled_repeated_partial_s2_cross_step_difference",
        "compiled_repeated_partial_s2_primitive_additive_sum",
        "legacy_analytic_proxy",
    ]


def _array_hash(array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(np.asarray(array, dtype=np.complex128))
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "sha256": hashlib.sha256(contiguous.view(np.uint8).tobytes()).hexdigest(),
    }


def normalize_involutory_tail(
    tail_id: str,
    terms: Sequence[InvolutoryTailTerm],
    *,
    atol: float = 0.0,
    validation_atol: float = 1e-10,
) -> NormalizedRTETail:
    """Validate, threshold, and sign-absorb an involutory tail.

    ``atol`` applies only to coefficients.  Its default is exactly zero, so a
    nonzero term is never silently discarded.  ``validation_atol`` is the
    independent numerical tolerance for Hermiticity and involution checks.
    The hash covers the canonical pre-threshold input and normalization policy.
    """
    if not tail_id:
        raise ValueError("tail_id must not be empty.")
    if not terms:
        raise ValueError("At least one nonzero involutory tail term is required.")
    if not math.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be finite and non-negative.")
    if not math.isfinite(validation_atol) or validation_atol < 0.0:
        raise ValueError("validation_atol must be finite and non-negative.")

    identifiers = [str(term.component_id) for term in terms]
    if any(not identifier for identifier in identifiers):
        raise ValueError("component_id must not be empty.")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("component_id values must be unique before thresholding.")

    canonical_terms = sorted(terms, key=lambda item: str(item.component_id))
    validated: list[tuple[InvolutoryTailTerm, float, np.ndarray]] = []
    hash_terms: list[dict[str, Any]] = []
    dimension: int | None = None
    for term in canonical_terms:
        coefficient = float(term.coefficient)
        if not math.isfinite(coefficient):
            raise ValueError(f"{term.component_id}: coefficient must be finite.")
        operator = np.asarray(term.operator, dtype=np.complex128)
        if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
            raise ValueError(f"{term.component_id}: operator must be square.")
        if dimension is None:
            dimension = int(operator.shape[0])
        elif operator.shape != (dimension, dimension):
            raise ValueError("All tail operators must have the same dimension.")
        identity = np.eye(operator.shape[0], dtype=np.complex128)
        if not np.allclose(
            operator, operator.conj().T, atol=validation_atol, rtol=0.0
        ):
            raise ValueError(f"{term.component_id}: operator must be Hermitian.")
        if not np.allclose(
            operator @ operator, identity, atol=validation_atol, rtol=0.0
        ):
            raise ValueError(
                f"{term.component_id}: RTE proof requires an involution P^2=I."
            )
        validated.append((term, coefficient, operator))
        hash_terms.append(
            {
                "component_id": str(term.component_id),
                "coefficient": coefficient,
                "df_fragment_id": term.df_fragment_id,
                "basis_id": term.basis_id,
                "basis_hash": term.basis_hash,
                "diagonal_pauli_support": term.diagonal_pauli_support,
                "basis_change_operations": [
                    asdict(operation) for operation in term.basis_change_operations
                ],
                "operator": _array_hash(operator),
            }
        )

    retained = [item for item in validated if abs(item[1]) > atol]
    dropped = [item for item in validated if abs(item[1]) <= atol]
    if not retained:
        raise ValueError("At least one nonzero involutory tail term is required.")
    lambda_r = float(sum(abs(coefficient) for _term, coefficient, _op in retained))
    components: list[RTEComponent] = []
    operators: list[np.ndarray] = []

    for term, coefficient, operator in retained:
        identity = np.eye(operator.shape[0], dtype=np.complex128)
        sign = 1 if coefficient > 0.0 else -1
        signed_operator = sign * operator
        is_identity = bool(
            np.allclose(
                signed_operator, identity, atol=validation_atol, rtol=0.0
            )
            or np.allclose(
                signed_operator, -identity, atol=validation_atol, rtol=0.0
            )
        )
        component = RTEComponent(
            component_id=str(term.component_id),
            probability=abs(coefficient) / lambda_r,
            coefficient_sign=sign,
            coefficient_abs=abs(coefficient),
            df_fragment_id=term.df_fragment_id,
            basis_id=term.basis_id,
            basis_hash=term.basis_hash,
            is_identity=is_identity,
            diagonal_pauli_support=term.diagonal_pauli_support,
            basis_change_operations=term.basis_change_operations,
        )
        components.append(component)
        operators.append(signed_operator)

    policy = "drop_abs_coefficient_lte_atol"
    payload = {
        "tail_id": tail_id,
        "pre_threshold_terms": hash_terms,
        "normalization_policy": policy,
        "coefficient_atol": float(atol),
        "validation_atol": float(validation_atol),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    dropped_l1 = float(sum(abs(coefficient) for _term, coefficient, _op in dropped))
    return NormalizedRTETail(
        tail_id=tail_id,
        tail_hash=hashlib.sha256(encoded).hexdigest(),
        lambda_r=lambda_r,
        components=tuple(components),
        operators=tuple(operators),
        normalization_metadata=TailNormalizationMetadata(
            coefficient_atol=float(atol),
            input_component_count=len(validated),
            retained_component_count=len(retained),
            dropped_component_count=len(dropped),
            dropped_coefficient_l1=dropped_l1,
            operator_error_bound=dropped_l1,
            normalization_policy=policy,
        ),
    )


def _paired_order_weight(dimensionless_step_time: float, order: int) -> float:
    order = require_integer_count(order, name="order")
    if order % 2:
        raise ValueError("RTE Taylor event order must be non-negative and even.")
    magnitude = abs(float(dimensionless_step_time))
    if magnitude == 0.0:
        return 1.0 if order == 0 else 0.0
    log_taylor = order * math.log(magnitude) - math.lgamma(order + 1)
    return math.exp(log_taylor) * math.hypot(1.0, magnitude / (order + 1))


def step_taylor_truncation_residual_bound(
    dimensionless_step_time: float,
    finite_taylor_order: int,
) -> float:
    """LCU 1-norm bound for omitted paired events.

    A cutoff at even ``K`` retains ordinary Taylor degrees through ``K+1``.
    Since each omitted paired 2-norm is no larger than the corresponding
    pair's 1-norm, the scalar exponential tail from degree ``K+2`` is a safe
    bound.
    """
    finite_taylor_order = require_integer_count(
        finite_taylor_order,
        name="finite_taylor_order",
    )
    if finite_taylor_order % 2:
        raise ValueError("finite_taylor_order must be a non-negative even integer.")
    magnitude = abs(float(dimensionless_step_time))
    if magnitude == 0.0:
        return 0.0
    first_degree = finite_taylor_order + 2
    term = math.exp(
        first_degree * math.log(magnitude) - math.lgamma(first_degree + 1)
    )
    total = term
    degree = first_degree
    for _ in range(100_000):
        next_term = term * magnitude / (degree + 1)
        next_ratio = magnitude / (degree + 2)
        if next_ratio < 1.0:
            remaining_upper_bound = next_term / (1.0 - next_ratio)
            if remaining_upper_bound <= max(1e-300, 1e-16 * total):
                return float(total + remaining_upper_bound)
        degree += 1
        term = next_term
        total += term
    raise RuntimeError("Taylor residual summation did not converge.")


def taylor_truncation_residual_bound(
    dimensionless_step_time: float,
    finite_taylor_order: int,
) -> float:
    """Backward-compatible alias for the one-short-step Taylor bound."""
    return step_taylor_truncation_residual_bound(
        dimensionless_step_time,
        finite_taylor_order,
    )


def choose_finite_taylor_order(
    dimensionless_step_time: float,
    truncation_tolerance: float,
    *,
    maximum_order: int = 10_000,
) -> int:
    """Choose the smallest even cutoff meeting the one-step residual bound."""
    if truncation_tolerance <= 0.0:
        raise ValueError("truncation_tolerance must be positive.")
    maximum_order = require_integer_count(
        maximum_order,
        name="maximum_order",
    )
    for order in range(0, maximum_order + 1, 2):
        if (
            step_taylor_truncation_residual_bound(dimensionless_step_time, order)
            <= truncation_tolerance
        ):
            return order
    raise ValueError("No finite Taylor cutoff met the requested tolerance.")


def occurrence_truncation_residual_bound(
    step_truncation_residual_bound: float,
    rte_steps: int,
) -> float:
    """Compose one short-step bound as ``(1 + epsilon_step)^r - 1``."""
    step_bound = float(step_truncation_residual_bound)
    if math.isnan(step_bound) or step_bound < 0.0:
        raise ValueError("step_truncation_residual_bound must be non-negative.")
    rte_steps = require_integer_count(rte_steps, name="rte_steps")
    if rte_steps == 0 or step_bound == 0.0:
        return 0.0
    if math.isinf(step_bound):
        return math.inf
    try:
        return float(math.expm1(rte_steps * math.log1p(step_bound)))
    except OverflowError:
        return math.inf


def compose_truncation_residual_bounds(
    occurrences: Sequence[tuple[float, int, int]],
) -> float:
    """Compose heterogeneous ``(step bound, steps, occurrence count)`` triples."""
    log_bound = 0.0
    for step_bound, rte_steps, occurrence_count in occurrences:
        step_bound = float(step_bound)
        if math.isnan(step_bound) or step_bound < 0.0:
            raise ValueError("Every step truncation bound must be non-negative.")
        rte_steps = require_integer_count(rte_steps, name="rte_steps")
        occurrence_count = require_integer_count(
            occurrence_count,
            name="occurrence_count",
        )
        if step_bound == 0.0 or rte_steps == 0 or occurrence_count == 0:
            continue
        if math.isinf(step_bound):
            return math.inf
        log_bound += rte_steps * occurrence_count * math.log1p(step_bound)
    try:
        return float(math.expm1(log_bound))
    except OverflowError:
        return math.inf


def _occurrence_truncation_record(
    parameters: RTEOccurrenceParameters,
    finite_taylor_order: int,
    *,
    allocated_step_error_bound: float | None,
) -> RTEOccurrenceTruncation:
    finite_taylor_order = require_integer_count(
        finite_taylor_order,
        name="finite_taylor_order",
    )
    tau = parameters.lambda_r * parameters.evolution_time / parameters.rte_steps
    step_bound = step_taylor_truncation_residual_bound(tau, finite_taylor_order)
    occurrence_bound = occurrence_truncation_residual_bound(
        step_bound,
        parameters.rte_steps,
    )
    round_bound = occurrence_truncation_residual_bound(
        step_bound,
        parameters.rte_steps * parameters.round_occurrence_count,
    )
    return RTEOccurrenceTruncation(
        occurrence_id=parameters.occurrence_id,
        tail_id=parameters.tail_id,
        tail_hash=parameters.tail_hash,
        lambda_r=parameters.lambda_r,
        evolution_time=parameters.evolution_time,
        rte_steps=parameters.rte_steps,
        finite_taylor_order=finite_taylor_order,
        dimensionless_step_time=float(tau),
        step_truncation_residual_bound=step_bound,
        occurrence_truncation_residual_bound=occurrence_bound,
        round_occurrence_count=parameters.round_occurrence_count,
        round_contribution_residual_bound=round_bound,
        allocated_step_error_bound=allocated_step_error_bound,
    )


def rte_occurrence_truncation_from_config(
    occurrence_id: str,
    config: RTEConfig,
    *,
    round_occurrence_count: int = 1,
) -> RTEOccurrenceTruncation:
    """Build occurrence accounting from an already selected finite RTE config."""
    parameters = RTEOccurrenceParameters(
        occurrence_id=occurrence_id,
        tail_id=config.tail_id,
        tail_hash=config.tail_hash,
        lambda_r=config.lambda_r,
        evolution_time=config.evolution_time,
        rte_steps=config.rte_steps,
        round_occurrence_count=round_occurrence_count,
    )
    return _occurrence_truncation_record(
        parameters,
        config.finite_taylor_order,
        allocated_step_error_bound=None,
    )


def _rpe_round_context(
    rpe_round: RPERound | None,
    occurrence_count: int,
) -> tuple[int, int | None, int | None]:
    if rpe_round is None:
        return 0, None, None
    if occurrence_count != rpe_round.tail_evolutions:
        raise ValueError(
            "Sum of round_occurrence_count values must match "
            "RPERound.tail_evolutions."
        )
    return (
        rpe_round.round_index,
        rpe_round.partial_s2_repetitions,
        rpe_round.tail_evolutions,
    )


def summarize_rpe_round_truncation(
    occurrences: Sequence[RTEOccurrenceTruncation],
    *,
    target_round_truncation_error: float,
    rpe_round: RPERound | None = None,
) -> RPETruncationSummary:
    """Summarize user-selected cutoffs without assuming common occurrence data."""
    if not occurrences:
        raise ValueError("occurrences must not be empty.")
    target = float(target_round_truncation_error)
    if not math.isfinite(target) or target < 0.0:
        raise ValueError(
            "target_round_truncation_error must be finite and non-negative."
        )
    total_occurrences = sum(item.round_occurrence_count for item in occurrences)
    round_index, partial_s2, expected_tail = _rpe_round_context(
        rpe_round,
        total_occurrences,
    )
    total_steps = sum(
        item.rte_steps * item.round_occurrence_count for item in occurrences
    )
    if rpe_round is not None and total_steps != rpe_round.rte_total_steps:
        raise ValueError(
            "Composed short-step count must match RPERound.rte_total_steps."
        )
    total_bound = compose_truncation_residual_bounds(
        tuple(
            (
                item.step_truncation_residual_bound,
                item.rte_steps,
                item.round_occurrence_count,
            )
            for item in occurrences
        )
    )
    budget = RPERoundTruncationBudget(
        round_index=round_index,
        target_round_truncation_error=target,
        allocation_policy="user_selected_orders",
        total_short_step_count=total_steps,
        allocated_log_error_per_short_step=None,
        allocated_step_error_bound=None,
        partial_s2_repetitions=partial_s2,
        expected_tail_evolutions=expected_tail,
    )
    return RPETruncationSummary(
        budget=budget,
        occurrences=tuple(occurrences),
        round_truncation_residual_bound=total_bound,
        meets_round_budget=total_bound <= target,
    )


def select_rpe_round_taylor_orders(
    occurrences: Sequence[RTEOccurrenceParameters],
    *,
    target_round_truncation_error: float,
    rpe_round: RPERound | None = None,
    maximum_order: int = 10_000,
) -> RPETruncationSummary:
    """Choose minimal even cutoffs under equal log budget per short step.

    This is a baseline allocation, not a circuit-cost optimum. Each cutoff is
    selected using the directly evaluated finite scalar Taylor residual.
    """
    if not occurrences:
        raise ValueError("occurrences must not be empty.")
    target = float(target_round_truncation_error)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("target_round_truncation_error must be finite and positive.")
    total_occurrences = sum(item.round_occurrence_count for item in occurrences)
    round_index, partial_s2, expected_tail = _rpe_round_context(
        rpe_round,
        total_occurrences,
    )
    total_short_steps = sum(
        item.rte_steps * item.round_occurrence_count for item in occurrences
    )
    if rpe_round is not None and total_short_steps != rpe_round.rte_total_steps:
        raise ValueError(
            "Composed short-step count must match RPERound.rte_total_steps."
        )
    allocated_log = math.log1p(target) / total_short_steps
    allocated_step_bound = math.expm1(allocated_log)
    records: list[RTEOccurrenceTruncation] = []
    for parameters in occurrences:
        tau = parameters.lambda_r * parameters.evolution_time / parameters.rte_steps
        cutoff = choose_finite_taylor_order(
            tau,
            allocated_step_bound,
            maximum_order=maximum_order,
        )
        records.append(
            _occurrence_truncation_record(
                parameters,
                cutoff,
                allocated_step_error_bound=allocated_step_bound,
            )
        )
    total_bound = compose_truncation_residual_bounds(
        tuple(
            (
                record.step_truncation_residual_bound,
                record.rte_steps,
                record.round_occurrence_count,
            )
            for record in records
        )
    )
    budget = RPERoundTruncationBudget(
        round_index=round_index,
        target_round_truncation_error=target,
        allocation_policy="equal_log_budget_per_short_step",
        total_short_step_count=total_short_steps,
        allocated_log_error_per_short_step=allocated_log,
        allocated_step_error_bound=allocated_step_bound,
        partial_s2_repetitions=partial_s2,
        expected_tail_evolutions=expected_tail,
    )
    meets_budget = total_bound <= target * (1.0 + 1e-14) + 1e-15
    if not meets_budget:
        raise RuntimeError("Selected finite Taylor orders missed the round budget.")
    return RPETruncationSummary(
        budget=budget,
        occurrences=tuple(records),
        round_truncation_residual_bound=total_bound,
        meets_round_budget=True,
    )


def finite_rte_distribution(
    dimensionless_step_time: float,
    finite_taylor_order: int,
) -> RTEFiniteDistribution:
    """Build the finite v2 Eqs. (A23)--(A26) event distribution."""
    finite_taylor_order = require_integer_count(
        finite_taylor_order,
        name="finite_taylor_order",
    )
    if finite_taylor_order % 2:
        raise ValueError("finite_taylor_order must be a non-negative even integer.")
    orders = tuple(range(0, finite_taylor_order + 1, 2))
    weights = tuple(
        _paired_order_weight(dimensionless_step_time, order) for order in orders
    )
    normalization = float(sum(weights))
    probabilities = tuple(weight / normalization for weight in weights)
    tau = abs(float(dimensionless_step_time))
    return RTEFiniteDistribution(
        dimensionless_step_time=float(dimensionless_step_time),
        finite_taylor_order=finite_taylor_order,
        orders=orders,
        unnormalized_order_weights=weights,
        order_probabilities=probabilities,
        exact_finite_distribution=normalization,
        truncation_residual_bound=step_taylor_truncation_residual_bound(
            dimensionless_step_time, finite_taylor_order
        ),
        paper_upper_bound=float(math.exp(tau * tau)),
    )


def make_rte_config(
    tail: RTETailLike,
    *,
    evolution_time: float,
    rte_steps: int,
    truncation_tolerance: float,
    finite_taylor_order: int | None = None,
    seed: int = 0,
) -> tuple[RTEConfig, RTEFiniteDistribution]:
    """Create a self-consistent config and its exact finite distribution."""
    rte_steps = require_integer_count(rte_steps, name="rte_steps", minimum=1)
    if tail.lambda_r == 0.0 or not tail.components:
        raise DeterministicOnlyRTETailError(
            "The tail has no randomized components; finite RTE is not required."
        )
    step_time = float(evolution_time) / rte_steps
    tau = tail.lambda_r * step_time
    cutoff = (
        choose_finite_taylor_order(tau, truncation_tolerance)
        if finite_taylor_order is None
        else require_integer_count(
            finite_taylor_order,
            name="finite_taylor_order",
        )
    )
    distribution = finite_rte_distribution(tau, cutoff)
    config = RTEConfig(
        tail_id=tail.tail_id,
        tail_hash=tail.tail_hash,
        lambda_r=tail.lambda_r,
        evolution_time=float(evolution_time),
        rte_steps=rte_steps,
        step_time=step_time,
        dimensionless_step_time=tau,
        taylor_distribution="rte_even_taylor_paired",
        finite_taylor_order=cutoff,
        truncation_tolerance=float(truncation_tolerance),
        distribution_normalization=distribution.exact_finite_distribution,
        truncation_residual_bound=distribution.step_truncation_residual_bound,
        seed=int(seed),
    )
    return config, distribution


def _basis_reuse_intervals(
    circuit_order_components: Sequence[RTEComponent],
) -> tuple[BasisReuseInterval, ...]:
    intervals: list[BasisReuseInterval] = []
    start = 0
    while start < len(circuit_order_components):
        first = circuit_order_components[start]
        basis_id = first.basis_id
        basis_hash = first.basis_hash
        basis_key = basis_hash or basis_id
        stop = start + 1
        while (
            basis_key is not None
            and stop < len(circuit_order_components)
            and (
                circuit_order_components[stop].basis_hash
                or circuit_order_components[stop].basis_id
            )
            == basis_key
        ):
            stop += 1
        if basis_key is not None and stop - start > 1:
            intervals.append(
                BasisReuseInterval(
                    start,
                    stop,
                    basis_id or basis_key,
                    basis_hash,
                )
            )
        start = stop
    return tuple(intervals)


def _validate_components(components: Sequence[RTEComponent]) -> None:
    if not components:
        raise ValueError("components must not be empty.")
    identifiers = [component.component_id for component in components]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("component_id values must be unique.")
    if not math.isclose(
        sum(component.probability for component in components),
        1.0,
        rel_tol=1e-14,
        abs_tol=1e-14,
    ):
        raise ValueError("component probabilities must sum to one.")


def _make_event(
    component_indices: Sequence[int],
    components: Sequence[RTEComponent],
    distribution: RTEFiniteDistribution,
    order_index: int,
) -> RTEEvent:
    order = distribution.orders[order_index]
    if len(component_indices) != order + 1:
        raise ValueError("An order-n event requires one rotation plus n products.")
    rotation = components[int(component_indices[0])]
    products = tuple(components[int(index)] for index in component_indices[1:])
    component_probability = math.prod(
        components[int(index)].probability for index in component_indices
    )
    coefficient = (
        distribution.unnormalized_order_weights[order_index]
        * component_probability
    )
    event_probability = coefficient / distribution.exact_finite_distribution
    circuit_order = (*products, rotation)
    selected = tuple(component.component_id for component in circuit_order)
    fragment_ids = tuple(component.df_fragment_id for component in circuit_order)
    bases = {component.basis_id for component in circuit_order}
    common_basis = next(iter(bases)) if len(bases) == 1 else None
    basis_hashes = {component.basis_hash for component in circuit_order}
    common_basis_hash = next(iter(basis_hashes)) if len(basis_hashes) == 1 else None
    tau = distribution.dimensionless_step_time
    application_sequence = tuple(
        RTEEventApplication(
            application_index=index,
            role="rotation" if index == len(circuit_order) - 1 else "product",
            component_id=component.component_id,
            coefficient_abs=component.coefficient_abs,
            coefficient_sign=component.coefficient_sign,
            is_identity=component.is_identity,
            df_fragment_id=component.df_fragment_id,
            basis_id=component.basis_id,
            basis_hash=component.basis_hash,
            diagonal_pauli_support=component.diagonal_pauli_support,
            basis_change_operations=component.basis_change_operations,
        )
        for index, component in enumerate(circuit_order)
    )
    return RTEEvent(
        taylor_order=order,
        rotation_component_id=rotation.component_id,
        product_component_ids=tuple(item.component_id for item in products),
        selected_component_ids=selected,
        df_fragment_ids=fragment_ids,
        event_probability=float(event_probability),
        event_coefficient=float(coefficient),
        phase=complex((-1) ** (order // 2)),
        # Pairing degrees n and n+1 algebraically gives
        # I - i*tau*P/(n+1) = sqrt(...) V(+atan(tau/(n+1))).
        # This sign is dense-reconstructed in the tests against v2 (A18)--(A23).
        rotation_angle=float(math.atan(tau / (order + 1))),
        event_normalization=distribution.exact_finite_distribution,
        basis_id=common_basis,
        basis_hash=common_basis_hash,
        basis_reuse_intervals=_basis_reuse_intervals(circuit_order),
        application_sequence=application_sequence,
    )


def enumerate_rte_events(
    components: Sequence[RTEComponent],
    distribution: RTEFiniteDistribution,
    *,
    max_events: int = 1_000_000,
) -> tuple[RTEEvent, ...]:
    """Enumerate all finite events for small systems only."""
    max_events = require_integer_count(max_events, name="max_events")
    _validate_components(components)
    count = sum(len(components) ** (order + 1) for order in distribution.orders)
    if count > max_events:
        raise ValueError(f"Event space has {count} events, above max_events={max_events}.")
    events: list[RTEEvent] = []
    for order_index, order in enumerate(distribution.orders):
        for indices in itertools.product(range(len(components)), repeat=order + 1):
            events.append(_make_event(indices, components, distribution, order_index))
    return tuple(events)


def sample_rte_events(
    components: Sequence[RTEComponent],
    distribution: RTEFiniteDistribution,
    *,
    sample_count: int,
    seed: int,
) -> tuple[RTEEvent, ...]:
    """Classically sample event circuits; this is not quantum shot sampling."""
    sample_count = require_integer_count(sample_count, name="sample_count")
    _validate_components(components)
    rng = np.random.default_rng(int(seed))
    component_probabilities = np.asarray(
        [component.probability for component in components], dtype=float
    )
    events: list[RTEEvent] = []
    for _ in range(sample_count):
        order_index = int(
            rng.choice(
                len(distribution.orders), p=distribution.order_probabilities
            )
        )
        order = distribution.orders[order_index]
        indices = rng.choice(
            len(components), size=order + 1, p=component_probabilities
        )
        events.append(_make_event(indices, components, distribution, order_index))
    return tuple(events)


def event_unitary(
    event: RTEEvent,
    operators: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Materialize ``(-1)^(n/2) V_l(phi_n) P_ln ... P_l1``."""
    rotation_operator = np.asarray(
        operators[event.rotation_component_id], dtype=np.complex128
    )
    identity = np.eye(rotation_operator.shape[0], dtype=np.complex128)
    product = identity
    for component_id in event.product_component_ids:
        product = np.asarray(operators[component_id], dtype=np.complex128) @ product
    rotation = (
        math.cos(event.rotation_angle) * identity
        - 1j * math.sin(event.rotation_angle) * rotation_operator
    )
    return event.phase * rotation @ product


def exact_enumerated_event_mean_operator(
    events: Sequence[RTEEvent],
    operators: Mapping[str, np.ndarray],
    *,
    probability_atol: float = 1e-12,
) -> np.ndarray:
    """Return ``sum_m q_m U_m`` for a complete finite enumeration.

    Repeated event records are allowed and contribute their stated probability
    mass.  The total mass must nevertheless be one; this catches accidentally
    passing Monte Carlo samples to the exact-enumeration API.
    """
    if not events:
        raise ValueError("events must not be empty.")
    if not math.isfinite(probability_atol) or probability_atol < 0.0:
        raise ValueError("probability_atol must be finite and non-negative.")
    probability_sum = math.fsum(float(event.event_probability) for event in events)
    if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=probability_atol):
        raise ValueError(
            "Exact enumerated event probabilities must sum to one; "
            f"received {probability_sum:.17g}."
        )
    if not operators:
        raise ValueError("operators must not be empty.")
    first = next(iter(operators.values()))
    result = np.zeros_like(first, dtype=np.complex128)
    for event in events:
        result += event.event_probability * event_unitary(event, operators)
    return result


def sample_event_mean_operator(
    sampled_events: Sequence[RTEEvent],
    operators: Mapping[str, np.ndarray],
) -> EventOperatorSampleEstimate:
    """Return the unweighted ``1/M`` Monte Carlo event-operator mean.

    Event probabilities are intentionally not multiplied a second time: they
    have already been used to draw ``sampled_events``.
    """
    if not sampled_events:
        raise ValueError("sampled_events must not be empty.")
    unitary_samples = np.stack(
        [event_unitary(event, operators) for event in sampled_events], axis=0
    )
    mean = np.mean(unitary_samples, axis=0)
    sample_count = len(sampled_events)
    if sample_count == 1:
        entrywise_standard_error = np.zeros_like(mean, dtype=float)
    else:
        centered = unitary_samples - mean
        entrywise_variance = np.sum(np.abs(centered) ** 2, axis=0) / (
            sample_count - 1
        )
        entrywise_standard_error = np.sqrt(entrywise_variance / sample_count)
    return EventOperatorSampleEstimate(
        operator_mean=mean,
        entrywise_standard_error=entrywise_standard_error,
        frobenius_standard_error=float(np.linalg.norm(entrywise_standard_error)),
        sample_count=sample_count,
    )


def finite_event_mean_operator(
    events: Sequence[RTEEvent],
    operators: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Deprecated exact-enumeration wrapper.

    Use :func:`exact_enumerated_event_mean_operator` for an enumeration or
    :func:`sample_event_mean_operator` for Monte Carlo samples.
    """
    warnings.warn(
        "finite_event_mean_operator is ambiguous; use the exact-enumeration "
        "or Monte Carlo sample-mean API explicitly.",
        DeprecationWarning,
        stacklevel=2,
    )
    return exact_enumerated_event_mean_operator(events, operators)


def finite_taylor_operator(
    normalized_hamiltonian: np.ndarray,
    dimensionless_step_time: float,
    finite_taylor_order: int,
) -> np.ndarray:
    """Dense paired-Taylor reference retained through degree ``K+1``."""
    finite_taylor_order = require_integer_count(
        finite_taylor_order,
        name="finite_taylor_order",
    )
    if finite_taylor_order % 2:
        raise ValueError("finite_taylor_order must be a non-negative even integer.")
    hamiltonian = np.asarray(normalized_hamiltonian, dtype=np.complex128)
    identity = np.eye(hamiltonian.shape[0], dtype=np.complex128)
    result = identity.copy()
    term = identity.copy()
    for degree in range(1, finite_taylor_order + 2):
        term = term @ ((-1j * dimensionless_step_time / degree) * hamiltonian)
        result += term
    return result


def finite_rte_corrected_operator(
    normalized_hamiltonian: np.ndarray,
    config: RTEConfig,
) -> np.ndarray:
    """Return the normalization-corrected finite Taylor operator for all steps."""
    one_step = finite_taylor_operator(
        normalized_hamiltonian,
        config.dimensionless_step_time,
        config.finite_taylor_order,
    )
    return np.linalg.matrix_power(one_step, config.rte_steps)


def finite_rte_operator_moments(
    normalized_hamiltonian: np.ndarray,
    config: RTEConfig,
) -> RTEOperatorMoments:
    """Return corrected and attenuated operators without normalization ambiguity."""
    corrected = finite_rte_corrected_operator(normalized_hamiltonian, config)
    normalization_product = float(
        math.exp(config.rte_steps * math.log(config.distribution_normalization))
    )
    return RTEOperatorMoments(
        corrected_operator=corrected,
        attenuated_event_mean_operator=corrected / normalization_product,
        normalization_product=normalization_product,
        attenuation_factor=1.0 / normalization_product,
    )


def compose_finite_rte_occurrences(
    occurrences: Sequence[tuple[np.ndarray, RTEConfig]],
) -> RTEOperatorMoments:
    """Compose distinct tail occurrences in the given circuit application order.

    Each occurrence may have its own tail, signed time, integer step count,
    cutoff, and finite normalization.  No common ``B`` is assumed.
    """
    if not occurrences:
        raise ValueError("occurrences must not be empty.")
    first_hamiltonian = np.asarray(occurrences[0][0], dtype=np.complex128)
    corrected = np.eye(first_hamiltonian.shape[0], dtype=np.complex128)
    log_normalization = 0.0
    for normalized_hamiltonian, config in occurrences:
        occurrence_operator = finite_rte_corrected_operator(
            normalized_hamiltonian, config
        )
        if occurrence_operator.shape != corrected.shape:
            raise ValueError("All RTE occurrences must have the same operator shape.")
        corrected = occurrence_operator @ corrected
        log_normalization += config.rte_steps * math.log(
            config.distribution_normalization
        )
    normalization_product = float(math.exp(log_normalization))
    return RTEOperatorMoments(
        corrected_operator=corrected,
        attenuated_event_mean_operator=corrected / normalization_product,
        normalization_product=normalization_product,
        attenuation_factor=1.0 / normalization_product,
    )


def finite_rte_multi_step_operator(
    normalized_hamiltonian: np.ndarray,
    config: RTEConfig,
) -> np.ndarray:
    """Deprecated alias for :func:`finite_rte_corrected_operator`."""
    warnings.warn(
        "finite_rte_multi_step_operator was ambiguously named; use "
        "finite_rte_corrected_operator.",
        DeprecationWarning,
        stacklevel=2,
    )
    return finite_rte_corrected_operator(normalized_hamiltonian, config)


def finite_rte_attenuation(
    config: RTEConfig,
    *,
    tail_evolutions: int = 1,
) -> float:
    """Return the finite-distribution normalization attenuation ``B_K^-r``."""
    tail_evolutions = require_integer_count(
        tail_evolutions,
        name="tail_evolutions",
    )
    exponent = config.rte_steps * tail_evolutions
    return float(math.exp(-exponent * math.log(config.distribution_normalization)))


def finite_rte_combined_attenuation(configs: Sequence[RTEConfig]) -> float:
    """Multiply attenuation for heterogeneous tail occurrences."""
    return float(
        math.exp(
            -math.fsum(
                config.rte_steps * math.log(config.distribution_normalization)
                for config in configs
            )
        )
    )
