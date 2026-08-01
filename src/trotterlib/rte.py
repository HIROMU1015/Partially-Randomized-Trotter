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
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Sequence, TypeAlias

import numpy as np


MeasurementAxis: TypeAlias = Literal["X", "Y"]
TaylorDistributionKind: TypeAlias = Literal["rte_even_taylor_paired"]
FidelityLevel: TypeAlias = Literal[0, 1, 2, 3, 4, 5, 6]


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
        if not self.qubits or any(qubit < 0 for qubit in self.qubits):
            raise ValueError("Basis-change operation qubits must be non-negative.")


@dataclass(frozen=True)
class RTEComponent:
    """One normalized, sign-absorbed Hermitian involution in ``H/lambda``."""

    component_id: str
    probability: float
    coefficient_abs: float
    coefficient_sign: int
    df_fragment_id: str | None = None
    basis_id: str | None = None
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
        if self.finite_taylor_order < 0 or self.finite_taylor_order % 2:
            raise ValueError("finite_taylor_order must be a non-negative even integer.")
        if self.orders != tuple(range(0, self.finite_taylor_order + 1, 2)):
            raise ValueError("orders must contain every even order through the cutoff.")
        if len(self.orders) != len(self.unnormalized_order_weights):
            raise ValueError("order weights length mismatch.")
        if len(self.orders) != len(self.order_probabilities):
            raise ValueError("order probabilities length mismatch.")
        if not math.isclose(sum(self.order_probabilities), 1.0, abs_tol=1e-14):
            raise ValueError("finite Taylor order probabilities must sum to one.")


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
        if self.lambda_r <= 0.0 or not math.isfinite(self.lambda_r):
            raise ValueError("lambda_r must be finite and positive.")
        if self.rte_steps <= 0:
            raise ValueError("rte_steps must be a positive integer.")
        if self.finite_taylor_order < 0 or self.finite_taylor_order % 2:
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
        if self.truncation_residual_bound > self.truncation_tolerance:
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
            self.truncation_residual_bound,
            finite.truncation_residual_bound,
            rel_tol=1e-14,
            abs_tol=1e-15,
        ):
            raise ValueError("truncation_residual_bound does not match the cutoff.")


@dataclass(frozen=True)
class BasisReuseInterval:
    """Maximal adjacent interval sharing a known DF basis in circuit order."""

    start: int
    stop: int
    basis_id: str


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


@dataclass(frozen=True)
class RTEOperatorMoments:
    """Normalization-corrected polynomial and its attenuated event mean."""

    corrected_operator: np.ndarray = field(repr=False, compare=False)
    attenuated_event_mean_operator: np.ndarray = field(repr=False, compare=False)
    normalization_product: float
    attenuation_factor: float


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
    if order < 0 or order % 2:
        raise ValueError("RTE Taylor event order must be non-negative and even.")
    magnitude = abs(float(dimensionless_step_time))
    if magnitude == 0.0:
        return 1.0 if order == 0 else 0.0
    log_taylor = order * math.log(magnitude) - math.lgamma(order + 1)
    return math.exp(log_taylor) * math.hypot(1.0, magnitude / (order + 1))


def taylor_truncation_residual_bound(
    dimensionless_step_time: float,
    finite_taylor_order: int,
) -> float:
    """LCU 1-norm bound for omitted paired events.

    A cutoff at even ``K`` retains ordinary Taylor degrees through ``K+1``.
    Since each omitted paired 2-norm is no larger than the corresponding
    pair's 1-norm, the scalar exponential tail from degree ``K+2`` is a safe
    bound.
    """
    if finite_taylor_order < 0 or finite_taylor_order % 2:
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


def choose_finite_taylor_order(
    dimensionless_step_time: float,
    truncation_tolerance: float,
    *,
    maximum_order: int = 10_000,
) -> int:
    """Choose the smallest even cutoff meeting the one-step residual bound."""
    if truncation_tolerance <= 0.0:
        raise ValueError("truncation_tolerance must be positive.")
    for order in range(0, int(maximum_order) + 1, 2):
        if (
            taylor_truncation_residual_bound(dimensionless_step_time, order)
            <= truncation_tolerance
        ):
            return order
    raise ValueError("No finite Taylor cutoff met the requested tolerance.")


def finite_rte_distribution(
    dimensionless_step_time: float,
    finite_taylor_order: int,
) -> RTEFiniteDistribution:
    """Build the finite v2 Eqs. (A23)--(A26) event distribution."""
    if finite_taylor_order < 0 or finite_taylor_order % 2:
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
        finite_taylor_order=int(finite_taylor_order),
        orders=orders,
        unnormalized_order_weights=weights,
        order_probabilities=probabilities,
        exact_finite_distribution=normalization,
        truncation_residual_bound=taylor_truncation_residual_bound(
            dimensionless_step_time, finite_taylor_order
        ),
        paper_upper_bound=float(math.exp(tau * tau)),
    )


def make_rte_config(
    tail: NormalizedRTETail,
    *,
    evolution_time: float,
    rte_steps: int,
    truncation_tolerance: float,
    finite_taylor_order: int | None = None,
    seed: int = 0,
) -> tuple[RTEConfig, RTEFiniteDistribution]:
    """Create a self-consistent config and its exact finite distribution."""
    if rte_steps <= 0:
        raise ValueError("rte_steps must be positive.")
    step_time = float(evolution_time) / int(rte_steps)
    tau = tail.lambda_r * step_time
    cutoff = (
        choose_finite_taylor_order(tau, truncation_tolerance)
        if finite_taylor_order is None
        else int(finite_taylor_order)
    )
    distribution = finite_rte_distribution(tau, cutoff)
    config = RTEConfig(
        tail_id=tail.tail_id,
        tail_hash=tail.tail_hash,
        lambda_r=tail.lambda_r,
        evolution_time=float(evolution_time),
        rte_steps=int(rte_steps),
        step_time=step_time,
        dimensionless_step_time=tau,
        taylor_distribution="rte_even_taylor_paired",
        finite_taylor_order=cutoff,
        truncation_tolerance=float(truncation_tolerance),
        distribution_normalization=distribution.exact_finite_distribution,
        truncation_residual_bound=distribution.truncation_residual_bound,
        seed=int(seed),
    )
    return config, distribution


def _basis_reuse_intervals(
    circuit_order_components: Sequence[RTEComponent],
) -> tuple[BasisReuseInterval, ...]:
    intervals: list[BasisReuseInterval] = []
    start = 0
    while start < len(circuit_order_components):
        basis_id = circuit_order_components[start].basis_id
        stop = start + 1
        while (
            basis_id is not None
            and stop < len(circuit_order_components)
            and circuit_order_components[stop].basis_id == basis_id
        ):
            stop += 1
        if basis_id is not None and stop - start > 1:
            intervals.append(BasisReuseInterval(start, stop, basis_id))
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
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative.")
    _validate_components(components)
    rng = np.random.default_rng(int(seed))
    component_probabilities = np.asarray(
        [component.probability for component in components], dtype=float
    )
    events: list[RTEEvent] = []
    for _ in range(int(sample_count)):
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
    if tail_evolutions < 0:
        raise ValueError("tail_evolutions must be non-negative.")
    exponent = int(config.rte_steps) * int(tail_evolutions)
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
