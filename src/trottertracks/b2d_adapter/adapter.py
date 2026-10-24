"""Describe G_D = P K**n P† and G_D V_R G_D without building operators.

All returned operations are metadata, in circuit APPEND order. The published
processor sequence is a mathematical product: it must be reversed before
append. Upstream floating-point coefficients are preserved as exact rational
representations of those floats; they are not repaired or reoptimized.

Tail payloads and their phase/normalization metadata remain opaque. This
module neither samples them nor asserts numerical controlled equivalence.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import math
import operator
from typing import Iterator, Sequence, TypeAlias

from ._published import (
    UPSTREAM_COMMIT,
    _iter_s2_steps,
    inverse_s2_sequence,
    morales_2025_yp8m8_kernel_list,
    morales_2025_yp8m8_processor_sequence,
    symmetric_s2_sequence,
)

Scalar: TypeAlias = Fraction | int | float


def _rational(value: Scalar) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (Fraction, int, float)):
        raise TypeError("Use a Fraction, integer, or finite float.")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite time or scalar coefficient.")
    return Fraction(value)


def _positive_integer(value: int) -> int:
    if isinstance(value, bool):
        raise TypeError("A positive integer is required, not bool.")
    count = operator.index(value)
    if count < 1:
        raise ValueError("A positive integer is required.")
    return count


@dataclass(frozen=True)
class ScalarTerm:
    label: str
    coefficient: Fraction

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("A scalar provenance label is required.")
        object.__setattr__(self, "coefficient", _rational(self.coefficient))


@dataclass(frozen=True)
class DContext:
    preparation_id: str
    tail_id: str
    block_ids: tuple[str, ...]
    num_system_qubits: int
    scalar_terms: tuple[ScalarTerm, ...] = ()
    controlled: bool = False
    ancilla_qubit: int | None = None

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (self.preparation_id, self.tail_id)
        ):
            raise ValueError("Preparation and tail identities are required.")
        if isinstance(self.block_ids, str):
            raise TypeError("Pass an ordered sequence of block labels, not a string.")
        blocks = tuple(self.block_ids)
        if not blocks or any(not isinstance(x, str) or not x for x in blocks):
            raise ValueError("Nonempty ordered D block labels are required.")
        if len(set(blocks)) != len(blocks):
            raise ValueError("D block labels must be unique.")
        object.__setattr__(self, "block_ids", blocks)
        object.__setattr__(
            self, "num_system_qubits", _positive_integer(self.num_system_qubits)
        )
        terms = tuple(self.scalar_terms)
        if any(not isinstance(t, ScalarTerm) for t in terms):
            raise TypeError("scalar_terms must contain ScalarTerm records.")
        if len({t.label for t in terms}) != len(terms):
            raise ValueError("Scalar provenance labels must be unique.")
        object.__setattr__(self, "scalar_terms", terms)
        if not isinstance(self.controlled, bool):
            raise TypeError("controlled must be bool.")
        if self.controlled:
            if isinstance(self.ancilla_qubit, bool) or self.ancilla_qubit is None:
                raise ValueError("A separate control ancilla is required.")
            ancilla = operator.index(self.ancilla_qubit)
            if ancilla < self.num_system_qubits:
                raise ValueError("Control ancilla overlaps the system register.")
            object.__setattr__(self, "ancilla_qubit", ancilla)
        elif self.ancilla_qubit is not None:
            raise ValueError("Uncontrolled context must not specify an ancilla.")


def context_from_preparation(
    preparation: object, *, controlled: bool = False, ancilla_qubit: int | None = None
) -> DContext:
    """Read existing DFPartialS2Preparation metadata, without importing it."""
    return DContext(
        preparation_id=preparation.preparation_hash,
        tail_id=preparation.tail_extraction.tail_hash,
        block_ids=tuple(preparation.deterministic_block_order),
        num_system_qubits=preparation.num_system_qubits,
        scalar_terms=(
            ScalarTerm("hamiltonian_constant", preparation.constant_coefficient),
            ScalarTerm(
                "extracted_tail_identity", preparation.extracted_identity_coefficient
            ),
        ),
        controlled=controlled,
        ancilla_qubit=ancilla_qubit,
    )


@dataclass(frozen=True)
class BlockExponential:
    block_id: str
    evolution_time: Fraction
    controlled: bool
    ancilla_qubit: int | None


def actual_adjoint(
    append_operations: Sequence[BlockExponential],
) -> tuple[BlockExponential, ...]:
    """Reverse the *realized descriptor list* and negate every signed time.

    Only D exponentials are accepted. An opaque tail is not a fresh draw of
    an inverse. Numerical gate/basis/phase equivalence remains untested.
    """
    operations = tuple(append_operations)
    if any(not isinstance(op, BlockExponential) for op in operations):
        raise TypeError("Only deterministic block descriptors have this adjoint.")
    return tuple(
        replace(op, evolution_time=-op.evolution_time)
        for op in reversed(operations)
    )


def published_math_sequences() -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    """Return (kernel, processor) in mathematical product order.

    This is the YP8m8 branch of the fixed upstream sequence construction.
    It uses the upstream generator bodies, not a new PF coefficient solver.
    """
    kernel = symmetric_s2_sequence(morales_2025_yp8m8_kernel_list())
    processor = morales_2025_yp8m8_processor_sequence()
    return tuple(map(_rational, kernel)), tuple(map(_rational, processor))


def _expand_append_s2(
    context: DContext, weights: Sequence[Fraction], tau: Fraction
) -> tuple[BlockExponential, ...]:
    # No merging across S2 blocks: avoid altering the upstream floating-point
    # weights or losing the exact correspondence of an actual inverse.
    return tuple(
        BlockExponential(
            context.block_ids[index],
            tau * weight,
            context.controlled,
            context.ancilla_qubit,
        )
        for stage in weights
        for index, weight in _iter_s2_steps(len(context.block_ids), stage)
    )


@dataclass(frozen=True)
class DRun:
    context: DContext
    tau: Fraction
    kernel_repetitions: int

    def __post_init__(self) -> None:
        if not isinstance(self.context, DContext):
            raise TypeError("DContext is required.")
        object.__setattr__(self, "tau", _rational(self.tau))
        object.__setattr__(
            self, "kernel_repetitions", _positive_integer(self.kernel_repetitions)
        )

    @property
    def nominal_time(self) -> Fraction:
        return self.tau * self.kernel_repetitions

    def mathematical_s2_weights(self) -> tuple[Fraction, ...]:
        kernel, processor = published_math_sequences()
        return (
            processor
            + kernel * self.kernel_repetitions
            + tuple(inverse_s2_sequence(processor))
        )

    def append_operations(self) -> tuple[BlockExponential, ...]:
        kernel, processor = published_math_sequences()
        p_append = _expand_append_s2(
            self.context, tuple(reversed(processor)), self.tau
        )
        k_append = _expand_append_s2(self.context, tuple(reversed(kernel)), self.tau)
        # Rightmost mathematical P† acts first. The ending P is the same
        # realization whose actual adjoint was placed at the beginning.
        return actual_adjoint(p_append) + k_append * self.kernel_repetitions + p_append


@dataclass(frozen=True)
class TailOccurrence:
    context: DContext
    occurrence_id: str
    evolution_time: Fraction
    payload: object
    event_phase_metadata: object
    normalization_metadata: object

    def __post_init__(self) -> None:
        if (
            not isinstance(self.context, DContext)
            or not isinstance(self.occurrence_id, str)
            or not self.occurrence_id
        ):
            raise ValueError("Tail context and occurrence identity are required.")
        if any(
            x is None
            for x in (
                self.payload, self.event_phase_metadata, self.normalization_metadata
            )
        ):
            raise ValueError(
                "Opaque tail payload, phase, and normalization must be explicit."
            )
        object.__setattr__(self, "evolution_time", _rational(self.evolution_time))


@dataclass(frozen=True)
class PhaseOperation:
    term: ScalarTerm
    evolution_time: Fraction
    controlled: bool
    ancilla_qubit: int | None

    @property
    def phase_argument(self) -> Fraction:
        """Argument of exp(i*argument); never evaluate the exponential here."""
        return -self.evolution_time * self.term.coefficient

    @property
    def destination(self) -> str:
        return "relative_ancilla_phase" if self.controlled else "global_phase"


RunItem: TypeAlias = DRun | TailOccurrence
AppendOperation: TypeAlias = BlockExponential | TailOccurrence | PhaseOperation


def fuse_adjacent_d_runs(items: Sequence[RunItem]) -> tuple[RunItem, ...]:
    """Fuse only adjacent identical-context, identical-tau D runs.

    An intervening tail is an unconditional barrier. Reuse the old P(tau)
    and add kernel repetitions; never construct P(2*tau).
    """
    result: list[RunItem] = []
    for item in items:
        if not isinstance(item, (DRun, TailOccurrence)):
            raise TypeError("Only D runs and opaque tails are supported.")
        if result and isinstance(item, DRun) and isinstance(result[-1], DRun):
            previous = result[-1]
            if previous.context == item.context and previous.tau == item.tau:
                result[-1] = replace(
                    previous,
                    kernel_repetitions=(
                        previous.kernel_repetitions + item.kernel_repetitions
                    ),
                )
                continue
        result.append(item)
    return tuple(result)


@dataclass(frozen=True)
class StructuralCostMetadata:
    kernel_repetitions: int
    processor_pair_count: int
    payment_unit: str = "per_contiguous_D_run_per_execution"
    price_status: str = "NO_NATIVE_COST_OR_SHOT_ESTIMATE"


@dataclass(frozen=True)
class B2DPlan:
    body: tuple[RunItem, ...]
    scalar_phases: tuple[PhaseOperation, ...]
    step_time: Fraction
    repetition_count: int
    n_d: int
    sequence_source_commit: str = UPSTREAM_COMMIT

    @property
    def structural_cost(self) -> StructuralCostMetadata:
        runs = [item for item in self.body if isinstance(item, DRun)]
        return StructuralCostMetadata(
            kernel_repetitions=sum(run.kernel_repetitions for run in runs),
            processor_pair_count=len(runs),
        )

    def iter_append_operations(self) -> Iterator[AppendOperation]:
        # Central scalar phases commute, so pay them once for total physical
        # time. They are not multiplied by processor/kernel S2 stage counts.
        yield from self.scalar_phases
        for item in self.body:
            if isinstance(item, DRun):
                yield from item.append_operations()
            else:
                yield item


def make_b2d_plan(
    context: DContext,
    *,
    step_time: Scalar,
    n_d: int,
    tail_occurrences: Sequence[TailOccurrence],
    fuse_boundaries: bool = False,
) -> B2DPlan:
    """Bind unchanged tails between same-direction processed D half runs.

    Each occurrence must already belong to the same preparation, tail, time,
    and control convention. Metadata binding is not scientific validation of
    an opaque payload. No sampler, state, operator, or circuit is created.
    """
    if not isinstance(context, DContext) or not isinstance(fuse_boundaries, bool):
        raise TypeError("DContext and a bool boundary policy are required.")
    delta = _rational(step_time)
    n_d = _positive_integer(n_d)
    tails = tuple(tail_occurrences)
    if not tails:
        raise ValueError("This adapter requires at least one explicit tail occurrence.")
    if any(not isinstance(t, TailOccurrence) for t in tails):
        raise TypeError("TailOccurrence records are required.")
    if len({t.occurrence_id for t in tails}) != len(tails):
        raise ValueError("Each tail occurrence needs a distinct identity.")
    if any(t.context != context or t.evolution_time != delta for t in tails):
        raise ValueError("Tail preparation/time/control metadata do not match.")
    half = DRun(context, delta / (2 * n_d), n_d)
    items = tuple(item for tail in tails for item in (half, tail, half))
    body = fuse_adjacent_d_runs(items) if fuse_boundaries else items
    phases = tuple(
        PhaseOperation(
            term, delta * len(tails), context.controlled, context.ancilla_qubit
        )
        for term in context.scalar_terms
    )
    return B2DPlan(body, phases, delta, len(tails), n_d)
