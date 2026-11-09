"""Exact occupation grids and phase-aware threshold-reflection gate plans.

Construction only: no molecular loader, dense operator, simulator or compiler.
The RTE bridge uses the existing paired order law and event data classes, but
keeps the potentially enormous threshold dictionary implicit. Gaussian frames
are supplied by an existing provider; they are never inferred from a name.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
from functools import reduce
from importlib import import_module
import json
import math
from operator import index
from typing import Any, Callable, Iterable, Literal, Protocol, Sequence


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    result = index(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def exact(value: Any) -> F:
    """Interpret a finite stored float by its exact ratio, without rounding."""
    if isinstance(value, bool):
        raise TypeError("Boolean coefficients are not supported")
    if hasattr(value, "as_integer_ratio"):
        return F(*value.as_integer_ratio())
    return F(value)


def _fraction_gcd(values: Iterable[F]) -> F:
    values = tuple(abs(v) for v in values if v)
    if not values:
        return F(1)
    denominator = math.lcm(*(v.denominator for v in values))
    numerator = reduce(math.gcd, (int(v * denominator) for v in values))
    return F(numerator, denominator)


def _bits(values: Sequence[int], count: int) -> tuple[int, ...]:
    values = tuple(_integer(v, "occupation") for v in values)
    if len(values) != count or any(v > 1 for v in values):
        raise ValueError("occupation must contain one bit per mode")
    return values


@dataclass(frozen=True)
class OccupationPolynomial:
    """constant + sum coefficient * product(n_i), of degree at most two."""

    num_modes: int
    constant: F = F(0)
    terms: tuple[tuple[tuple[int, ...], F], ...] = ()

    def __post_init__(self) -> None:
        n = _integer(self.num_modes, "num_modes", 1)
        constant = exact(self.constant)
        collected: dict[tuple[int, ...], F] = {}
        for support, coefficient in self.terms:
            # n_i^2 = n_i, including full-ij number-CDF diagonal entries.
            support = tuple(sorted(set(_integer(i, "mode") for i in support)))
            if len(support) > 2 or any(i >= n for i in support):
                raise ValueError("only valid linear/quadratic occupation terms are supported")
            coefficient = exact(coefficient)
            if support:
                collected[support] = collected.get(support, F(0)) + coefficient
            else:
                constant += coefficient
        object.__setattr__(self, "num_modes", n)
        object.__setattr__(self, "constant", constant)
        object.__setattr__(self, "terms", tuple(sorted((s, c) for s, c in collected.items() if c)))

    @classmethod
    def from_number_cdf(cls, matrix: Sequence[Sequence[Any]], *, constant: Any = 0) -> OccupationPolynomial:
        """Use the full-ij convention sum J_ij n_i n_j, not upper-triangle J."""
        rows = tuple(tuple(row) for row in matrix)
        if not rows or any(len(row) != len(rows) for row in rows):
            raise ValueError("number-CDF matrix must be nonempty and square")
        return cls(len(rows), exact(constant), tuple(
            ((i, j), exact(value)) for i, row in enumerate(rows) for j, value in enumerate(row)
        ))

    def value(self, occupation: Sequence[int]) -> F:
        bits = _bits(occupation, self.num_modes)
        return self.constant + sum((c * math.prod(bits[i] for i in s) for s, c in self.terms), F(0))


@dataclass(frozen=True)
class ThresholdGrid:
    """f = lower + spacing*z, with a full-Fock integer enclosure 0 <= z <= M.

    General quadratic grids use termwise bounds. A DF square instead keeps an
    integer weighted charge, whose square is evaluated reversibly without an
    occupation-spectrum table. No fixed-particle-number assumption is made.
    """

    num_modes: int
    lower: F
    spacing: F
    extent: int
    origin: int
    terms: tuple[tuple[tuple[int, ...], int], ...] = ()
    charge_weights: tuple[int, ...] | None = None
    square_sign: int = 1

    def __post_init__(self) -> None:
        n = _integer(self.num_modes, "num_modes", 1)
        lower, spacing = exact(self.lower), exact(self.spacing)
        extent, origin = _integer(self.extent, "extent"), _integer(self.origin, "origin")
        if spacing <= 0:
            raise ValueError("spacing must be positive")
        if self.square_sign not in (-1, 1):
            raise ValueError("square_sign must be -1 or +1")
        if self.charge_weights is not None:
            weights = tuple(index(v) for v in self.charge_weights)
            if len(weights) != n or self.terms:
                raise ValueError("a square grid needs exactly one charge weight per mode")
            lo, hi = sum(v for v in weights if v < 0), sum(v for v in weights if v > 0)
            bound = max(lo * lo, hi * hi)
            if extent != bound or origin != (0 if self.square_sign == 1 else bound):
                raise ValueError("square grid enclosure is inconsistent")
            object.__setattr__(self, "charge_weights", weights)
        else:
            canonical: dict[tuple[int, ...], int] = {}
            for support, value in self.terms:
                support = tuple(_integer(i, "mode") for i in support)
                if not 1 <= len(support) <= 2 or tuple(sorted(set(support))) != support or support[-1] >= n:
                    raise ValueError("invalid integer monomial support")
                canonical[support] = canonical.get(support, 0) + index(value)
            terms = tuple(sorted((s, v) for s, v in canonical.items() if v))
            if origin + sum(v for _, v in terms if v < 0) < 0 or origin + sum(v for _, v in terms if v > 0) > extent:
                raise ValueError("termwise enclosure is insufficient")
            object.__setattr__(self, "terms", terms)
        object.__setattr__(self, "num_modes", n)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "spacing", spacing)
        object.__setattr__(self, "extent", extent)
        object.__setattr__(self, "origin", origin)

    @classmethod
    def from_polynomial(cls, polynomial: OccupationPolynomial) -> ThresholdGrid:
        step = _fraction_gcd(c for _, c in polynomial.terms)
        terms = tuple((s, int(c / step)) for s, c in polynomial.terms)
        negative = sum(v for _, v in terms if v < 0)
        return cls(polynomial.num_modes, polynomial.constant + step * negative,
                   step, sum(abs(v) for _, v in terms), -negative, terms)

    @classmethod
    def from_df_square(cls, eta: Sequence[Any], lam: Any, *, constant: Any = 0) -> ThresholdGrid:
        """Exact stored-factor target constant + lam*(sum eta_i*n_i)^2.

        Products here are exact rational products of the stored factors. They
        need not be bit-identical to a previously rounded float coefficient map.
        """
        eta = tuple(exact(v) for v in eta)
        if not eta:
            raise ValueError("DF square needs at least one mode")
        lam, constant = exact(lam), exact(constant)
        if not lam or not any(eta):
            return cls(len(eta), constant, F(1), 0, 0)
        unit = _fraction_gcd(eta)
        weights = tuple(int(v / unit) for v in eta)
        lo, hi = sum(v for v in weights if v < 0), sum(v for v in weights if v > 0)
        extent, step = max(lo * lo, hi * hi), abs(lam) * unit * unit
        sign = 1 if lam > 0 else -1
        return cls(len(eta), constant if sign == 1 else constant - step * extent,
                   step, extent, 0 if sign == 1 else extent, charge_weights=weights, square_sign=sign)

    @classmethod
    def from_df_block(cls, block: Any, *, constant: Any = 0) -> ThresholdGrid:
        """Consume existing DFBlock eta/lam fields; do not load molecular data."""
        return cls.from_df_square(block.eta, block.lam, constant=constant)

    @property
    def center(self) -> F:
        return self.lower + self.spacing * self.extent / 2

    @property
    def reflection_mass(self) -> F:
        return self.spacing * self.extent / 2

    @property
    def value_width(self) -> int:
        return max(1, self.extent.bit_length())

    @property
    def charge_width(self) -> int:
        if self.charge_weights is None or not self.extent:
            return 0
        lo = sum(v for v in self.charge_weights if v < 0)
        hi = sum(v for v in self.charge_weights if v > 0)
        return max(1, hi.bit_length() + 1, (~lo).bit_length() + 1 if lo < 0 else 1)

    def value_integer(self, occupation: Sequence[int]) -> int:
        bits = _bits(occupation, self.num_modes)
        if self.charge_weights is not None:
            charge = sum(v * bit for v, bit in zip(self.charge_weights, bits))
            return self.origin + self.square_sign * charge * charge
        return self.origin + sum(v * math.prod(bits[i] for i in s) for s, v in self.terms)

    def value(self, occupation: Sequence[int]) -> F:
        return self.lower + self.spacing * self.value_integer(occupation)


def uniform_below(bound: int, randbits: Callable[[int], int]) -> int:
    """Exact rejection from unbiased classical bits. No finite retry fallback.

    A caller may interrupt and report failure, but must not return a biased draw.
    Tests supply scripted bits; this module does not start a trajectory run.
    """
    bound = _integer(bound, "bound", 1)
    width = (bound - 1).bit_length()
    if not width:
        return 0
    while True:
        draw = _integer(randbits(width), "random bits")
        if draw >= 1 << width:
            raise ValueError("randbits returned a value outside its requested width")
        if draw < bound:
            return draw


@dataclass(frozen=True)
class ThresholdFragment:
    fragment_id: str
    grid: ThresholdGrid
    basis_id: str | None = None
    basis_hash: str | None = None


@dataclass(frozen=True)
class ThresholdAtom:
    fragment: ThresholdFragment | None
    threshold: int | None = None
    identity_sign: int = 1

    def __post_init__(self) -> None:
        if self.fragment is None:
            if self.threshold is not None or self.identity_sign not in (-1, 1):
                raise ValueError("invalid scalar identity atom")
        else:
            if self.identity_sign != 1:
                raise ValueError("a threshold atom carries its sign in the predicate")
            if not 1 <= _integer(self.threshold, "threshold", 1) <= self.fragment.grid.extent:
                raise ValueError("threshold is outside 1..M")

    @property
    def component_id(self) -> str:
        return "threshold:identity" if self.fragment is None else "threshold:" + json.dumps(
            [self.fragment.fragment_id, format(self.threshold, "x")], separators=(",", ":")
        )


ScalarPolicy = Literal["faithful_identity_in_tail", "extract_identity_phase"]


@dataclass(frozen=True)
class RTEOrderParameters:
    exact_dimensionless_time: F
    time_conversion_error: F
    distribution: Any


@dataclass(frozen=True)
class ThresholdTail:
    fragments: tuple[ThresholdFragment, ...]
    identity_policy: ScalarPolicy = "faithful_identity_in_tail"

    def __post_init__(self) -> None:
        fragments = tuple(self.fragments)
        ids = [f.fragment_id for f in fragments]
        if not fragments or any(not s for s in ids) or len(set(ids)) != len(ids):
            raise ValueError("fragments need distinct nonempty IDs")
        if len({f.grid.num_modes for f in fragments}) != 1:
            raise ValueError("all fragments must use the same system register")
        for fragment in fragments:
            if fragment.basis_id == "" or (fragment.basis_hash is not None and fragment.basis_id is None):
                raise ValueError("a basis hash needs a nonempty provider basis ID")
        if self.identity_policy not in ("faithful_identity_in_tail", "extract_identity_phase"):
            raise ValueError("unknown identity policy")
        object.__setattr__(self, "fragments", fragments)

    @property
    def scalar(self) -> F:
        return sum((f.grid.center for f in self.fragments), F(0))

    @property
    def deterministic_scalar(self) -> F:
        return self.scalar if self.identity_policy == "extract_identity_phase" else F(0)

    @property
    def mass(self) -> F:
        return sum((f.grid.reflection_mass for f in self.fragments), F(0)) + (
            abs(self.scalar) if self.identity_policy == "faithful_identity_in_tail" else F(0)
        )

    def identity_atom(self) -> ThresholdAtom:
        if self.identity_policy != "faithful_identity_in_tail" or not self.scalar:
            raise ValueError("this tail has no sampled identity")
        return ThresholdAtom(None, identity_sign=1 if self.scalar > 0 else -1)

    def coefficient(self, atom: ThresholdAtom) -> F:
        if atom.fragment is None:
            if atom != self.identity_atom():
                raise ValueError("identity sign disagrees with tail")
            return abs(self.scalar)
        if atom.fragment not in self.fragments:
            raise ValueError("atom does not belong to this tail")
        return atom.fragment.grid.spacing / 2

    def probability(self, atom: ThresholdAtom) -> F:
        return self.coefficient(atom) / self.mass

    def draw_atom(self, randbits: Callable[[int], int]) -> ThresholdAtom:
        """Draw a group by exact rational mass, then uniform J; never expand M."""
        groups: list[tuple[ThresholdFragment | None, F]] = [
            (f, f.grid.reflection_mass) for f in self.fragments if f.grid.extent
        ]
        if self.identity_policy == "faithful_identity_in_tail" and self.scalar:
            groups.append((None, abs(self.scalar)))
        if not groups:
            raise ValueError("deterministic-only tail has no RTE draws")
        unit = _fraction_gcd(v for _, v in groups)
        weights = [int(v / unit) for _, v in groups]
        selected = uniform_below(sum(weights), randbits)
        for (fragment, _), weight in zip(groups, weights):
            if selected < weight:
                return self.identity_atom() if fragment is None else ThresholdAtom(
                    fragment, 1 + uniform_below(fragment.grid.extent, randbits)
                )
            selected -= weight
        raise AssertionError("unreachable group selection")

    def order_parameters(self, step_time: Any, cutoff: int, *, rte_api: Any = None) -> RTEOrderParameters:
        """Reuse the canonical order law; scalar extraction changes its target.

        Building a distribution for scientific inputs remains a scientific
        evaluation. This construction helper grants no execution permission.
        """
        if not self.mass:
            raise ValueError("deterministic-only tail needs no RTE distribution")
        api = rte_api if rte_api is not None else import_module("trotterlib.rte")
        tau = self.mass * exact(step_time)
        tau_float = _float_nonzero(tau)
        return RTEOrderParameters(tau, abs(exact(tau_float) - tau),
                                  api.finite_rte_distribution(tau_float, cutoff))


def _float_nonzero(value: F) -> float:
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError("value is not representable in the existing float RTE interface") from exc
    if not math.isfinite(result) or (value and not result):
        raise ValueError("value would overflow or silently underflow in the float RTE interface")
    return result


@dataclass(frozen=True)
class BoundRTEEvent:
    event: Any
    rotation_first_atoms: tuple[ThresholdAtom, ...]
    conditional_atom_probability: F
    event_probability_roundoff: F
    event_coefficient_roundoff: F


def bind_rte_event(tail: ThresholdTail, rotation_first_atoms: Sequence[ThresholdAtom],
                   distribution: Any, order_index: int, *, rte_api: Any = None) -> BoundRTEEvent:
    """Bind already drawn atoms to the unchanged RTEEvent schema.

    The old _make_event sums the entire component list, so it is unsuitable for
    an implicit dictionary. Reuse its event/application classes and basis-run
    convention instead. Do not pretend that just the drawn atoms sum to one.
    No order draw, signal calculation or circuit compilation happens here.
    """
    api = rte_api if rte_api is not None else import_module("trotterlib.rte")
    oi = _integer(order_index, "order_index")
    order = distribution.orders[oi]
    atoms = tuple(rotation_first_atoms)
    if len(atoms) != order + 1 or order % 2:
        raise ValueError("paired order needs one rotation plus an even number of products")
    components = []
    probability = F(1)
    for atom in atoms:
        p = tail.probability(atom)
        probability *= p
        f = atom.fragment
        components.append(api.RTEComponent(
            component_id=atom.component_id, probability=_float_nonzero(p),
            coefficient_abs=_float_nonzero(tail.coefficient(atom)),
            coefficient_sign=atom.identity_sign if f is None else 1,
            is_identity=f is None, df_fragment_id=None if f is None else f.fragment_id,
            basis_id=None if f is None else f.basis_id,
            basis_hash=None if f is None else f.basis_hash,
            # A threshold predicate is not an I/Z/ZZ support.
            diagonal_pauli_support=None, basis_change_operations=(),
        ))
    coefficient = exact(distribution.unnormalized_order_weights[oi]) * probability
    normalization = exact(distribution.exact_finite_distribution)
    event_probability = coefficient / normalization
    products, rotation = tuple(components[1:]), components[0]
    circuit_order = (*products, rotation)
    applications = tuple(api.RTEEventApplication(
        application_index=i, role="rotation" if i == order else "product",
        component_id=c.component_id, coefficient_abs=c.coefficient_abs,
        coefficient_sign=c.coefficient_sign, is_identity=c.is_identity,
        df_fragment_id=c.df_fragment_id, basis_id=c.basis_id, basis_hash=c.basis_hash,
        diagonal_pauli_support=None, basis_change_operations=(),
    ) for i, c in enumerate(circuit_order))
    bases, hashes = {c.basis_id for c in circuit_order}, {c.basis_hash for c in circuit_order}
    fp, fc = _float_nonzero(event_probability), _float_nonzero(coefficient)
    event = api.RTEEvent(
        taylor_order=order, rotation_component_id=rotation.component_id,
        product_component_ids=tuple(c.component_id for c in products),
        selected_component_ids=tuple(c.component_id for c in circuit_order),
        df_fragment_ids=tuple(c.df_fragment_id for c in circuit_order),
        event_probability=fp, event_coefficient=fc, phase=complex((-1) ** (order // 2)),
        rotation_angle=math.atan(distribution.dimensionless_step_time / (order + 1)),
        event_normalization=float(normalization), basis_id=next(iter(bases)) if len(bases) == 1 else None,
        basis_hash=next(iter(hashes)) if len(hashes) == 1 else None,
        basis_reuse_intervals=api._basis_reuse_intervals(circuit_order),
        application_sequence=applications,
    )
    return BoundRTEEvent(event, atoms, probability, abs(exact(fp) - event_probability), abs(exact(fc) - coefficient))


@dataclass(frozen=True)
class Angle:
    """An exact pi multiple plus a stored rational radian value."""

    pi: F = F(0)
    radians: F = F(0)

    def scaled(self, factor: int) -> Angle:
        return Angle(self.pi * factor, self.radians * factor)


@dataclass(frozen=True)
class Instruction:
    kind: Literal["x", "phase", "basis"]
    controls: tuple[int, ...] = ()
    target: int | None = None
    angle: Angle = Angle()
    basis_id: str | None = None
    basis_hash: str | None = None
    inverse: bool = False
    system: tuple[int, ...] = ()


class CircuitSink(Protocol):
    def x(self, target: int, controls: tuple[int, ...]) -> None: ...
    def phase(self, angle: Angle, controls: tuple[int, ...]) -> None: ...
    def basis(self, basis_id: str, basis_hash: str | None, inverse: bool, system: tuple[int, ...]) -> None: ...


@dataclass(frozen=True)
class ReflectionPlan:
    num_qubits: int
    system: tuple[int, ...]
    value: tuple[int, ...]
    charge: tuple[int, ...]
    flag: int | None
    control: int | None
    instructions: tuple[Instruction, ...]

    def emit(self, sink: CircuitSink, *, qubit_map: Sequence[int] | None = None) -> None:
        mapping = tuple(range(self.num_qubits)) if qubit_map is None else tuple(qubit_map)
        if len(mapping) != self.num_qubits or len(set(mapping)) != len(mapping):
            raise ValueError("qubit map must be one-to-one and cover the plan")
        for op in self.instructions:
            controls = tuple(mapping[i] for i in op.controls)
            if op.kind == "x":
                sink.x(mapping[op.target], controls)
            elif op.kind == "phase":
                sink.phase(op.angle, controls)
            else:
                sink.basis(op.basis_id, op.basis_hash, op.inverse, tuple(mapping[i] for i in op.system))

    def structural_counts(self) -> dict[str, Any]:
        """Logical MCX arities, not decomposed native prices or T estimates."""
        arities: dict[int, int] = {}
        for op in self.instructions:
            if op.kind == "x":
                arities[len(op.controls)] = arities.get(len(op.controls), 0) + 1
        return {"system_qubits": len(self.system), "value_qubits": len(self.value),
                "charge_qubits": len(self.charge), "comparison_flag": int(self.flag is not None),
                "control_qubits": int(self.control is not None), "x_by_control_arity": arities,
                "phase_operations": sum(op.kind == "phase" for op in self.instructions),
                "basis_calls": sum(op.kind == "basis" for op in self.instructions)}


def _add_constant(register: tuple[int, ...], value: int, controls: tuple[int, ...] = ()) -> list[Instruction]:
    """Controlled modular constant add. Port of N2 increment's reversible ladder.

    N2: ae743445.../representation_exploration/algorithm_design.py:increment.
    Generalized to arbitrary constants and one/two occupation controls; no
    floating fit, native Qiskit construction, or hidden MCX decomposition.
    """
    operations = []
    for bit in range(len(register)):
        if not (abs(value) >> bit) & 1:
            continue
        sub = register[bit:]
        ladder = [Instruction("x", controls + sub[:i], sub[i]) for i in reversed(range(1, len(sub)))]
        ladder.append(Instruction("x", controls, sub[0]))
        operations.extend(ladder if value >= 0 else reversed(ladder))
    return operations


def _evaluate(grid: ThresholdGrid, system: tuple[int, ...], value: tuple[int, ...],
              charge: tuple[int, ...]) -> list[Instruction]:
    # The charge stays live through comparison and is cleared by the outer
    # inverse after the phase. Computing it twice within each direction is wasteful.
    operations = _add_constant(value, grid.origin)
    if grid.charge_weights is None:
        for support, coefficient in grid.terms:
            operations.extend(_add_constant(value, coefficient, tuple(system[i] for i in support)))
        return operations
    compute = []
    for qubit, coefficient in zip(system, grid.charge_weights):
        compute.extend(_add_constant(charge, coefficient, (qubit,)))
    operations.extend(compute)
    signed_bits = [(q, -(1 << i) if i == len(charge) - 1 else 1 << i) for i, q in enumerate(charge)]
    for i, (q, weight) in enumerate(signed_bits):
        operations.extend(_add_constant(value, grid.square_sign * weight * weight, (q,)))
        for r, other in signed_bits[i + 1:]:
            operations.extend(_add_constant(value, grid.square_sign * 2 * weight * other, (q, r)))
    return operations


def _compare(value: tuple[int, ...], flag: int, threshold: int) -> list[Instruction]:
    """XOR [z >= J] into flag, restore z; flag may start in either bit state."""
    return (_add_constant((*value, flag), -threshold) + [Instruction("x", target=flag)]
            + _add_constant(value, threshold))


def reflection_plan(atom: ThresholdAtom, *, num_modes: int | None = None,
                    angle: Any = None, controlled: bool = True) -> ReflectionPlan:
    """Plan R (angle=None), or exp(-i angle R), with clean arithmetic workspace.

    The product reflection is (-1)*(-1)^predicate. Its pi phase is relative
    when controlled. A rotation is exp(+i angle)*exp(-2i angle*predicate).
    Gaussian inverse is applied first, then the actual forward transformation.
    """
    if atom.fragment is not None:
        if num_modes is not None and atom.fragment.grid.num_modes != _integer(num_modes, "num_modes", 1):
            raise ValueError("system register size disagrees with fragment")
        return _threshold_run_plan((atom,), (angle,), controlled=controlled)
    n = _integer(num_modes, "num_modes", 1)
    control = n if controlled else None
    phase = Angle(pi=F(atom.identity_sign == -1)) if angle is None else Angle(radians=-exact(angle) * atom.identity_sign)
    return ReflectionPlan(n + int(controlled), tuple(range(n)), (), (), None, control,
                          (Instruction("phase", () if control is None else (control,), angle=phase),))


def _threshold_run_plan(atoms: Sequence[ThresholdAtom], angles: Sequence[Any], *, controlled: bool) -> ReflectionPlan:
    """Share value computation and a full Gaussian frame for one exact fragment.

    Only consecutive occurrences of the identical fragment are combined. This
    is existing compute/uncompute and basis cancellation, not a new sampler.
    """
    fragment = atoms[0].fragment
    if fragment is None or len(atoms) != len(angles) or any(a.fragment != fragment for a in atoms):
        raise ValueError("a shared run must use one identical threshold fragment")
    grid, n = fragment.grid, fragment.grid.num_modes
    system = tuple(range(n))
    value = tuple(range(n, n + grid.value_width))
    charge = tuple(range(n + len(value), n + len(value) + grid.charge_width))
    flag = n + len(value) + len(charge)
    next_qubit = flag + 1
    control = next_qubit if controlled else None
    on = () if control is None else (control,)
    operations: list[Instruction] = []
    frame = fragment.basis_id
    if frame is not None:
        operations.append(Instruction("basis", basis_id=frame, basis_hash=fragment.basis_hash,
                                      inverse=True, system=system))
    compute = _evaluate(grid, system, value, charge)
    operations.extend(compute)
    for atom, angle in zip(atoms, angles):
        compare = _compare(value, flag, atom.threshold)
        operations.extend(compare)
        if angle is None:
            operations.extend((Instruction("phase", on, angle=Angle(pi=F(1))),
                               Instruction("phase", (*on, flag), angle=Angle(pi=F(1)))))
        else:
            theta = exact(angle)
            operations.extend((Instruction("phase", on, angle=Angle(radians=theta)),
                               Instruction("phase", (*on, flag), angle=Angle(radians=-2 * theta))))
        operations.extend(reversed(compare))
    operations.extend(reversed(compute))
    if frame is not None:
        operations.append(Instruction("basis", basis_id=frame, basis_hash=fragment.basis_hash,
                                      inverse=False, system=system))
    return ReflectionPlan(next_qubit + int(controlled), system, value, charge, flag, control, tuple(operations))


def rte_event_plans(bound: BoundRTEEvent, *, num_modes: int, controlled: bool = True) -> tuple[ReflectionPlan, ...]:
    """Return plans in existing event circuit order, sharing exact fragment runs."""
    event = bound.event
    atoms = (*bound.rotation_first_atoms[1:], bound.rotation_first_atoms[0])
    if tuple(a.component_id for a in atoms) != event.selected_component_ids:
        raise ValueError("bound atom order disagrees with event")
    plans = []
    start = 0
    while start < len(atoms):
        stop = start + 1
        while stop < len(atoms) and atoms[start].fragment is not None and atoms[stop].fragment == atoms[start].fragment:
            stop += 1
        angles = tuple(event.rotation_angle if i == event.taylor_order else None for i in range(start, stop))
        if atoms[start].fragment is None:
            plan = reflection_plan(atoms[start], num_modes=num_modes, angle=angles[0], controlled=controlled)
        else:
            if atoms[start].fragment.grid.num_modes != _integer(num_modes, "num_modes", 1):
                raise ValueError("system register size disagrees with fragment")
            plan = _threshold_run_plan(atoms[start:stop], angles, controlled=controlled)
        plans.append(plan)
        start = stop
    return tuple(plans)


def emit_rte_event(bound: BoundRTEEvent, sink: CircuitSink, *, num_modes: int,
                   controlled: bool = True) -> int:
    """Emit products, rotation and paired-order phase; return register size.

    One maximum-sized workspace is reused between runs. The optional RPE
    control has one fixed index even when runs require different bit widths.
    This is an emitter interface, not a native builder/compiler or full wrapper.
    """
    plans = rte_event_plans(bound, num_modes=num_modes, controlled=controlled)
    size = max(p.num_qubits for p in plans)
    for plan in plans:
        mapping = list(range(plan.num_qubits))
        if controlled:
            mapping[plan.control] = size - 1
        plan.emit(sink, qubit_map=mapping)
    sink.phase(Angle(pi=F((bound.event.taylor_order // 2) % 2)), (size - 1,) if controlled else ())
    return size
