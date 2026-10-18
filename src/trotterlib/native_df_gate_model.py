"""Compiler-independent native DF counts; standard library only.

This module consumes prepared primitive metadata. It never prepares a Hamiltonian,
samples a trajectory, constructs a Qiskit circuit, or estimates compiled costs.
Profile ``explicit_rz_sx_x_cx_v1`` is a declared lowering, not Qiskit's optimizer.
See docs/research/track_a_native_df_theoretical_gate_model_20261016.md.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from typing import Iterable

PROFILE = "explicit_rz_sx_x_cx_v1"


@dataclass(frozen=True)
class Gate:
    name: str
    qubits: tuple[int, ...]
    angle: float = 0.0
    beta: float = 0.0

    def __post_init__(self):
        arities = {"rz": 1, "p": 1, "z": 1, "h": 1, "sx": 1,
                   "sxdg": 1, "x": 1, "sdg": 1, "cx": 2, "cz": 2,
                   "rzz": 2, "crz": 2, "crzz": 3, "rxx": 2,
                   "ryy": 2, "xx_plus_yy": 2}
        if self.name not in arities:
            raise ValueError(f"Unsupported gate: {self.name}; no opaque cost fallback")
        if (len(self.qubits) != arities[self.name] or
                len(set(self.qubits)) != len(self.qubits) or
                any(type(q) is not int or q < 0 for q in self.qubits)):
            raise ValueError("Invalid ordered qubit binding")
        if not math.isfinite(self.angle) or not math.isfinite(self.beta):
            raise ValueError("Only finite numeric parameters are supported")

    def inverse(self):
        if self.name in ("x", "z", "h", "cx", "cz"):
            return self
        if self.name in ("sx", "sxdg"):
            return Gate("sxdg" if self.name == "sx" else "sx", self.qubits)
        if self.name == "sdg":
            return Gate("p", self.qubits, math.pi / 2)
        return Gate(self.name, self.qubits, -self.angle, self.beta)


@dataclass(frozen=True)
class Circuit:
    operations: tuple[Gate, ...] = ()
    global_phase: float = 0.0

    def __post_init__(self):
        if not math.isfinite(self.global_phase):
            raise ValueError("Nonfinite global phase")


@dataclass(frozen=True)
class Cost:
    """raw = source primitive instructions, others = declared lowered counts."""
    raw: int | Fraction = 0
    rz: int | Fraction = 0
    cx: int | Fraction = 0
    sx: int | Fraction = 0
    x: int | Fraction = 0

    def __add__(self, other):
        return Cost(*(a + b for a, b in zip(self.values(), other.values())))

    def __mul__(self, scale):
        return Cost(*(a * scale for a in self.values()))

    __rmul__ = __mul__

    def values(self):
        return self.raw, self.rz, self.cx, self.sx, self.x

    def as_dict(self):
        return dict(zip(("raw", "rz", "cx", "sx", "x"), self.values()))


def lower_gate(g: Gate) -> Circuit:
    """Explicit temporal-order decomposition, including its global phase.

    RYY uses SX on both inputs, RZZ, then SXdg on both outputs.
    XX+YY uses RZ_0(beta), RXX(theta/2), RYY(theta/2), RZ_0(-beta).
    No zero-angle pruning/rotation fusion occurs at this lowering layer.
    """
    n, q, a = g.name, g.qubits, g.angle
    if n in ("rz", "sx", "x", "cx"):
        return Circuit((g,))
    if n in ("p", "z", "sdg"):
        a = a if n == "p" else math.pi if n == "z" else -math.pi / 2
        return Circuit((Gate("rz", q, a),), a / 2)
    if n == "h":
        return Circuit((Gate("rz", q, math.pi / 2), Gate("sx", q),
                        Gate("rz", q, math.pi / 2)), math.pi / 4)
    if n == "sxdg":
        return Circuit((Gate("sx", q), Gate("x", q)))
    if n == "rzz":
        word = (Gate("cx", q), Gate("rz", (q[1],), a), Gate("cx", q))
    elif n == "crz":
        word = (Gate("rz", (q[1],), a / 2), Gate("cx", q),
                Gate("rz", (q[1],), -a / 2), Gate("cx", q))
    elif n == "crzz":
        pair = (q[1], q[2])
        word = (Gate("cx", pair), Gate("crz", (q[0], q[2]), a), Gate("cx", pair))
    elif n == "cz":
        word = (Gate("h", (q[1],)), Gate("cx", q), Gate("h", (q[1],)))
    elif n == "rxx":
        word = (Gate("h", (q[0],)), Gate("h", (q[1],)), Gate("rzz", q, a),
                Gate("h", (q[0],)), Gate("h", (q[1],)))
    elif n == "ryy":
        word = (Gate("sx", (q[0],)), Gate("sx", (q[1],)), Gate("rzz", q, a),
                Gate("sxdg", (q[0],)), Gate("sxdg", (q[1],)))
    elif n == "xx_plus_yy":
        word = (Gate("rz", (q[0],), g.beta), Gate("rxx", q, a / 2),
                Gate("ryy", q, a / 2), Gate("rz", (q[0],), -g.beta))
    else:
        raise ValueError(n)
    return lower(Circuit(word))


def lower(circuit: Circuit) -> Circuit:
    word, phase = [], circuit.global_phase
    for g in circuit.operations:
        result = lower_gate(g)
        word.extend(result.operations)
        phase += result.global_phase
    return Circuit(tuple(word), phase)


@lru_cache(maxsize=4096)
def gate_cost(gate: Gate) -> Cost:
    histogram = Counter(g.name for g in lower_gate(gate).operations)
    return Cost(1, *(histogram[n] for n in ("rz", "cx", "sx", "x")))


def count(circuit: Circuit) -> Cost:
    # Lower at most one bounded-size primitive at a time, never a full word.
    total = Cost()
    for gate in circuit.operations:
        total += gate_cost(gate)
    return total


def primitive_histogram(circuit: Circuit) -> dict[str, int]:
    return dict(sorted(Counter(g.name for g in circuit.operations).items()))


@dataclass(frozen=True)
class Basis:
    basis_id: str
    source_hash: str
    operations: tuple[Gate, ...]
    control_scope: str = "uncontrolled"

    def __post_init__(self):
        if not self.basis_id or not self.source_hash:
            raise ValueError("Explicit source identity is required")
        if self.control_scope != "uncontrolled":
            raise ValueError("Native basis changes must be uncontrolled")

    @property
    def operation_digest(self):
        payload = [(g.name, g.qubits, float(g.angle).hex(), float(g.beta).hex())
                   for g in self.operations]
        return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()

    def circuit(self, inverse=False):
        return Circuit(tuple(g.inverse() for g in reversed(self.operations))
                       if inverse else self.operations)


def basis_from_runtime(basis_id, source_hash, runtime_operations) -> Basis:
    """Read existing (Gate, ordered qubits) metadata without importing Qiskit.

    This adapter does not certify a registry hash or a support-restricted basis.
    The caller must validate those against the original source/registry first.
    Unknown gates (including opaque dense ``unitary``) fail closed.
    """
    result = []
    for gate, qubits in runtime_operations:
        name = gate.name
        params = tuple(float(p) for p in gate.params)
        wanted = 2 if name == "xx_plus_yy" else 1 if name in (
            "rz", "p", "rzz", "crz", "crzz", "rxx", "ryy") else 0
        if len(params) != wanted:
            raise ValueError("Unsupported gate parameter schema")
        result.append(Gate(name, tuple(qubits), *(params or (0.0,))))
    return Basis(basis_id, source_hash, tuple(result))


@dataclass(frozen=True)
class Sandwich:
    basis: Basis
    central: Circuit


@dataclass(frozen=True)
class Barrier:
    """Explicit basis-cache flush; identity applications are barriers in source."""


def _check_basis_identities(bases):
    seen = {}
    for basis in bases:
        if basis is None:
            continue
        key = basis.basis_id, basis.source_hash
        if key in seen and seen[key] != basis:
            raise ValueError("Same basis ID/hash with different operations or binding")
        seen[key] = basis


def assemble(parts: Iterable[Sandwich | Circuit | Barrier], model="A") -> Circuit:
    """A emits every sandwich. B cancels only adjacent equal basis boundaries.

    A/B share the source's phase aggregation and identity barriers. A disables
    optional basis reuse even if a particular runtime request enabled it.
    No commuting, rotation merging, or approximate identity removal is performed.
    """
    if model not in ("A", "B"):
        raise ValueError("model must be A or B")
    parts = tuple(parts)
    _check_basis_identities(p.basis for p in parts if isinstance(p, Sandwich))
    word, phase, active = [], 0.0, None
    for part in (*parts, Barrier()):
        if isinstance(part, Sandwich):
            if active != part.basis or model == "A":
                if active is not None:
                    word.extend(active.circuit().operations)
                word.extend(part.basis.circuit(inverse=True).operations)
            active = part.basis
            word.extend(part.central.operations)
            phase += part.central.global_phase
            if model == "A":
                word.extend(active.circuit().operations)
                active = None
        else:
            if active is not None:
                word.extend(active.circuit().operations)
                active = None
            if isinstance(part, Circuit):
                word.extend(part.operations)
                phase += part.global_phase
            elif not isinstance(part, Barrier):
                raise TypeError("Unknown part")
    return Circuit(tuple(word), phase)


def diagonal_from_primitives(primitives, mode="ordinary", ancilla=None) -> Circuit:
    """Duck-type existing DiagonalEvolutionPrimitives; preserve source emission.

    Never recompute DF coefficients here: use the original producer's lists and
    phase, including its floating-point order and sign conventions.
    """
    if mode not in ("ordinary", "uncontrolled", "directional"):
        raise ValueError("Unknown control mode")
    rz = tuple((q, float(a)) for q, a in primitives.rz)
    rzz = tuple((p, q, float(a)) for p, q, a in primitives.rzz)
    phase = float(primitives.global_phase)
    if not all(math.isfinite(a) for *_, a in (*rz, *rzz)) or not math.isfinite(phase):
        raise ValueError("Nonfinite diagonal parameter")
    system = {q for q, _ in rz} | {q for p, q, _ in rzz} | {p for p, _, _ in rzz}
    if mode != "uncontrolled" and (type(ancilla) is not int or ancilla < 0 or ancilla in system):
        raise ValueError("A separate ancilla is required")
    word = []
    for left, right, angle in rzz:
        if angle == 0:
            continue
        if mode == "directional":
            pair, control = (left, right), (ancilla, right)
            word.extend((Gate("cx", pair), Gate("cx", control),
                         Gate("rz", (right,), -angle), Gate("cx", control), Gate("cx", pair)))
        else:
            word.append(Gate("rzz" if mode == "uncontrolled" else "crzz",
                             (left, right) if mode == "uncontrolled" else (ancilla, left, right), angle))
    for qubit, angle in rz:
        # Existing appenders retain every RZ entry, including explicit zero.
        # Normal primitive producers already prune zero RZ entries upstream.
        if mode == "directional":
            control = (ancilla, qubit)
            word.extend((Gate("cx", control), Gate("rz", (qubit,), -angle), Gate("cx", control)))
        else:
            word.append(Gate("rz" if mode == "uncontrolled" else "crz",
                             (qubit,) if mode == "uncontrolled" else (ancilla, qubit), angle))
    if mode != "uncontrolled" and phase != 0:
        word.append(Gate("p" if mode == "ordinary" else "rz", (ancilla,),
                         phase if mode == "ordinary" else 2 * phase))
    return Circuit(tuple(word), phase if mode == "uncontrolled" else 0.0)


def structural_instruction_bound(num_system_qubits, runtime_basis_size, one_body=False,
                                  mode="ordinary"):
    """Reference v5 high-level block bound, NOT a native RZ/CX bound."""
    if (type(num_system_qubits) is not int or num_system_qubits <= 0 or
            type(runtime_basis_size) is not int or runtime_basis_size < 0):
        raise ValueError("Invalid structural dimensions")
    n = num_system_qubits
    pairs = 0 if one_body else n * (n - 1) // 2
    if mode == "uncontrolled":
        return 2 * runtime_basis_size + n + pairs
    if mode == "ordinary":
        return 2 * runtime_basis_size + n + pairs + 1
    if mode == "directional":
        return 2 * runtime_basis_size + 3 * n + 5 * pairs + 1
    raise ValueError("Unknown control mode")


def hadamard_wrapper(evolution: Circuit, ancilla: int, axis="X") -> Circuit:
    if axis not in ("X", "Y"):
        raise ValueError("axis must be X or Y")
    tail = (Gate("sdg", (ancilla,)),) if axis == "Y" else ()
    return Circuit((Gate("h", (ancilla,)), *evolution.operations, *tail,
                    Gate("h", (ancilla,))), evolution.global_phase)


@dataclass(frozen=True)
class PFStage:
    block: int
    time: float
    mode: str


def pf_schedule(block_count, total_time, q, formula="2nd", policy="ordinary"):
    """Unmerged S2 / global Yoshida S4 schedule of reference native provider."""
    if type(q) is not int or q <= 0 or type(block_count) is not int or block_count < 0:
        raise ValueError("Invalid schedule size")
    if not math.isfinite(total_time) or formula not in ("2nd", "4th"):
        raise ValueError("Invalid time/formula")
    if policy not in ("ordinary", "symmetric_directional"):
        raise ValueError("Invalid control policy")
    a = 2 ** (1 / 3)
    w = 1 / (2 - a)
    # Match the reference native provider's binary64 operations, not merely
    # the algebraically equivalent Yoshida coefficients and stage times.
    pieces = (1.0,) if formula == "2nd" else (w, -a / (2 - a), w)
    modes = ("ordinary", "ordinary") if policy == "ordinary" else ("uncontrolled", "directional")
    return tuple(PFStage(j, total_time * weight / q / 2, mode)
                 for _ in range(q) for weight in pieces
                 for indices, mode in ((range(block_count), modes[0]),
                                       (reversed(range(block_count)), modes[1])) for j in indices)


def deterministic_circuit(bases, primitive_factory, total_time, q, formula="2nd",
                          policy="ordinary", ancilla=0, scalar_phase=0.0, model="A"):
    """Factory must return existing, already prepared primitive metadata per time.

    ``scalar_phase`` is the actual single final native ancilla phase, e.g. -c*T.
    Neither the molecular preparation nor scalar extraction is performed here.
    """
    if any(ancilla in g.qubits for b in bases for g in b.operations):
        raise ValueError("Ancilla overlaps Gaussian system binding")
    parts = [Sandwich(bases[s.block], diagonal_from_primitives(
        primitive_factory(s.block, s.time), s.mode, ancilla))
        for s in pf_schedule(len(bases), total_time, q, formula, policy)]
    if scalar_phase != 0:
        parts.append(Circuit((Gate("p", (ancilla,), scalar_phase),)))
    return assemble(parts, model)


def partial_prefix_circuit(bases, primitive_factory, total_time, q, ancilla,
                           policy="ordinary", scalar_phase=0.0, model="A"):
    """Fixed q prefix halves only, closing at EVERY omitted tail/step boundary.

    Add q closed stochastic occurrences with expected_partial_trajectory. Unlike
    removing a tail from a PF circuit, this does not invent a cancellable boundary.
    These deliberately closed boundaries define the expectation model B scope.
    """
    if policy not in ("ordinary", "symmetric_directional"):
        raise ValueError("Invalid control policy")
    if type(q) is not int or q <= 0 or not math.isfinite(total_time):
        raise ValueError("Invalid q/time")
    if any(ancilla in g.qubits for b in bases for g in b.operations):
        raise ValueError("Ancilla overlaps Gaussian system binding")
    modes = ("ordinary", "ordinary") if policy == "ordinary" else ("uncontrolled", "directional")
    halves = []
    for _ in range(q):
        for indices, mode in ((range(len(bases)), modes[0]), (reversed(range(len(bases))), modes[1])):
            halves.append(assemble((Sandwich(bases[j], diagonal_from_primitives(
                primitive_factory(j, total_time / q / 2), mode, ancilla)) for j in indices), model))
    if scalar_phase != 0:
        halves.append(Circuit((Gate("p", (ancilla,), scalar_phase),)))
    return assemble(halves, model)


@dataclass(frozen=True)
class Component:
    component_id: str
    probability: Fraction
    basis: Basis | None
    support: tuple[int, ...] = ()
    sign: int = 1

    def __post_init__(self):
        if not isinstance(self.probability, Fraction) or self.probability < 0:
            raise ValueError("Use nonnegative Fraction probabilities")
        if self.sign not in (-1, 1) or not self.component_id:
            raise ValueError("Invalid component ID/sign")
        if (self.basis is None) != (not self.support):
            raise ValueError("Only identity has empty support/no basis")
        if self.basis is not None:
            Gate("rz" if len(self.support) == 1 else "rzz", self.support)


def canonical_order_probabilities(tau: float, maximum_order: int):
    """All even finite-RTE orders. Fractions normalize binary64 log-weights.

    These probabilities are numerical evaluations, not exact transcendental
    values/error certificates. Any positive weight lost to underflow is rejected.
    """
    if not math.isfinite(tau) or type(maximum_order) is not int or maximum_order < 0 or maximum_order % 2:
        raise ValueError("Finite tau and nonnegative even K required")
    if tau == 0:
        return {k: Fraction(k == 0) for k in range(0, maximum_order + 1, 2)}
    t = abs(tau)
    logs = {k: k * math.log(t) - math.lgamma(k + 1) + math.log(math.hypot(1, t / (k + 1)))
            for k in range(0, maximum_order + 1, 2)}
    peak = max(logs.values())
    weights = {k: math.exp(v - peak) for k, v in logs.items()}
    if any(v == 0 or not math.isfinite(v) for v in weights.values()):
        raise ArithmeticError("Lost finite-RTE order: increase precision externally")
    weights = {k: Fraction(v) for k, v in weights.items()}
    z = sum(weights.values())
    return {k: v / z for k, v in weights.items()}


def _validate_law(components, orders, ancilla):
    if not components or sum(c.probability for c in components) != 1:
        raise ValueError("Component probabilities must sum exactly to one")
    if len({c.component_id for c in components}) != len(components):
        raise ValueError("Duplicate component ID")
    if not orders or any(type(k) is not int or k < 0 or k % 2 or
                         not isinstance(p, Fraction) or p < 0 for k, p in orders.items()):
        raise ValueError("Finite even-order Fraction law required")
    if sum(orders.values()) != 1:
        raise ValueError("Order probabilities must sum exactly to one")
    _check_basis_identities(c.basis for c in components)
    system = {q for c in components for q in c.support}
    system |= {q for c in components if c.basis for g in c.basis.operations for q in g.qubits}
    if type(ancilla) is not int or ancilla < 0 or ancilla in system:
        raise ValueError("Separate ancilla required")


def application(component, role, theta, ancilla):
    """Central instructions only. Identity is handled as a source barrier."""
    if role not in ("product", "rotation"):
        raise ValueError("Unknown application role")
    if component.basis is None:
        return Circuit()
    s = component.support
    if role == "product":
        return Circuit(tuple(Gate("cz", (ancilla, q)) for q in s))
    return Circuit((Gate("crz" if len(s) == 1 else "crzz", (ancilla, *s),
                         2 * component.sign * theta),))


def explicit_rte_occurrence(events, tau, ancilla, model="A"):
    """Events are (even k, ordered k product Components, rotation Component).

    Existing producer's floating accumulation order is retained: event phases
    first, then application phases. Identity closes basis reuse even with phase0.
    """
    if not math.isfinite(tau):
        raise ValueError("Invalid tau")
    events = tuple(events)
    phase = 0.0
    for k, products, rotation in events:
        if type(k) is not int or k < 0 or k % 2 or len(products) != k:
            raise ValueError("Invalid paired event")
        phase += math.pi if (k // 2) % 2 else 0.0
    parts = []
    for k, products, rotation in events:
        theta = math.atan(tau / (k + 1))
        for role, comp in [*(("product", c) for c in products), ("rotation", rotation)]:
            if comp.basis is None:
                parts.append(Barrier())
                phase += (math.pi if comp.sign == -1 else 0.0) if role == "product" else -comp.sign * theta
            else:
                parts.append(Sandwich(comp.basis, application(comp, role, theta, ancilla)))
                if role == "product" and comp.sign == -1:
                    phase += math.pi
    if phase != 0:
        parts.append(Circuit((Gate("p", (ancilla,), phase),)))
    return assemble(parts, model)


def existing_rte_occurrence(events, basis_lookup, ancilla, model="A"):
    """Count supplied RTEEvent.application_sequence metadata, without sampling.

    ``basis_lookup(application)`` must resolve/validate the ORIGINAL registry
    identity, operation metadata and any basis-plan certificate. This adapter
    does not manufacture support-restricted bases. It uses event angles/phases
    as stored, instead of reconstructing them from tau.
    """
    events = tuple(events)
    phase, parts = 0.0, []
    for event in events:
        k = event.taylor_order
        if type(k) is not int or k < 0 or k % 2:
            raise ValueError("Invalid event Taylor order")
        apps = tuple(event.application_sequence)
        if (len(apps) != k + 1 or
                tuple(a.application_index for a in apps) != tuple(range(k + 1)) or
                tuple(a.role for a in apps) != ("product",) * k + ("rotation",) or
                tuple(a.component_id for a in apps) != (*event.product_component_ids, event.rotation_component_id)):
            raise ValueError("Event application order/identity mismatch")
        if event.phase not in (1, -1):
            raise ValueError("Only real paired-event signs are supported")
        phase += math.pi if event.phase == -1 else 0.0
    for event in events:
        theta = float(event.rotation_angle)
        for app in event.application_sequence:
            basis = None if app.is_identity else basis_lookup(app)
            if not app.is_identity and (not isinstance(basis, Basis) or
                    (basis.basis_id, basis.source_hash) != (app.basis_id, app.basis_hash)):
                raise ValueError("Resolved basis identity mismatch; explicit plan adapter required")
            comp = Component(app.component_id, Fraction(0), basis,
                             () if app.is_identity else tuple(app.diagonal_pauli_support), app.coefficient_sign)
            if basis is None:
                parts.append(Barrier())
                phase += (math.pi if comp.sign == -1 else 0.0) if app.role == "product" else -comp.sign * theta
            else:
                if any(ancilla in g.qubits for g in basis.operations) or ancilla in comp.support:
                    raise ValueError("Ancilla overlaps system")
                parts.append(Sandwich(basis, application(comp, app.role, theta, ancilla)))
                if app.role == "product" and comp.sign == -1:
                    phase += math.pi
    if phase != 0:
        parts.append(Circuit((Gate("p", (ancilla,), phase),)))
    return assemble(parts, model)


@dataclass(frozen=True)
class ExpectedCost:
    """Bounds differ only by at most one aggregated P per RTE occurrence."""
    lower: Cost
    upper: Cost
    phase_probability_lower: Fraction
    phase_probability_upper: Fraction
    transitions: int
    probability_status: str = "EXACT_FOR_SUPPLIED_FRACTION_LAW"

    @property
    def exact(self):
        return self.lower == self.upper


def expected_rte_occurrence(components, orders, r, tau, ancilla, model="A",
                            max_transitions=1_000_000) -> ExpectedCost:
    """Basis-state DP includes ALL supplied orders/components; no sampling.

    Component draws are iid within and between events conditional on k, with the
    same categorical law for products and rotation. k is iid across events.
    Cost probabilities use Fractions; angles only determine nonzero source gates.
    General identity-rotation phase cancellation is bounded, not guessed exact.
    """
    components = tuple(components)
    _validate_law(components, orders, ancilla)
    if type(r) is not int or r <= 0 or model not in ("A", "B") or not math.isfinite(tau):
        raise ValueError("Invalid event count/model/tau")
    if type(max_transitions) is not int or max_transitions <= 0:
        raise ValueError("Positive transition cap required")
    bases = {c.basis for c in components if c.basis is not None}
    forward = {b: count(b.circuit()) for b in bases}
    inverse = {b: count(b.circuit(True)) for b in bases}
    calls = 0
    # Reuse deterministic lowering costs; DP does not build an event sequence.
    central = {(k, role, c): count(application(c, role, math.atan(tau / (k + 1)), ancilla))
               for k in orders for role in ("product", "rotation") for c in components}

    def draw(states, role, k):
        nonlocal calls
        output = {}
        for active, (mass, weighted) in states.items():
            for c in components:
                if c.probability == 0:
                    continue
                calls += 1
                if calls > max_transitions:
                    raise RuntimeError("DP transition cap; no approximation fallback")
                basis = c.basis
                cost = central[k, role, c]
                if model == "A":
                    if basis is not None:
                        cost += inverse[basis] + forward[basis]
                    end = None
                else:
                    if active != basis:
                        if active is not None:
                            cost += forward[active]
                        if basis is not None:
                            cost += inverse[basis]
                    end = basis
                p, w = output.get(end, (Fraction(0), Cost()))
                output[end] = (p + mass * c.probability,
                               w + (weighted + cost * mass) * c.probability)
        return output

    total = Cost()
    if model == "A":
        # Linear expectation: products/rotation have separate central costs.
        for k, probability in orders.items():
            event_mean = Cost()
            for c in components:
                pair = inverse[c.basis] + forward[c.basis] if c.basis else Cost()
                event_mean += ((pair + central[k, "product", c]) * k +
                               pair + central[k, "rotation", c]) * c.probability
            total += event_mean * probability * r
    else:
        states = {None: (Fraction(1), Cost())}
        for _ in range(r):
            combined = {}
            for k, probability in sorted(orders.items()):
                if probability == 0:
                    continue
                conditional = states
                for _ in range(k):
                    conditional = draw(conditional, "product", k)
                conditional = draw(conditional, "rotation", k)
                for basis, (p, cost) in conditional.items():
                    old_p, old_cost = combined.get(basis, (Fraction(0), Cost()))
                    combined[basis] = old_p + p * probability, old_cost + cost * probability
            states = combined
        assert sum(p for p, _ in states.values()) == 1
        for basis, (p, cost) in states.items():
            total += cost + (forward[basis] * p if basis is not None else Cost())
    # All event signs and negative product signs contribute nonnegative pi.
    # Without identity rotation phases there is no possible cancellation.
    if tau == 0 or not any(c.basis is None and c.probability for c in components):
        positive = sum(c.probability for c in components if c.sign == 1)
        zero_event_phase = sum(p * positive ** k for k, p in orders.items() if not (k // 2) % 2)
        lo = hi = 1 - zero_event_phase ** r
    else:
        lo, hi = Fraction(0), Fraction(1)
    phase_cost = count(Circuit((Gate("p", (ancilla,), 1.0),)))
    return ExpectedCost(total + phase_cost * lo, total + phase_cost * hi, lo, hi, calls)


def expected_partial_trajectory(occurrence: ExpectedCost, q: int, deterministic: Circuit,
                                ancilla: int, axis="X") -> ExpectedCost:
    """q closed iid RTE occurrences of r events each (R=q*r).

    ``deterministic`` must already count the q native prefix forward/reverse
    pieces plus the one global scalar phase, with occurrence boundaries closed.
    No cancellation across a stochastic occurrence boundary is credited here.
    """
    if type(q) is not int or q <= 0:
        raise ValueError("Positive q required")
    fixed = count(hadamard_wrapper(deterministic, ancilla, axis))
    return ExpectedCost(fixed + occurrence.lower * q, fixed + occurrence.upper * q,
                        occurrence.phase_probability_lower, occurrence.phase_probability_upper,
                        occurrence.transitions, occurrence.probability_status)
