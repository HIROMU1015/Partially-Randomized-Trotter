"""One-slot label-averaged measurement, artificial dynamic-circuit prototype.

No state, probability, RNG, molecular loader or compiler is accepted here.
Exact structural eligibility is sufficient, not a complete operator-equivalence
solver. Binary64 readout angles/weights are NOT a certified scientific engine.
"""
from dataclasses import dataclass, replace
from fractions import Fraction as F
import math

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit import ControlledGate
from qiskit.circuit.library import RZGate, RZZGate, XXPlusYYGate

from .a1_control_variates import CategoricalLaw
from .a2_lcu_access import _check_system_circuit
from trotterlib.rpe_hadamard_interrogation import RPE_HADAMARD_BIT_VALUE_MAPPING

_SIGNS = dict(RPE_HADAMARD_BIT_VALUE_MAPPING)


class Ineligible(ValueError):
    """Stable reason code followed by the offending label or field."""


def _stop(code, detail):
    raise Ineligible(f'{code}: {detail}')


@dataclass(frozen=True)
class Angle:
    """Explicit radians or exact rational multiples of pi; no tolerance snapping."""
    unit: str
    value: float | F

    def __post_init__(self):
        if self.unit == 'pi' and type(self.value) in (int, F):
            object.__setattr__(self, 'value', F(self.value))
        elif self.unit == 'rad' and type(self.value) in (int, float):
            object.__setattr__(self, 'value', float(self.value))
        else:
            _stop('ANGLE_REPRESENTATION', repr((self.unit, self.value)))
        if not math.isfinite(self.numeric):
            _stop('NONFINITE_ANGLE', self.unit)

    @property
    def numeric(self):
        return float(self.value) * (math.pi if self.unit == 'pi' else 1.)

    @property
    def key(self):
        return self.unit, (self.value if self.unit == 'pi' else self.value.hex())

    def cos_sin(self):
        if self.unit == 'pi' and (2*self.value).denominator == 1:
            return ((1., 0.), (0., 1.), (-1., 0.), (0., -1.))[int(2*self.value) % 4]
        return math.cos(self.numeric), math.sin(self.numeric)


ZERO = Angle('pi', F(0))


def _gate_key(op):
    if isinstance(op, ControlledGate):
        return ('controlled', op.num_ctrl_qubits, op.ctrl_state, _gate_key(op.base_gate))
    return (type(op).__module__, type(op).__qualname__,
            tuple(float(x).hex() for x in op.params))


@dataclass(frozen=True)
class LiteralCircuit:
    """Read-only system operation metadata; never a quantum-circuit builder.

    Operations use application order and binary64 hex parameters. Ordered scalar
    phase terms are retained without rounding their sum or fitting a phase.
    """
    num_qubits: int
    operations: tuple
    global_phase_terms: tuple = ()

    def __post_init__(self):
        if type(self.num_qubits) is not int or self.num_qubits < 1:
            _stop('LITERAL_WIDTH', self.num_qubits)
        if type(self.operations) is not tuple or type(self.global_phase_terms) is not tuple:
            _stop('LITERAL_IMMUTABILITY', 'tuple descriptions required')
        for name, params, wires in self.operations:
            arity = {'rz': (1,1), 'rzz': (1,2), 'xx_plus_yy': (2,2)}.get(name)
            if (arity is None or type(params) is not tuple or type(wires) is not tuple
                    or (len(params),len(wires)) != arity or len(set(wires)) != len(wires)
                    or any(type(q) is not int or not 0 <= q < self.num_qubits for q in wires)):
                _stop('LITERAL_OPERATION', name)
            for p in params:
                self._hex(p)
        for p in self.global_phase_terms:
            self._hex(p)

    @staticmethod
    def _hex(p):
        if not isinstance(p, str) or not math.isfinite(float.fromhex(p)) or float.fromhex(p).hex() != p:
            _stop('LITERAL_PARAMETER', p)

    @property
    def key(self):
        classes = {'rz': RZGate, 'rzz': RZZGate, 'xx_plus_yy': XXPlusYYGate}
        phase = ('ordered_scalar_terms_v1', self.global_phase_terms)
        return self.num_qubits, phase, tuple(
            ((classes[name].__module__, classes[name].__qualname__, params), wires)
            for name, params, wires in self.operations)

    def copy(self):
        return self

    def adjoint(self):
        operations = tuple((name, ((-float.fromhex(params[0])).hex(), *params[1:]), wires)
                           for name, params, wires in reversed(self.operations))
        phases = tuple((-float.fromhex(p)).hex() for p in reversed(self.global_phase_terms))
        return LiteralCircuit(self.num_qubits, operations, phases)


def circuit_key(circuit):
    """Literal ordered standard-gate description, including coherent phase."""
    if isinstance(circuit, LiteralCircuit):
        return circuit.key
    _check_system_circuit(circuit)
    return (circuit.num_qubits, float(circuit.global_phase).hex(), tuple(
        (_gate_key(item.operation), tuple(circuit.find_bit(q).index for q in item.qubits))
        for item in circuit.data))


@dataclass(frozen=True)
class SlotLabel:
    label: str
    context_id: str
    slot_id: str
    policy_id: str
    law_id: str
    actual_mass: F
    A: QuantumCircuit
    B: QuantumCircuit
    V: QuantumCircuit
    system_order: tuple[str, ...]
    selected_basis_id: str
    support: tuple[int, ...]
    sign: int = 1
    kind: str = 'product'
    phi: Angle | None = None
    scalar_phase: Angle = ZERO
    event_phase: Angle = ZERO
    original_basis_id: str = 'reference-only'


def _row_key(row, law):
    if not isinstance(row, SlotLabel):
        _stop('LABEL_DESCRIPTION', type(row).__name__)
    for name in ('label', 'context_id', 'slot_id', 'policy_id', 'law_id', 'selected_basis_id'):
        if not isinstance(getattr(row, name), str) or not getattr(row, name):
            _stop('MISSING_IDENTITY', name)
    if row.law_id != law.source:
        _stop('LAW_ID_MISMATCH', row.label)
    if type(row.actual_mass) not in (int, F) or row.actual_mass != law.mass(row.label):
        _stop('ACTUAL_MASS_MISMATCH', row.label)
    if row.actual_mass <= 0:
        _stop('ZERO_ACTUAL_SUPPORT', row.label)
    keys = tuple(circuit_key(c) for c in (row.A, row.B, row.V))
    n = row.A.num_qubits
    if any(c.num_qubits != n for c in (row.B, row.V)):
        _stop('SYSTEM_WIDTH', row.label)
    if (type(row.system_order) is not tuple or len(row.system_order) != n
            or len(set(row.system_order)) != n
            or any(not isinstance(x, str) or not x for x in row.system_order)):
        _stop('QUBIT_MAPPING', row.label)
    if (type(row.support) is not tuple or len(row.support) > 2
            or any(type(q) is not int or not 0 <= q < n for q in row.support)
            or len(set(row.support)) != len(row.support)
            or type(row.sign) is not int or row.sign not in (-1, 1)):
        _stop('SIGNED_DIAGONAL_SUPPORT', row.label)
    if row.kind not in ('product', 'rotation'):
        _stop('SLOT_KIND', row.label)
    if ((row.kind == 'product' and row.phi is not None)
            or (row.kind == 'rotation' and not isinstance(row.phi, Angle))):
        _stop('ROTATION_ANGLE', row.label)
    if any(not isinstance(x, Angle) for x in (row.scalar_phase, row.event_phase)):
        _stop('PHASE_DESCRIPTION', row.label)
    return (row.context_id, row.slot_id, row.policy_id, row.law_id,
            row.system_order, row.selected_basis_id, *keys, row.kind,
            None if row.phi is None else row.phi.key,
            row.scalar_phase.key, row.event_phase.key)


def _law_check(law):
    if not isinstance(law, CategoricalLaw):
        _stop('ACTUAL_LAW_REQUIRED', type(law).__name__)
    if any((p*2**53).denominator != 1 for p in law.masses):
        _stop('ACTUAL_GRID_MASS', law.source)


@dataclass(frozen=True)
class EligibleGroup:
    rows: tuple[SlotLabel, ...]
    law: CategoricalLaw
    signature: tuple
    mass: F
    conditional: tuple[F, ...]

    def check(self):
        # Circuit objects are mutable; detect edits even to a singleton group.
        if any(_row_key(r, self.law) != self.signature for r in self.rows):
            _stop('QUALIFIED_GROUP_MUTATED', self.rows[0].slot_id)

    @property
    def system_qubits(self):
        return len(self.rows[0].system_order)

    def r(self, z):
        if type(z) is not int or not 0 <= z < 2**self.system_qubits:
            _stop('SYSTEM_OUTCOME', z)
        return sum((p*row.sign*(-1)**sum((z >> q) & 1 for q in row.support)
                    for p, row in zip(self.conditional, self.rows, strict=True)), F(0))


def qualify_group(rows, law):
    """Check only the supplied slot descriptions; never discover molecular groups.

    The caller must bind each description to its original selected target. IDs
    are metadata, not a proof of that external binding. No equivalence fitting,
    zero-support dropping or hidden renormalization occurs here.
    """
    _law_check(law)
    rows = tuple(rows)
    if any(not isinstance(r, SlotLabel) for r in rows):
        _stop('LABEL_DESCRIPTION', 'group')
    if not rows or len({r.label for r in rows}) != len(rows):
        _stop('EMPTY_OR_DUPLICATE_LABELS', 'group')
    keys = tuple(_row_key(r, law) for r in rows)
    names = ('CONTEXT', 'SLOT', 'POLICY', 'LAW', 'QUBIT_ORDER', 'SELECTED_BASIS',
             'A_SEQUENCE', 'B_SEQUENCE', 'V_SEQUENCE', 'KIND', 'PHI',
             'SCALAR_PHASE', 'EVENT_PHASE')
    for row, key in zip(rows[1:], keys[1:], strict=True):
        for name, first, other in zip(names, keys[0], key, strict=True):
            if first != other:
                _stop(name+'_MISMATCH', row.label)
    # Freeze caller-owned circuit descriptions by copying; retain a mutation guard.
    copies = tuple(replace(r, A=r.A.copy(), B=r.B.copy(), V=r.V.copy()) for r in rows)
    mass = sum((F(r.actual_mass) for r in copies), F(0))
    return EligibleGroup(copies, law, keys[0], mass,
                         tuple(F(r.actual_mass)/mass for r in copies))


def partition_supplied_rows(rows, law):
    """Diagnostic split of explicitly supplied descriptions; no optimizer/search.

    Invalid rows raise rather than disappear. Outer context masses still belong
    to the caller; this function neither samples nor changes their law.
    """
    _law_check(law)
    rows = tuple(rows)
    if any(not isinstance(r, SlotLabel) for r in rows):
        _stop('LABEL_DESCRIPTION', 'partition')
    if len({r.label for r in rows}) != len(rows):
        _stop('EMPTY_OR_DUPLICATE_LABELS', 'partition')
    if not rows or len({r.slot_id for r in rows}) != 1:
        _stop('ONE_SLOT_REQUIRED', 'partition')
    buckets = {}
    for row in rows:
        buckets.setdefault(_row_key(row, law), []).append(row)
    return tuple(qualify_group(group, law) for group in buckets.values())


@dataclass(frozen=True)
class Readout:
    z: int
    r: F
    angle: float
    weight: float | F

    def decode(self, bit):
        if type(bit) is not int or bit not in _SIGNS:
            _stop('ANCILLA_BIT', bit)
        return self.weight*_SIGNS[bit]


def readout(group, axis, z):
    if axis not in ('X', 'Y'):
        _stop('MEASUREMENT_AXIS', axis)
    row = group.rows[0]
    r = group.r(z)
    cp, sp = row.scalar_phase.cos_sin()
    ce, se = row.event_phase.cos_sin()
    cp, sp = cp*ce-sp*se, sp*ce+cp*se
    if row.kind == 'product':
        x, y = (cp, -sp) if axis == 'X' else (sp, cp)
        weight = r  # Keep the stipulated product decoder s_a*r, including sign.
    else:
        c, s = row.phi.cos_sin()
        t = s*float(r)
        x, y = (cp*c+sp*t, cp*t-sp*c) if axis == 'X' else (sp*c-cp*t, sp*t+cp*c)
        weight = math.hypot(x, y)
    return Readout(z, r, math.atan2(y, x) if (x or y) else 0., weight)


def _append_axis(circuit, ancilla, angle):
    if angle == math.pi/2:
        circuit.sdg(ancilla)  # Original Y convention: Sdg then H.
    elif angle:
        circuit.rz(-angle, ancilla)
    circuit.h(ancilla)


@dataclass(frozen=True)
class MInstrument:
    circuit: QuantumCircuit
    preparation: QuantumCircuit
    readouts: tuple[Readout, ...]
    system_qubits: int
    ancilla: int
    axis: str
    numerically_certified: bool = False

    def decode(self, system_z, ancilla_bit):
        if type(system_z) is not int or not 0 <= system_z < len(self.readouts):
            _stop('SYSTEM_OUTCOME', system_z)
        return self.readouts[system_z].decode(ancilla_bit)


def build_instrument(group, axis, *, feed_forward=False,
                     measurement_mode='qiskit_dynamic_reference'):
    """Build the real dynamic-circuit instruction sequence, NOT hardware support.

    System wires 0:n, ancilla n. The system input is supplied externally; it is
    destroyed by n computational measurements. Fresh input per attempt is needed.
    Enumerated outcome controls are intentionally limited to invented 1/2-qubit
    prototypes; no H4 instrument is activated by this function.
    """
    group.check()
    if any(isinstance(c, LiteralCircuit) for c in (group.rows[0].A, group.rows[0].B, group.rows[0].V)):
        _stop('METADATA_ONLY_NO_QUANTUM_BUILD', group.rows[0].slot_id)
    if (measurement_mode != 'qiskit_dynamic_reference' or type(feed_forward) is not bool
            or (group.rows[0].kind == 'rotation' and not feed_forward)):
        _stop('UNSUPPORTED_MEASUREMENT_FEED_FORWARD', measurement_mode)
    n = group.system_qubits
    if n not in (1, 2):
        _stop('ARTIFICIAL_WIDTH_ONLY', n)
    settings = tuple(readout(group, axis, z) for z in range(2**n))
    row = group.rows[0]
    prep = QuantumCircuit(n+1)
    prep.h(n)
    prep.append(row.A.inverse().to_gate().control(1, ctrl_state=0), [n, *range(n)])
    prep.append(row.B.to_gate().control(1), [n, *range(n)])
    prep.compose(row.V, qubits=list(range(n)), inplace=True)
    circuit = prep.copy()
    system = ClassicalRegister(n, 'm_system')
    ancilla = ClassicalRegister(1, 'm_ancilla')
    circuit.add_register(system, ancilla)
    circuit.measure(list(range(n)), list(system))
    if row.kind == 'product':
        _append_axis(circuit, n, settings[0].angle)
    else:
        for setting in settings:
            with circuit.if_test((system, setting.z)):
                _append_axis(circuit, n, setting.angle)
    circuit.measure(n, ancilla[0])
    circuit.metadata = dict(scope='M_ONE_SLOT_ARTIFICIAL_DYNAMIC_PROTOTYPE',
        state_oracle=False, inverse_state_preparation=False,
        input_state_preparation_included=False, system_measurements=n,
        ancilla_measurements=1, system_readout='destructive',
        feed_forward_required=row.kind == 'rotation',
        hardware_feed_forward_qualified=False, outcome_controls=len(settings) if row.kind == 'rotation' else 0,
        numerical_error_bound='UNKNOWN', molecular_binding='NOT_QUALIFIED',
        scientific_execution=False)
    return MInstrument(circuit, prep, settings, n, n, axis)
