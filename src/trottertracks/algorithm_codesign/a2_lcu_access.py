"""Explicit two-unitary LCU adapter; no state, RNG, loader or compiler.

Inputs are system-only circuits with explicit standard gates, not black-box
oracles. Exact algebra is documented separately. Binary64 angles/decoding here
are an artificial correctness prototype, NOT a certified molecular engine.
"""
from dataclasses import dataclass
from fractions import Fraction
import math

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit import ControlledGate
from qiskit.circuit.library import (
    HGate, IGate, PhaseGate, RXGate, RYGate, RZGate, RZZGate,
    SGate, SdgGate, TGate, TdgGate, XGate, YGate, ZGate,
    SwapGate, XXPlusYYGate,
)
from trotterlib.rpe_hadamard_interrogation import RPE_HADAMARD_BIT_VALUE_MAPPING

_STANDARD = (HGate, IGate, PhaseGate, RXGate, RYGate, RZGate, RZZGate,
             SGate, SdgGate, TGate, TdgGate, XGate, YGate, ZGate,
             SwapGate, XXPlusYYGate)
_SIGNS = dict(RPE_HADAMARD_BIT_VALUE_MAPPING)


def _supported(gate):
    if isinstance(gate, ControlledGate):
        return _supported(gate.base_gate)
    return isinstance(gate, _STANDARD)


def _check_system_circuit(circuit):
    if not isinstance(circuit, QuantumCircuit) or circuit.num_qubits < 1:
        raise ValueError('explicit nonempty system circuit required')
    if circuit.num_clbits or circuit.parameters:
        raise ValueError('no classical bits or unbound parameters allowed')
    if not math.isfinite(float(circuit.global_phase)):
        raise ValueError('nonfinite global phase')
    for item in circuit.data:
        op = item.operation
        if (item.clbits or getattr(op, 'condition', None) is not None
                or not _supported(op)):
            raise ValueError('unsupported access; explicit standard unitary gates only')
        if not all(math.isfinite(float(x)) for x in op.params):
            raise ValueError('nonfinite gate parameter')


def order0_projection(order, labels):
    """Metadata pushforward only; does not construct a law, event or random word.

    Labels are in existing application order: products then final rotation.
    The input fine event is never modified; its phase/angle stay in that event.
    """
    labels = tuple(labels)
    if type(order) is not int or order not in (0, 2) or len(labels) != order + 1:
        raise ValueError('only the existing one-step K2 application metadata')
    if any(not isinstance(x, str) or not x for x in labels):
        raise ValueError('nonempty component identities required')
    return 0, (labels[-1],)


@dataclass(frozen=True)
class LCUScale:
    zeta: float
    s: float
    beta: float
    beta2: float
    prepare_angle: float
    coarse_block_angle: float
    readout_angle: float
    eigenvalues: tuple

    @classmethod
    def from_zeta(cls, zeta):
        if isinstance(zeta, bool) or not isinstance(zeta, (int, float)):
            raise ValueError('finite positive explicit zeta required')
        zeta = float(zeta)
        if not math.isfinite(zeta) or zeta <= 0:
            raise ValueError('finite positive explicit zeta required')
        try:
            s = math.hypot(zeta, 1.)
            beta = zeta + s
            beta2 = beta * beta
            inv = 1. / zeta
            a = inv * inv
            plus = (a + math.hypot(a, 2. * inv)) / 2.
            minus = -a / plus
            values = (s, beta, beta2, a, plus, minus, beta2 * plus)
            if not all(math.isfinite(x) for x in values) or a <= 0 or minus >= 0:
                raise ValueError('unresolved numeric scale')
            return cls(zeta, s, beta, beta2,
                       2. * math.atan2(math.sqrt(s), math.sqrt(zeta)),
                       2. * math.atan2(1., -zeta),
                       math.atan2(2. * zeta, 1.), (plus, minus))
        except (OverflowError, ZeroDivisionError) as exc:
            raise ValueError('unresolved numeric scale') from exc

    def outcome(self, flag, block_bit, axis_bit, *, common_weight=Fraction(1)):
        """One all-attempt value. Never normalize by the number of successes.

        Only a common exact real weight is supported, not unequal level weights.
        """
        if any(type(x) is not int or x not in (0, 1)
               for x in (flag, block_bit, axis_bit)):
            raise ValueError('three explicit binary outcomes required')
        if isinstance(common_weight, bool) or not isinstance(common_weight, (int, Fraction)):
            raise ValueError('common exact real weight required')
        weight = float(common_weight)
        if not math.isfinite(weight):
            raise ValueError('nonfinite weight')
        if flag:
            return 0.
        result = weight * self.beta2 * self.eigenvalues[block_bit] * _SIGNS[axis_bit]
        if not math.isfinite(result):
            raise ValueError('nonfinite decoded outcome')
        return result

    def absolute_range(self, common_weight=Fraction(1)):
        return abs(self.outcome(0, 0, 0, common_weight=common_weight))


@dataclass(frozen=True)
class LCUCircuit:
    circuit: QuantumCircuit
    scale: LCUScale
    system_qubits: int
    signal_ancilla: int
    block_ancilla: int
    herald_ancilla: int
    axis: str | None
    numerically_certified: bool = False


def build_access(fine, coarse, *, zeta, axis=None, measure=False):
    """PREP--SELECT--PREP† with explicit doubly controlled system circuits.

    Layout: system[0:n], signal a=n, block b=n+1, herald h=n+2, all ancillas
    initially zero. Supply the system state separately. No state inverse,
    quantum probability, synthesis, sampling, or real-bundle access occurs here.
    Deferred flag measurement allows a unitary reference; outcome decoder
    zero-fills h=1. Physical feed-forward may omit readout on those failures.
    """
    _check_system_circuit(fine)
    _check_system_circuit(coarse)
    if fine.num_qubits != coarse.num_qubits:
        raise ValueError('fine/coarse system ordering and width must match')
    if axis not in (None, 'X', 'Y') or type(measure) is not bool or (measure and axis is None):
        raise ValueError('X/Y axis required for measurement')
    scale = LCUScale.from_zeta(zeta)
    n = fine.num_qubits
    a, b, h = n, n + 1, n + 2
    circuit = QuantumCircuit(n + 3)
    circuit.h(a)
    circuit.ry(scale.prepare_angle, h)
    # Qiskit little endian: controls [h,a] have states (0,1)->2, (1,1)->3.
    # to_gate/control preserve the full global phase inside each SELECT branch.
    circuit.append(fine.to_gate().control(2, ctrl_state=2), [h, a, *range(n)])
    circuit.append(coarse.to_gate().control(2, ctrl_state=3), [h, a, *range(n)])
    circuit.cry(scale.coarse_block_angle, h, b)
    circuit.ry(-scale.prepare_angle, h)
    if axis is not None:
        circuit.ry(-scale.readout_angle, b)
        if axis == 'Y':
            circuit.sdg(a)
        circuit.h(a)
    if measure:
        bits = ClassicalRegister(3, 'a2_readout')
        circuit.add_register(bits)
        circuit.measure(h, bits[0])
        circuit.measure(b, bits[1])
        circuit.measure(a, bits[2])
    circuit.metadata = dict(scope='A2_EXPLICIT_LCU_ARTIFICIAL_PROTOTYPE',
        state_oracle=False, inverse_state_preparation=False,
        fine_system_calls=1, coarse_system_calls=1, system_control_count=2,
        extra_ancillas=2, retries=0, numerical_error_bound='UNKNOWN',
        molecular_binding='NOT_QUALIFIED', scientific_execution=False)
    return LCUCircuit(circuit, scale, n, a, b, h, axis)
