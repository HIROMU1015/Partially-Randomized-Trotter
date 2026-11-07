"""Two-label positive LCU with a flag before a paid suffix.

Artificial 1/2-system-qubit adapter only. Inputs are explicit system-only
unitaries and an exact rational probability; no state/probability oracle,
sampler, molecular loader, compiler or state inverse is accepted. Binary64
PREP and gates are NOT a certified scientific measurement engine.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import math

from qiskit import ClassicalRegister, QuantumCircuit

from .a2_lcu_access import _check_system_circuit
from trotterlib.rpe_hadamard_interrogation import RPE_HADAMARD_BIT_VALUE_MAPPING

_SIGNS = dict(RPE_HADAMARD_BIT_VALUE_MAPPING)


@dataclass(frozen=True)
class EInstrument:
    circuit: QuantumCircuit
    prefix: QuantumCircuit
    system_qubits: int
    signal_ancilla: int
    pool_ancilla: int
    probability: F
    axis: str
    early_abort: bool
    numerically_certified: bool = False

    def decode(self, flag, axis_bit=None):
        """One all-attempt value; a failed attempt has no signal measurement.

        flag=1 is always zero, including the late-flag comparison which does
        measure the discarded signal. No success-probability compensation.
        """
        if type(flag) is not int or flag not in (0, 1):
            raise ValueError('explicit binary flag required')
        if axis_bit is not None and (type(axis_bit) is not int or axis_bit not in _SIGNS):
            raise ValueError('explicit binary axis outcome or missing failure readout')
        if flag:
            return 0
        if axis_bit is None:
            raise ValueError('successful attempt requires measured axis bit')
        return _SIGNS[axis_bit]


def build_instrument(A, B, W0, W1, *, p, axis, early_abort=True):
    """Build H_a, cB, PREP, SELECT, PREP†, flag, conditional cA/readout.

    Layout: system[0:n], signal a=n, pool k=n+1; all ancillas start at zero.
    Classical bit0 is flag, bit1 is signal (unmeasured on early failure).
    early_abort=False is the same coherent instrument with the full suffix
    executed on every attempt. Neither version retries or prepares |psi>.
    """
    for circuit in (A, B, W0, W1):
        _check_system_circuit(circuit)
    n = A.num_qubits
    if n not in (1, 2) or any(c.num_qubits != n for c in (B, W0, W1)):
        raise ValueError('matching invented 1/2-qubit system descriptions only')
    if type(p) not in (int, F) or not 0 < p < 1:
        raise ValueError('exact positive two-label rational probability required')
    p = F(p)
    if axis not in ('X', 'Y') or type(early_abort) is not bool:
        raise ValueError('X/Y axis and explicit early-abort choice required')
    pf = float(p)
    if not 0 < pf < 1:
        raise ValueError('probability unresolved in prototype numeric precision')
    angle = 2 * math.atan2(math.sqrt(float(1-p)), math.sqrt(pf))
    a, k = n, n+1
    prefix = QuantumCircuit(n+2, name='e_prefix')
    prefix.h(a)
    prefix.append(B.to_gate().control(1, label='common_B'), [a, *range(n)])
    prefix.ry(angle, k)
    # Qiskit little-endian controls [k,a]: (k,a)=(0,1)->2, (1,1)->3.
    # Controlling full circuits retains their scalar/identity/event phases.
    for j, W in enumerate((W0, W1)):
        prefix.append(W.to_gate().control(2, ctrl_state=2+j, label=f'SELECT_W{j}'),
                      [k, a, *range(n)])
    prefix.ry(-angle, k)
    circuit = prefix.copy(name='e_early' if early_abort else 'lcu_late')
    bits = ClassicalRegister(2, 'e_outcome')
    circuit.add_register(bits)

    def suffix():
        circuit.append(A.to_gate().control(1, label='common_A'), [a, *range(n)])
        if axis == 'Y':
            circuit.sdg(a)
        circuit.h(a)
        circuit.measure(a, bits[1])

    if early_abort:
        circuit.measure(k, bits[0])
        with circuit.if_test((bits[0], 0)):
            suffix()
    else:
        suffix()
        circuit.measure(k, bits[0])
    circuit.metadata = dict(scope='E_TWO_LABEL_ARTIFICIAL_INSTRUMENT',
        exact_probability=str(p), system_order='Qiskit_little_endian',
        bit_order=['pool_flag', 'signal_axis'], axis=axis, range=[-1, 1],
        all_attempt_denominator=True, automatic_retries=0,
        common_B_calls=1, common_A_calls='success_only' if early_abort else 1,
        SELECT_system_branches=2, SELECT_control_count=2,
        pool_PREP_calls=1, pool_inverse_PREP_calls=1, extra_pool_ancillas=1,
        input_state_preparation_included=False, fresh_state_consumed_each_attempt=True,
        failure_axis_measurement=not early_abort, feed_forward_required=early_abort,
        hardware_feed_forward_qualified=False, numerical_error_bound='UNKNOWN',
        molecular_binding='NOT_QUALIFIED', scientific_execution=False)
    return EInstrument(circuit, prefix, n, a, k, p, axis, early_abort)
