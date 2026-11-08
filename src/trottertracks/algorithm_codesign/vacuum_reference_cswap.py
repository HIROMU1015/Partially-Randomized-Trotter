"""Small explicit vacuum-reference/SE gadget, not a molecular engine.

No state, Born, RNG, scientific loader or compiler here. The phase ledger uses
exact binary64-as-rational input angles; conversion to the correction gate is
still a non-certified binary64 prototype. All operations are paid, including
the reference register and optional fanout. Only invented 1--3-qubit inputs.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import math

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit import ControlledGate
from qiskit.circuit.library import (
    IGate, PhaseGate, RZGate, RZZGate, SGate, SdgGate, TGate, TdgGate,
    ZGate, SwapGate, XXPlusYYGate,
)

from .a2_lcu_access import _check_system_circuit
from trotterlib.rpe_hadamard_interrogation import RPE_HADAMARD_BIT_VALUE_MAPPING

_SIGNS = dict(RPE_HADAMARD_BIT_VALUE_MAPPING)


def _vacuum_gate_phase(gate):
    if isinstance(gate, ControlledGate):
        # Vacuum controls are all zero; only that control state executes base.
        return _vacuum_gate_phase(gate.base_gate) if gate.ctrl_state == 0 else F(0)
    if isinstance(gate, (RZGate, RZZGate)):
        return -F(float(gate.params[0]))/2
    if isinstance(gate, (IGate, PhaseGate, SGate, SdgGate, TGate, TdgGate,
                         ZGate, SwapGate, XXPlusYYGate)):
        return F(0)
    raise ValueError('gate has no supported gatewise vacuum-eigenstate proof')


def vacuum_phase(circuit):
    """Exact phase expression, not a matrix evaluation or vacuum energy guess.

    Gatewise preservation is sufficient. A circuit that restores vacuum only
    through cancellation of unsupported gates is deliberately rejected.
    """
    _check_system_circuit(circuit)
    if circuit.num_qubits not in (1, 2, 3):
        raise ValueError('invented 1--3 system qubits only')
    return F(float(circuit.global_phase)) + sum(
        (_vacuum_gate_phase(item.operation) for item in circuit.data), F(0))


def append_gadget(circuit, L, R, *, target, reference, ancilla, fanout=()):
    """Append CSWAP, L†/R on disjoint lanes, CSWAP, P(ell+r).

    Caller supplies arbitrary target state and fresh vacuum reference/fanout.
    Returns the classical phase ledger. No reference postselection or retry.
    This block alone neither prepares nor reads the signal ancilla.
    """
    ell, r = vacuum_phase(L), vacuum_phase(R)
    n = L.num_qubits
    target, reference, fanout = tuple(target), tuple(reference), tuple(fanout)
    wires = (*target, *reference, ancilla, *fanout)
    if (R.num_qubits != n or len(target) != n or len(reference) != n
            or len(fanout) not in (0, n-1) or len(set(wires)) != len(wires)
            or any(type(q) is not int or not 0 <= q < circuit.num_qubits for q in wires)):
        raise ValueError('disjoint explicit target/reference/control/fanout mapping required')
    angle = float(ell+r)
    if not math.isfinite(angle):
        raise ValueError('unresolved finite correction angle')
    controls = (ancilla, *fanout)
    edges = [(controls[(i-1)//2], controls[i]) for i in range(1,len(controls))]
    for a,b in edges: circuit.cx(a,b)
    for i,(t,v) in enumerate(zip(target,reference)):
        circuit.cswap(controls[i] if fanout else ancilla,t,v)
    circuit.compose(L.inverse(),qubits=target,inplace=True)
    circuit.compose(R,qubits=reference,inplace=True)
    for i,(t,v) in enumerate(zip(target,reference)):
        circuit.cswap(controls[i] if fanout else ancilla,t,v)
    for a,b in reversed(edges): circuit.cx(a,b)
    circuit.p(angle,ancilla)
    return dict(L_vacuum_phase=str(ell),R_vacuum_phase=str(r),
                correction_phase=str(ell+r),correction_binary64_hex=angle.hex())


@dataclass(frozen=True)
class VacuumInstrument:
    circuit: QuantumCircuit
    target: tuple
    reference: tuple
    ancilla: int
    fanout: tuple
    axis: str | None
    ledger: dict
    numerically_certified: bool = False

    def decode(self, bit):
        if type(bit) is not int or bit not in _SIGNS:
            raise ValueError('explicit measured ancilla bit required')
        return _SIGNS[bit]


def build_instrument(L, R, *, axis=None, measure=False, mapping=None, fanout=False,
                     initialize_reference=False):
    """One fresh-state X/Y experiment. State preparation is supplied externally.

    Optional reset instructions explicitly charge candidate-specific reference
    initialization in resource circuits. No system state preparation is built.
    """
    n = L.num_qubits
    vacuum_phase(L);vacuum_phase(R)
    if (R.num_qubits != n or axis not in (None,'X','Y')
            or any(type(x) is not bool for x in (measure,fanout,initialize_reference))
            or (measure and axis is None)):
        raise ValueError('matching inputs and explicit X/Y readout required')
    width = 2*n+1+(n-1 if fanout else 0)
    mapping = tuple(range(width)) if mapping is None else tuple(mapping)
    if sorted(mapping)!=list(range(width)) or any(type(q) is not int for q in mapping):
        raise ValueError('full wire permutation required')
    target,reference = mapping[:n],mapping[n:2*n]
    a,fans = mapping[2*n],mapping[2*n+1:]
    c = QuantumCircuit(width)
    if initialize_reference:
        c.reset([*reference,*fans])
    c.h(a)
    ledger = append_gadget(c,L,R,target=target,reference=reference,ancilla=a,fanout=fans)
    if axis == 'Y':c.sdg(a)
    if axis is not None:c.h(a)
    if measure:
        bit=ClassicalRegister(1,'vacuum_signal');c.add_register(bit);c.measure(a,bit[0])
    c.metadata=dict(scope='VACUUM_CSWAP_ARTIFICIAL_PROTOTYPE',system_qubits=n,
        reference_qubits=n,fanout_qubits=len(fans),cswap_calls=2*n,
        fanout_CX_calls=2*len(fans),phase_ledger=ledger,
        system_state_preparation_included=False,
        reference_initialization='explicit_reset' if initialize_reference else 'external_vacuum_required',
        reference_postselection=False,automatic_retries=0,
        hardware_qualified=False,numerical_error_bound='UNKNOWN',molecular_binding='NOT_QUALIFIED')
    return VacuumInstrument(c,target,reference,a,fans,axis,ledger)
