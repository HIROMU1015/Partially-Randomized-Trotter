"""D2 candidate: lift a saved Pauli body using central-rotation control.

No imports of Qiskit, I/O, grant issuance or scientific execution on import.
This is an unlaunched implementation candidate, not a certificate. The future
caller must authorize and seal the saved body and diagnostic scope first.
"""
from dataclasses import dataclass
import math


def _real(value):
    z = complex(value)
    if z.imag != 0 or not math.isfinite(z.real):
        raise ValueError('D2_FINITE_REAL_REQUIRED')
    return float(z.real)


def _tokens(body):
    """Read only explicit operands; never request lazy gate definitions."""
    from qiskit.circuit.library import PauliEvolutionGate, XGate, YGate, ZGate
    from qiskit.synthesis import LieTrotter
    if body.num_qubits < 1 or body.num_clbits or body.parameters:
        raise ValueError('D2_NUMERIC_UNITARY_BODY_REQUIRED')
    phase = _real(body.global_phase)
    tokens = []
    for item in body.data:
        op = item.operation
        if item.clbits or getattr(op, 'condition', None) is not None:
            raise ValueError('D2_CLASSICAL_OPERATION')
        wires = tuple(body.find_bit(q).index for q in item.qubits)
        if type(op) is PauliEvolutionGate:
            terms = op.operator.to_list()
            if (len(terms) != 1 or type(op.synthesis) is not LieTrotter
                    or op.synthesis.settings != LieTrotter().settings):
                raise ValueError('D2_SINGLE_PAULI_DEFAULT_SYNTHESIS_REQUIRED')
            label, coefficient = terms[0]
            if not label or len(label) != len(wires) or set(label)-set('IXYZ'):
                raise ValueError('D2_PAULI_LABEL_OR_WIRES')
            if set(label) == {'I'}:
                # Fixed Pauli source separates scalar constants from nonidentity
                # terms. Do not extend that scope: this runtime's all-I synthesis
                # has a different phase from the ideal single-Pauli formula.
                raise ValueError('D2_IDENTITY_EVOLUTION_OUTSIDE_FIXED_SOURCE')
            coefficient, time = _real(coefficient), _real(op.time)
            angle = _real(2 * coefficient * time)
            tokens.append(dict(kind='evolution', label=label, coefficient=coefficient,
                               time=time, central_rz_angle=angle, wires=wires))
        elif op.base_class in (XGate, YGate, ZGate) and len(wires) == 1 and not op.params:
            tokens.append(dict(kind='product', axis=op.name, wires=wires))
        else:
            raise ValueError('D2_UNSUPPORTED_BODY_OPERATION:'+op.name)
    return tokens, phase


def _rotation(circuit, label, angle, wires, ancilla):
    active = [(wires[j], axis) for j, axis in enumerate(reversed(label)) if axis != 'I']
    for q, axis in active:
        if axis == 'X':
            circuit.h(q)
        elif axis == 'Y':
            circuit.sdg(q)
            circuit.h(q)
    target = active[-1][0]
    for q, _ in active[:-1]:
        circuit.cx(q, target)
    circuit.crz(angle, ancilla, target)
    for q, _ in reversed(active[:-1]):
        circuit.cx(q, target)
    for q, axis in reversed(active):
        if axis == 'X':
            circuit.h(q)
        elif axis == 'Y':
            circuit.h(q)
            circuit.s(q)


@dataclass
class ControlledBody:
    body: object
    sidecar: dict


def central_control_body(body):
    """Return diag(I,U) with ancilla n. Source body is never mutated.

    Products retain chronological order. Negative product/order/constant phases
    already accumulated in the saved body are transferred once, as P(phi) on
    the ancilla. No per-event phase is added a second time.
    """
    from qiskit import QuantumCircuit
    tokens, phase = _tokens(body)  # Validate the whole source before emitting.
    ancilla = body.num_qubits
    controlled = QuantumCircuit(ancilla + 1)
    mapping = []
    for index, token in enumerate(tokens):
        start = len(controlled.data)
        if token['kind'] == 'evolution':
            _rotation(controlled, token['label'], token['central_rz_angle'], token['wires'], ancilla)
        else:
            getattr(controlled, 'c'+token['axis'])(ancilla, token['wires'][0])
        mapping.append(dict(source_instruction=index, output_start=start,
                            output_stop=len(controlled.data), source_token=token))
    phase_start = len(controlled.data)
    controlled.p(phase, ancilla)
    return ControlledBody(controlled, dict(
        control_policy='D2_candidate_central_rotation_control', ancilla=ancilla,
        source_instruction_count=len(tokens), instruction_mapping=mapping,
        aggregate_source_global_phase=phase,
        phase_owner='saved_body_aggregate_only_no_event_phase_readdition',
        phase_instruction=phase_start, numerical_certificate=False))


def full_wrapper(body):
    """Reuse the original cosine wrapper; no generic whole-body control call."""
    from .fixed_signal_evaluation import hadamard_wrapper
    result = central_control_body(body)
    circuit = hadamard_wrapper(result.body, 'cosine', already_controlled=True)
    return circuit, result
