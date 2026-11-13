"""Efficient controlled ZZ rotation for DF RTE tails: CX(s0,s1) . CRZ(2u)(anc,s1) . CX(s0,s1).

Legacy (provider c31ef424, unchanged): df_rte_qiskit._append_central_application appends
RZZGate(2u).control(1) on [ancilla, s0, s1], which Qiskit 1.3.0 lowers through generic
add_control to CCX.CRZ.CCX. The two are the same unitary including global and control
phase (P0, 442cafa1); only the gate cost differs. Here a subclass overrides only the
controlled two-wire rotation and delegates every other case (one-wire rotation, products,
uncontrolled events, validation) to the original static method. Activation is an explicit,
reversible swap of the provider module attribute, so the existing call-time imports in the
PR compose (h46_long_time_native.compose) and Full-H RTE (h46_full_h_pure_rte) pick it up
without any provider or carrier file being edited.
"""
from contextlib import contextmanager

LOWERING_ID = 'CONTROLLED_ZZ_CX_CRZ_CX_V1'
LEGACY_LOWERING_ID = 'CONTROLLED_ZZ_GENERIC_RZZ_CONTROL1_PROVIDER_C31EF424'
PROVIDER = 'c31ef424175c02379eafebe8fc6c7548c1d9aabe'
P0 = dict(result='442cafa14f75aeee9e04696df54aaaff1f53260e', source='a287b8e6cb629c6a089c99b66b758751910ee15a')


def make_builder_class(base):
    """Subclass of the provider QiskitDFRTEEventCircuitBuilder with the efficient controlled ZZ."""
    from qiskit.circuit.library import CRZGate
    legacy = base._append_central_application

    class EfficientControlledZZBuilder(base):
        lowering_id = LOWERING_ID
        legacy_class = base

        @staticmethod
        def _append_central_application(circuit, event, application, *, controlled, ancilla_qubit):
            support = application.diagonal_pauli_support
            if controlled and application.role == 'rotation' and support is not None and len(support) == 2:
                s0, s1 = support
                circuit.cx(s0, s1)
                circuit.append(CRZGate(2.0 * event.unsigned_rotation_angle), [ancilla_qubit, s1])
                circuit.cx(s0, s1)
                return 0.0                       # same return as the legacy rotation branch
            return legacy(circuit, event, application, controlled=controlled, ancilla_qubit=ancilla_qubit)

    EfficientControlledZZBuilder.__name__ = EfficientControlledZZBuilder.__qualname__ = 'EfficientControlledZZBuilder'
    return EfficientControlledZZBuilder


@contextmanager
def efficient_controlled_zz(module):
    """Swap module.QiskitDFRTEEventCircuitBuilder for the efficient subclass; always restored."""
    original = module.QiskitDFRTEEventCircuitBuilder
    if getattr(original, 'lowering_id', None) == LOWERING_ID:
        raise ValueError('Efficient controlled ZZ already active; nested activation is not allowed')
    module.QiskitDFRTEEventCircuitBuilder = make_builder_class(original)
    try:
        yield module.QiskitDFRTEEventCircuitBuilder
    finally:
        module.QiskitDFRTEEventCircuitBuilder = original


def instruction_parity(legacy_circuit, new_circuit):
    """New tail = legacy tail with each controlled RZZ replaced by cx,crz,cx on the same wires/angle.

    Every other instruction must be identical (name, qubits, parameters) and in the same order.
    """
    def ops(qc):
        return [(i.operation, [qc.find_bit(q).index for q in i.qubits]) for i in qc.data]
    old, new = ops(legacy_circuit), ops(new_circuit)
    if legacy_circuit.num_qubits != new_circuit.num_qubits or float(legacy_circuit.global_phase) != float(new_circuit.global_phase):
        return dict(status='FAIL', reason='QUBITS_OR_GLOBAL_PHASE')
    i = j = replaced = 0
    while i < len(old):
        op, qubits = old[i]
        if op.name == 'crzz' and len(qubits) == 3:
            anc, s0, s1 = qubits; theta = float(op.base_gate.params[0])
            want = [('cx', [s0, s1], []), ('crz', [anc, s1], [theta]), ('cx', [s0, s1], [])]
            got = [(o.name, q, [float(x) for x in o.params]) for o, q in new[j:j + 3]]
            if got != want:
                return dict(status='FAIL', reason='REPLACEMENT_DIFFERS', legacy_index=i)
            i += 1; j += 3; replaced += 1; continue
        if j >= len(new):
            return dict(status='FAIL', reason='NEW_TAIL_SHORTER', legacy_index=i)
        o2, q2 = new[j]
        if op.name != o2.name or qubits != q2 or [float(x) for x in op.params] != [float(x) for x in o2.params]:
            return dict(status='FAIL', reason='OTHER_INSTRUCTION_DIFFERS', legacy_index=i)
        i += 1; j += 1
    if j != len(new):
        return dict(status='FAIL', reason='NEW_TAIL_LONGER')
    return dict(status='PASS', replaced_controlled_zz=replaced, legacy_instructions=len(old), new_instructions=len(new))
