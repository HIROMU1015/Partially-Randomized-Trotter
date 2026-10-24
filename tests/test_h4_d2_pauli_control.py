"""D2 artificial tests only. No real H4 reads, builds or compiles."""
from contextlib import ExitStack
import importlib
import itertools
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_h4_d1_adapter import A, toy, event
from d1_private import h4_d2_pauli_control as D
from qiskit import QuantumCircuit, qpy
from qiskit.circuit import Gate, Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.synthesis import LieTrotter
import numpy as np


def evolution(body, label, coefficient, time, wires=None, **kw):
    body.append(PauliEvolutionGate(SparsePauliOp.from_list([(label, coefficient)]),
                                  time=time, **kw), list(range(len(label))) if wires is None else wires)


def oracle(body):
    u = Operator(body).data
    result = np.eye(2 * len(u), dtype=complex)
    result[len(u):, len(u):] = u
    return result


class CentralTests(unittest.TestCase):
    def equal(self, left, right):
        self.assertLess(float(np.max(np.abs(left-right))), 2e-12)

    def unequal(self, left, right):
        self.assertGreater(float(np.max(np.abs(left-right))), 1e-5)

    def test_all_two_qubit_labels_and_signed_rotations(self):
        for chars in itertools.product('IXYZ', repeat=2):
            if set(chars) == {'I'}:
                continue
            for coefficient in (-.7, .9):
                for time in (-.21, .13):
                    body = QuantumCircuit(2)
                    evolution(body, ''.join(chars), coefficient, time)
                    result = D.central_control_body(body)
                    self.equal(Operator(result.body).data, oracle(body))

    def test_mixed_xyz_little_endian_and_permuted_wires(self):
        body = QuantumCircuit(3)
        evolution(body, 'XYZ', -.31, .4, [2, 0, 1])
        evolution(body, 'YIX', .13, -.6, [1, 2, 0])
        body.y(0)
        self.equal(Operator(D.central_control_body(body).body).data, oracle(body))

    def test_products_negative_and_aggregate_phase(self):
        body = QuantumCircuit(2)
        body.x(0); body.y(1); body.z(0); body.y(0)
        body.global_phase = math.pi + .37
        result = D.central_control_body(body)
        self.equal(Operator(result.body).data, oracle(body))
        no_phase = result.body.copy()
        no_phase.data.pop()
        self.unequal(Operator(no_phase).data, oracle(body))

    def test_noncommuting_order_is_preserved(self):
        body = QuantumCircuit(2)
        evolution(body, 'XY', .7, .2)
        body.x(1)
        evolution(body, 'YZ', -.3, .8)
        self.equal(Operator(D.central_control_body(body).body).data, oracle(body))
        wrong = QuantumCircuit(2)
        evolution(wrong, 'YZ', -.3, .8); wrong.x(1); evolution(wrong, 'XY', .7, .2)
        self.unequal(oracle(wrong), oracle(body))

    def test_reused_d1_pf_pr_and_rte_orders_0_2_4(self):
        for method in ('PF', 'PR'):
            for k in (0, 2, 4):
                model, join = toy('pauli_jw', method)
                if method == 'PR':
                    join['events'] = [dict(event(k), outer_step_index=s, inner_RTE_position=0) for s in range(2)]
                original = A.build(model, method, join)
                circuit, controlled = D.full_wrapper(original.body)
                self.equal(Operator(controlled.body).data, oracle(original.body))
                self.equal(Operator(circuit.remove_final_measurements(inplace=False)).data,
                           Operator(original.circuit.remove_final_measurements(inplace=False)).data)

    def test_pure_scalar_phase_including_wraparound(self):
        for phase in (0., math.pi, -.41, 9 * math.pi + .17):
            body = QuantumCircuit(2)
            body.global_phase = phase
            result = D.central_control_body(body)
            self.equal(Operator(result.body).data, oracle(body))
            self.assertEqual(result.sidecar['aggregate_source_global_phase'], float(body.global_phase))
            self.assertEqual(float(result.body.global_phase), 0.)

    def test_transform_does_not_sample_build_compile_or_generic_control(self):
        model, join = toy('pauli_jw', 'PR')
        body = A.build(model, 'PR', join).body
        targets = [('rte', 'sample_rte_events'), ('rte', '_make_event'),
                   ('pauli_signal_adapter', 'sample_rte_events'),
                   ('df_partial_s2', 'prepare_df_partial_s2'), ('rte_compiled_cost', 'transpile')]
        with ExitStack() as stack:
            for module, name in targets:
                stack.enter_context(patch.object(importlib.import_module(A.__package__+'.'+module),
                                                  name, side_effect=AssertionError('FORBIDDEN:'+name)))
            stack.enter_context(patch.object(A, 'build', side_effect=AssertionError('REBUILD')))
            stack.enter_context(patch.object(QuantumCircuit, 'to_gate', side_effect=AssertionError('WHOLE_BODY_GATE')))
            stack.enter_context(patch.object(Gate, 'control', side_effect=AssertionError('GENERIC_CONTROL')))
            D.full_wrapper(body)

    def test_transform_never_hydrates_pauli_evolution_definition(self):
        body = QuantumCircuit(2)
        evolution(body, 'YX', -.8, .3)
        with patch.object(PauliEvolutionGate, '_define', side_effect=AssertionError('LAZY_SYNTHESIS')):
            D.central_control_body(body)

    def test_unsupported_and_nondefault_programs_stop(self):
        cases = []
        b = QuantumCircuit(1); b.h(0); cases.append(b)
        b = QuantumCircuit(1, 1); b.measure(0, 0); cases.append(b)
        b = QuantumCircuit(1); b.rz(Parameter('t'), 0); cases.append(b)
        b = QuantumCircuit(1); evolution(b, 'X', 1., .2)
        b.data[0].operation.operator.coeffs[0] = 1j; cases.append(b)
        b = QuantumCircuit(2); evolution(b, 'II', -.7, -.21); cases.append(b)
        b = QuantumCircuit(1); evolution(b, 'X', 1., .2, synthesis=LieTrotter(reps=2)); cases.append(b)
        b = QuantumCircuit(1)
        b.append(PauliEvolutionGate(SparsePauliOp.from_list([('X', .2), ('Z', .3)]), time=.4), [0]); cases.append(b)
        b = QuantumCircuit(1); b.append(Gate('x', 1, []), [0]); cases.append(b)
        for body in cases:
            with self.assertRaises(ValueError):
                D.central_control_body(body)

    def test_source_unchanged_and_exact_instruction_mapping(self):
        model, join = toy('pauli_jw', 'PR')
        body = A.build(model, 'PR', join).body
        before = A.api().canonical_qiskit_circuit_fingerprint(body)
        result = D.central_control_body(body)
        self.assertEqual(before, A.api().canonical_qiskit_circuit_fingerprint(body))
        self.assertEqual([m['source_instruction'] for m in result.sidecar['instruction_mapping']], list(range(len(body.data))))
        for mapping, instruction in zip(result.sidecar['instruction_mapping'], body.data):
            if mapping['source_token']['kind'] == 'evolution':
                self.assertEqual(mapping['source_token']['time'], float(instruction.operation.time))
                self.assertEqual(mapping['source_token']['coefficient'], instruction.operation.operator.to_list()[0][1].real)

    def test_qpy_roundtrip_and_original_wrapper_profile(self):
        model, join = toy('pauli_jw', 'PR')
        source = A.build(model, 'PR', join).body
        wrapper, result = D.full_wrapper(source)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'toy.qpy'
            with path.open('wb') as f: qpy.dump([source, result.body, wrapper], f)
            with path.open('rb') as f: restored = qpy.load(f)
        self.equal(Operator(restored[1]).data, oracle(restored[0]))
        self.equal(Operator(restored[2].remove_final_measurements(inplace=False)).data,
                   Operator(wrapper.remove_final_measurements(inplace=False)).data)
        profile = A.api().default_compiler()
        self.assertEqual(tuple(profile.basis_gates), ('rz', 'sx', 'x', 'cx'))
        self.assertEqual((profile.optimization_level, profile.transpiler_seed), (1, 17))
        self.assertEqual(wrapper.num_clbits, 1)

    def test_wrong_control_angle_and_y_basis_are_detected(self):
        body = QuantumCircuit(1)
        evolution(body, 'Y', .7, .3)
        expected = oracle(body)
        wrong = QuantumCircuit(2)
        wrong.sdg(0); wrong.h(0); wrong.crz(.42, 0, 1); wrong.h(0); wrong.s(0)
        self.unequal(Operator(wrong).data, expected)
        wrong = QuantumCircuit(2)
        wrong.sdg(0); wrong.h(0); wrong.crz(-.42, 1, 0); wrong.h(0); wrong.s(0)
        self.unequal(Operator(wrong).data, expected)
        wrong = QuantumCircuit(2)
        wrong.h(0); wrong.crz(.42, 1, 0); wrong.h(0)
        self.unequal(Operator(wrong).data, expected)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CentralTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if len(sys.argv) == 2:
        Path(sys.argv[1]).write_text(json.dumps(dict(scope='ARTIFICIAL_ONLY_MAX_3_SYSTEM_QUBITS',
            tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
            real_H4_build=0, real_H4_compile=0, scientific_grants_issued=0), indent=2)+'\n')
    sys.exit(not result.wasSuccessful())
