"""Synthetic circuits only; no saved molecular QPY, state or result inputs."""
from fractions import Fraction
import json
from pathlib import Path
import sys
import unittest

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate, Parameter
from qiskit.circuit.library import RYGate, UnitaryGate, XXPlusYYGate
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Operator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import m5_phase_repair_prototype as candidate


def toy():
    q = QuantumCircuit(3)
    q.global_phase = .17
    q.h(2)
    q.p(.7, 0)
    q.p(-.7, 1)  # Different qubits: retain both rotations.
    q.rzz(.14, 0, 1)
    q.append(XXPlusYYGate(.23, .9), [0, 1])
    q.cx(2, 1)
    q.p(-.8, 2)  # Relative phase, not a freely discardable scalar.
    q.sx(0)
    q.h(2)
    return q


def bytes_metadata(q):
    return json.dumps(dict(phase=float(q.global_phase).hex(),
        operations=[(i.operation.name, [float(p).hex() for p in i.operation.params],
                      [q.find_bit(x).index for x in i.qubits]) for i in q.data]),
                      sort_keys=True).encode()


class PhaseRepairTests(unittest.TestCase):
    def test_default_rejects_real_use(self):
        with self.assertRaisesRegex(ValueError, 'ARTIFICIAL_ONLY'):
            candidate.lower_p_once(toy())
        with self.assertRaisesRegex(ValueError, 'ARTIFICIAL_ONLY'):
            candidate.ArtificialPassRecorder()
        with self.assertRaisesRegex(ValueError, 'ARTIFICIAL_ONLY'):
            candidate.lower_native_once(toy())

    def test_input_unchanged_and_no_pair_deletion(self):
        q = toy()
        before = bytes_metadata(q)
        lower, receipt = candidate.lower_p_once(q, artificial=True)
        self.assertEqual(bytes_metadata(q), before)
        self.assertEqual(len(q.data), len(lower.data))
        self.assertEqual(receipt['replaced_P'], 3)
        self.assertEqual(receipt['deleted_P_pairs'], 0)
        for old, new in zip(q.data, lower.data):
            self.assertEqual([q.find_bit(x).index for x in old.qubits],
                             [lower.find_bit(x).index for x in new.qubits])
            self.assertEqual(old.operation.params, new.operation.params)
            self.assertEqual(new.operation.name,
                             'rz' if old.operation.name == 'p' else old.operation.name)

    def test_complete_operator_and_controlled_operator(self):
        q = toy()
        lower, _ = candidate.lower_p_once(q, artificial=True)
        self.assertLess(np.linalg.norm(Operator(q).data - Operator(lower).data), 1e-12)
        original_controlled = Operator(q.to_gate().control(1)).data
        lowered_controlled = Operator(lower.to_gate().control(1)).data
        self.assertLess(np.linalg.norm(original_controlled - lowered_controlled), 1e-12)

    def test_relative_phase_cannot_be_dropped(self):
        q = QuantumCircuit(3)
        q.h(2); q.p(.8, 2); q.h(2)
        correct, _ = candidate.lower_p_once(q, artificial=True)
        missing = QuantumCircuit(3); missing.h(2); missing.h(2)
        self.assertGreater(np.linalg.norm(Operator(q).data-Operator(missing).data), .1)
        self.assertLess(np.linalg.norm(Operator(q).data-Operator(correct).data), 1e-12)

    def test_exact_sum_before_single_rounding(self):
        q = QuantumCircuit(3)
        for angle in (1e16, 1., -1e16): q.p(angle, 0)
        _, receipt = candidate.lower_p_once(q, artificial=True)
        self.assertEqual(receipt['exact_global_plus_half_P_sum'],
                         candidate.payload(Fraction(1, 2)))
        self.assertEqual(receipt['assigned_global_phase_binary64_hex'], (.5).hex())

    def test_large_sum_and_signed_wrap(self):
        from mpmath import mp
        context = mp.clone(); context.dps = 450
        for number in (-.7, .7, 0., 1e200, -1e200):
            f = candidate.exact(number)
            value = context.mpf(f.numerator)/f.denominator
            period = 2*context.pi
            reference = float(value-context.floor(value/period)*period)
            self.assertEqual(candidate._rounded_phase(f).hex(), reference.hex())

    def test_unknown_symbolic_and_nonfinite_refuse(self):
        q = QuantumCircuit(3); q.p(Parameter('theta'), 0)
        with self.assertRaisesRegex(ValueError, 'SYMBOLIC'):
            candidate.lower_p_once(q, artificial=True)
        q = QuantumCircuit(3); q.rx(.2, 0)
        with self.assertRaisesRegex(ValueError, 'UNSUPPORTED'):
            candidate.lower_p_once(q, artificial=True)
        for value in (True, float('nan'), float('inf')):
            with self.assertRaises(ValueError): candidate.exact(value)

    def test_custom_gate_cannot_spoof_p(self):
        q = QuantumCircuit(3); q.append(Gate('p', 1, [.2]), [0])
        with self.assertRaisesRegex(ValueError, 'UNSUPPORTED'):
            candidate.lower_p_once(q, artificial=True)
        snap = candidate.snapshot(circuit_to_dag(q))
        self.assertEqual(snap['status'], 'UNSUPPORTED_DETERMINANT_NULL')
        self.assertIsNone(snap['F_global_plus_half_P_exact'])

    def test_measurement_and_classical_condition_guard(self):
        q = QuantumCircuit(3, 1); q.p(.3, 2); q.measure(2, 0)
        lower, _ = candidate.lower_p_once(q, artificial=True)
        self.assertEqual(lower.data[-1].operation.name, 'measure')
        q.x(2)
        with self.assertRaisesRegex(ValueError, 'MID_CIRCUIT'):
            candidate.lower_p_once(q, artificial=True)
        q = QuantumCircuit(3, 1); q.p(.2, 0).c_if(q.cregs[0], 1)
        with self.assertRaisesRegex(ValueError, 'UNSUPPORTED'):
            candidate.lower_p_once(q, artificial=True)

    def test_trace_copies_metadata_not_mutable_DAG(self):
        q = toy(); dag = circuit_to_dag(q)
        recorder = candidate.ArtificialPassRecorder(artificial=True)
        class Pass:
            def name(self): return 'ARTIFICIAL'
        recorder(pass_=Pass(), dag=dag, time=0., property_set={}, count=0)
        frozen = json.dumps(recorder.rows, sort_keys=True)
        dag.global_phase = 2.
        self.assertEqual(json.dumps(recorder.rows, sort_keys=True), frozen)
        self.assertEqual(recorder.rows[0]['snapshot']['gate_matrices_evaluated'], 0)

    def test_trace_unknown_or_wrong_size_not_complete_evidence(self):
        q = QuantumCircuit(3); q.rx(.1, 0)
        snap = candidate.snapshot(circuit_to_dag(q))
        self.assertIsNone(snap['P_angle_sum_exact'])
        self.assertFalse(snap['determinant_modulo_evaluated'])
        with self.assertRaisesRegex(ValueError, 'THREE_QUBITS'):
            candidate.snapshot(circuit_to_dag(QuantumCircuit(2)))

    def test_unknown_matrix_gate_remains_null_without_evaluation(self):
        q = QuantumCircuit(3)
        q.append(UnitaryGate([[0, 1], [1, 0]]), [0])
        snap = candidate.snapshot(circuit_to_dag(q))
        self.assertEqual(snap['unknown_gates'], {'unitary': 1})
        self.assertIsNone(snap['F_global_plus_half_P_exact'])

    def test_topological_measurement_of_independent_wire_is_terminal(self):
        q = QuantumCircuit(3, 1)
        q.p(.3, 0); q.p(.2, 1); q.measure(2, 0)
        snap = candidate.snapshot(circuit_to_dag(q))
        self.assertEqual(snap['status'], 'SUPPORTED_SCALAR_METADATA')

    def test_real_callback_on_synthetic_opt1_does_not_change_output(self):
        q, _ = candidate.lower_native_once(toy(), artificial=True)
        recorder = candidate.ArtificialPassRecorder(artificial=True)
        recorder.boundary(q, 'INPUT')
        settings = dict(basis_gates=['rz','sx','x','cx'], optimization_level=1,
                        seed_transpiler=17, num_processes=1)
        recorded = transpile(q, callback=recorder, **settings)
        recorder.boundary(recorded, 'OUTPUT')
        plain = transpile(q, **settings)
        self.assertEqual(bytes_metadata(recorded), bytes_metadata(plain))
        self.assertTrue(any('pass_name' in row for row in recorder.rows))
        self.assertEqual(recorder.rows[0]['boundary'], 'INPUT')
        self.assertEqual(recorder.rows[-1]['boundary'], 'OUTPUT')
        self.assertEqual(recorder.rows[-1]['snapshot']['status'], 'SUPPORTED_SCALAR_METADATA')
        self.assertLess(np.linalg.norm(Operator(q).data-Operator(recorded).data), 1e-12)

    def test_P_only_does_not_fix_XX_equivalence_path(self):
        q = toy(); lower, receipt = candidate.lower_p_once(q, artificial=True)
        compiled = transpile(lower, basis_gates=['rz','sx','x','cx'],
                             optimization_level=1, seed_transpiler=17, num_processes=1)
        # Regression of a discovered failure; this is not an acceptance PASS.
        self.assertGreater(np.linalg.norm(Operator(q).data-Operator(compiled).data), 1.)
        self.assertLess(np.linalg.norm(1j*Operator(q).data-Operator(compiled).data), 1e-12)
        self.assertFalse(receipt['production_installed'])
        self.assertFalse(receipt['postcompile_correction'])

    def test_explicit_native_basis_before_and_after_opt1(self):
        q = toy(); before = bytes_metadata(q)
        lower, receipt = candidate.lower_native_once(q, artificial=True)
        self.assertEqual(bytes_metadata(q), before)
        self.assertLess(np.linalg.norm(Operator(q).data-Operator(lower).data), 1e-12)
        compiled = transpile(lower, basis_gates=['rz','sx','x','cx'],
                             optimization_level=1, seed_transpiler=17, num_processes=1)
        self.assertLess(np.linalg.norm(Operator(q).data-Operator(compiled).data), 1e-12)
        self.assertLess(np.linalg.norm(Operator(q.to_gate().control()).data-
                                      Operator(lower.to_gate().control()).data), 1e-12)
        self.assertTrue(receipt['changed_compiler_input_policy'])
        self.assertFalse(receipt['old_cost_compatibility'])

    def test_explicit_RY_and_XX_small_representative_cases(self):
        for gate in (RYGate(.0), RYGate(-.4), XXPlusYYGate(.23, .9),
                     XXPlusYYGate(-.31, -.8)):
            with self.subTest(gate=gate.name, params=gate.params):
                q = QuantumCircuit(3); q.append(gate, range(gate.num_qubits))
                lower, _ = candidate.lower_native_once(q, artificial=True)
                compiled = transpile(lower, basis_gates=['rz','sx','x','cx'],
                                     optimization_level=1, seed_transpiler=17, num_processes=1)
                self.assertLess(np.linalg.norm(Operator(q).data-Operator(compiled).data), 1e-12)

    def test_parameterized_constant_phase_minimal_reproducer(self):
        from qiskit.circuit.equivalence import EquivalenceLibrary
        from qiskit.transpiler import PassManager
        from qiskit.transpiler.passes import BasisTranslator
        theta = Parameter('theta')
        for kind in ('numeric', 'expression', 'zero'):
            definition = QuantumCircuit(1)
            definition.global_phase = .123 if kind == 'numeric' else theta/2 if kind == 'expression' else 0.
            definition.rz(theta, 0)
            library = EquivalenceLibrary()
            library.add_equivalence(Gate('phase_probe', 1, [theta]), definition)
            q = QuantumCircuit(3); q.append(Gate('phase_probe', 1, [.2]), [0])
            compiled = PassManager(BasisTranslator(library, ['rz'])).run(q)
            # The fixed installed translator drops numeric constant phase.
            # Retain this as observed FAIL evidence, not a relaxed assertion.
            self.assertEqual(float(compiled.global_phase), .1 if kind == 'expression' else 0.)
            ideal = QuantumCircuit(3)
            ideal.compose(definition.assign_parameters({theta:.2}), [0], inplace=True)
            error = np.linalg.norm(Operator(ideal).data-Operator(compiled).data)
            self.assertGreater(error, .1) if kind == 'numeric' else self.assertLess(error, 1e-12)


if __name__ == '__main__': unittest.main()
