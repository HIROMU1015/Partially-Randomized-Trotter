"""Artificial operator tests; no H8 inputs, samples, or scientific launcher."""
import copy
import math
import sys
from pathlib import Path
import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import Operator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import h8_compiler_phase_bookkeeping as phase


class PhaseTests(unittest.TestCase):
    def fixture(self, number):
        wrapper = QuantumCircuit(2)
        for _ in range(number):
            wrapper.append(XXPlusYYGate(.37, .19), [0, 1])
        # A basis circuit with a known extra phase; no compiler bug is needed
        # for this artificial accounting test.
        compiled = QuantumCircuit(2)
        compiled.global_phase = (number % 4) * math.pi / 2
        records = {key: dict(name=op.name, extra_phase_quarters=1, error=0.)
                   for key, op in phase.nonbasis_gates(wrapper).items()}
        proof = dict(compiler=phase.a.COMPILER, policy=phase.POLICY, records=records, maximum_error=0.)
        return wrapper, compiled, proof

    def test_operator_phase_and_counts(self):
        for n in (1, 2, 4):
            with self.subTest(n=n):
                wrapper, compiled, proof = self.fixture(n)
                original = float(compiled.global_phase)
                corrected, meta = phase.correct_phase(wrapper, compiled, proof)
                self.assertLess(np.linalg.norm(Operator(corrected).data - np.eye(4)), 1e-12)
                self.assertEqual(meta['delta_phase_quarters'], (-n) % 4)
                self.assertEqual(compiled.global_phase, original)
                self.assertEqual(corrected.count_ops(), compiled.count_ops())
                self.assertFalse(meta['statevector_phase_alignment'])

    def test_uncertified_angle_rejected(self):
        wrapper, compiled, proof = self.fixture(2)
        wrapper.append(XXPlusYYGate(.38, .19), [0, 1])
        with self.assertRaisesRegex(ValueError, 'ACTUAL_GATE_PARAMETERS_CERTIFIED'):
            phase.correct_phase(wrapper, compiled, proof)

    def test_failed_operator_or_wrong_compiler_rejected(self):
        wrapper, compiled, proof = self.fixture(2)
        bad = copy.deepcopy(proof)
        next(iter(bad['records'].values()))['error'] = 1.
        with self.assertRaisesRegex(ValueError, 'CERTIFIED_GATE_OPERATOR_PHASE'):
            phase.correct_phase(wrapper, compiled, bad)
        bad = copy.deepcopy(proof)
        bad['compiler'] = 'another compiler'
        with self.assertRaisesRegex(ValueError, 'FIXED_COMPILER_PHASE_POLICY'):
            phase.correct_phase(wrapper, compiled, bad)

    def test_actual_small_operator_and_reuse(self):
        wrapper = QuantumCircuit(2)
        wrapper.append(XXPlusYYGate(.37, .19), [0, 1])
        wrapper.h(0)
        cache = {}
        proof = phase.certify(wrapper, cache)
        self.assertEqual(len(cache), 2)
        self.assertLess(proof['maximum_error'], 1e-10)
        self.assertEqual(phase.certify(wrapper, cache), proof)

    def test_full_artificial_wrapper_operator(self):
        from qiskit import transpile
        for n in (1, 2, 4):
            with self.subTest(n=n):
                wrapper = QuantumCircuit(2)
                wrapper.h(0)
                for j in range(n):
                    wrapper.append(XXPlusYYGate(.37 + j * .03, .19), [0, 1])
                    wrapper.rz(.13, 0)
                compiled = transpile(wrapper, basis_gates=['rz', 'sx', 'x', 'cx'],
                    optimization_level=1, seed_transpiler=17, num_processes=1)
                proof = phase.certify(wrapper, {})
                corrected, _ = phase.correct_phase(wrapper, compiled, proof)
                self.assertGreater(compiled.count_ops().get('rz', 0), 0)
                self.assertEqual(corrected.count_ops(), compiled.count_ops())
                self.assertLess(np.linalg.norm(Operator(corrected).data - Operator(wrapper).data), 1e-12)


if __name__ == '__main__':
    unittest.main()
