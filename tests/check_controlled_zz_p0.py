"""P0 design/structure checks without compiling: published metadata, circuits, operators, budget guard.

No transpile, sampling, fit or price. The compile function is replaced by a fake
object in the budget test; the real compiles are only made by the approved runner.
"""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
from trottertracks import controlled_zz_p0 as Z
import qiskit


class P0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = Z.p.SavedArtifacts(ROOT)
        cls.data = Z.inputs(cls.store); cls.design = Z.design(cls.data)

    def case(self, form, context, m):
        return next(c for c in self.design['cases'] if (c['form'], c['context'], c['m']) == (form, context, m))

    def test_01_design_is_fixed_from_published_metadata(self):
        d = self.design
        self.assertEqual(len(d['cases']), Z.COMPILE_BUDGET)
        self.assertEqual(len(d['component']['support']), 2)
        zz = [c for c in self.data['components'] if len(c['support']) == 2]
        self.assertEqual(d['component']['probability'], max(c['probability'] for c in zz))
        self.assertEqual({c['R_outer'] for c in d['cases']}, {2, 128})
        for c in d['cases']:
            law = self.data['laws'][c['m']]
            self.assertEqual(c['rotation_angle'], law['rotation_angles']['0'])
            self.assertEqual(c['gate_angle'], 2 * d['component']['sign'] * law['rotation_angles']['0'])
        self.assertEqual(Z.design(Z.inputs(self.store)), d)                 # deterministic

    def test_02_circuits_follow_the_provider_and_counter_forms(self):
        g = Z.circuit(self.case('GENERIC_CONTROL', 'ALONE', 0), self.design, self.data)
        a = Z.circuit(self.case('ANALYTIC_CX_CRZ_CX', 'ALONE', 0), self.design, self.data)
        self.assertEqual(dict(g.count_ops()), {'crzz': 1}); self.assertEqual(dict(a.count_ops()), {'cx': 2, 'crz': 1})
        lowering = Z.decomposition(g)[0]
        self.assertEqual([x[0] for x in lowering['definition']], ['ccx', 'crz', 'ccx'])
        self.assertEqual(lowering['nested_counts']['ccx'], {'cx': 6, 't': 4, 'tdg': 3, 'h': 2})
        sandwich = Z.circuit(self.case('GENERIC_CONTROL', 'SAVED_BASIS_PAIR', 6), self.design, self.data)
        self.assertEqual(sum(1 for i in sandwich.data if i.operation.name in ('p', 'xx_plus_yy')), 2 * self.design['basis']['operations'])
        self.assertEqual(sandwich.data[0].operation.name, 'p')                      # inverse: last saved op first
        import subprocess
        provider = subprocess.check_output(['git', 'show', Z.PROVIDER + ':src/trotterlib/df_rte_qiskit.py'], cwd=ROOT)
        for text in (b'self._append_basis(circuit, definition, inverse=True)', b'gate = RZZGate(2.0 * angle)',
                     b'circuit.append(gate.control(1), [ancilla_qubit, *support])'):
            self.assertIn(text, provider)

    def test_03_operators_equal_including_control_and_global_phase(self):
        s0, s1 = self.design['component']['support']
        for m in Z.LAW_ROUNDS:
            for context in Z.CONTEXTS:
                g = Z.circuit(self.case('GENERIC_CONTROL', context, m), self.design, self.data)
                a = Z.circuit(self.case('ANALYTIC_CX_CRZ_CX', context, m), self.design, self.data)
                out = Z.operators(g, a, s0=s0, s1=s1)
                self.assertTrue(out['equal_including_global_phase'] and out['controlled_structure'])
                if context == 'ALONE':
                    self.assertLess(out['independent_matrix_error_generic'], Z.OPERATOR_ATOL)
                    self.assertLess(out['independent_matrix_error_analytic'], Z.OPERATOR_ATOL)

    def test_04_structural_S_counts_the_cheap_form(self):
        alone = Z.structural(self.case('GENERIC_CONTROL', 'ALONE', 0), self.design, self.data)
        self.assertEqual(alone['structural_rz'], 2)
        self.assertEqual(alone, Z.structural(self.case('ANALYTIC_CX_CRZ_CX', 'ALONE', 0), self.design, self.data))
        pair = Z.structural(self.case('GENERIC_CONTROL', 'SAVED_BASIS_PAIR', 6), self.design, self.data)
        self.assertGreater(pair['structural_rz'], alone['structural_rz'])

    def test_05_compile_budget_is_eight_and_never_exceeded(self):
        calls = []
        def fake(qc, settings):
            calls.append(qc.name)
            return SimpleNamespace(compiler_settings_hash=Z.p.PROFILE['compiler'], rz_count=0, cx_count=0, rz_depth=0, cx_depth=0,
                                   total_depth=0, circuit_size=0, posttranspile_gate_counts=(), transpiled_circuit=qiskit.QuantumCircuit(9))
        with patch.object(qiskit, 'transpile', side_effect=AssertionError('No real transpile in tests')):
            out = Z.execute(self.store, fake, settings=None)
        self.assertEqual(len(calls), 8); self.assertEqual(out['compile_calls'], calls)
        self.assertEqual((out['sampling'], out['fit'], out['prices']), (0, 0, 0))
        bad = lambda qc, s: SimpleNamespace(**dict(fake(qc, s).__dict__, compiler_settings_hash='OTHER'))
        with self.assertRaisesRegex(ValueError, 'Compiler settings hash'):
            Z.execute(self.store, bad, settings=None)

    def test_06_counter_and_provider_bytes_are_the_fixed_ones(self):
        import subprocess
        for commit, path in ((Z.COUNTER, 'src/trottertracks/h4_df10_price_v1.py'), (Z.COUNTER, 'src/trottertracks/track_d_native_resources.py'),
                             (Z.PROVIDER, 'src/trotterlib/rte_compiled_cost.py'), (Z.PROVIDER, 'src/trotterlib/df_rte_qiskit.py')):
            self.assertEqual(subprocess.check_output(['git', 'rev-parse', commit + ':' + path], cwd=ROOT).strip(),
                             subprocess.check_output(['git', 'hash-object', path], cwd=ROOT).strip())
        self.assertFalse([m for m in sys.modules if m.split('.')[0] in ('trotterlib', 'pyscf')])


if __name__ == '__main__':
    unittest.main(verbosity=2)
