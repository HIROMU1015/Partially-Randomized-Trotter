"""Efficient controlled ZZ: artificial equivalence/regression checks and the fixed P1 plan.

Synthetic events/applications only (no sampling, no compile). The provider builder is the
worktree copy, byte-identical to provider c31ef424 (checked).
"""
import ast
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from trottertracks import efficient_controlled_zz as E
from trottertracks import p1_efficient_zz as P
import trotterlib.df_rte_qiskit as dfq

LEGACY = dfq.QiskitDFRTEEventCircuitBuilder


def app(support, role='rotation', sign=1):
    return SimpleNamespace(diagonal_pauli_support=tuple(support), role=role, coefficient_sign=sign, is_identity=False)


def central(cls, support, role='rotation', sign=1, angle=0.3, controlled=True):
    qc = QuantumCircuit(4)
    ret = cls._append_central_application(qc, SimpleNamespace(unsigned_rotation_angle=angle), app(support, role, sign),
                                          controlled=controlled, ancilla_qubit=3 if controlled else None)
    return qc, ret


class EfficientZZTests(unittest.TestCase):
    def test_01_controlled_ZZ_is_the_same_unitary_for_both_signs(self):
        new = E.make_builder_class(LEGACY)
        for angle in (0.14218194063365502, -0.14218194063365502, 0.0022366815002455784, -1.3):
            for support in ((0, 1), (2, 0)):
                old_qc, old_ret = central(LEGACY, support, angle=angle)
                new_qc, new_ret = central(new, support, angle=angle)
                self.assertEqual(old_ret, new_ret)
                diff = np.abs(Operator(old_qc).data - Operator(new_qc).data).max()
                self.assertLess(diff, 1e-12)                                   # including global phase
                self.assertEqual(dict(new_qc.count_ops()), {'cx': 2, 'crz': 1})
                U = Operator(new_qc).data; dim = 8                              # ancilla = qubit 3 (most significant)
                self.assertLess(np.abs(U[:dim, :dim] - np.eye(dim)).max(), 1e-12)   # ancilla-0 branch is identity
                self.assertLess(max(np.abs(U[:dim, dim:]).max(), np.abs(U[dim:, :dim]).max()), 1e-12)

    def test_02_other_cases_are_delegated_unchanged(self):
        new = E.make_builder_class(LEGACY)
        cases = [dict(support=(1,)), dict(support=(0, 1), role='product', sign=-1), dict(support=(0,), role='product', sign=1),
                 dict(support=(0, 1), controlled=False), dict(support=(2,), controlled=False)]
        for kw in cases:
            old_qc, old_ret = central(LEGACY, **kw); new_qc, new_ret = central(new, **kw)
            self.assertEqual(old_ret, new_ret)
            self.assertEqual([(i.operation.name, [old_qc.find_bit(q).index for q in i.qubits], list(i.operation.params)) for i in old_qc.data],
                             [(i.operation.name, [new_qc.find_bit(q).index for q in i.qubits], list(i.operation.params)) for i in new_qc.data])
        for bad in (dict(support=(0, 1, 2)), dict(support=(0, 1), role='other')):
            with self.assertRaises(ValueError):
                central(new, **bad)

    def test_03_activation_is_explicit_reversible_and_not_nested(self):
        self.assertIs(dfq.QiskitDFRTEEventCircuitBuilder, LEGACY)
        with E.efficient_controlled_zz(dfq) as cls:
            self.assertIs(dfq.QiskitDFRTEEventCircuitBuilder, cls)
            self.assertEqual(cls.lowering_id, E.LOWERING_ID); self.assertTrue(issubclass(cls, LEGACY))
            with self.assertRaisesRegex(ValueError, 'nested'):
                with E.efficient_controlled_zz(dfq):
                    pass
        self.assertIs(dfq.QiskitDFRTEEventCircuitBuilder, LEGACY)
        try:
            with E.efficient_controlled_zz(dfq):
                raise RuntimeError('boom')
        except RuntimeError:
            pass
        self.assertIs(dfq.QiskitDFRTEEventCircuitBuilder, LEGACY)
        self.assertFalse(hasattr(LEGACY, 'lowering_id'))

    def test_04_instruction_parity_accepts_only_the_planned_replacement(self):
        old = QuantumCircuit(4); old.p(0.2, 0); old.append(__import__('qiskit').circuit.library.RZZGate(0.5).control(1), [3, 0, 1]); old.cz(3, 2)
        new = QuantumCircuit(4); new.p(0.2, 0); new.cx(0, 1); new.crz(0.5, 3, 1); new.cx(0, 1); new.cz(3, 2)
        self.assertEqual(E.instruction_parity(old, new), dict(status='PASS', replaced_controlled_zz=1, legacy_instructions=3, new_instructions=5))
        wrong = QuantumCircuit(4); wrong.p(0.2, 0); wrong.cx(0, 1); wrong.crz(-0.5, 3, 1); wrong.cx(0, 1); wrong.cz(3, 2)
        self.assertEqual(E.instruction_parity(old, wrong)['status'], 'FAIL')
        other = QuantumCircuit(4); other.p(0.3, 0); other.cx(0, 1); other.crz(0.5, 3, 1); other.cx(0, 1); other.cz(3, 2)
        self.assertEqual(E.instruction_parity(old, other)['reason'], 'OTHER_INSTRUCTION_DIFFERS')

    def test_05_PR_and_Full_H_paths_import_the_builder_at_call_time(self):
        for commit, path in ((P.CARRIER, 'scripts/h46_long_time_native.py'),
                             ('9b83b96b83f62142c9d87858dc24a94db7ac5d61', 'scripts/h46_full_h_pure_rte.py')):
            tree = ast.parse(subprocess.check_output(['git', 'show', commit + ':' + path], cwd=ROOT))
            nested = [n for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) for n in ast.walk(f)
                      if isinstance(n, ast.ImportFrom) and n.module == 'trotterlib.df_rte_qiskit'
                      and any(a.name == 'QiskitDFRTEEventCircuitBuilder' for a in n.names)]
            top = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == 'trotterlib.df_rte_qiskit']
            self.assertTrue(nested and not top, path)

    def test_06_legacy_provider_bytes_are_unchanged(self):
        for path in ('src/trotterlib/df_rte_qiskit.py', 'src/trotterlib/rte_compiled_cost.py'):
            self.assertEqual(subprocess.check_output(['git', 'rev-parse', E.PROVIDER + ':' + path], cwd=ROOT).strip(),
                             subprocess.check_output(['git', 'hash-object', path], cwd=ROOT).strip())
        self.assertIn(b'gate.control(1)', subprocess.check_output(['git', 'show', E.PROVIDER + ':src/trotterlib/df_rte_qiskit.py'], cwd=ROOT))

    def test_07_P1_plan_is_fixed_and_bounded(self):
        a = P.make_plan(ROOT, 'TEST'); b = P.make_plan(ROOT, 'TEST')
        self.assertEqual(a, b)
        self.assertEqual([c['m'] for c in a['cells']], [0, 6]); self.assertEqual(a['compile_budget'], 8); self.assertEqual(a['trajectory_paths'], 4)
        seed = P.trial_seed_function(ROOT)
        for cell, row in zip(a['cells'], a['menus']['H4']['candidates']):
            c = row['candidate']
            self.assertEqual(len(cell['seeds']), 2); self.assertEqual([len(s) for s in cell['seeds']], [c['q'], c['q']])
            self.assertEqual(cell['seeds'][1][0], seed(c, P.COHORT, 1, 0, 0))
            self.assertEqual(row['V2_candidate_id'][:5], 'RMSE:')
        self.assertEqual(a['lowering']['id'], E.LOWERING_ID)

    def test_08_event_counts_and_finite_summary(self):
        ev = [dict(taylor_order=0, application_sequence=[dict(is_identity=False, role='rotation', diagonal_pauli_support=[0, 1])]),
              dict(taylor_order=2, application_sequence=[dict(is_identity=False, role='product', diagonal_pauli_support=[2]),
                   dict(is_identity=False, role='product', diagonal_pauli_support=[0, 3]),
                   dict(is_identity=False, role='rotation', diagonal_pauli_support=[1])])]
        self.assertEqual(P.event_metrics(ev), dict(events=2, rotation_ZZ=1, rotation_Z=1, identity_applications=0,
                                                   product_applications=2, order2_events=1, orders={'0': 1, '2': 1}))
        plan = P.make_plan(ROOT, 'TEST')
        jobs = [dict(cell_id=plan['cells'][0]['id'], index=i, C=[10 + i, 10 + i], S=[8, 8], events=dict(rotation_ZZ=2, order2_events=0)) for i in (0, 1)]
        jobs = [dict(j, D=[c - s for c, s in zip(j['C'], j['S'])]) for j in jobs]
        s = P.summarize(dict(cells=plan['cells'][:1]), jobs)[0]
        self.assertEqual((s['n_paths'], s['X']['mean_D'], s['X_equals_Y']), (2, '5/2', True))
        self.assertEqual(s['expectation'], 'FINITE_DEVELOPMENT_COHORT_ONLY')


if __name__ == '__main__':
    unittest.main(verbosity=2)
