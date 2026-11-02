"""Artificial counter/phase checks plus missing-input and display safeguards."""
import importlib.util
from pathlib import Path
from fractions import Fraction as F
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('current_s2', ROOT/'src/trotterlib/current_s2_theory.py')
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
m = a.m


class FixedNativeS2Tests(unittest.TestCase):
    def setUp(self):
        self.basis = m.Basis('synthetic', 'synthetic-source', (m.Gate('rz', (0,), .23),))
        self.primitive = lambda j, t: NS(rz=((0, t),), rzz=(), global_phase=t*.3)
        self.pf = dict(method='B0', formula='2nd', q=3)

    def test_saved_B0I_scalar_required(self):
        with self.assertRaisesRegex(ValueError, 'MISSING_ORIGINAL'):
            a.scalar_phase('B0I', constant=.1, T=.8)

    def test_B0I_uses_original_saved_sum(self):
        self.assertEqual(a.scalar_phase('B0I', constant=999., T=.8, saved_B0I_scalar=.35), -.35*.8)

    def test_PR_identity_is_included_once(self):
        self.assertEqual(a.scalar_phase('B2', constant=.1, extracted=.25, T=.8), -(.1+.25)*.8)

    def test_B0_does_not_recover_tail_identity(self):
        self.assertEqual(a.scalar_phase('B0', constant=.1, extracted=.25, T=.8), -.1*.8)

    def test_scalar_emit_is_final_not_per_outer_step(self):
        c = m.deterministic_circuit((self.basis,), self.primitive, .8, 3, '2nd',
                                    'symmetric_directional', 1, -.2, 'A')
        self.assertEqual(sum(g.name == 'p' for g in c.operations), 1)
        no = m.deterministic_circuit((self.basis,), self.primitive, .8, 3, '2nd',
                                     'symmetric_directional', 1, 0., 'A')
        self.assertEqual(m.count(c).rz, m.count(no).rz+1)

    def test_identity_cancelled_saved_scalar_emits_no_extra_P(self):
        p = a.scalar_phase('B0I', constant=.1, T=.8, saved_B0I_scalar=0.)
        self.assertEqual(p, 0.)

    def test_native_units_wrapper_axis(self):
        u = a.native_units(candidate=self.pf, bases=(self.basis,), primitive_factory=self.primitive,
                           ancilla=1, constant=.1)
        self.assertEqual(F(u['sine']['lower']) - F(u['cosine']['lower']), 1)

    def test_grouped_reference_exact_for_toy_law(self):
        cs = (m.Component('z', F(2,3), self.basis, (0,), 1),
              m.Component('identity', F(1,3), None, (), -1))
        orders = {0:F(3,4), 2:F(1,4)}
        fast = a.grouped.expected_rte_grouped(m, cs, orders, 2, .2, 1, 'B')
        slow = m.expected_rte_occurrence(cs, orders, 2, .2, 1, 'B')
        self.assertEqual(fast.lower, slow.lower)
        self.assertEqual(fast.upper, slow.upper)

    def test_missing_law_fails(self):
        with self.assertRaisesRegex(ValueError, 'MISSING_COMPLETE_PR_LAW'):
            a.native_units(candidate=dict(method='B2',formula='2nd',q=1,r=2),
                           bases=(self.basis,),primitive_factory=self.primitive,ancilla=1,constant=.1,extracted=.2)

    def test_shot_accounting_has_both_axes(self):
        units={'cosine':dict(lower='2',upper='3'),'sine':dict(lower='4',upper='5')}
        self.assertEqual(a.total_from_saved_shots(units,dict(cosine=7,sine=11)),dict(lower='58',upper='76'))
        with self.assertRaises(ValueError):a.total_from_saved_shots(units,dict(cosine=7))

    def test_missing_theory_remains_null(self):
        r=a.fixed_pair_display(dict(compiled='3',theory=None),dict(compiled='4',theory=None))
        self.assertIsNone(r['rho_theory_lower']); self.assertEqual(r['rho_compiled'],'3/4')

    def test_negative_relative_change_is_preserved(self):
        r=a.fixed_pair_display(dict(compiled='6',theory=dict(lower='4',upper='4')),
                               dict(compiled='2',theory=dict(lower='4',upper='4')))
        self.assertEqual(r['PR_change']['lower'],'-50')
        self.assertEqual(r['rho_theory_lower'],'1');self.assertEqual(r['rho_compiled'],'3')

    def test_S4_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'S2_ONLY'):
            a.native_units(candidate=dict(method='B0',formula='4th',q=1),bases=(),primitive_factory=None,ancilla=1,constant=0.)


if __name__ == '__main__':unittest.main()
