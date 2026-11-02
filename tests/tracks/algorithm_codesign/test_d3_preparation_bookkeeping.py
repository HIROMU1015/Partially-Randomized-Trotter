"""Invented scalars/words only; no registered inputs, budgets or backends."""
from fractions import Fraction as F
import importlib.util
from math import comb, factorial
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('d3_plan', ROOT / 'scripts/tracks/algorithm_codesign/prepare_d3_guarantee_price_plan.py')
plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan)


class ProofBookkeeping(unittest.TestCase):
    def test_integral_remainder_constant(self):
        # Integrate (1-s)^5 term by term, independently of the bound formula.
        integral = sum((F((-1)**k * comb(5, k), k+1) for k in range(6)), F(0))
        self.assertEqual(integral / factorial(5), F(1, factorial(6)))

    def test_norm_factor_is_required(self):
        x, norm = F(1, 11), F(2)
        self.assertEqual(x**6 * norm**6 / factorial(6), 64*x**6/factorial(6))
        self.assertGreater(x**6 * norm**6 / factorial(6), x**6/factorial(6))

    def test_product_telescope_nonunitary(self):
        for delta, error, r in ((F(1, 53), F(1, 97), 3), (F(1, 71), F(1, 89), 4)):
            a, b = 1+delta+error, 1+delta
            telescope = error*sum((a**k*b**(r-1-k) for k in range(r)), F(0))
            self.assertEqual(telescope, a**r-b**r)
            self.assertGreater(telescope, r*error)

    def test_total_bias_and_native_reserve_separate(self):
        delta, error, r = F(1, 61), F(1, 103), 3
        truncation = (1+delta)**r-1
        implementation = (1+delta+error)**r-(1+delta)**r
        self.assertEqual(2*(truncation+implementation), 2*((1+delta+error)**r-1))
        self.assertGreater(implementation, 0)

    def test_guarantee_monotonicity_at_fixed_native_error(self):
        delta, error, r = F(1, 83), F(1, 101), 3
        old, new = 2*((1+3*delta+error)**r-1), 2*((1+delta+error)**r-1)
        self.assertLess(new, old)
        self.assertEqual(error, F(1, 101))


class StaticWordCertificate(unittest.TestCase):
    def test_literal_inverse_only(self):
        c = plan.no_rotation_cancellation_certificate([[('H',0),('R',0,'2/9',1),('CX',1,0)], [('CX',1,0),('R',0,'1/11',1),('H',0)]])
        self.assertTrue(c['no_R_can_be_cancelled'])
        self.assertEqual(c['event_pairs_evaluated'], 0)

    def test_rotation_cancellation_is_detected(self):
        c = plan.no_rotation_cancellation_certificate([[('R',0,'2/9',1)], [('R',0,'2/9',-1)]])
        self.assertFalse(c['no_R_can_be_cancelled'])

    def test_same_sign_rotation_is_not_an_inverse(self):
        c = plan.no_rotation_cancellation_certificate([[('R',0,'2/9',1)]])
        self.assertTrue(c['no_R_can_be_cancelled'])

    def test_phase_gate_barrier_is_kept(self):
        c = plan.no_rotation_cancellation_certificate([[('Z',1),('R',0,'2/9',1),('Z',1),('R',0,'2/9',-1)]])
        self.assertTrue(c['no_R_can_be_cancelled'])

    def test_empty_domain_fails(self):
        with self.assertRaises(ValueError):
            plan.no_rotation_cancellation_certificate([])


class InventoryIdentity(unittest.TestCase):
    def setUp(self):
        self.options = dict(seed=0, up_to_phase=False)
        self.contract = dict(primitive_error='1/1000000', synthesizer_options=self.options,
                             tool_identity=dict(sha256='a'*64))
        self.old_spec = plan.key_spec('2/9', '1/1000000', self.options, 'a'*64)
        self.old = dict(keys=[dict(ratio='2/9', key=plan.key_hash(self.old_spec))])
        self.rows = [dict(row=0, ratios=['2/9']), dict(row=1, ratios=['2/9'])]

    def test_rational_normalization_and_one_common_key(self):
        a = plan.key_spec('4/18', '1/1000000', self.options, 'a'*64)
        self.assertEqual(plan.key_hash(a), plan.key_hash(self.old_spec))
        inventory = plan.inventory_from_rows(self.rows, self.old, self.contract)
        self.assertEqual(inventory['used_unique_keys'], 1)
        self.assertEqual(inventory['keys'][0]['rows'], [0,1])
        self.assertIsNone(inventory['keys'][0]['new_price'])

    def test_precision_is_part_of_identity(self):
        a = plan.key_spec('2/9', '1/10000000', self.options, 'a'*64)
        self.assertNotEqual(plan.key_hash(a), plan.key_hash(self.old_spec))

    def test_tool_and_seed_are_part_of_identity(self):
        a = plan.key_spec('2/9', '1/1000000', dict(seed=1, up_to_phase=False), 'a'*64)
        b = plan.key_spec('2/9', '1/1000000', self.options, 'b'*64)
        self.assertNotEqual(plan.key_hash(a), plan.key_hash(self.old_spec))
        self.assertNotEqual(plan.key_hash(b), plan.key_hash(self.old_spec))

    def test_missing_or_tampered_old_key_fails(self):
        with self.assertRaises(ValueError):
            plan.inventory_from_rows(self.rows, dict(keys=[]), self.contract)
        with self.assertRaises(ValueError):
            plan.inventory_from_rows(self.rows, dict(keys=[dict(ratio='2/9',key='bad')]), self.contract)

    def test_projective_guard_and_zero_precision_rejected(self):
        with self.assertRaises(ValueError):
            plan.key_spec('2/9','1/1000000',dict(up_to_phase=True),'a'*64)
        with self.assertRaises(ValueError):
            plan.key_spec('2/9','0',self.options,'a'*64)


if __name__ == '__main__':
    unittest.main()
