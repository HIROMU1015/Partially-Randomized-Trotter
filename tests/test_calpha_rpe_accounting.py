"""Artificial raw-radius and branch checks; no scientific inputs."""
import cmath
from fractions import Fraction
import unittest
from trotterlib.calpha_interval_bounds import IntervalBounds
from trotterlib.calpha_rpe_accounting import account_raw_radius,branch_sufficient_conditions


class AccountingTests(unittest.TestCase):
    def test_raw_radius_counts_are_integer_and_not_reattenuated(self):
        a=account_raw_radius(Fraction(1,2),IntervalBounds())
        b=account_raw_radius(Fraction(1,4),IntervalBounds())
        self.assertEqual(a['attenuation_applied'],1)
        self.assertIsInstance(a['N_X'],int)
        self.assertEqual(a['N_round'],2*a['N_X'])
        self.assertTrue(3.99 < b['N_X']/a['N_X'] < 4.01)
        self.assertGreaterEqual(a['N_X'],a['interval_ceiling'])

    def test_nonpositive_radius_has_no_accounting(self):
        with self.assertRaises(ValueError):account_raw_radius(0,IntervalBounds())

    def test_adversarial_synthetic_phases_use_existing_decoder(self):
        from trotterlib.b2_p1_branch_validation import decode_raw_means
        q=[1<<m for m in range(18)]
        for energy in (-127.999,0.,127.999):
            signals=[cmath.exp(1j*(-.02*energy*x+(.4 if m%2 else -.4))) for m,x in enumerate(q)]
            decoded=decode_raw_means(signals,q)
            self.assertLessEqual(abs(decoded['final_energy_estimate_Ha']-energy),.4/(.02*q[-1])+1e-12)
        self.assertTrue(branch_sufficient_conditions(IntervalBounds())['sufficient'])


if __name__=='__main__':unittest.main()
