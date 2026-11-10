"""Artificial scalar/matrix enclosure checks; no molecular data."""
import math
import unittest
from fractions import Fraction
from trotterlib.calpha_interval_bounds import IntervalBounds, endpoint


class IntervalTests(unittest.TestCase):
    def setUp(self):
        self.b = IntervalBounds()

    def test_binary64_import_is_exact(self):
        x = self.b.real(.1)
        self.assertEqual(endpoint(x),Fraction.from_float(.1))
        self.assertEqual(endpoint(x,True),Fraction.from_float(.1))

    def test_complex_dagger_preserves_interval_norm(self):
        vector=self.b.matrix([[1+2j],[3-4j]])
        norm=(self.b.dagger(vector)*vector)[0,0]
        self.assertEqual(endpoint(norm.real),30)
        self.assertEqual(endpoint(norm.imag),0)

    def test_exp_remainder_covers_nonterminating_series(self):
        # Degree 3 is intentionally inaccurate: the explicit remainder matters.
        actual=self.b.c.exp(self.b.real(1))
        enclosed=self.b.expm(self.b.matrix([[1]]),degree=3)[0,0].real
        self.assertLessEqual(endpoint(enclosed),endpoint(actual))
        self.assertGreaterEqual(endpoint(enclosed,True),endpoint(actual,True))

    def test_tail_log_scalar_encloses_direct_functions(self):
        rt,gamma=self.b.tail_log(self.b.matrix([[.4]]),self.b.real(.6),self.b.real(.005),self.b.real(.4))
        reference=IntervalBounds(80)
        expected_rt=reference.c.atan2(reference.real(.005)*reference.real(.4),reference.real(1))/reference.real(.005)
        expected_gamma=(reference.c.ln(1+(reference.real(.6)*reference.real(.005))**2)
                        -reference.c.ln(1+(reference.real(.4)*reference.real(.005))**2))/(2*reference.real(.005))
        for value,target in [(rt[0,0].real,expected_rt),(gamma[0,0].real,expected_gamma)]:
            self.assertLessEqual(endpoint(value),endpoint(target))
            self.assertGreaterEqual(endpoint(value,True),endpoint(target,True))

    def test_weighted_scalar_geometric_bound(self):
        import numpy as np
        c=self.b.c
        z=self.b.matrix([[-.2]])
        out=self.b.weighted_state_pf(self.b.matrix([[.01]]),z,
            np.array([-.2+0j]),np.array([[1.+0j]]),np.array([1.+0j]),
            self.b.matrix([[1]]),self.b.real(.1),self.b.real(.1),10)
        point=sum(.01*math.exp(-.1*.1*(9-k)) * math.exp(-.2*.1*k) for k in range(10))
        self.assertLess(abs(float(out.b)-point),1e-15)


if __name__=='__main__':
    unittest.main()
