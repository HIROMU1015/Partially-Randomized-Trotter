"""Scalar technical checks on artificial rationals; no F2 fixture or circuits.

The rational log series is an independent test oracle, not a replacement for
the production interval provider and not a finite-bit circuit certificate.
"""
from fractions import Fraction as F
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks.pr_dictionary_validation.intervals import (
    endpoints, rational_interval, scalar_context,
)
from trottertracks.pr_dictionary_validation.gates import shot_count_from_enclosure
from trottertracks.pr_dictionary_validation.pricing import UnpricedError


def log_bounds(value, terms=256):
    """Exact rational bounds via log(y)=2*atanh((y-1)/(y+1)).

    Reduce x=2**k*y, 1<=y<2. For 0<=z<=1/3 the positive tail after
    `terms` summands is <=2*z**(2*terms+1)/((2*terms+1)*(1-z*z)).
    """
    x = F(value)
    if x <= 0:
        raise ValueError('Positive log argument required')
    k = 0
    while x >= 2:
        x /= 2
        k += 1
    while x < 1:
        x *= 2
        k -= 1

    def reduced(y):
        z = (y - 1) / (y + 1)
        term = z
        total = F(0)
        for j in range(terms):
            total += 2 * term / (2*j + 1)
            term *= z*z
        return total, total + 2*term / ((2*terms + 1)*(1-z*z))

    lo, hi = reduced(x)
    two_lo, two_hi = reduced(F(2))
    if k >= 0:
        return lo + k*two_lo, hi + k*two_hi
    return lo + k*two_hi, hi + k*two_lo


class ScalarIntervalTests(unittest.TestCase):
    def setUp(self):
        self.ctx = scalar_context()

    def tearDown(self):
        # Importing the scalar primitives must not start the research pipeline.
        for module in ('numpy', 'scipy', 'qiskit', 'trotterlib'):
            self.assertNotIn(module, sys.modules)

    def assert_contains(self, interval, exact_lo, exact_hi=None):
        lo, hi = endpoints(interval)
        self.assertLessEqual(lo, exact_lo)
        self.assertGreaterEqual(hi, exact_lo if exact_hi is None else exact_hi)

    def test_exact_rational_conversion_and_directed_arithmetic(self):
        for x in (F(1, 10), F(-7, 13), F(0), F(2**400+1, 3**160)):
            self.assert_contains(rational_interval(self.ctx, x), x)
        x, y = F(7, 13), F(11, 17)
        a, b = (rational_interval(self.ctx, v) for v in (x, y))
        for actual, expected in ((a+b, x+y), (a-b, x-y), (a*b, x*y),
                                 (a/b, x/y), (a**2, x*x)):
            self.assert_contains(actual, expected)

    def test_log_encloses_independent_exact_series_bounds(self):
        # 80/320 also cover the two still-unadopted alpha=1/20 scope
        # candidates. Evaluating these scalar constants does not adopt either.
        for x in (F(1), F(2), F(1, 2), F(3, 2), F(10), F(64), F(1, 64),
                  F(80), F(320)):
            with self.subTest(x=str(x)):
                lo, hi = log_bounds(x)
                self.assert_contains(self.ctx.ln(rational_interval(self.ctx, x)), lo, hi)

    def test_ceil_resolves_integer_and_rejects_boundary_straddles(self):
        for lo, hi, expected in ((F(17), F(17), 17),
                                 (F(16999, 1000), F(17), 17),
                                 (F(17001, 1000), F(17999, 1000), 18)):
            self.assertEqual(shot_count_from_enclosure(lo, hi), expected)
        for lo, hi in ((F(17), F(17001, 1000)),
                       (F(16999, 1000), F(17001, 1000)),
                       (F(-1), F(0)), (F(2), F(1))):
            with self.assertRaisesRegex(ValueError, 'SHOT_COUNT_UNRESOLVED'):
                shot_count_from_enclosure(lo, hi)

    def test_nonfinite_endpoints_are_rejected(self):
        for value in ('inf', '-inf', 'nan'):
            with self.assertRaises(UnpricedError):
                endpoints(self.ctx.mpf(value))

    def test_artificial_shot_expression_matches_exact_oracle(self):
        # Neither the F2 candidate policy nor an arm's B/lambda is evaluated.
        bound, eta, alpha = F(7, 5), F(1, 8), F(1, 16)
        factor = 4*bound*bound/(eta*eta)
        for alpha_arm in (alpha, alpha/4):
            log_lo, log_hi = log_bounds(4/alpha_arm)
            oracle = shot_count_from_enclosure(factor*log_lo, factor*log_hi)
            b = rational_interval(self.ctx, bound)
            actual = 4*b*b/rational_interval(self.ctx, eta)**2 * self.ctx.ln(
                rational_interval(self.ctx, 4/alpha_arm))
            self.assert_contains(actual, factor*log_lo, factor*log_hi)
            self.assertEqual(shot_count_from_enclosure(*endpoints(actual)), oracle)


if __name__ == '__main__':
    unittest.main()
