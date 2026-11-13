"""Artificial distributions only: no H4 probabilities, paths, kernels or prices."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import unittest

import numpy as np

from trottertracks import finite_k_sampler_law as m

ROOT = str(Path(__file__).resolve().parents[1])


class CategoricalTests(unittest.TestCase):
    def audit(self, p, **kwargs):
        return m.categorical_audit(p, cdf_law=m.saved_cdf_helper(ROOT), **kwargs)

    def test_dyadic_mass_zero_plateau_and_exhaustive_small_grid(self):
        a = self.audit([.25, 0., .375, .375])
        counts = [0] * 4
        for i in range(8):
            counts[np.searchsorted(a['cdf'], F(i, 8), side='right')] += 1
        self.assertEqual(a['G'], tuple(F(n, 8) for n in counts))
        self.assertEqual(a['TV']['total'], 0)
        self.assertEqual(a['grid_counts'][1], 0)
        self.assertEqual(sum(a['grid_counts']), 2**53)

    def test_right_boundary_including_zero(self):
        a = self.audit([0., .25, 0., .75])
        cdf = np.asarray([float(x) for x in a['cdf']])
        self.assertEqual(np.searchsorted(cdf, 0., side='right'), 1)
        self.assertEqual(np.searchsorted(cdf, .25, side='right'), 3)
        self.assertEqual(np.searchsorted(cdf, np.nextafter(.25, 0.), side='right'), 1)
        self.assertEqual(np.searchsorted(cdf, np.nextafter(1., 0.), side='right'), 3)

    def test_tiny_first_interval_receives_zero_grid_point(self):
        a = self.audit([2**-60, 1.])
        self.assertEqual(a['grid_counts'][0], 1)
        self.assertGreater(a['G'][0], a['F'][0])

    def test_positive_last_probability_can_disappear(self):
        a = self.audit([1., 1e-25])
        self.assertGreater(a['P'][1], 0)
        self.assertEqual(a['G'][1], 0)
        self.assertGreater(a['TV']['total'], 0)
        self.assertIsNone(a['physical_law_distance'])
        self.assertFalse(a['qualified_for_compiled_price'])

    def test_exact_decomposition_and_existing_helper_agree(self):
        a = self.audit([.1, .2, .7])
        helper = m.saved_cdf_helper(ROOT)([.1, .2, .7])
        self.assertEqual(float(2 * m.tv(a['Q'], a['G'])), helper['L1_roundoff'])
        self.assertLessEqual(a['TV']['total'], a['triangle_bound'])
        self.assertLessEqual(a['TV']['grid'], F(2, 2**53))
        self.assertEqual(sum(a['F']), 1)
        self.assertFalse(a['PRNG_independence_proved'])

    def test_reference_normalization_is_separate(self):
        a = self.audit([.5, .5], ideal_weights=[1., 3.])
        self.assertEqual(a['TV']['normalization'], F(1, 4))
        self.assertEqual(a['TV']['cdf_rounding'], 0)
        self.assertEqual(a['TV']['grid'], 0)

    def test_invalid_input_not_repaired(self):
        for p in ([], [0., 0.], [-.1, 1.1], [float('nan'), 1.], [float('inf')], [.2, .2]):
            with self.subTest(p=p), self.assertRaises(ValueError):
                self.audit(p)
        with self.assertRaises(ValueError):
            self.audit([1.], ideal_weights=[1., 1.])


class BoundTests(unittest.TestCase):
    def test_k2_bound_against_exhaustive_artificial_event_law(self):
        p, q = (F(3, 4), F(1, 4)), (F(2, 3), F(1, 3))
        c, d = (F(1, 3), F(2, 3)), (F(1, 2), F(1, 2))
        pp, qq = [], []
        for j, order in enumerate((0, 2)):
            for word in product(range(2), repeat=order + 1):
                x, y = p[j], q[j]
                for i in word:
                    x *= c[i]; y *= d[i]
                pp.append(x); qq.append(y)
        bound = m.k2_event_bound(p, q, m.tv(c, d), iid_components=True)
        self.assertLessEqual(m.tv(pp, qq), bound)
        self.assertLessEqual(bound, 1)
        self.assertIsNone(m.k2_event_bound(p, q, m.tv(c, d)))
        self.assertIsNone(m.k2_event_bound(p, q, None, iid_components=True))

    def test_equal_laws_and_k2_shape_validation(self):
        self.assertEqual(m.k2_event_bound([1, 0], [1, 0], 0, iid_components=True), 0)
        with self.assertRaises(ValueError):
            m.k2_event_bound([1], [1], 0, iid_components=True)
        with self.assertRaises(ValueError):
            m.tv([F(1, 2)], [1])

    def test_large_position_count_no_sequence_and_no_unproved_prng_bound(self):
        delta = F(1, 2**53)
        self.assertEqual(m.trajectory_bound(delta, 33_423_360, uniform_conditional_bound=True), 33_423_360 * delta)
        self.assertEqual(m.trajectory_bound(F(1, 2), 10, uniform_conditional_bound=True), 1)
        self.assertIsNone(m.trajectory_bound(delta, 33_423_360))
        self.assertIsNone(m.trajectory_bound(None, 2, uniform_conditional_bound=True))
        with self.assertRaises(ValueError):
            m.trajectory_bound(delta, -1)

    def test_equal_one_draw_marginals_do_not_imply_product_law(self):
        self.assertEqual(m.tv([F(1, 2)] * 2, [F(1, 2)] * 2), 0)
        self.assertEqual(m.tv([F(1, 4)] * 4, [F(1, 2), 0, 0, F(1, 2)]), F(1, 2))

    def test_cost_error_requires_known_full_support_range(self):
        self.assertIsNone(m.expectation_shift_bound(F(1, 4), None))
        self.assertIsNone(m.expectation_shift_bound(None, 10))
        self.assertEqual(m.expectation_shift_bound(F(1, 4), 10), F(5, 2))
        with self.assertRaises(ValueError):
            m.expectation_shift_bound(F(1, 4), -1)


if __name__ == '__main__':
    unittest.main()
