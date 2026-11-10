"""Synthetic checks only; no molecular inputs, saved signals or circuits."""
import unittest
from fractions import Fraction

import numpy as np
from scipy.linalg import expm
from trotterlib.calpha_macrostep_bounds import (
    macrostep_diagnostics, rational_inertia, certify_gap_from_brackets)


class MacrostepTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[0., 1.], [1., 0.]])
        self.z = np.diag([1., -1.])
        self.gamma = np.diag([.03, .05])

    def test_matches_existing_append_order(self):
        from trotterlib.b2_p1_signal_validation import compose_step
        heads = [.7*self.x, -.2*self.z, .1*(self.x+self.z)]
        info, step, _ = macrostep_diagnostics(heads, .3*self.z, self.gamma, .13)
        halves = [expm(-.5j*.13*h) for h in heads]
        central = expm(.13*(-.3j*self.z-self.gamma))
        np.testing.assert_allclose(step, compose_step(halves, central, 1), atol=2e-15)
        self.assertEqual([x['outer_to_inner_index'] for x in info['stage_terms_central_to_outer']], [2,1,0])

    def test_noncommuting_defect_is_locally_cubic(self):
        coarse, _, _ = macrostep_diagnostics([self.x, .2*self.z], .3*self.z, self.gamma, .02)
        fine, _, _ = macrostep_diagnostics([self.x, .2*self.z], .3*self.z, self.gamma, .01)
        self.assertTrue(7.9 < coarse['eta_macro']/fine['eta_macro'] < 8.1)
        self.assertLess(coarse['eta_macro'], coarse['local_commutator_bound'])

    def test_commuting_case_has_no_pf_error(self):
        info, _, _ = macrostep_diagnostics([self.z, .2*self.z], .3*self.z, self.gamma, .1)
        self.assertEqual(info['C_multi'], 0)
        self.assertLess(info['eta_macro'], 3e-16)
        self.assertIn('NOT_CERTIFIED', info['qualification'])

    def test_invalid_shape_rejected(self):
        with self.assertRaisesRegex(ValueError, 'COMMON_SQUARE'):
            macrostep_diagnostics([np.eye(3)], self.z, self.gamma, .1)


class InertiaTests(unittest.TestCase):
    def test_exact_mixed_inertia(self):
        # Eigenvalues -sqrt(2), +sqrt(2); integer arithmetic needs no solve.
        self.assertEqual(rational_inertia([[1,1],[1,-1]], 0),
                         dict(negative=1,positive=1,zero=0))

    def test_certifies_ordered_lowest_eigenvalues(self):
        certificate = certify_gap_from_brackets(
            [[-2,0,0],[0,Fraction(-7,4),0],[0,0,4]],
            ['-2.01','-1.99'], ['-1.76','-1.74'])
        self.assertEqual(Fraction(certificate['gap_lower_exact']), Fraction(23,100))

    def test_no_false_certificate_for_wrong_counts(self):
        with self.assertRaisesRegex(ValueError, 'COUNTS_NOT_CERTIFIED'):
            certify_gap_from_brackets([[-2,0],[0,-1]],[-3,-2.5],[-1.1,-.9])

    def test_asymmetric_matrix_and_zero_pivot_rejected(self):
        with self.assertRaisesRegex(ValueError, 'SYMMETRY'):
            rational_inertia([[1,2],[0,1]],0)
        with self.assertRaisesRegex(ValueError, 'UNDETERMINED'):
            rational_inertia([[0,1],[1,0]],0)


if __name__ == '__main__':
    unittest.main()
