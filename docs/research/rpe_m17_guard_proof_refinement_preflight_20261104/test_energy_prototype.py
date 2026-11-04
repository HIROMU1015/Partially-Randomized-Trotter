"""Artificial proof examples, countermodels and small floating matrices only."""
from fractions import Fraction as Q
import unittest
from unittest.mock import patch

import numpy as np
from flint import ctx

import energy_prototype as p
import virtual_bridge_adapter as old


class ProofRefinementExamples(unittest.TestCase):
    def setUp(self):
        self.prec = ctx.prec
        ctx.prec = 96
        self.enterContext(patch.object(np, 'load', side_effect=AssertionError('No saved arrays')))
        self.enterContext(patch.object(old, 'numeric_components', side_effect=AssertionError('No real context')))
        self.enterContext(patch.object(old, 'bound_context', side_effect=AssertionError('No real bounds')))

    def tearDown(self):
        ctx.prec = self.prec

    def test_real_row_envelope_retains_additive_underflow(self):
        bounds = p.kernel.path_bounds(1, 2)
        self.assertEqual(p.real_row_error(Q(3, 8), 2), bounds.mv_rel*Q(3, 16)+bounds.dot_abs/2)
        self.assertGreater(p.real_row_error(0, 2), 0)

    def test_row_information_can_improve_on_frobenius_cauchy(self):
        # Artificial real A=(1/2)[[1,1],[1,-1]], x=e0. Imaginary row sums are 0.
        real, imaginary = p.real_row_error(Q(1, 2), 2), p.real_row_error(0, 2)
        b = old.b
        row_norm = b.upper((2*b.real(real)**2+2*b.real(imaginary)**2).sqrt())
        k = p.kernel.path_bounds(2, 2)
        old_norm = b.real(k.mv_rel+k.mv_abs)  # ||A||F=||x||=1.
        self.assertLess(row_norm, old_norm)

    def test_signed_energy_retains_phase_cancellation(self):
        # A=I, v=e0, e=(i-1)e0: action error is nonzero; output norm remains 1.
        error = complex(-1, 1)
        radial = Q.from_float(error.real)
        error_squared = Q.from_float(error.real)**2+Q.from_float(error.imag)**2
        self.assertGreater(error_squared, 0)
        self.assertEqual(p.energy_step(1, 0, radial, error_squared), 1)

    def test_unknown_radial_term_cannot_be_set_to_zero(self):
        delta = Q(1, 1024)
        self.assertEqual(p.energy_step(1, 0, delta, delta**2), (1+delta)**2)
        self.assertLess(p.energy_step(1, 0, 0, delta**2), (1+delta)**2)
        with self.assertRaises(ValueError): p.energy_step(1, 0, None, delta**2)

    def test_norm_error_model_does_not_imply_square_root_accumulation(self):
        # Adversarial errors allowed by the norm-only model; not actual fl(Iv).
        delta, count = Q(1, 1024), 16
        coherent = (1+delta)**count
        self.assertGreater(coherent, 1+4*delta)  # sqrt(16)=4 is unjustified.

    def test_exact_fr_inverse_does_not_cancel_unconstrained_roundoff(self):
        delta = Q(1, 1024)
        # Exact F=R=I but two independently bounded radial error maps.
        bound = p.compose(p.primitive(1+delta), p.primitive(1+delta))
        self.assertEqual(bound.final_scale, (1+delta)**2)
        self.assertGreater(bound.final_scale, 1)

    def test_events_between_f_and_r_prevent_algebraic_elimination(self):
        f = np.array([[0, 1], [1, 0]], complex)
        event = np.diag([1, -1]).astype(complex)
        vector = np.array([1, 0], complex)
        np.testing.assert_array_equal(f@f, np.eye(2))
        np.testing.assert_array_equal(f@(event@(f@vector)), -(event@vector))

    def test_contracting_block_tracks_internal_prefix_expansion(self):
        block = p.compose(p.primitive(Q(1, 8)), p.primitive(4))
        chain = p.power(block, 32)
        self.assertEqual(chain.final_scale, Q(1, 1 << 32))
        self.assertGreaterEqual(chain.prefix_scale, 4)
        self.assertGreater(chain.prefix_scale, chain.final_scale)

    def test_additive_affine_prefix_envelope_covers_every_toy_step(self):
        primitive = p.primitive(Q(1, 2), Q(1, 8))
        chain, value, peak = p.power(primitive, 12), Q(1), Q(1)
        for _ in range(12):
            value = primitive.final_scale*value+primitive.final_offset
            peak = max(peak, value)
        self.assertEqual(chain.final_scale+chain.final_offset, value)
        self.assertGreaterEqual(chain.prefix_scale+chain.prefix_offset, peak)
        self.assertGreater(chain.final_offset, 0)

    def test_signed_permutation_orbit_has_exact_long_chain_norm(self):
        maps = [np.array([[0, 1], [1, 0]], complex), np.diag([1, -1]).astype(complex), 1j*np.eye(2)]
        orbit = [np.array([a, 0], complex) for a in [1, -1, 1j, -1j]]
        orbit += [np.array([0, a], complex) for a in [1, -1, 1j, -1j]]
        for matrix in maps:
            for state in orbit:
                self.assertTrue(any(np.array_equal(matrix@state, target) for target in orbit))
        exact = p.power(p.primitive(1), 10**12)
        self.assertEqual(exact.final_scale, 1)
        self.assertEqual(exact.prefix_scale, 1)
        self.assertTrue(old.guard_completion(1, abs_evaluation_error=p.kernel.complex_abs_error(Q(1)),
                                             finite_intermediates=True))

    def test_nonidentity_metric_can_inflate_the_final_observable_bound(self):
        # H=diag(4,1), psi=(1,1). Products below are exact rationals.
        primal_squared, dual_squared, raw_norm_squared = Q(5), Q(5, 4), Q(2)
        self.assertGreater(primal_squared*dual_squared, raw_norm_squared**2)

    def test_invalid_envelopes_stop_and_norm_type_is_not_action_growth(self):
        for scale, offset in [(-1, 0), (1, -1), (True, 0)]:
            with self.assertRaises(ValueError): p.primitive(scale, offset)
        with self.assertRaises(ValueError): p.power(p.primitive(1), True)
        with self.assertRaises(ValueError): p.real_row_error(-1, 2)
        with self.assertRaises(ValueError): p.NormAffine(2, 0, 1, 0)
        self.assertFalse(hasattr(p.primitive(Q(1, 2)), 'gain'))
        self.assertFalse(hasattr(p.primitive(Q(1, 2)), 'additive'))


if __name__ == '__main__':
    unittest.main()
