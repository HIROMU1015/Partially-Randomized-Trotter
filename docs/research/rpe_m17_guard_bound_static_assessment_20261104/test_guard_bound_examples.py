"""Exact/ball artificial examples of proof conservatism; no scientific loader.

These are mathematical examples, not a new norm adapter or scientific runner.
The real KernelBounds coefficients and H4 arrays are never evaluated here.
"""
from fractions import Fraction as Q
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from flint import acb, acb_mat, ctx

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'docs/research/rpe_m17_virtual_measurement_bridge_preparation_20261104'))
import virtual_bridge_adapter as v

b = v.b


def toy_kernel(**fields):
    return v.KernelBounds(**(dict.fromkeys(v.KernelBounds.__dataclass_fields__, 0) | fields))


class StaticBoundExamples(unittest.TestCase):
    def setUp(self):
        self.prec, self.threads = ctx.prec, ctx.threads
        ctx.prec, ctx.threads = 96, 1  # Only constant-size artificial matrices.
        for owner, name in [(v.np, 'load'), (v, 'numeric_components'), (v, 'bound_context')]:
            self.enterContext(patch.object(owner, name, side_effect=AssertionError('No real input')))

    def tearDown(self):
        ctx.prec, ctx.threads = self.prec, self.threads

    def test_phase_error_is_not_norm_inflation_and_not_a_mean_bound(self):
        # A=-I vs U=I, applied three times to a unit vector: exact norm=1,
        # exact signal=-1, ideal signal=+1. Norm gain cannot bound mean error.
        action = v.power(v.action_growth(2, 1, toy_kernel()), 3)
        error, old_magnitude = v.signal_error({'lower': '1', 'upper': '1'}, action, toy_kernel())
        self.assertGreaterEqual(error, 2)
        self.assertGreater(old_magnitude, 1)
        norm_only = v.Growth(1)
        wrong_error, norm_magnitude = v.signal_error({'lower': '1', 'upper': '1'}, norm_only, toy_kernel())
        self.assertEqual(wrong_error, 0)  # Not a valid action-error input here.
        self.assertLess(wrong_error, abs((-1)**3-1))
        self.assertEqual(norm_magnitude, 1)
        self.assertTrue(v.guard_completion(norm_magnitude, abs_evaluation_error=0, finite_intermediates=True))

    def test_defect_bound_covers_nonunitary_matrix_on_exact_toy_vectors(self):
        a = acb_mat([[1, b.real(Q(1, 8))], [0, 1]])
        rho = b.frobenius(b.dagger(a)*a-b.identity(2))
        p = b.upper((1+rho).sqrt())
        for vector in [acb_mat([[1], [0]]), acb_mat([[0], [1]]),
                       acb_mat([[b.real(Q(3, 5))], [b.real(Q(4, 5))]]),
                       acb_mat([[1], [acb(0, 1)]])]:
            self.assertLessEqual(b.frobenius(a*vector), b.upper(p*b.frobenius(vector)))
        self.assertGreater(rho, 0)

    def test_rotation_identity_needs_nonhermitian_cross_term(self):
        p = acb_mat([[acb(0, 1), 0], [0, acb(0, -1)]])
        identity = b.identity(2)
        c = s = b.real(Q(3, 4))
        t = identity*c-p*acb(0, s)
        lhs = b.dagger(t)*t-identity
        rhs = identity*(c*c+s*s-1)+(b.dagger(p)*p-identity)*(s*s)+(b.dagger(p)-p)*acb(0, c*s)
        self.assertEqual(lhs, rhs)
        self.assertEqual(b.dagger(p)*p, identity)
        # Omitting the skew-Hermitian term gives an invalid norm bound.
        omitted = b.upper((1+abs(c*c+s*s-1)).sqrt())
        true_norm_on_e0 = b.frobenius(t*acb_mat([[1], [0]]))
        self.assertGreater(true_norm_on_e0, omitted)
        full_rho = b.upper(abs(c*c+s*s-1)+abs(c*s)*b.frobenius(b.dagger(p)-p))
        self.assertGreaterEqual(b.upper((1+full_rho).sqrt()), true_norm_on_e0)

    def test_rotation_rounding_formula_covers_injected_exact_scalar_errors(self):
        # Scalar P=1; errors saturate the declared artificial kernel bounds.
        c, s = Q(3, 4), Q(1, 2)
        d, k, sub = Q(1, 16), Q(1, 32), Q(1, 64)
        am, asc, at = Q(1, 128), Q(1, 256), Q(1, 512)
        vector = b.real(1)
        mv = vector*(1+b.real(d))+b.real(am)
        first = vector*b.real(c)*(1+b.real(k))+b.real(asc)
        second = acb(0, -(mv*b.real(s)*(1+b.real(k))+b.real(asc)))
        # Exact rational/complex injected errors, not actual installed rounding.
        subtraction_error = b.real(sub)*(abs(first)+abs(second))+b.real(at)
        output = acb(first)+second+subtraction_error
        a0, b0 = (1+k)*c, (1+k)*s*(1+d)
        erel = k*c+s*d+k*s*(1+d)+sub*(a0+b0)
        eabs = (1+sub)*((1+k)*s*am+2*asc)+at
        # P=1 makes ||cI-isP||=sqrt(c²+s²), known exactly as a ball.
        upper = b.upper((b.real(c*c+s*s)).sqrt()+b.real(erel+eabs))
        self.assertLessEqual(b.upper(abs(output)), upper)
        self.assertGreater(eabs, 0)

    def test_positive_rare_mass_stays_in_full_support(self):
        rare = Q(1, 1 << 40)
        average, worst = v.trajectory_growth(v.Growth(1), v.Growth(1),
            [v.Growth(1), v.Growth(2)], [1-rare, rare], [0, 2], r=1, q=1)
        self.assertEqual(b.dyadic(average.gain), 1+rare)
        self.assertEqual(worst.gain, 2)
        _, excluded = v.trajectory_growth(v.Growth(1), v.Growth(1),
            [v.Growth(1), v.Growth(2)], [Q(1), Q(0)], [0, 2], r=1, q=1)
        self.assertEqual(excluded.gain, 1)

    def test_kernel_term_remains_with_zero_construction_residual(self):
        # Artificial unitary scalar repeated eight times, no H4 count/coefficients.
        kappa = Q(1, 1 << 20)
        kernel = toy_kernel(mv_rel=kappa)
        action = v.power(v.action_growth(0, 1, kernel), 8)
        _, magnitude = v.signal_error({'lower': '1', 'upper': '1'}, action, kernel)
        self.assertGreaterEqual(magnitude, b.real(1+8*kappa))
        self.assertFalse(v.guard_completion(magnitude, abs_evaluation_error=0, finite_intermediates=True))

    def test_component_extrema_coupling_can_tighten_a_valid_bound(self):
        residuals, frobenius, kappa = [Q(1, 8), Q(0)], [Q(1), Q(2)], Q(1, 8)
        paired = max(delta+kappa*f for delta, f in zip(residuals, frobenius))
        separate = max(residuals)+kappa*max(frobenius)
        self.assertLess(paired, separate)
        self.assertTrue(all(delta+kappa*f <= paired for delta, f in zip(residuals, frobenius)))

    def test_unproved_guard_keeps_both_means_null_despite_small_action_bound(self):
        axes = v.classify_axes({c: {'lower': '0', 'upper': '0'} for c in ['real', 'imag']},
            dict(real=0, imag=0), {a: {'lower': '0', 'upper': '0'} for a in ['cosine', 'sine']},
            dict(cosine='1', sine='1'), 0, completion=False, arithmetic_resolved=True)
        for axis in axes.values():
            self.assertIsNone(axis['U_virtual_IID_upper'])
            self.assertIsNone(axis['U_exec_upper'])
            self.assertEqual(axis['decision'], 'UNDETERMINED')


if __name__ == '__main__':
    unittest.main()
