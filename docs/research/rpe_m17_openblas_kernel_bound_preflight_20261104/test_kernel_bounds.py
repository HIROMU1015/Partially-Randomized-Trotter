"""Artificial arrays and exact-rational proof checks; no real scientific data."""
from fractions import Fraction as Q
import contextlib
import io
import itertools
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import kernel_bounds as k
import virtual_bridge_adapter as v
import run_virtual_bridge as runner


def pair(z):
    return Q.from_float(float(z.real)), Q.from_float(float(z.imag))


def exact_dot(x, y, conjugate=False):
    real, imag = Q(0), Q(0)
    for a, b in zip(x, y, strict=True):
        ar, ai = pair(a); br, bi = pair(b)
        if conjugate: ai = -ai
        real += ar*br-ai*bi
        imag += ar*bi+ai*br
    return real, imag


def norm2(values):
    return sum((a*a+b*b for a, b in map(pair, values)), Q(0))


def check_error(test, actual, exact, upper):
    ar, ai = pair(actual)
    squared = (ar-exact[0])**2+(ai-exact[1])**2
    bound = v.b.dyadic(v.b.upper(upper))
    test.assertLessEqual(squared, bound*bound)


class KernelTests(unittest.TestCase):
    def test_profile_and_fp_mode_do_not_accept_ftz_daz_or_directed_rounding(self):
        for state in [0x1f80, 0x1fa0, 0x1fa2]: self.assertTrue(k.fp_mode_supported(state))
        for bit in [6, 13, 14, 15]: self.assertFalse(k.fp_mode_supported(0x1f80 | (1 << bit)))
        self.assertFalse(k.fp_mode_supported(0))
        self.assertFalse(k.fp_mode_supported(True))
        self.assertIsNone(k.selected_path_contract({}, rows=4, columns=4, layout='C', mxcsr_values=[0x1f80])['bounds'])
        self.assertIsNone(k.selected_path_contract(k.PROFILE, rows=4, columns=4, layout='F', mxcsr_values=[0x1f80])['bounds'])
        out = k.selected_path_contract(k.PROFILE, rows=4, columns=4, layout='C', mxcsr_values=[0x1f80, 0x1fa2])
        self.assertEqual(out['status'], 'CURRENT_KERNEL_BOUND_DERIVED')
        self.assertFalse(out['scientific_execution_ready'])

    def test_coefficients_have_positive_underflow_terms_and_valid_scope(self):
        bounds = k.path_bounds(3, 4)
        self.assertIsInstance(bounds, v.KernelBounds)
        for name in ('mv_abs', 'dot_abs', 'scale_abs', 'sub_abs'):
            self.assertGreater(getattr(bounds, name), 0)
        for n in [0, 71, True]:
            with self.assertRaises(ValueError): k.path_bounds(n, 4)

    def test_gemv_against_exact_rational_on_cancellation_and_subnormal_toys(self):
        smallest = float.fromhex('0x0.0000000000001p-1022')
        for small in [1., 2.**-100, smallest]:
            matrix = np.array([[small+small*1j, -small, small*.5j],
                               [small*.25, small-small*.5j, -small]], dtype=complex)
            vector = np.array([.1+.2j, -.3+.125j, .25-.1j], dtype=complex)
            result, bounds = matrix @ vector, k.path_bounds(2, 3)
            upper = (v.b.real(bounds.mv_rel)*v.b.real(norm2(matrix.ravel())).sqrt()
                     *v.b.real(norm2(vector)).sqrt()+v.b.real(bounds.mv_abs))
            error2 = Q(0)
            for row, actual in zip(matrix, result, strict=True):
                target = exact_dot(row, vector)
                ar, ai = pair(actual)
                error2 += (ar-target[0])**2+(ai-target[1])**2
            self.assertLessEqual(error2, v.b.dyadic(v.b.upper(upper))**2)

    def test_dot_conjugation_and_underflow(self):
        tiny = float.fromhex('0x0.0000000000001p-1022')
        for scale in [1., 2.**100, tiny]:
            x = np.array([scale*(.5+.25j), scale*(-.75+.5j), scale*(.25-.5j)], complex)
            y = np.array([.1+.2j, .3-.4j, -.5+.125j], complex)
            bounds = k.path_bounds(3, 3)
            upper = (v.b.real(bounds.dot_rel)*v.b.real(norm2(x)).sqrt()*v.b.real(norm2(y)).sqrt()
                     +v.b.real(bounds.dot_abs))
            check_error(self, np.vdot(x, y), exact_dot(x, y, conjugate=True), upper)

    def test_larger_artificial_gemv_uses_same_exact_reference_bound(self):
        # Dimension alone is shared with the future context; no H/state/DF input.
        size = 70
        indices = np.arange(size*size).reshape(size, size)
        matrix = np.asarray((indices % 7-.3)/32 + 1j*(indices.T % 5+.1)/64, dtype=complex)
        matrix = np.ascontiguousarray(matrix)
        vector = np.asarray((np.arange(size) % 3-.1)/16+1j/32, dtype=complex)
        result, bounds = matrix @ vector, k.path_bounds(size, size)
        upper = (v.b.real(bounds.mv_rel)*v.b.real(norm2(matrix.ravel())).sqrt()
                 *v.b.real(norm2(vector)).sqrt()+v.b.real(bounds.mv_abs))
        squared = Q(0)
        for row, actual in zip(matrix, result, strict=True):
            re, im = exact_dot(row, vector)
            ar, ai = pair(actual)
            squared += (ar-re)**2+(ai-im)**2
        self.assertLessEqual(squared, v.b.dyadic(v.b.upper(upper))**2)

    def test_scale_and_subtract_against_exact_complex_arithmetic(self):
        scalar = complex(.1, -.3)
        tiny = float.fromhex('0x0.0000000000001p-1022')
        x = np.array([.25+.125j, 2.**100-(2.**98)*1j, tiny+tiny*1j], complex)
        bounds = k.path_bounds(3, 3)
        scale_bound = v.b.real(bounds.scale_rel)*v.b.real(norm2([scalar])).sqrt()*v.b.real(norm2(x)).sqrt()+v.b.real(bounds.scale_abs)
        for actual, value in zip(scalar*x, x, strict=True):
            check_error(self, actual, exact_dot([scalar], [value]), scale_bound)
        y = x*np.float64(.75)
        sub_bound = v.b.real(bounds.sub_rel)*(v.b.real(norm2(x)).sqrt()+v.b.real(norm2(y)).sqrt())+v.b.real(bounds.sub_abs)
        for a, b, actual in zip(x, y, x-y, strict=True):
            ar, ai = pair(a); br, bi = pair(b)
            check_error(self, actual, (ar-br, ai-bi), sub_bound)

    def test_arbitrary_reduction_orders_stay_within_tree_bound(self):
        terms = [1., 2.**-53, -1., 2.**-54]
        target = sum(map(Q.from_float, terms))
        bound = k.gamma(2*len(terms)+16)*sum(abs(Q.from_float(x)) for x in terms)
        for order in itertools.permutations(terms):
            sequential = 0.
            for x in order: sequential += x
            pairwise = (order[0]+order[1])+(order[2]+order[3])
            for value in (sequential, pairwise):
                self.assertLessEqual(abs(Q.from_float(value)-target), bound)

    def test_fused_and_unfused_product_sum_fit_same_budget(self):
        a, b, c = 1.+2.**-27, 1.-2.**-27, -1.
        target = Q.from_float(a)*Q.from_float(b)+Q.from_float(c)
        fused = float(target)  # rational -> nearest binary64, simulates one FMA
        unfused = a*b+c
        self.assertNotEqual(fused, unfused)
        bound = k.gamma(2)*(abs(Q.from_float(a)*Q.from_float(b))+abs(Q.from_float(c)))
        for value in (fused, unfused): self.assertLessEqual(abs(Q.from_float(value)-target), bound)

    def test_hypot_polynomial_budget_is_exact_rational_proof(self):
        self.assertLess(k.hypot_polynomial_error(), 256*k.U)
        self.assertLess(k.hypot_relative_proof_bound(), 1024*k.U)
        self.assertEqual(k.complex_abs_error(1), 1024*k.U+k.ETA)

    def test_hypot_subnormal_zero_axis_and_near_unit_toys(self):
        # Exact squared check; avoids using float hypot as its own reference.
        tiny = float.fromhex('0x0.0000000000001p-1022')
        for x, y in [(0., 0.), (tiny, tiny), (tiny, 0.), (.75, .25), (1., 2.**-54), (2.**100, 2.**99)]:
            rho = v.b.real(Q.from_float(x)**2+Q.from_float(y)**2).sqrt()
            magnitude = v.b.dyadic(v.b.upper(rho))
            upper = v.b.real(k.complex_abs_error(magnitude))
            self.assertLessEqual(v.b.upper(abs(v.b.real(abs(complex(x, y)))-rho)), v.b.upper(upper))

    def test_overflow_screen_and_completion_are_separate(self):
        self.assertTrue(k.finite_intermediate_envelope(matrix_frobenius=2, initial_norm=1, boundary_norm=2, scalar_modulus=1))
        self.assertFalse(k.finite_intermediate_envelope(matrix_frobenius=1 << 201, initial_norm=1, boundary_norm=2, scalar_modulus=1))
        self.assertTrue(v.guard_completion(1, abs_evaluation_error=k.complex_abs_error(1), finite_intermediates=True))
        self.assertFalse(v.guard_completion(v.b.real('1.000000000002'), abs_evaluation_error=k.complex_abs_error(2), finite_intermediates=True))
        self.assertFalse(v.guard_completion(1, abs_evaluation_error=k.complex_abs_error(1), finite_intermediates=False))

    def test_old_source_status_and_scientific_run_remain_blocked(self):
        self.assertIsNone(v.installed_kernel_contract()['bounds'])
        with patch.object(runner, 'metadata', return_value=[]), patch.object(np, 'load', side_effect=AssertionError('No real arrays')), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(['--run']), 2)

    def test_recorded_dispatch_covers_runtime_ufunc_and_thread_modes(self):
        text = (Path(__file__).parent/'dispatch_observation.txt').read_text()
        for token in ['TRACE GEMVT m=70 n=4', 'TRACE complex DOT n=70', 'CDOUBLE_multiply_FMA3__AVX2',
                      'CDOUBLE_subtract_FMA3__AVX2', 'TRACE _Py_c_abs', 'TRACE hypot', 'norm real DOT']:
            self.assertIn(token, text)
        states = [int(line.split('mxcsr=')[-1], 16) for line in text.splitlines() if 'mxcsr=' in line]
        self.assertTrue(states)
        self.assertTrue(all(k.fp_mode_supported(state) for state in states))


if __name__ == '__main__':
    unittest.main()
