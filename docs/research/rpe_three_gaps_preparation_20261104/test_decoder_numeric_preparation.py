"""Artificial B-only tests. Never loads H4 arrays/means or runs science."""
from fractions import Fraction as Q
import math
import unittest
from unittest.mock import patch

import numpy as np
from flint import ctx
from common import D, real
import decoder_finite as finite
import decoder_numeric_preparation as numeric


def row(q=1, center=('1', '0'), eps=('3/5', '3/5'), n=4):
    return dict(q=q, N_cosine=n, N_sine=n, mu_cosine=center[0],
                mu_sine=center[1], eps_cosine=eps[0], eps_sine=eps[1])


class NumericPreparationTests(unittest.TestCase):
    def setUp(self):
        self.prec = ctx.prec
        ctx.prec = 192
        self.functions, _ = D.existing_functions()

    def tearDown(self):
        ctx.prec = self.prec

    def test_mean_paths_differ_and_each_exact_error_is_bounded(self):
        self.assertNotEqual(finite.represented_mean(1, 3, 'count_grid'),
                            finite.represented_mean(1, 3, 'integer_sum'))
        for n in range(1, 20):
            for k in range(n+1):
                exact = Q(2*k, n)-1
                for style in ('integer_sum', 'count_grid'):
                    error = abs(Q.from_float(finite.represented_mean(k, n, style))-exact)
                    self.assertLessEqual(error, numeric.mean_bound_under_rn(n, style))

    def test_int8_mean_and_fsum_match_integer_sum_on_entire_small_grid(self):
        for n in (3, 7, 16):
            for k in range(n+1):
                outcomes = np.array([1]*k+[-1]*(n-k), dtype=np.int8)
                expected = finite.represented_mean(k, n, 'integer_sum')
                self.assertEqual(float(np.mean(outcomes)), expected)
                self.assertEqual(math.fsum(outcomes)/n, expected)

    def test_unproved_mean_domain_or_path_stops(self):
        for n, style in ((0, 'integer_sum'), (True, 'count_grid'),
                         (2**52+1, 'integer_sum'), (4, 'unspecified')):
            with self.assertRaisesRegex(ValueError, 'UNKNOWN/STOP'):
                numeric.mean_bound_under_rn(n, style)

    def test_exact_open_success_set_excludes_boundary(self):
        r = row(center=('0', '0'), eps=('1/2', '1/2'))
        self.assertEqual(finite.feasible_pairs(r), [(2, 2)])
        self.assertEqual(numeric.finite_input_envelope(r, 'integer_sum')['decision'], 'UNKNOWN/STOP')

    def test_whole_round_input_covered_once_with_wrap(self):
        for style in ('integer_sum', 'count_grid'):
            original = np.angle
            with patch.object(numeric.np, 'angle', wraps=original) as angle:
                out = numeric.finite_input_envelope(row(), style)
            self.assertEqual(angle.call_count, len(finite.feasible_pairs(row())))
            self.assertEqual(out['successful_count_pairs'], 6)
            self.assertEqual(out['decision'], 'EXHAUSTIVE_SUPPLIED_ROUND_INPUT_ONLY')
            self.assertLess(Q(out['a_supplied_grid_upper']['upper']), Q(1, 10**12))
            self.assertEqual(out['H4_uniform_decoder_decision'], 'UNKNOWN/STOP')

    def test_negative_real_axis_and_both_pi_seams_are_circular(self):
        out = numeric.finite_input_envelope(row(center=('-1', '0')), 'integer_sum')
        self.assertEqual(out['successful_count_pairs'], 6)
        self.assertLess(Q(out['a_supplied_grid_upper']['upper']), Q(1, 10**12))
        for imag in (0.0, -0.0):
            phase = float(np.angle(complex(-1., imag)))
            principal = self.functions['_wrap_phase'](phase)
            self.assertLess(D.circle_distance(principal, D.arb.pi()), real('1e-14'))

    def test_outward_grid_bound_contains_higher_precision_comparisons(self):
        r = row(n=3, eps=('1', '1'))
        bound = Q(numeric.finite_input_envelope(r, 'count_grid')['a_supplied_grid_upper']['upper'])
        with ctx.workprec(384):
            from flint import acb
            for k, l in finite.feasible_pairs(r):
                exact = acb(real(Q(2*k, 3)-1), real(Q(2*l, 3)-1)).arg()
                z = complex(finite.represented_mean(k, 3), finite.represented_mean(l, 3))
                actual = self.functions['_wrap_phase'](float(np.angle(z)))
                self.assertGreaterEqual(real(bound), D.circle_distance(actual, exact))

    def test_entire_n3_set_uses_exact_means_for_each_path(self):
        r = row(n=3, center=('1', '0'), eps=('1', '1'))
        for style in ('integer_sum', 'count_grid'):
            out = numeric.finite_input_envelope(r, style)
            self.assertEqual(out['successful_count_pairs'], 4)
            self.assertGreater(Q(out['coordinate_error_exact'][1]), 0)
            self.assertLessEqual(Q(out['coordinate_error_exact'][1]),
                                 numeric.mean_bound_under_rn(3, style))

    def test_empty_set_nonfinite_angle_and_wrap_stay_unknown(self):
        empty = row(center=('1/4', '1/4'), eps=('1/100', '1/100'))
        self.assertEqual(numeric.finite_input_envelope(empty, 'count_grid')['decision'], 'UNKNOWN/STOP')
        with patch.object(numeric.np, 'angle', return_value=float('nan')):
            self.assertIsNone(numeric.finite_input_envelope(row(), 'integer_sum')['a_supplied_grid_upper'])
        with patch.object(numeric.D, 'existing_functions', return_value=({'_wrap_phase': lambda x: float('inf')}, {})):
            self.assertIsNone(numeric.finite_input_envelope(row(), 'integer_sum')['a_supplied_grid_upper'])

    def test_prewrap_analytic_constant_against_artificial_complete_roots(self):
        for q in (2, 4, 8, 16, 32):
            bound = real(numeric.prewrap_bound_under_rn(q))
            for principal in (0., math.pi, -math.pi, math.nextafter(math.pi, 0.), 5e-324):
                for k in range(q):
                    actual = (principal+2.0*math.pi*k)/q
                    ideal = (real(principal)+2*D.arb.pi()*k)/q
                    self.assertLess(abs(real(actual)-ideal), bound)
        # All-q theorem, not a q_max candidate enumeration or H4 evaluation.
        self.assertEqual(numeric.prewrap_bound_under_rn(2**17), 64*numeric.UNIT_ROUNDOFF)

    def test_prewrap_domain_does_not_claim_full_candidate_error(self):
        for q in (1, 3, 2**18):
            with self.assertRaisesRegex(ValueError, 'UNKNOWN/STOP'):
                numeric.prewrap_bound_under_rn(q)
        self.assertFalse(hasattr(numeric, 'production_float_envelopes'))

    def test_original_candidate_generation_has_complete_toy_root_correspondence(self):
        decoder = self.functions['reconstruct_rpe_phase']
        distance = self.functions['_phase_distance']
        captured = []
        def capture(left, right):
            captured.append(np.asarray(left).copy())
            return distance(left, right)
        for phase in (0., math.nextafter(math.pi, 0.), -math.pi):
            captured.clear()
            with patch.dict(decoder.__globals__, _phase_distance=capture):
                out = decoder([phase]*5, [1, 2, 4, 8, 16])
            for m, candidates in enumerate(captured, 1):
                q = 2**m
                self.assertEqual(len(candidates), q)
                self.assertEqual(len(set(map(float, candidates))), q)
                principal = real(out['rounds'][m]['measured_round_phase'])
                for k, candidate in enumerate(candidates):
                    ideal = (principal+2*D.arb.pi()*k)/q
                    self.assertLess(D.circle_distance(float(candidate), ideal),
                                    real(128*numeric.UNIT_ROUNDOFF))

    def test_far_candidate_can_win_without_all_distance_contract(self):
        decoder = self.functions['reconstruct_rpe_phase']
        distance = self.functions['_phase_distance']
        def corrupted(left, right):
            if len(left) != 4:
                return distance(left, right)
            # Nonnegative finite distances; far candidate spuriously wins.
            return np.array([1., 1., 0., 1.])
        with patch.dict(decoder.__globals__, _phase_distance=corrupted):
            out = decoder([0., 0., 0.], [1, 2, 4])
        self.assertEqual(out['rounds'][2]['selected_branch'], 2)

    def test_strict_tie_is_unknown_even_when_argmin_returns_index(self):
        rows = [row(q, eps=('1/4', '1/4')) for q in (1, 2)]
        out = finite.finite_prefix_check(rows, D.arb.pi()/2)
        self.assertEqual(out['decision'], 'UNKNOWN/STOP')
        self.assertIn('tie', out['reason'])

    def test_input_only_bound_does_not_promote_decoder_or_global_status(self):
        out = D.branch_conditions(['1/10', '1/10'], [1, 2])
        self.assertEqual(out['implementation_decision'], 'UNKNOWN/STOP')
        with self.assertRaisesRegex(ValueError, 'UNKNOWN/STOP'):
            D.branch_conditions(['1/10', '1/10'], [1, 2],
                                [{'angle': 0, 'candidate': 0}]*2)
        self.assertFalse(D.STATUS['RPE_tolerance_certified'])
        self.assertIsNone(D.STATUS['U_exec_upper'])

    def test_nonpositive_branch_margin_is_not_certified(self):
        out = D.branch_conditions([D.arb.pi()/3]*2, [1, 2])
        self.assertEqual(out['ideal_decision'], 'UNKNOWN/STOP')
        self.assertEqual(out['implementation_decision'], 'UNKNOWN/STOP')


if __name__ == '__main__':
    unittest.main()
