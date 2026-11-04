"""Artificial inputs only. No H4 mean/interval is loaded or evaluated."""
from fractions import Fraction as Q
import math
import unittest

from flint import acb, arb, ctx
import numpy as np

import adapter as a


class ConnectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f, cls.records = a.existing_functions()

    def setUp(self):
        self.prec = ctx.prec
        ctx.prec = 160

    def tearDown(self):
        ctx.prec = self.prec

    def test_positive_nonunit_radius_and_disk_bound(self):
        result = a.round_geometry('3/5', '0', '3/50', '1/25', 0, 1)
        self.assertTrue(a.interval(result['radius']).contains(a.real('3/5')))
        with ctx.workprec(320):
            expected = (a.real('13/2500').sqrt() / a.real('3/5')).asin()
            self.assertGreaterEqual(a.real(result['disk_phase_upper']['upper']), expected)
        self.assertLess(a.phase_upper(result), a.real('1/5'))

    def test_disk_contains_tangent_phase_error(self):
        # Exact tangent to a radius-D disk about a positive center.
        d = a.real('1/5')
        theta = d.asin()
        z = acb(0, theta).exp() * (1-d*d).sqrt()
        self.assertTrue(abs(z-acb(1, 0)).contains(d))
        geometry = a.round_geometry(1, 0, '3/25', '4/25', 0, 1)
        self.assertGreaterEqual(a.real(geometry['disk_phase_upper']['upper']), theta)

    def test_rectangle_is_tighter_for_anisotropic_box(self):
        result = a.round_geometry(1, 0, '3/10', '1/10', 0, 1)
        self.assertEqual(result['method'], 'rectangle')
        self.assertLess(a.phase_upper(result), a.real('1/5'))
        self.assertGreater(a.real(result['disk_phase_upper']['upper']), a.real('3/10'))

    def test_minimal_rectangle_fallback_when_disk_cannot_decide(self):
        result = a.round_geometry(1, 0, '99/100', '9/10', 0, 1)
        self.assertIsNone(result['disk_phase_upper'])
        self.assertEqual(result['method'], 'rectangle')
        self.assertLess(a.phase_upper(result), arb.pi()/2)

    def test_origin_and_uncertain_inputs_stop(self):
        with self.assertRaisesRegex(ValueError, 'UNKNOWN/STOP'):
            a.round_geometry(0, 0, '1/10', '1/10', 0, 1)
        result = a.round_geometry(1, 0, 1, 1, 0, 1)
        self.assertEqual(result['decision'], 'UNKNOWN/STOP')
        for value in (float('nan'), float('inf'), -1):
            with self.assertRaisesRegex(ValueError, 'UNKNOWN/STOP'):
                a.nonnegative(value)
        with self.assertRaisesRegex(ValueError, 'UNKNOWN/STOP'):
            a.interval({'lower': '1', 'upper': '0'})

    def test_systematic_phase_is_included_once(self):
        result = a.round_geometry(1, 0, '1/100', '1/100', '-1/10', 2)
        self.assertGreaterEqual(a.real(result['systematic_phase_upper']['upper']), a.real('1/5'))
        self.assertLess(a.phase_upper(result), a.real('1/4'))

    def test_circular_distance_across_cut_and_antipodes(self):
        self.assertLess(a.circle_distance(arb.pi()-a.real('1/100'), -arb.pi()+a.real('1/100')), a.real('21/1000'))
        self.assertGreaterEqual(a.circle_distance(0, arb.pi()), arb.pi())
        result = a.round_geometry(-1, '1/10000', '1/10000', '1/10000', -arb.pi()+a.real('1/10000'), 1)
        self.assertLess(a.phase_upper(result), a.real('1/1000'))

    def test_strict_variable_branch_bound_and_tie(self):
        result = a.branch_conditions(['1/10', '1/5', '1/3'], [1, 2, 4])
        self.assertEqual(result['ideal_decision'], 'SUFFICIENT')
        tie = a.branch_conditions([arb.pi()/4, arb.pi()/2], [1, 2])
        self.assertEqual(tie['ideal_decision'], 'UNKNOWN/STOP')
        self.assertEqual(a.branch_conditions([1, 2], [1, 2])['ideal_decision'], 'UNKNOWN/STOP')
        for q in ([2], [1, 4], [1, 2, 2], [1, 2.0]):
            with self.assertRaisesRegex(ValueError, 'UNKNOWN/STOP'):
                a.branch_conditions([0]*len(q), q)

    def test_missing_uniform_float_proof_stays_unknown(self):
        result = a.branch_conditions(['1/10', '1/10'], [1, 2])
        self.assertEqual(result['implementation_decision'], 'UNKNOWN/STOP')
        self.assertIsNone(result['implementation_final_phase_upper'])
        with self.assertRaisesRegex(ValueError, 'UNKNOWN/STOP'):
            a.branch_conditions([0, 0], [1, 2], [None, None])

    def test_float_error_can_consume_an_ideal_strict_margin(self):
        good = [{'angle': '1/1000', 'candidate': '1/10000', 'distance': '1/10000'}]*2
        result = a.branch_conditions(['1/10', '1/10'], [1, 2], good)
        self.assertEqual(result['implementation_decision'], 'CONDITIONAL_ON_SUPPLIED_UNIFORM_ENVELOPES')
        self.assertGreater(a.real(result['implementation_final_phase_upper']['upper']), a.real('1/20'))
        bad = [dict(x, distance=2) for x in good]
        result = a.branch_conditions(['1/10', '1/10'], [1, 2], bad)
        self.assertEqual(result['ideal_decision'], 'SUFFICIENT')
        self.assertEqual(result['implementation_decision'], 'UNKNOWN/STOP')

    def test_existing_argmin_tie_does_not_certify_uniqueness(self):
        result = self.f['reconstruct_rpe_phase']([0, math.pi], [1, 2])
        last = result['rounds'][-1]
        self.assertAlmostEqual(last['distance_from_previous'], last['next_smallest_branch_distance'])
        self.assertEqual(a.branch_conditions([0, arb.pi()], [1, 2])['ideal_decision'], 'UNKNOWN/STOP')

    def test_original_decoder_18_artificial_rounds_and_rounding_diagnostic(self):
        # SOURCE code is reused; the physical saved signals are never loaded.
        q = [1 << m for m in range(18)]
        phi = a.real('173/1000')
        measured = [float(acb(0, k*phi).exp().arg().mid()) for k in q]
        result = self.f['reconstruct_rpe_phase'](measured, q)
        self.assertLess(a.circle_distance(result['final_phase_estimate'], phi), a.real('1e-11'))
        self.assertEqual(len(result['rounds']), 18)
        # Diagnostic residual only: this finite set does not produce an envelope.
        self.assertIsNone(a.branch_conditions(['1/10']*18, q)['implementation_final_phase_upper'])

    def test_float_angle_wrap_and_distance_at_seams(self):
        for theta in (math.nextafter(math.pi, 0), math.pi, math.nextafter(math.pi, math.inf),
                      -math.pi, math.nextafter(-math.pi, 0), 0.0):
            wrapped = self.f['_wrap_phase'](theta)
            self.assertLess(a.circle_distance(wrapped, theta), a.real('1e-12'))
        for x, y in ((1e-300, 1e-300), (-1.0, 1e-30), (-1.0, -1e-30), (0.0, 1.0)):
            computed = float(np.angle(complex(x, y)))
            self.assertLess(a.circle_distance(computed, acb(a.real(x), a.real(y)).arg()), a.real('1e-12'))
        computed = self.f['_phase_distance'](math.pi-1e-7, -math.pi+1e-7)
        self.assertAlmostEqual(computed, 2e-7, delta=1e-14)

    def test_alias_and_lift_need_strict_boundary_margin(self):
        good = a.energy_conditions('1/5', '1/100', '1/10', 1)
        self.assertEqual(good['literal_target_decision'], 'SUFFICIENT')
        boundary = a.energy_conditions(arb.pi(), '1/100', '1/10', 1)
        self.assertEqual(boundary['reference_alias_decision'], 'UNKNOWN/STOP')
        crossing = a.energy_conditions(arb.pi()-a.real('1/100'), '1/50', '1/10', 1)
        self.assertEqual(crossing['reference_alias_decision'], 'SUFFICIENT')
        self.assertEqual(crossing['representative_lift_decision'], 'UNKNOWN/STOP')
        self.assertEqual(crossing['literal_target_decision'], 'UNKNOWN/STOP')

    def test_sign_and_reference_propagation(self):
        energy, delta = a.real('-3/2'), a.real('1/8')
        phase = -energy*delta
        self.assertTrue((-phase/delta).contains(energy))
        base = a.energy_conditions(phase, '1/1000', delta, '1/100')
        self.assertIsNone(base['physical_energy_upper'])
        good = a.energy_conditions(phase, '1/1000', delta, '1/100', '1/1000')
        self.assertEqual(good['physical_target_decision'], 'CONDITIONAL_ON_REFERENCE_BOUND')
        self.assertGreaterEqual(a.real(good['physical_energy_upper']['upper']), a.real('9/1000'))
        bad = a.energy_conditions(phase, '1/1000', delta, '1/100', '3/1000')
        self.assertEqual(bad['physical_target_decision'], 'UNKNOWN/STOP')

    def test_reference_uncertainty_can_destroy_alias_condition(self):
        result = a.energy_conditions(arb.pi()-a.real('1/100'), '1/1000', '1/10', 1, '1/5')
        self.assertEqual(result['physical_alias_decision'], 'UNKNOWN/STOP')

    def test_exact_alpha_sum_does_not_silently_round_to_budget(self):
        good = a.union_bound(['1/100', '2/100'], '3/100')
        self.assertEqual(good['alpha_sum_exact'], '3/100')
        self.assertEqual(good['success_lower_exact'], '97/100')
        excess = a.union_bound(['0.010000000000000000001', '0.02'], '0.03')
        self.assertEqual(excess['saved_budget_decision'], 'UNKNOWN/STOP')
        self.assertGreaterEqual(Q.from_float(float.fromhex(excess['alpha_sum_float_upper'])), Q(excess['alpha_sum_exact']))

    def test_reuse_accounting_outward_binary64_sum(self):
        values = (0.1, 0.2, 0.3)
        original = self.f['_conservative_nonnegative_sum'](values)
        result = a.union_bound(values, 0.9)
        self.assertEqual(original.hex(), result['alpha_sum_float_upper'])
        self.assertTrue(all(x['unchanged_ast'] for x in self.records.values()))
        self.assertNotEqual(a.union_bound(['0.1'], '0.9')['alpha_sum_exact'],
                            a.union_bound([0.1], 0.9)['alpha_sum_exact'])

    def test_hoeffding_bridge_is_already_inside_epsilon(self):
        sufficient = a.local_axis_condition({'lower': '49/100', 'upper': '51/100'}, .5, 10000, '.01', '.1')
        self.assertEqual(sufficient['categorical_decision'], 'SUFFICIENT_CONDITION_CONFIRMED')
        excluded = a.local_axis_condition({'lower': '9/10', 'upper': '1'}, .5, 10000, '.01', '.1')
        self.assertNotEqual(excluded['categorical_decision'], 'SUFFICIENT_CONDITION_CONFIRMED')
        self.assertIsNone(sufficient['U_exec_upper'])

    def test_global_status_is_never_promoted_by_toy_passes(self):
        self.assertFalse(a.STATUS['RPE_tolerance_certified'])
        self.assertEqual(a.STATUS['scientific_status'], 'UNDETERMINED')
        self.assertIsNone(a.STATUS['U_virtual_IID_upper'])
        self.assertIsNone(a.STATUS['U_exec_upper'])


if __name__ == '__main__':
    unittest.main()
