"""Analytic artificial-region certificates; finite points are reference checks only."""
from fractions import Fraction as Q
import unittest
from unittest.mock import patch

import numpy as np
from flint import ctx

import invariant_adapter as a


def fixture():
    f = np.array([[0, 2], [2, 0]], complex)  # Exact power-of-two/swap, expands norm.
    rev = np.array([[.125, .0625], [.03125, .125]], complex)  # Dense, non-Hermitian contraction.
    components = [np.array([[.25, .125], [.0625, .25]], complex),
                  np.array([[.125, -.0625j], [.125j, .1875]], complex)]
    return dict(forward=f, reverse=rev, components=components, component_masses=[Q(1, 2)]*2,
                orders=[a.OrderFamily(0, .5, .25, Q(3, 4)), a.OrderFamily(2, .25, .5, Q(1, 4))],
                state_norm_squared=dict(lower='1', upper='1'), r=32, q=131072,
                runtime=dict(profile=a.k.PROFILE, mxcsr=[0x1f80], stable_all_threads=True,
                             forest_family_confirmed=True, deterministic_graph=False,
                             kind='SYNTHETIC_THEOREM_ASSUMPTIONS_NOT_LIVE_RUNTIME'))


def action(error):
    return dict(logical={c: dict(lower='0', upper='0') for c in ['real', 'imag']},
                means=dict(real=0, imag=0), shots={x: dict(lower='0', upper='0') for x in ['cosine', 'sine']},
                epsilon=dict(cosine='1', sine='1'), error=error)


class InvariantAdapterTests(unittest.TestCase):
    def setUp(self):
        self.prec = ctx.prec; ctx.prec = 96
        self.enterContext(patch.object(np, 'load', side_effect=AssertionError('No saved arrays')))
        self.enterContext(patch.object(a.old, 'analyze', side_effect=AssertionError('No state enumeration')))
        self.enterContext(patch.object(a.old.bridge, 'bound_context', side_effect=AssertionError('No H4')))

    def tearDown(self):
        ctx.prec = self.prec

    def test_dense_nonhermitian_box_and_ball_certificates(self):
        matrix = fixture()['components'][0]
        self.assertFalse(np.array_equal(matrix, matrix.conj().T))
        certificate = a.matrix_certificate(matrix)
        self.assertTrue(a.box_invariant(certificate, [(-Q(1), Q(1))]*4))
        self.assertLess(a.image(certificate['norm_map'], Q(1)), 1)
        envelope = a.energy_envelope(certificate, 1)
        self.assertLess(envelope['drift_coefficient_upper'], 0)
        self.assertGreater(envelope['radial_interval'][1], 0)  # No negative sign assumption.

    def test_non_enumerative_all_support_guard_and_prefix(self):
        result = a.prove(**fixture())
        self.assertTrue(result['conditional_guard_proved'], result.get('reason'))
        self.assertEqual(result['state_enumerations'], 0)
        self.assertEqual(result['trajectory_enumerations'], 0)
        self.assertGreaterEqual(result['prefix_norm_upper'], 2)
        self.assertTrue(result['allowed_graph_variation_enclosed'])
        self.assertFalse(result['runtime_live_observed'])
        self.assertIsNone(result['U_virtual_IID_upper'])
        self.assertIsNone(result['U_exec_upper'])

    def test_cost_and_certificate_do_not_scale_with_q_or_r(self):
        records = []
        for count in [1, 10**12]:
            data = fixture(); data.update(r=count, q=count)
            with patch.object(a, 'matrix_certificate', wraps=a.matrix_certificate) as method:
                result = a.prove(**data)
            self.assertTrue(result['conditional_guard_proved'], result.get('reason'))
            records.append((method.call_count, result['prefix_norm_upper'], result['signal_magnitude_upper']))
        self.assertEqual(records[0], records[1])

    def test_forward_reverse_order_changes_the_prefix_envelope(self):
        data = fixture(); first = a.prove(**data)
        data['forward'], data['reverse'] = data['reverse'], data['forward']
        second = a.prove(**data)
        self.assertTrue(first['conditional_guard_proved'], first.get('reason'))
        self.assertTrue(second['conditional_guard_proved'], second.get('reason'))
        self.assertGreater(first['prefix_norm_upper'], second['prefix_norm_upper'])

    def test_point_reference_residuals_are_inside_universal_envelope(self):
        matrix = fixture()['components'][1]
        certificate = a.matrix_certificate(matrix)
        for state in [np.array([1, 0], complex), np.array([.25+.5j, -.125j], complex)]:
            point = a.old.gemv(matrix, state)
            r = a.old.sqrt_upper(point['energy']['input_energy'])
            envelope = a.energy_envelope(certificate, r)
            self.assertLessEqual(point['energy']['drift'],
                                 certificate['drift_coefficient_upper']*point['energy']['input_energy'])
            self.assertLessEqual(point['energy']['output_energy'], envelope['energy_upper'])
            self.assertLessEqual(point['energy']['error_squared'], envelope['error_squared_upper'])
            self.assertLessEqual(point['energy']['radial'], envelope['radial_interval'][1])
            self.assertGreaterEqual(point['energy']['radial'], envelope['radial_interval'][0])

    def test_rotation_bound_covers_nonhermitian_numeric_graph(self):
        cases = [(fixture()['components'][0], .5, .25, np.array([.25+.5j, -.125j], complex)),
                 (np.diag([1j, -1j]), .75, .75, np.array([1, 0], complex))]
        for matrix, cosine, sine, state in cases:
            certificate = a.matrix_certificate(matrix)
            bound = a.rotation_map(certificate, cosine, sine)
            point = a.old.rotation(matrix, state, cosine, sine)
            radius = a.old.sqrt_upper(a.old.norm2(a.old.vec(state)))
            self.assertLessEqual(a.old.sqrt_upper(point['energy']['output_energy']), a.image(bound, radius))
            self.assertLessEqual(a.old.sqrt_upper(point['prefix_energy']), a.prefix(bound, radius))

    def test_unitary_dense_case_exposes_symmetric_radial_obstruction(self):
        # Exact dyadic complex unitary, dense and non-Hermitian.
        unitary = np.array([[.5+.5j, .5+.5j], [.5+.5j, -.5-.5j]], complex)
        certificate = a.matrix_certificate(unitary)
        self.assertEqual(certificate['norm_squared_upper'], 1)
        self.assertGreater(certificate['norm_map'].final_scale, 1)
        self.assertGreater(certificate['error_additive'], 0)
        self.assertFalse(a.box_invariant(certificate, [(-Q(1), Q(1))]*4))
        data = fixture(); data.update(forward=unitary, reverse=unitary.conj().T.copy(),
            components=[unitary], component_masses=[Q(1)], orders=[a.OrderFamily(0, 1., 0., Q(1))])
        result = a.prove(**data)
        self.assertFalse(result['conditional_guard_proved'])
        self.assertIn('UNKNOWN_INVARIANCE', result['reason'])

    def test_rounding_cell_signed_radial_has_nonempty_binary64_witness(self):
        u = a.k.U; x = 1.+2.**-52
        radius = u*u/4
        for coefficient, sign in [(1.-2.**-53, -1), (1.-2.**-52, 1)]:
            cell = a.single_product_cell([coefficient], [(Q.from_float(x)-radius, Q.from_float(x)+radius)], 1.)
            self.assertEqual(cell['status'], 'PROVED_SINGLE_PRODUCT_CELL')
            # Known representable witness; a favorable cell is not assumed global.
            self.assertEqual(coefficient*x, 1.)
            if sign < 0: self.assertLess(cell['radial_interval'][1], 0)
            else: self.assertGreater(cell['radial_interval'][0], 0)
        multi = a.single_product_cell([.5, .5], [(Q(1), Q(1))]*2, 1.)
        self.assertEqual(multi['status'], 'UNKNOWN_MULTIPRODUCT_CELL')

    def test_underflow_cells_and_ties_to_even_are_preserved(self):
        tiny = Q(1, 1 << 1074)
        zero = a.single_product_cell([.5], [(tiny, tiny)], 0.)
        self.assertEqual(zero['residual_interval'], (-a.k.ETA, -a.k.ETA))
        self.assertLess(zero['radial_interval'][1], 0)
        odd = a.single_product_cell([.5], [(3*tiny, 3*tiny)], float(tiny))
        even = a.single_product_cell([.5], [(3*tiny, 3*tiny)], float(2*tiny))
        self.assertEqual(odd['status'], 'UNKNOWN_CELL_CROSSING')
        self.assertEqual(even['status'], 'PROVED_SINGLE_PRODUCT_CELL')
        certificate = a.matrix_certificate(np.array([[.5]], complex))
        self.assertGreater(certificate['error_additive'], 0)

    def test_exact_scaling_up_does_not_imply_exact_scaling_down(self):
        up = a.matrix_certificate(np.array([[2.]], complex))
        down = a.matrix_certificate(np.array([[.5]], complex))
        self.assertEqual(up['error_relative'], 0)
        self.assertEqual(up['error_additive'], 0)
        self.assertGreater(down['error_relative'], 0)
        self.assertGreater(down['error_additive'], 0)

    def test_positive_rare_component_cannot_be_removed(self):
        for rare in [Q(0), Q(1, 1 << 53)]:
            data = fixture()
            data['components'].append(np.eye(2, dtype=complex)*8)
            data['component_masses'] = [(1-rare)/2, (1-rare)/2, rare]
            result = a.prove(**data)
            if rare: self.assertIn('UNKNOWN_INVARIANCE', result['reason'])
            else: self.assertTrue(result['conditional_guard_proved'], result.get('reason'))

    def test_missing_runtime_or_outside_forest_is_unknown(self):
        for runtime in [None, dict(profile=a.k.PROFILE, mxcsr=[0x1f80], stable_all_threads=True),
                        dict(profile=a.k.PROFILE, mxcsr=[0x9fc0], stable_all_threads=True, forest_family_confirmed=True)]:
            data = fixture(); data['runtime'] = runtime
            with patch.object(a, 'matrix_certificate', side_effect=AssertionError('Stop before proof work')):
                result = a.prove(**data)
            self.assertIn('UNKNOWN_RUNTIME', result['reason'])

    def test_guard_failure_and_missing_action_do_not_change_global_status(self):
        result = a.prove(**fixture(), action=action(None))
        self.assertTrue(result['conditional_guard_proved'], result.get('reason'))
        self.assertTrue(all(axis['U_virtual_IID_upper'] is None for axis in result['artificial_axis_check'].values()))
        data = fixture(); data['state_norm_squared'] = dict(lower='4', upper='4')
        result = a.prove(**data)
        self.assertEqual(result['status'], 'UNKNOWN_GUARD_STOP')
        self.assertEqual(result['scientific_status'], 'UNDETERMINED')

    def test_finite_screen_and_invalid_inputs_stop(self):
        identity = a.matrix_certificate(np.eye(2, dtype=complex))
        self.assertFalse(a.box_invariant(identity, [(-Q(2)**201, Q(2)**201)]*4))
        data = fixture(); data['forward'] = np.eye(2, dtype=complex)*(2.**201)
        data['reverse'] = np.zeros((2, 2), complex)
        data['components'] = [np.zeros((2, 2), complex)]; data['component_masses'] = [Q(1)]
        data['orders'] = [a.OrderFamily(0, 1., 0., Q(1))]
        self.assertIn('UNKNOWN_FINITE', a.prove(**data)['reason'])
        data = fixture(); data['state_norm_squared'] = dict(lower='2', upper='1')
        self.assertEqual(a.prove(**data)['status'], 'UNKNOWN_STOP')
        with self.assertRaises(ValueError): a.single_product_cell([Q(1, 3)], [(Q(0), Q(1))], 0.)


if __name__ == '__main__':
    unittest.main()
