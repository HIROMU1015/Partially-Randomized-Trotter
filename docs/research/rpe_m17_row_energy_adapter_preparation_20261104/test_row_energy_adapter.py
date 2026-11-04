"""Exhaustive small artificial state domains; no saved inputs or scientific CLI."""
from fractions import Fraction as Q
import itertools
import unittest
from unittest.mock import patch

import numpy as np
from flint import ctx

import row_energy_adapter as a


def domain():
    # Preserve all raw signed-zero variants; no tolerance or equivalence merge.
    result = []
    for position, amplitude in [(0, 2.), (1, 1.)]:
        for coordinate in [2*position, 2*position+1]:
            for sign in [-1, 1]:
                for zeros in itertools.product([-0., 0.], repeat=3):
                    values, pending = [], iter(zeros)
                    for slot in range(4): values.append(sign*amplitude if slot == coordinate else next(pending))
                    result.append(np.array([complex(*values[:2]), complex(*values[2:])], dtype=complex))
    return result


def fixture():
    f = np.array([[0, 2], [.5, 0]], complex)  # Nonunitary/non-Hermitian, f^2=I.
    z = np.diag([1, -1]).astype(complex)
    events = [a.Event((), 'Z', 1., 0., 1, Q(1, 2)),
              a.Event(('Z', 'A'), 'A', 0., 1., -1, Q(1, 2))]
    runtime = dict(profile=a.k.PROFILE, mxcsr=[0x1f80], stable_all_threads=True, deterministic_graph=True,
                   kind='SYNTHETIC_THEOREM_ASSUMPTIONS_NOT_LIVE_RUNTIME')
    return dict(states=domain(), psi=np.array([0, 1], complex), forward=f, reverse=f.copy(),
                components=dict(A=f, Z=z), events=events, r=32, q=131072, runtime=runtime)


def action(error):
    return dict(logical={c: dict(lower='0', upper='0') for c in ['real', 'imag']}, means=dict(real=0, imag=0),
                shots={axis: dict(lower='0', upper='0') for axis in ['cosine', 'sine']},
                epsilon=dict(cosine='3', sine='3'), error=error)


class RowEnergyAdapterTests(unittest.TestCase):
    def setUp(self):
        self.prec = ctx.prec
        ctx.prec = 96
        self.enterContext(patch.object(np, 'load', side_effect=AssertionError('No saved arrays')))
        self.enterContext(patch.object(a.bridge, 'numeric_components', side_effect=AssertionError('No H4 context')))
        self.enterContext(patch.object(a.bridge, 'bound_context', side_effect=AssertionError('No H4 evaluation')))

    def tearDown(self):
        ctx.prec = self.prec

    def test_underflow_has_signed_radial_and_nonzero_additive_bound(self):
        tiny = float.fromhex('0x0.0000000000001p-1022')
        proof = a.gemv(np.array([[.5]], complex), np.array([tiny], complex))
        self.assertEqual(proof['rows'][0]['error'][0], -a.k.ETA)
        self.assertGreater(proof['rows'][0]['forest_bounds'][0], 0)
        self.assertLess(proof['energy']['radial'], 0)
        self.assertGreater(proof['energy']['error_squared'], 0)
        self.assertEqual(proof['energy']['output_energy'], 0)

    def test_structural_sign_swap_is_exact_including_subnormal_input(self):
        tiny = float.fromhex('0x0.0000000000001p-1022')
        matrix = np.array([[0, 1j], [-1, 0]], complex)
        proof = a.gemv(matrix, np.array([tiny, .25+.5j], complex))
        self.assertTrue(all(row['structurally_exact'] for row in proof['rows']))
        self.assertEqual(proof['energy']['error_squared'], 0)
        self.assertEqual(proof['energy']['input_energy'], proof['energy']['output_energy'])

    def test_generic_rows_keep_actual_residuals_and_signed_energy(self):
        matrix = np.array([[.1, .2], [.3, .4]], complex)
        state = np.array([.7+.1j, -.25+.125j], complex)
        proof = a.gemv(matrix, state)
        self.assertEqual(len(proof['rows']), 2)
        self.assertGreater(proof['energy']['error_squared'], 0)
        e = proof['energy']
        self.assertEqual(e['input_energy']+e['drift']+2*e['radial']+e['error_squared'], e['output_energy'])

    def test_nonhermitian_rotation_term_is_not_dropped(self):
        proof = a.rotation(np.diag([1j, -1j]), np.array([1, 0], complex), .75, .75)
        self.assertEqual(proof['nonhermitian_term'], Q(9, 8))
        self.assertEqual(proof['energy']['drift'], Q(5, 4))
        self.assertEqual(proof['energy']['output_energy'], Q(9, 4))

    def test_rotation_checks_both_scales_subtract_and_gemv(self):
        matrix = np.array([[.1+.2j, .25], [.125j, .7]], complex)
        state = np.array([.3+.2j, -.7+.125j], complex)
        proof = a.rotation(matrix, state, .8, .3)
        self.assertEqual(len(proof['gemv']['rows']), 2)
        self.assertEqual(len(proof['arithmetic']), 3)
        self.assertTrue(all(item['norm_bound'] > 0 for item in proof['arithmetic']))
        self.assertGreaterEqual(proof['prefix_energy'], proof['energy']['output_energy'])

    def test_full_support_long_chain_certifies_artificial_guard_with_prefix_peak(self):
        result = a.analyze(**fixture())
        self.assertTrue(result['conditional_guard_proved'], result.get('reason'))
        self.assertEqual(result['prefix_energy_upper'], 4)
        self.assertEqual(result['signal_magnitude_upper'], 1)
        self.assertGreaterEqual(result['endpoint_count'], 2)
        self.assertIsNone(result['U_virtual_IID_upper'])
        self.assertIsNone(result['U_exec_upper'])
        self.assertEqual(result['scope'], 'ARTIFICIAL_FINITE_DOMAIN_ONLY')
        self.assertFalse(result['scientific_execution_ready'])

    def test_product_order_is_retained_between_noncommuting_f_and_r(self):
        data = fixture(); data.update(r=1, q=1)
        outputs = []
        for order in [('Z', 'A'), ('A', 'Z')]:
            data['events'] = [a.Event(order, 'A', 0., 1., -1, Q(1))]
            result = a.analyze(**data)
            self.assertTrue(result['conditional_guard_proved'], result.get('reason'))
            values = {proof['output'] for proof in result['dot_evidence']}
            self.assertEqual(len(values), 1)
            outputs.append(values.pop())
        self.assertEqual(outputs[0], -outputs[1])

    def test_final_inverse_action_does_not_erase_intermediate_energy(self):
        data = fixture(); data.update(events=[a.Event((), 'Z', 1., 0., 1, Q(1))], r=1, q=1)
        result = a.analyze(**data)
        self.assertTrue(result['conditional_guard_proved'], result.get('reason'))
        self.assertEqual(result['signal_magnitude_upper'], 1)
        self.assertEqual(result['prefix_energy_upper'], 4)
        self.assertTrue(any(proof['energy']['drift'] < 0 for proof in result['evidence']))
        self.assertTrue(any(proof['energy']['drift'] > 0 for proof in result['evidence']))

    def test_complete_contraction_retains_peak_from_an_earlier_prefix(self):
        data = fixture()
        zeros = [np.array([complex(*v[:2]), complex(*v[2:])], complex)
                 for v in itertools.product([-0., 0.], repeat=4)]
        data['states'] += zeros
        data['reverse'] = np.zeros((2, 2), complex)
        data.update(r=1, q=1024, events=[a.Event((), 'Z', 1., 0., 1, Q(1))])
        result = a.analyze(**data)
        self.assertTrue(result['conditional_guard_proved'], result.get('reason'))
        self.assertEqual(result['signal_magnitude_upper'], 0)
        self.assertEqual(result['prefix_energy_upper'], 4)

    def test_missing_or_bad_runtime_stops_before_matrix_actions(self):
        for runtime in [None, False, dict(profile=a.k.PROFILE, mxcsr=[0x9fc0], stable_all_threads=True, deterministic_graph=True),
                        dict(profile=a.k.PROFILE, mxcsr=[0x1f80], stable_all_threads=True)]:
            data = fixture(); data['runtime'] = runtime
            with patch.object(a, 'gemv', side_effect=AssertionError('No arithmetic')):
                result = a.analyze(**data)
            self.assertEqual(result['status'], 'UNKNOWN_STOP')
            self.assertIn('UNKNOWN_RUNTIME', result['reason'])

    def test_missing_finite_closure_is_unknown(self):
        data = fixture()
        data['states'] = [s for s in data['states'] if s[0] == 0]
        result = a.analyze(**data)
        self.assertFalse(result['conditional_guard_proved'])
        self.assertIn('UNKNOWN_CLOSURE', result['reason'])

    def test_raw_state_is_not_normalized_to_force_guard_pass(self):
        data = fixture(); data['psi'] = np.array([2, 0], complex)
        data.update(r=1, q=1, events=[a.Event((), 'Z', 1., 0., 1, Q(1))])
        result = a.analyze(**data, action=action(0))
        self.assertEqual(result['status'], 'UNKNOWN_GUARD_STOP')
        self.assertEqual(result['signal_magnitude_upper'], 4)
        self.assertTrue(all(axis['U_virtual_IID_upper'] is None for axis in result['artificial_axis_check'].values()))

    def test_positive_rare_event_is_not_removed_but_exact_zero_mass_is(self):
        for rare in [Q(0), Q(1, 1 << 53)]:
            data = fixture()
            data['events'] = [a.Event((), 'Z', 1., 0., 1, 1-rare), a.Event((), 'Z', 2., 0., 1, rare)]
            result = a.analyze(**data)
            if rare:
                self.assertIn('UNKNOWN_CLOSURE', result['reason'])
            else:
                self.assertTrue(result['conditional_guard_proved'], result.get('reason'))

    def test_unknown_action_bound_is_not_replaced_by_norm_certificate(self):
        for spec in [action(None), {'error': 0}]:
            result = a.analyze(**fixture(), action=spec)
            self.assertTrue(result['conditional_guard_proved'], result.get('reason'))
            self.assertTrue(all(axis['U_virtual_IID_upper'] is None for axis in result['artificial_axis_check'].values()))

    def test_old_action_bound_and_strict_classifier_stay_independent(self):
        zero = a.bridge.KernelBounds(**dict.fromkeys(a.bridge.KernelBounds.__dataclass_fields__, 0))
        error, _ = a.bridge.signal_error(dict(lower='1', upper='1'), a.bridge.Growth(3), zero)
        spec = action(error)
        result = a.analyze(**fixture(), action=spec)
        self.assertTrue(result['conditional_guard_proved'], result.get('reason'))
        self.assertTrue(all(Q(axis['U_virtual_IID_upper']) >= 2 for axis in result['artificial_axis_check'].values()))
        equal = str(Q(2)+3*Q(1, 1 << 53))
        axes = a.bridge.classify_axes(spec['logical'], spec['means'], spec['shots'],
            dict(cosine=equal, sine=equal), error, completion=True, arithmetic_resolved=True)
        self.assertTrue(all(axis['decision'] == 'NOT_CERTIFIED_BY_THIS_BOUND' for axis in axes.values()))

    def test_finite_envelope_failure_stops_before_actions(self):
        data = fixture(); data['forward'] = np.eye(2, dtype=complex)*(2.**201)
        with patch.object(a, 'gemv', side_effect=AssertionError('No arithmetic')):
            result = a.analyze(**data)
        self.assertIn('UNKNOWN_FINITE', result['reason'])

    def test_invalid_layout_and_event_proof_are_unknown(self):
        data = fixture(); data['forward'] = np.asfortranarray(data['forward'])
        self.assertEqual(a.analyze(**data)['status'], 'UNKNOWN_STOP')
        data = fixture(); data['events'] = [a.Event(('A',), 'Z', 1., 0., 1, Q(1))]
        self.assertIn('Invalid fixed K2', a.analyze(**data)['reason'])

    def test_relation_powers_match_brute_force_paths_and_prefixes(self):
        energies = [Q(1), Q(4)]
        f = [{1: Q(4)}, {0: Q(4)}]
        alternatives = [a.identity(energies), f]
        block = a.compose(f, a.compose(a.power(a.union(alternatives), 2, energies), f))
        relation = a.power(block, 2, energies)
        for initial in range(2):
            expected = {}
            for word in itertools.product([0, 1], repeat=4):
                state, peak, offset = initial, energies[initial], 0
                for _ in range(2):
                    state = 1-state; peak = max(peak, energies[state])
                    for choice in word[offset:offset+2]:
                        state = state if choice == 0 else 1-state
                        peak = max(peak, energies[state])
                    offset += 2
                    state = 1-state; peak = max(peak, energies[state])
                expected[state] = max(expected.get(state, Q(0)), peak)
            self.assertEqual(relation[initial], expected)


if __name__ == '__main__':
    unittest.main()
