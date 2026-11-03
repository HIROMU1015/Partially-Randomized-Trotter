"""Artificial 2x2 matrices/exact laws only; no real-data numerical tests."""
from fractions import Fraction as Q
import contextlib
import copy
import io
import itertools
import json
import math
import sys
import unittest
from unittest.mock import patch

import numpy as np
from flint import acb_mat, ctx

import virtual_bridge_adapter as v
import run_virtual_bridge as run


def kernel(**kwargs):
    # Artificial theorem coefficients, never installed-BLAS error allowances.
    fields = dict.fromkeys(v.KernelBounds.__dataclass_fields__, 0)
    return v.KernelBounds(**(fields | kwargs))


def toy_export():
    return {'n_qubits': 2, 'n_electrons': 1, 'ordered_sector': [1, 2],
            'bases': {'u': {'operations': [{'name': 'rz', 'qubits': [0], 'params': [0..hex()],
                'local_binary64_matrix': [[[1..hex(), 0..hex()], [0..hex(), 0..hex()]],
                                          [[0..hex(), 0..hex()], [1..hex(), 0..hex()]]]}]}},
            'components': [{'id': 'a', 'basis_id': 'u', 'support': [0], 'sign': -1}]}


class BridgeAdapterTests(unittest.TestCase):
    def setUp(self):
        self.prec, self.threads = ctx.prec, ctx.threads
        ctx.prec, ctx.threads = 96, 1

    def tearDown(self):
        ctx.prec, ctx.threads = self.prec, self.threads

    def test_numeric_basis_uses_saved_local_matrix_and_wire_order(self):
        export = toy_export()
        np.testing.assert_array_equal(v.numeric_basis(export['bases']['u'], [1, 2], 2, 1), np.eye(2))
        # Actual saved-array composition is compared to the reused legacy API.
        from qiskit.circuit.library import XXPlusYYGate, RZGate
        sys.path.insert(0, str(v.ROOT/'docs/research/rpe_fresh_iid_measurement_a_preparation_20261103'))
        from run_fresh_iid_a_v2 import restricted_basis
        ops = [(XXPlusYYGate(.25, .125), [0, 1]), (RZGate(.5), [1])]
        spec = {'operations': []}
        for gate, wires in ops:
            matrix = np.asarray(gate)
            pairs = [[[float(z.real).hex(), float(z.imag).hex()] for z in row] for row in matrix]
            spec['operations'].append({'name': gate.name, 'qubits': wires,
                                      'local_binary64_matrix': pairs})
        expected, _ = restricted_basis(ops, 2, np.array([1, 2]), np.array([1, 0]))
        np.testing.assert_array_equal(v.numeric_basis(spec, [2, 1], 2, 1), expected)

    def test_signed_component_and_ideal_residual(self):
        export = toy_export()
        numeric = v.numeric_components(export)
        np.testing.assert_array_equal(numeric['a'], np.diag([1., -1.]))
        result = v.component_bounds(export, numeric)
        self.assertEqual(result['a']['residual'], 0)
        changed = {'a': numeric['a']+np.eye(2)*.125}
        bound = v.component_bounds(export, changed)['a']['residual']
        self.assertGreaterEqual(bound, v.b.real('.125'))

    def test_saved_ball_global_radius_and_center_uncertainty_retained(self):
        target = v.b.recenter(acb_mat([[1, 0], [0, 1]]), v.b.real(Q(1, 100)), norm=1)
        decoded = v.decode_matrix_ball(v.b.encode_matrix_ball(target))
        self.assertGreaterEqual(v.matrix_residual(np.eye(2, dtype=complex), decoded), v.b.real(Q(1, 100)))
        saved = {'shape': [1, 1], 'center': [['1/3', '0']], 'spectral_radius_upper': '0', 'target_norm_upper': '1'}
        self.assertGreater(v.decode_matrix_ball(saved).radius, 0)

    def test_affine_composition_order_and_additive_error(self):
        a, b = v.Growth(2, 3), v.Growth(5, 7)
        ab, ba = v.compose(a, b), v.compose(b, a)
        self.assertEqual(ab.gain, 10); self.assertEqual(ab.additive, 17)
        self.assertEqual(ba.additive, 22)
        out = v.power(v.Growth(1, Q(1, 100)), 100)
        self.assertGreaterEqual(out.additive, 1)
        self.assertEqual(out.gain, 1)

    def test_growth_binary_power_matches_exact_rational_reference(self):
        for count in [0, 1, 2, 7, 1000000]:
            # Additive-only exact toy remains efficient for a long abstract chain.
            out = v.power(v.Growth(1, Q(1, 1 << 20)), count)
            self.assertEqual(v.b.dyadic(v.nonnegative(out.additive)), Q(count, 1 << 20))
        out = v.power(v.Growth(Q(101, 100), Q(1, 1000)), 7)
        gain = Q(101, 100)**7
        additive = sum(Q(101, 100)**j*Q(1, 1000) for j in range(7))
        self.assertGreaterEqual(out.gain, v.b.real(gain))
        self.assertGreaterEqual(out.additive, v.b.real(additive))

    def test_iid_affine_average_matches_artificial_enumeration(self):
        events, masses = [v.Growth(1, 1), v.Growth(2, 3)], [Q(3, 4), Q(1, 4)]
        averaged, worst = v.trajectory_growth(v.Growth(1), v.Growth(1), events, masses, [0, 2], r=2, q=2)
        gain, additive = Q(0), Q(0)
        for word in itertools.product([0, 1], repeat=4):
            prob = math.prod(masses[i] for i in word)
            value = v.Growth(1)
            for i in word:
                value = v.compose(events[i], value)
            gain += prob*v.b.dyadic(value.gain)
            additive += prob*v.b.dyadic(value.additive)
        self.assertEqual(v.b.dyadic(averaged.gain), gain)
        self.assertEqual(v.b.dyadic(averaged.additive), additive)
        self.assertGreaterEqual(worst.gain, averaged.gain)
        self.assertGreaterEqual(worst.additive, averaged.additive)

    def test_zero_probability_order_does_not_enlarge_uniform_support(self):
        _, worst = v.trajectory_growth(v.Growth(1), v.Growth(1), [v.Growth(1), v.Growth(100)], [Q(1), Q(0)], [0, 2], r=1, q=1)
        self.assertEqual(worst.gain, 1)

    def test_rotation_includes_trig_matrix_and_arithmetic_errors(self):
        exact = v.rotation_growth(0., 1., 0., 0, 1, kernel())
        self.assertEqual(exact.gain, 1); self.assertEqual(exact.additive, 0)
        trig = v.rotation_growth(0., .875, 0., 0, 1, kernel())
        self.assertGreaterEqual(trig.gain, v.b.real('1.125'))
        rounded = v.rotation_growth(.25, math.cos(.25), math.sin(.25), Q(1, 100), 2,
            kernel(mv_rel=Q(1, 100), mv_abs=Q(1, 1000), scale_rel=Q(1, 100), sub_rel=Q(1, 100)))
        self.assertGreater(rounded.gain, 1); self.assertGreater(rounded.additive, 0)

    def test_raw_norm_and_dot_error_each_counted_once(self):
        error, magnitude = v.signal_error({'lower': '4', 'upper': '4'}, v.Growth(1), kernel())
        self.assertEqual(error, 3); self.assertEqual(magnitude, 4)
        error, magnitude = v.signal_error({'lower': '1', 'upper': '1'}, v.Growth(1), kernel(dot_abs=Q(1, 4)))
        self.assertEqual(error, v.b.real('1/4')); self.assertEqual(magnitude, v.b.real('5/4'))

    def test_matrix_error_bound_covers_exact_perturbed_toy_action(self):
        # Pure dyadic arithmetic: P64=(1+1/8)P. This tests perturbation,
        # rather than making a statement about the installed BLAS's roundoff.
        p = np.diag([1., -1.]).astype(complex)
        numeric = (1+1/8)*p
        delta = v.matrix_residual(numeric, v.b.unit(acb_mat([[1, 0], [0, -1]])))
        growth = v.power(v.action_growth(delta, v.b.frobenius(v.exact_matrix(numeric)), kernel()), 3)
        error, _ = v.signal_error({'lower': '1', 'upper': '1'}, growth, kernel())
        true_error = Q(9, 8)**3-1
        self.assertGreaterEqual(error, v.b.real(true_error))

    def test_interval_radius_not_discarded_by_bridge_reconstruction(self):
        saved = {'lower': '1/3', 'upper': '2/3'}
        rebuilt = v.interval(saved)
        self.assertTrue(rebuilt.contains(v.b.real('1/3')))
        self.assertTrue(rebuilt.contains(v.b.real('2/3')))

    def test_guard_requires_uniform_not_average_and_abs_contract(self):
        self.assertFalse(v.guard_completion(1, finite_intermediates=True))
        self.assertFalse(v.guard_completion(1, abs_evaluation_error=0))
        self.assertTrue(v.guard_completion(1, abs_evaluation_error=0, finite_intermediates=True))
        self.assertFalse(v.guard_completion(v.b.real('1.000000000002'), abs_evaluation_error=0, finite_intermediates=True))

    def test_null_and_strict_budget_do_not_certify_fixed_pcg(self):
        logical = {coord: {'lower': '0', 'upper': '0'} for coord in ['real', 'imag']}
        means, shots, epsilon = dict(real=0, imag=0), {a: {'lower': '0', 'upper': '0'} for a in ['cosine', 'sine']}, dict(cosine='1', sine='1')
        for completion, arithmetic in [(False, True), (True, False)]:
            out = v.classify_axes(logical, means, shots, epsilon, 0, completion=completion, arithmetic_resolved=arithmetic)
            self.assertIsNone(out['cosine']['U_virtual_IID_upper'])
        epsilon = dict.fromkeys(['cosine', 'sine'], str(Q(3, 1 << 53)))
        out = v.classify_axes(logical, means, shots, epsilon, 0, completion=True, arithmetic_resolved=True)
        self.assertEqual(out['cosine']['decision'], 'NOT_CERTIFIED_BY_THIS_BOUND')
        self.assertIsNone(out['cosine']['U_exec_upper'])
        self.assertFalse(out['cosine']['fixed_PCG64_IID_certified'])

    def test_combined_toy_pass_uses_numeric_context_and_saved_halves(self):
        export = toy_export(); numeric = v.numeric_components(export)
        identity = np.eye(2, dtype=complex)
        halves = {key: v.b.encode_matrix_ball(v.b.unit(v.b.identity(2))) for key in ['forward', 'reverse']}
        out = v.bound_context(export, numeric, identity, identity, halves, [0., 0.], [(1., 0.), (1., 0.)],
            [Q(3, 4), Q(1, 4)], [0, 2], r=2, q=4, norm_squared={'lower': '1', 'upper': '1'}, kernel=kernel())
        self.assertEqual(out['average_error'], 0); self.assertEqual(out['uniform_magnitude'], 1)

    def test_default_and_run_stop_before_any_real_numerical_loader(self):
        with (patch.object(run, 'metadata', return_value=[]), patch.object(v, 'numeric_components', side_effect=AssertionError('No H4')),
             patch.object(v, 'bound_context', side_effect=AssertionError('No H4')), patch.object(np, 'load', side_effect=AssertionError('No arrays')),
             patch.object(np.random, 'PCG64', side_effect=AssertionError('No RNG')), contextlib.redirect_stdout(io.StringIO()) as output):
            self.assertEqual(run.main([]), 0)
            self.assertFalse(json.loads(output.getvalue())['scientific_execution_ready'])
            output.seek(0); output.truncate()
            self.assertEqual(run.main(['--run']), 2)
            self.assertEqual(json.loads(output.getvalue())['new_real_numerical_calls'], 0)

    def test_invalid_shapes_probabilities_and_bounds(self):
        for bad in [-1, float('nan'), True]:
            with self.assertRaises(ValueError): v.Growth(bad)
        with self.assertRaises(ValueError): kernel(dot_abs=-1)
        with self.assertRaises(ValueError): v.power(v.Growth(1), True)
        with self.assertRaises(ValueError): v.exact_matrix(np.eye(2, dtype=np.float32))
        with self.assertRaises(ValueError): v.trajectory_growth(v.Growth(1), v.Growth(1), [v.Growth(1)], ['1/2'], [0], r=1, q=1)
        with self.assertRaises(ValueError): v.decode_matrix_ball({'shape': [1, 1], 'center': []})
        bad = copy.deepcopy(toy_export()); bad['components'][0]['support'] = [2]
        with self.assertRaises(ValueError): v.numeric_components(bad)


if __name__ == '__main__':
    unittest.main()
