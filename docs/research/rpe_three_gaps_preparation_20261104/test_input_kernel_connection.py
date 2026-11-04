"""Only artificial observations and current-process metadata; no H4 loader."""
from fractions import Fraction as Q
import cmath
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from flint import ctx
from common import D, ROOT, real
import decoder_finite as finite
import inspect_decoder_kernels as kernels

sys.path.insert(0, str(ROOT/'docs/research/rpe_18round_measurement_connection_20261103'))
import measurement_adapter as connection
from test_measurement_adapter import fixture


class InputKernelTests(unittest.TestCase):
    def setUp(self):
        self.prec = ctx.prec
        ctx.prec = 192

    def tearDown(self):
        ctx.prec = self.prec

    def test_existing_callback_route_uses_integer_sum_not_count_grid(self):
        for k in range(4):
            s = fixture(shots=3)
            z = connection.collect_virtual_round(s, lambda r: None,
                    lambda r, t: 1 if r.shot < (k if r.axis == 'cosine' else 2) else -1)
            self.assertIs(type(z.cosine), float)
            self.assertIs(type(z.signal), complex)
            self.assertEqual(z.cosine, finite.represented_mean(k, 3, 'integer_sum'))
            self.assertEqual(z.sine, finite.represented_mean(2, 3, 'integer_sum'))
        self.assertNotEqual(finite.represented_mean(1, 3, 'integer_sum'),
                            finite.represented_mean(1, 3, 'count_grid'))

    def test_existing_decode_passes_cmath_float_phases_to_original_decoder(self):
        rounds = [fixture(m, shots=3) for m in range(3)]
        means = [connection.collect_virtual_round(s, lambda r: None,
                    lambda r, t: 1 if r.shot < (1 if r.axis == 'cosine' else 2) else -1)
                 for s in rounds]
        original = connection.original_decoder()
        with patch.object(connection, 'original_decoder', return_value=original), \
             patch.object(connection.cmath, 'phase', wraps=cmath.phase) as phase:
            out = connection.decode_measurements(rounds, means)
        self.assertEqual(phase.call_count, 3)
        self.assertEqual(len(out['reconstruction']['rounds']), 3)
        self.assertTrue(all(type(cmath.phase(z.signal)) is float for z in means))
        self.assertFalse(out['statistical_iid_verified'])
        self.assertFalse(out['success_probability_evaluated'])

    def test_callback_annotation_does_not_enforce_python_int(self):
        s = fixture(shots=3)
        # Existing SOURCE accepts float +/-1 too. No new production policy.
        z = connection.collect_virtual_round(s, lambda r: None, lambda r, t: 1.0)
        self.assertEqual(z.signal, 1+1j)
        with self.assertRaises(ValueError):
            connection.collect_virtual_round(s, lambda r: None, lambda r, t: True)

    def test_cmath_seams_and_zero_keep_existing_adapter_status(self):
        s = fixture()
        for imag in (0.0, -0.0):
            value = connection.MeasuredMeans(s, -1., imag)
            out = connection.decode_measurements([s], [value])
            self.assertEqual(out['status'], 'ALIAS_BOUNDARY')
            self.assertIsNone(out['energy'])
        self.assertEqual(connection.decode_measurements([s],
            [connection.MeasuredMeans(s, 0., 0.)])['status'], 'ZERO_SIGNAL')

    def test_resolved_scalar_and_array_angle_loops_and_remainder_slot(self):
        for strides in ((8, 8, 8), (16, 16, 8)):
            resolved = kernels.resolved_loop(np.arctan2, ('f8', 'f8', None), strides)
            self.assertEqual(resolved['inner']['symbols'], ['DOUBLE_arctan2'])
        resolved = kernels.resolved_loop(np.exp, ('c16', None), (16, 16))
        self.assertEqual(resolved['user_data']['symbols'], ['nc_exp'])
        self.assertEqual(kernels.scalar_remainder_slot()['symbols'], ['double_remainder'])

    def test_numpy_bound_math_slots_resolve_to_libm_but_supply_no_accuracy(self):
        data = kernels.snapshot()
        targets = [b['target'] for b in data['images'][0]['bindings']]
        self.assertTrue(targets)
        self.assertTrue(all(Path(x['path']).name == 'libm.so.6' for x in targets))
        self.assertIsNone(data['precision_contract'])
        self.assertEqual(data['H4_uniform_decoder_decision'], 'UNKNOWN/STOP')

    def test_fp_control_is_observed_and_not_modified(self):
        first = kernels.fp_environment()
        kernels.resolved_loop(np.exp, ('c16', None), (16, 16))
        second = kernels.fp_environment()
        for key in ('x87_control', 'x87_nearest', 'sse_nearest',
                    'flush_to_zero', 'denormals_are_zero'):
            self.assertEqual(first[key], second[key])
        self.assertTrue(first['sse_nearest'])
        self.assertFalse(first['flush_to_zero'])
        self.assertFalse(first['denormals_are_zero'])

    def test_unknown_inspection_abi_stops_before_private_layout_access(self):
        with patch.object(kernels.np, '__version__', 'unsupported'):
            with self.assertRaisesRegex(RuntimeError, 'UNKNOWN/STOP'):
                kernels.resolved_loop(np.exp, ('c16', None), (16, 16))

    def test_fmod_boundary_fixtures_equal_exact_truncation_remainder(self):
        d = 2*math.pi
        values = [0., -0., 5e-324, -5e-324, d, -d]
        for x in (d, -d, 2*d, -2*d):
            values.extend([math.nextafter(x, 0.), math.nextafter(x, math.copysign(math.inf, x))])
        for x in values:
            ratio = Q.from_float(x)/Q.from_float(d)
            expected = Q.from_float(x)-int(ratio)*Q.from_float(d)
            self.assertEqual(Q.from_float(math.fmod(x, d)), expected)
        self.assertLess(math.copysign(1., math.fmod(-d, d)), 0)
        wrap = D.existing_functions()[0]['_wrap_phase']
        for x in (-4., 8., -math.pi, math.pi, math.nextafter(math.pi, 0.)):
            self.assertLess(D.circle_distance(wrap(x), real(x)), real(Q(64, 2**53)))

    def test_original_distance_shapes_and_complex_special_boundaries(self):
        f = D.existing_functions()[0]
        original = np.exp
        received = []
        def observe(z):
            received.append((z.dtype, z.shape, z.strides))
            return original(z)
        with patch.object(f['_phase_distance'].__globals__['np'], 'exp', observe):
            out = f['_phase_distance'](np.array([-math.pi, 0., math.pi]), .125)
        self.assertEqual(received, [(np.dtype('c16'), (3,), (16,))])
        self.assertTrue(np.isfinite(out).all())
        for x in (-2*math.pi, -math.pi, -0., 0., 5e-324, math.pi, 2*math.pi):
            z = np.exp(complex(0., x))
            self.assertTrue(np.isfinite(z))
            self.assertTrue(math.isfinite(float(np.angle(z))))
        # These point checks provide no all-domain libm error contract.
        self.assertEqual(D.branch_conditions(['.1', '.1'], [1, 2])['implementation_decision'], 'UNKNOWN/STOP')


if __name__ == '__main__':
    unittest.main()
