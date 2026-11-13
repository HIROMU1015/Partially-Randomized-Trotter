"""Artificial weights/events only. No saved physical probability/price reads."""
from collections import defaultdict
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from trottertracks import structural_feature_means as m
from trottertracks.h4_df10_price_v1 import Accounting
from trottertracks.mean_compiled_price import MeanKernel
from trottertracks.r_price_phase1_science import functions

ROOT = Path(__file__).resolve().parents[1]
P = (F(1, 4), F(3, 4))
O = (F(3, 4), F(1, 4))
KEYS = (('ART_A', 'hash_A'), ('ART_B', 'hash_B'))


def event(indices, order):
    # Source sampler draws rotation first; Accounting applies products first.
    sequence = (*indices[1:], indices[0])
    apps = []
    for position, i in enumerate(sequence):
        key = KEYS[i]
        apps.append(dict(application_index=position, is_identity=False, coefficient_sign=1,
            role='rotation' if position == len(sequence) - 1 else 'product',
            basis_id=key[0], basis_hash=key[1],
            basis_change_operations=[dict(name='p', qubits=[0], parameters=[str(.1 + i * .2)])],
            diagonal_pauli_support=[0] if i == 0 else [0, 1]))
    return dict(taylor_order=order, phase=dict(real=(-1)**(order//2), imag=0.),
                rotation_angle=.13 if order == 0 else .05, application_sequence=apps)


def alphabet():
    out = []
    for j, order in enumerate((0, 2)):
        for indices in product(range(2), repeat=order + 1):
            weight = O[j]
            for i in indices:
                weight *= P[i]
            out.append((event(indices, order), weight))
    return out


class Tests(unittest.TestCase):
    def test_exact_linear_means_match_pinned_D_event_metrics(self):
        scope = {}
        functions(ROOT, '39ddd59edf54a3a901e5302207b2765ddbd03fbb',
                  'src/trottertracks/p1_efficient_zz.py', ('event_metrics',), scope)
        expected = defaultdict(F)
        for e, weight in alphabet():
            counts = scope['event_metrics']([e])
            for key in ('events', 'rotation_Z', 'rotation_ZZ', 'product_applications', 'identity_applications'):
                expected[key] += weight * counts[key]
            expected['event_order_2'] += weight * counts['order2_events']
        actual = m.k2_linear_counts(P, (1, 2), O, q=1, R_outer=1)
        for key, value in expected.items():
            self.assertEqual(actual[key], value)
        self.assertEqual(actual['product_CZ'], F(7, 8))
        self.assertEqual(actual['all_applications'], F(3, 2))

    def test_basis_formula_and_kernel_match_original_Accounting_enumeration(self):
        expected = defaultdict(F)
        for (a, pa), (b, pb) in product(alphabet(), repeat=2):
            counter = Accounting()
            counter.tail([a, b])
            for name in ('tail_basis_groups', 'tail_shared_basis_pairs'):
                expected[name] += pa * pb * counter.features[name]
        formula = m.k2_basis_counts(P, KEYS, O, q=1, R_outer=2)
        for name in expected:
            kernel = m.k2_basis_kernel(P, KEYS, O, feature=name).repeat(2)
            value, _ = kernel.expected([1., 0., 0.], [0., 0., 0.])
            self.assertEqual(formula[name], expected[name])
            self.assertAlmostEqual(value, float(expected[name]))

    def test_inhomogeneous_laws_keep_state_between_events(self):
        a = m.k2_basis_kernel((1., 0.), KEYS, (1., 0.), feature='tail_basis_groups')
        b = m.k2_basis_kernel((0., 1.), KEYS, (1., 0.), feature='tail_basis_groups')
        value = a.then(a).then(b).then(b).expected([1., 0., 0.], [0.] * 3)[0]
        self.assertEqual(value, 2)
        self.assertEqual(a.expected([1., 0., 0.], [0.] * 3)[0] * 4, 4)

    def test_tail_reset_has_explicit_source_semantics_not_native_counter_reset(self):
        e = event((0,), 0)
        a = Accounting()
        a.tail([e]); a.tail([e])
        b = Accounting()
        b.tail([e, e])
        self.assertEqual(a.features['tail_basis_groups'], 2)
        self.assertEqual(b.features['tail_basis_groups'], 1)
        # Kernel reset here represents the *active* variable of the next tail,
        # not pending Z/native accounting, which this small state omits.
        k = m.k2_basis_kernel((1., 0.), KEYS, (1., 0.), feature='tail_basis_groups')
        reset = MeanKernel(k.states, [[1., 0., 0.]] * 3, [0.] * 3)
        self.assertEqual(k.then(reset).then(k).expected([1., 0., 0.], [0.] * 3)[0], 2)

    def test_source_basis_keys_not_collapsed_by_support(self):
        keys = (('A', 'h1'), ('A', 'h2'))
        result = m.k2_basis_counts(P, keys, (1., 0.), q=1, R_outer=2)
        self.assertGreater(result['tail_basis_groups'], 1)
        self.assertEqual(m.k2_basis_counts(P, ('same', 'same'), O, q=1, R_outer=2)['tail_basis_groups'], 1)

    def test_zero_rare_and_positive_rare_remain_distinct(self):
        no = m.k2_linear_counts(P, (1, 2), (1, 0), q=3, R_outer=2)
        rare = m.k2_linear_counts(P, (1, 2), (2**60, 1), q=3, R_outer=2)
        self.assertEqual(no['product_applications'], 0)
        self.assertGreater(rare['product_applications'], 0)
        self.assertGreater(rare['event_order_2'], 0)

    def test_fraction_formula_large_positions_without_event_materialization(self):
        q, R = 4096, 8160  # schedule-sized integers, ARTIFICIAL weights only
        with patch.object(np.random, 'Generator', side_effect=AssertionError('No RNG')):
            counts = m.k2_linear_counts(P, (1, 2), O, q=q, R_outer=R)
            basis = m.k2_basis_counts(P, KEYS, O, q=q, R_outer=R)
        self.assertEqual(counts['events'], q * R)
        self.assertEqual(basis['tail_basis_groups'] + basis['tail_shared_basis_pairs'], counts['all_applications'])
        kernel = m.k2_basis_kernel(P, KEYS, O, feature='tail_basis_groups')
        stats = {}
        kernel.repeat(R, stats).expected([1., 0., 0.], [0.] * 3)
        self.assertLessEqual(stats['kernel_compositions'], 2 * R.bit_length())
        self.assertEqual(kernel.transition.shape, (3, 3))

    def test_expected_nonlinear_model_cannot_use_only_feature_means(self):
        # Mutually exclusive event categories: E[N_Z*N_ZZ] differs from E[N_Z]E[N_ZZ].
        counts = m.k2_linear_counts(P, (1, 2), O, q=1, R_outer=1)
        self.assertGreater(counts['rotation_Z'] * counts['rotation_ZZ'], 0)
        self.assertEqual(sum(weight * (len(e['application_sequence'][-1]['diagonal_pauli_support']) == 1)
                             * (len(e['application_sequence'][-1]['diagonal_pauli_support']) == 2)
                             for e, weight in alphabet()), 0)

    def test_bad_or_unknown_features_not_approximated(self):
        for sizes in ((0, 2), (1, 3), (1,), (True, 2)):
            with self.assertRaises(ValueError):
                m.k2_linear_counts(P, sizes, O, q=1, R_outer=1)
        with self.assertRaises(ValueError):
            m.k2_basis_kernel(P, KEYS, O, feature='COMPILED_RZ')
        with self.assertRaises(ValueError):
            m.k2_basis_counts(P, KEYS, O, q=0, R_outer=2)
        with self.assertRaises(ValueError):
            m.k2_basis_kernel(P, KEYS, (.5, .7), feature='tail_basis_groups')
        with self.assertRaises(ValueError):
            m.k2_basis_counts(P, (None, 'b'), O, q=1, R_outer=2)


if __name__ == '__main__':
    unittest.main()
