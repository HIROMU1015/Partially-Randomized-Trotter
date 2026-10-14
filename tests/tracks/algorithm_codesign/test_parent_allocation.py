"""Off-domain unit controls only. No registered D1/D2/D3 input or evaluator.

Reference matrices are plain Python 2x2/4x4 algebra on synthetic involutions.
No NumPy/Qiskit/synthesizer, circuit compiler, scientific runner, or shot forecast.
"""
from fractions import Fraction as F
import copy
import importlib.util
import json
import math
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from trottertracks.algorithm_codesign.parent_allocation import (
    ParentAllocation, RootAllocationGenerator, digital_proposal, reweight_events,
    event_statistics, error_accounting, native_plan, adjoint_plan, native_ir_g9,
    rotation_key, check_cache_metadata)
from trottertracks.algorithm_codesign.g9_p5 import P5Closed
from trottertracks.algorithm_codesign.g7_generator import CanonicalGenerator, DeterministicBits
from trottertracks.algorithm_codesign import g9_native

P = (F(2, 7), F(5, 7))
X = F(1, 3)
PRECISION = dict(root_bits=96, probability_bits=80, eta=F(1, 10**8), rho=F(1, 10**10))
I = [[1+0j, 0j], [0j, 1+0j]]
Q = [[[1+0j, 0j], [0j, -1+0j]],
     [[0.5+0j, math.sqrt(3)/2+0j], [math.sqrt(3)/2+0j, -0.5+0j]],
     [[0j, -1j], [1j, 0j]]]


def add(a, b):
    return [[x+y for x, y in zip(row, other)] for row, other in zip(a, b)]


def scale(a, factor):
    return [[factor*x for x in row] for row in a]


def multiply(a, b):
    n = len(a)
    return [[sum(a[i][k]*b[k][j] for k in range(n)) for j in range(n)] for i in range(n)]


def dagger(a):
    return [[a[j][i].conjugate() for j in range(len(a))] for i in range(len(a))]


def distance(a, b):
    return max(abs(x-y) for row, other in zip(a, b) for x, y in zip(row, other))


def reference_event(e):
    word = I
    for label in e['word']:
        word = multiply(word, Q[label])
    kind = e.get('event_kind', 'rotation')
    if kind == 'rotation':
        angle = math.atan(float(e['ratio']))
        rotation = add(scale(I, math.cos(angle)),
                       scale(Q[e['child']], -1j*e.get('rotation_sign', 1)*math.sin(angle)))
        word = multiply(rotation, word)
    elif kind == 'pure_child':
        word = multiply(Q[e['child']], word)
    return scale(word, 1j**e['phase_i_power'])


def target_parent(a, b, word=(), sigma=1):
    result = scale(I, float(a))
    for label, bi in enumerate(b):
        result = add(result, scale(Q[label], -1j*sigma*float(bi)))
    for label in word:
        result = multiply(result, Q[label])
    return scale(result, (-1)**(len(word)//2))


def target_p5(p, x, sigma):
    r = scale(I, 0)
    for pi, qi in zip(p, Q):
        r = add(r, scale(qi, float(pi)))
    result, power = copy.deepcopy(I), copy.deepcopy(I)
    for n in range(1, 6):
        power = multiply(power, r)
        result = add(result, scale(power, (-1j*sigma*float(x))**n/math.factorial(n)))
    return result


def corrected_mean(events):
    result = scale(I, 0)
    for e in events:
        result = add(result, scale(reference_event(e), float(e['proposal']*e['weight'])))
    return result


def controlled(matrix):
    out = [[0j]*4 for _ in range(4)]
    for i in range(2):
        out[i][i] = 1
        for j in range(2):
            out[i+2][j+2] = matrix[i][j]
    return out


def evaluate_plan(plan):
    result = [[complex(i == j) for j in range(4)] for i in range(4)]
    for gate in plan:
        if gate[0] in ('S', 's', 'Z'):
            phase = {'S': 1j, 's': -1j, 'Z': -1}[gate[0]]
            matrix = controlled(scale(I, phase))
        elif gate[0] == 'Q':
            matrix = controlled(Q[gate[1]])
        elif gate[0] == 'Rot':
            angle = math.atan(float(gate[2]))
            matrix = controlled(add(scale(I, math.cos(angle)),
                                    scale(Q[gate[1]], -1j*gate[3]*math.sin(angle))))
        else:
            raise ValueError('unknown mock gate')
        result = multiply(matrix, result)
    return result


def cq(label):
    return [('Q', label)]


def cr(label, ratio, sign):
    return [('Rot', label, ratio, sign)]


def inverse(g):
    if g[0] == 'Rot':
        return (*g[:-1], -g[-1])
    if g[0] in ('S', 's'):
        return ('s' if g[0] == 'S' else 'S', g[1])
    return g


class ParentControls(unittest.TestCase):
    def test_arbitrary_interior_mean(self):
        a = ParentAllocation(7, (3, 4), (4, 3), **PRECISION)
        self.assertLess(distance(corrected_mean([a.event(i) for i in range(len(a.atoms))]),
                                 target_parent(7, (3, 4))), 1e-12)

    def test_signed_noncommuting_parent_mean(self):
        for sigma in (-1, 1):
            a = ParentAllocation(7, (3, 4), (4, 3), word=(1, 0), sigma=sigma, **PRECISION)
            self.assertLess(distance(corrected_mean([a.event(i) for i in range(len(a.atoms))]),
                                     target_parent(7, (3, 4), (1, 0), sigma)), 1e-12)

    def test_each_atom_unitary(self):
        a = ParentAllocation(4, (1, 2), (4, 0), word=(1, 0), **PRECISION)
        for i in range(len(a.atoms)):
            matrix = reference_event(a.event(i))
            self.assertLess(distance(multiply(matrix, dagger(matrix)), I), 1e-12)

    def test_pure_child_both_signs(self):
        for sigma in (-1, 1):
            a = ParentAllocation(0, (1, 2), (0, 0), sigma=sigma, **PRECISION)
            for i in range(2):
                e = a.event(i)
                self.assertEqual(e['event_kind'], 'pure_child')
                self.assertIsNone(e['ratio'])
                self.assertLess(distance(reference_event(e), scale(Q[i], -1j*sigma)), 1e-12)

    def test_pure_parent_and_zero_child(self):
        a = ParentAllocation(3, (0, 2, 0), (1, 0, 2), **PRECISION)
        self.assertEqual([v.kind for v in a.atoms], ['pure_child', 'pure_parent'])
        self.assertEqual(a.atoms[1].coefficient, 3)
        self.assertLess(distance(corrected_mean([a.event(i) for i in range(2)]), target_parent(3, (0, 2, 0))), 1e-12)

    def test_all_zero_b_collapses_one_parent(self):
        a = ParentAllocation(3, (0, 0), (1, 2), word=(1, 0), **PRECISION)
        self.assertEqual(len(a.atoms), 1)
        self.assertEqual(a.atoms[0].kind, 'pure_parent')
        self.assertEqual(a.atoms[0].coefficient, 3)

    def test_zero_event_removed(self):
        a = ParentAllocation(0, (0, 2), (0, 0), **PRECISION)
        self.assertEqual(len(a.atoms), 1)
        self.assertEqual(a.atoms[0].child, 1)

    def test_zero_contribution_no_sampling(self):
        a = ParentAllocation(0, (0, 0), (0, 0), **PRECISION)
        self.assertEqual(a.mass.lo, 0)
        self.assertEqual(a.child_law, ())
        with self.assertRaises(ValueError):
            a.sample(DeterministicBits('off-domain-empty'))

    def test_common_angle_factorized(self):
        a = ParentAllocation(4, (1, 2), **PRECISION)
        self.assertTrue(a.is_common_angle)
        self.assertEqual(a.allocation, (F(4, 3), F(8, 3)))
        self.assertEqual([v.ratio for v in a.atoms], [F(3, 4)]*2)
        self.assertEqual([v.coefficient for v in a.atoms], [F(5, 3), F(10, 3)])

    def test_bad_allocation_rejected(self):
        for allocation in [(-1, 5), (1, 1), (), (1, 2, 1)]:
            with self.subTest(allocation=allocation), self.assertRaises(ValueError):
                ParentAllocation(4, (1, 2), allocation, **PRECISION)

    def test_a_zero_forbids_nonzero_allocation(self):
        with self.assertRaises(ValueError):
            ParentAllocation(0, (1, 2), (0, 1), **PRECISION)

    def test_float_and_boolean_rejected(self):
        for value in [0.5, True]:
            with self.subTest(value=value), self.assertRaises(TypeError):
                ParentAllocation(value, (1, 2), **PRECISION)
        with self.assertRaises(TypeError):
            RootAllocationGenerator.closed_p5((0.5, 0.5), X)

    def test_negative_coefficient_rejected(self):
        for a, b in [(-1, (1, 2)), (1, (-1, 2)), (1, ())]:
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                ParentAllocation(a, b, **PRECISION)

    def test_bad_sign_and_word_rejected(self):
        for options in [dict(sigma=0), dict(sigma=True), dict(word=(0,)),
                        dict(word=(0, 0)), dict(labels=(0, 0))]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                ParentAllocation(4, (1, 2), **options, **PRECISION)

    def test_insufficient_root_precision_fails(self):
        with self.assertRaises(ValueError):
            ParentAllocation(2, (1,), root_bits=2, probability_bits=80, rho=F(1, 10**10))

    def test_invalid_event_index_and_group_mass(self):
        a = ParentAllocation(4, (1, 2), **PRECISION)
        for index in [-1, True, 10]:
            with self.subTest(index=index), self.assertRaises(ValueError):
                a.event(index)
        for mass in [F(0), F(2)]:
            with self.subTest(mass=mass), self.assertRaises(ValueError):
                a.event(0, group_proposal=mass)


class ProposalControls(unittest.TestCase):
    def test_dyadic_full_support_and_exact_weight(self):
        a = ParentAllocation(7, (3, 4), (4, 3), **PRECISION)
        self.assertEqual(sum(a.child_law), 1)
        for i in range(2):
            e = a.event(i)
            self.assertGreater(e['proposal'], 0)
            self.assertEqual(e['proposal']*e['weight'], e['coefficient'])
            self.assertEqual((e['proposal']*(1 << a.H)).denominator, 1)

    def test_coefficient_enclosure_not_conflated_with_ideal(self):
        a = ParentAllocation(4, (1, 2), (4, 0), **PRECISION)
        bound = a.atoms[0].coefficient_interval
        self.assertLessEqual(bound.lo**2, 17)
        self.assertGreaterEqual(bound.hi**2, 17)
        self.assertGreater(a.coefficient_error_upper, 0)

    def test_statistics_use_actual_proposal(self):
        a = ParentAllocation(4, (1, 2), **PRECISION)
        events = [a.event(i) for i in range(2)]
        result = event_statistics(events)
        self.assertEqual(result['weight_second_moment'], sum(e['coefficient']**2/e['proposal'] for e in events))
        self.assertEqual(result['acceptance'], 1)
        self.assertEqual(result['range'], max(e['weight'] for e in events))

    def test_partial_statistics_are_explicit(self):
        a = ParentAllocation(4, (1, 2), **PRECISION)
        events = [a.event(i, group_proposal=F(1, 3)) for i in range(2)]
        self.assertEqual(event_statistics(events, complete=False)['acceptance'], F(1, 3))
        with self.assertRaises(ValueError):
            event_statistics(events)

    def test_cost_aware_positive_cost(self):
        law = digital_proposal((F(2), F(1)), mode='cost_aware', costs=(F(1), F(4)), **PRECISION)
        self.assertLess(abs(law[0]-F(4, 5)), F(1, 10**8))
        self.assertEqual(sum(law), 1)

    def test_zero_cost_analytic_limit_rejected(self):
        for costs in [(0, 1), (0, 0)]:
            with self.subTest(costs=costs), self.assertRaises(ValueError):
                digital_proposal((1, 2), mode='cost_aware', costs=costs, **PRECISION)

    def test_regularization_explicit_and_not_cost_charge(self):
        law = digital_proposal((1, 2), mode='regularized', costs=(0, 0), regularization=4, **PRECISION)
        canonical = digital_proposal((1, 2), **PRECISION)
        self.assertEqual(law, canonical)
        for lam in [None, 0, -1]:
            with self.subTest(lam=lam), self.assertRaises(ValueError):
                digital_proposal((1, 2), mode='regularized', costs=(0, 1), regularization=lam, **PRECISION)

    def test_invalid_costs_and_unknown_mode(self):
        for options in [dict(mode='cost_aware', costs=(1,)), dict(mode='cost_aware', costs=(-1, 2)),
                        dict(mode='wrong'), dict(mode='canonical', regularization=1)]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                digital_proposal((1, 2), **options, **PRECISION)

    def test_full_support_bit_failure_no_rescue(self):
        with self.assertRaises(ValueError):
            digital_proposal((1, 100), probability_bits=1)

    def test_reweight_preserves_coefficients_and_corrected_mean(self):
        a = ParentAllocation(7, (3, 4), (4, 3), **PRECISION)
        old = [a.event(i) for i in range(2)]
        new = reweight_events(old, mode='cost_aware', costs=(1, 4), **PRECISION)
        self.assertEqual([e['coefficient'] for e in old], [e['coefficient'] for e in new])
        self.assertLess(distance(corrected_mean(old), corrected_mean(new)), 1e-12)
        self.assertNotEqual([e['proposal'] for e in old], [e['proposal'] for e in new])

    def test_weight_tamper_rejected(self):
        a = ParentAllocation(4, (1, 2), **PRECISION)
        events = [a.event(i) for i in range(2)]
        events[0] = dict(events[0], weight=F(1))
        with self.assertRaises(ValueError):
            event_statistics(events)

    def test_error_accounting_external_coverage(self):
        a = ParentAllocation(4, (1, 2), **PRECISION)
        events = [a.event(i) for i in range(2)]
        result = error_accounting(events, coefficient_error_upper=F(1, 1000),
                                  angle_error_bounds=(F(1, 1000), F(2, 1000)),
                                  native_error_bounds=(F(3, 1000), F(4, 1000)))
        expected = F(1, 1000)+sum(e['coefficient']*F(v, 1000) for e, v in zip(events, (4, 6)))
        self.assertEqual(result['operator_mean_error_upper'], expected)
        self.assertFalse(result['confidence_or_shots_computed'])

    def test_missing_error_not_assumed_zero(self):
        a = ParentAllocation(4, (1, 2), **PRECISION)
        events = [a.event(i) for i in range(2)]
        with self.assertRaises(ValueError):
            error_accounting(events, coefficient_error_upper=0, angle_error_bounds=(0,), native_error_bounds=(0, 0))
        with self.assertRaises(TypeError):
            error_accounting(events, coefficient_error_upper=0, angle_error_bounds=(None, 0), native_error_bounds=(0, 0))


class GeneratorControls(unittest.TestCase):
    def test_common_angle_p5_byte_equivalent_metadata(self):
        old = P5Closed(P, X, **PRECISION)
        new = RootAllocationGenerator(old)
        self.assertTrue(new.legacy_exact)
        self.assertEqual(list(old.reference_events()), list(new.reference_events()))
        self.assertEqual(new.B, old.B)
        self.assertEqual(new.group_law, old.group_law)

    def test_common_angle_ordinary_metadata(self):
        old = CanonicalGenerator(P, X, 5, 'ordinary', **PRECISION)
        new = RootAllocationGenerator(old)
        for index, group in enumerate(old.groups):
            for child in range(2):
                self.assertEqual(new.event(index, (0,)*group.degree, child), old.event(index, (0,)*group.degree, child))
        self.assertEqual(new.B, old.B)

    def test_common_angle_sampling_bit_replay(self):
        for base in [P5Closed(P, X, **PRECISION), CanonicalGenerator(P, X, 5, 'ordinary', **PRECISION)]:
            new = RootAllocationGenerator(base)
            left, right = DeterministicBits('parent-off-domain-replay'), DeterministicBits('parent-off-domain-replay')
            for _ in range(8):
                self.assertEqual(base.sample(left), new.sample(right))
                self.assertEqual(left.consumed_bits, right.consumed_bits)

    def test_both_representations_arbitrary_nonvertex_mean(self):
        for factory in [RootAllocationGenerator.ordinary, RootAllocationGenerator.closed_p5]:
            baseline = factory(P, X, **PRECISION)
            allocation = (baseline.root.a/F(3), baseline.root.a*F(2, 3))
            for sigma in (-1, 1):
                new = factory(P, X, allocation, sigma=sigma, **PRECISION)
                records = list(new.reference_events())
                self.assertEqual(event_statistics(records)['acceptance'], 1)
                self.assertLess(distance(corrected_mean(records), target_p5(P, X, sigma)), 2e-12)

    def test_both_representations_vertex_endpoints(self):
        for factory in [RootAllocationGenerator.ordinary, RootAllocationGenerator.closed_p5]:
            baseline = factory(P, X, **PRECISION)
            for sigma in (-1, 1):
                new = factory(P, X, (baseline.root.a, 0), sigma=sigma, **PRECISION)
                self.assertEqual(new.root.atoms[1].kind, 'pure_child')
                self.assertLess(distance(corrected_mean(list(new.reference_events())), target_p5(P, X, sigma)), 2e-12)

    def test_nonroot_coefficients_and_ratios_preserved(self):
        for base in [P5Closed(P, X, **PRECISION), CanonicalGenerator(P, X, 5, 'ordinary', **PRECISION)]:
            a = base.a0 if isinstance(base, P5Closed) else F(1)
            new = RootAllocationGenerator(base, (a, 0))
            self.assertEqual(new.roots[1:], base.roots[1:])
            for e in new.reference_events():
                if e['raw_word']:
                    group = next(i for i in range(1, len(new.roots))
                                 if (base.groups[i].degree == len(e['raw_word']) if isinstance(base, CanonicalGenerator)
                                     else (base.groups[i][0] == 'two' and tuple(base.groups[i][1:3]) == e['raw_word'])
                                     or (base.groups[i][0] == 'four' and base.groups[i][1] == e['raw_word'][0] and len(e['raw_word']) == 4)))
                    old = base.event(group, e['raw_word'], e['child'])
                    self.assertEqual(e['coefficient'], old['coefficient'])
                    self.assertEqual(e['ratio'], old['ratio'])

    def test_sample_matches_reference_event_law(self):
        baseline = RootAllocationGenerator.closed_p5(P, X, **PRECISION)
        new = RootAllocationGenerator.closed_p5(P, X, (baseline.root.a, 0), **PRECISION)
        def key(e):
            return e['raw_word'], e['child'], e.get('event_kind', 'rotation')
        reference = {key(e): e for e in new.reference_events()}
        bits = DeterministicBits('parent-off-domain-interface')
        for _ in range(24):
            e = new.sample(bits)
            self.assertEqual(e, reference[key(e)])

    def test_each_group_is_forced_without_statistical_sampling(self):
        class ForcedBits:
            def __init__(self, first):
                self.first = first
            def bits(self, n):
                value = self.first
                self.first = 0
                return value
        for factory in [RootAllocationGenerator.ordinary, RootAllocationGenerator.closed_p5]:
            baseline = factory(P, X, **PRECISION)
            new = factory(P, X, (baseline.root.a/F(4), baseline.root.a*F(3, 4)), **PRECISION)
            reference = list(new.reference_events())
            cumulative = F(0)
            for mass in new.group_law:
                position = int((cumulative+mass/2)*(1 << new.H))
                e = new.sample(ForcedBits(position))
                self.assertIn(e, reference)
                cumulative += mass

    def test_sampling_does_not_enumerate(self):
        baseline = RootAllocationGenerator.closed_p5(P, X, **PRECISION)
        new = RootAllocationGenerator.closed_p5(P, X, (baseline.root.a, 0), **PRECISION)
        with patch.object(new, 'reference_events', side_effect=AssertionError('no table')), \
             patch.object(new.base, 'reference_events', side_effect=AssertionError('no table')):
            new.sample(DeterministicBits('parent-off-domain-no-table'))

    def test_reference_cap_and_invalid_group(self):
        new = RootAllocationGenerator.closed_p5(P, X, **PRECISION)
        with self.assertRaises(ValueError):
            list(new.reference_events(max_events=1))
        for index in [-1, True, 100]:
            with self.subTest(index=index), self.assertRaises(ValueError):
                new.event(index, (), 0)

    def test_root_cost_aware_uses_same_adapter(self):
        for factory in [RootAllocationGenerator.ordinary, RootAllocationGenerator.closed_p5]:
            new = factory(P, X, proposal_mode='regularized', root_costs=(0, 4), regularization=3, **PRECISION)
            self.assertFalse(new.legacy_exact)
            self.assertLess(distance(corrected_mean(list(new.reference_events())), target_p5(P, X, 1)), 2e-12)

    def test_single_label_off_domain(self):
        new = RootAllocationGenerator.closed_p5((F(1),), F(1, 5), **PRECISION)
        self.assertEqual(new.group_law, (F(1),))
        self.assertEqual(len(list(new.reference_events())), 1)

    def test_resource_inputs_do_not_forecast_or_invent_cost(self):
        new = RootAllocationGenerator.closed_p5(P, X, **PRECISION)
        cert = new.certificate()
        self.assertEqual(cert['shot_forecast'], 'NOT_IMPLEMENTED')
        self.assertEqual(cert['new_angle_acquisition_cost'], 'NOT_ACQUIRED')
        self.assertEqual(cert['online_sampling_cost'], 'NOT_MEASURED')


class NativeControls(unittest.TestCase):
    def test_controlled_noncommuting_plan_and_all_endpoints(self):
        for b, z in [((3, 4), (4, 3)), ((3, 4), (7, 0)), ((0, 4), (7, 0)), ((0, 0), (4, 3))]:
            for sigma in (-1, 1):
                allocation = ParentAllocation(7, b, z, word=(1, 0), sigma=sigma, **PRECISION)
                for i in range(len(allocation.atoms)):
                    event = allocation.event(i)
                    plan = native_plan(event, cq, cr, control=1)
                    self.assertLess(distance(evaluate_plan(plan), controlled(reference_event(event))), 1e-12)

    def test_phase_drop_is_detected_independently(self):
        e = ParentAllocation(0, (1,), sigma=1, **PRECISION).event(0)
        plan = native_plan(e, cq, cr, control=1)
        self.assertEqual(plan[0], ('s', 1))
        self.assertGreater(distance(evaluate_plan(plan[1:]), controlled(reference_event(e))), 0.5)

    def test_rotation_order_is_detected(self):
        e = ParentAllocation(7, (3, 4), (4, 3), word=(1, 0), **PRECISION).event(0)
        plan = native_plan(e, cq, cr, control=1)
        wrong = [plan[0], plan[-1], *plan[1:-1]]
        self.assertGreater(distance(evaluate_plan(wrong), controlled(reference_event(e))), 0.1)

    def test_full_adjoint_inverts_order_and_signed_rotation(self):
        e = ParentAllocation(7, (3, 4), (4, 3), word=(1, 0), sigma=-1, **PRECISION).event(0)
        plan = native_plan(e, cq, cr, control=1)
        self.assertLess(distance(evaluate_plan(adjoint_plan(plan, inverse)), dagger(evaluate_plan(plan))), 1e-12)

    def test_endpoint_never_requests_rotation(self):
        for a, b, z in [(0, (1,), (0,)), (2, (0,), (2,))]:
            e = ParentAllocation(a, b, z, **PRECISION).event(0)
            native_plan(e, cq, lambda *args: self.fail('endpoint requested rotation'), control=1)

    def test_g9_legacy_binding_with_mock_provider(self):
        event = {'word': (0, 1), 'child': 1, 'ratio': F(2, 9), 'phase_i_power': 2}
        with patch.object(g9_native, 'CQ', side_effect=lambda child, control=3: [('CZ', control, child)]), \
             patch.object(g9_native, 'V', side_effect=lambda child: [('T', child)]), \
             patch.object(g9_native, 'crot', side_effect=lambda axis, ratio, sign=1, control=3: [('R', 0, str(ratio), sign)]):
            for helper in (False, True):
                self.assertEqual(native_ir_g9(event, helper), g9_native.native_ir(event, helper))

    def test_g9_endpoint_phase_binding_is_not_silently_ignored(self):
        event = ParentAllocation(0, (1,), sigma=-1, **PRECISION).event(0)
        with patch.object(g9_native, 'CQ', return_value=[('CZ', 3, 0)]):
            plan = native_ir_g9(event)
            self.assertEqual(plan[0], ('S', 3))
            self.assertFalse(any(g[0] == 'R' for g in plan))

    def test_bad_phase_kind_and_endpoint_surrogate(self):
        base = ParentAllocation(0, (1,), **PRECISION).event(0)
        for change in [dict(phase_i_power=4), dict(phase_i_power=0), dict(event_kind='unknown'), dict(ratio=F(10**9))]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                native_plan(dict(base, **change), cq, cr, control=1)

    def test_g9_out_of_provider_label_rejected(self):
        event = ParentAllocation(0, (1,), labels=(7,), **PRECISION).event(0)
        with self.assertRaises(ValueError):
            native_ir_g9(event)


class CacheControls(unittest.TestCase):
    def key_and_entry(self):
        e = ParentAllocation(4, (1, 2), **PRECISION).event(0)
        expected = rotation_key(e, precision=F(1, 10**6),
                                tool_identity={'name': 'mock', 'version': 'off-domain',
                                               'source_sha256': '0'*64, 'seed': 'fixed-mock'})
        entry = dict(expected, sequence='HT', sequence_sha256=hashlib.sha256(b'HT').hexdigest(),
                     up_to_phase=False, complete=True, operator_error_upper='1/2000000')
        return e, expected, entry

    def test_signed_inverse_shares_positive_key(self):
        e, expected, entry = self.key_and_entry()
        reversed_key = rotation_key(dict(e, rotation_sign=-1), precision=F(1, 10**6), tool_identity=expected['spec']['tool'])
        self.assertEqual(expected, reversed_key)
        self.assertEqual(expected['spec']['scale'], '1')
        self.assertEqual(check_cache_metadata(entry, expected)['metadata_identity'], 'PASS')

    def test_precision_or_tool_change_has_different_key(self):
        e, expected, _ = self.key_and_entry()
        other = rotation_key(e, precision=F(1, 10**7), tool_identity=expected['spec']['tool'])
        self.assertNotEqual(expected['key'], other['key'])
        tool = dict(expected['spec']['tool'], seed='different')
        self.assertNotEqual(expected['key'], rotation_key(e, precision=F(1, 10**6), tool_identity=tool)['key'])

    def test_endpoints_have_no_key(self):
        for a, b, z in [(0, (1,), (0,)), (1, (0,), (1,))]:
            self.assertIsNone(rotation_key(ParentAllocation(a, b, z, **PRECISION).event(0),
                                          precision=F(1, 10**6), tool_identity={}))

    def test_cache_tamper_or_missing_guard_fails(self):
        _, expected, entry = self.key_and_entry()
        for changed in [dict(entry, sequence='TH'), dict(entry, up_to_phase=True),
                        dict(entry, complete=False), dict(entry, operator_error_upper='1/1000')]:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                check_cache_metadata(changed, expected)

    def test_expected_spec_cannot_be_changed_behind_key(self):
        _, expected, entry = self.key_and_entry()
        tampered = dict(expected, spec=dict(expected['spec'], precision='1/1000'))
        with self.assertRaises(ValueError):
            check_cache_metadata(dict(entry, spec=tampered['spec']), tampered)

    def test_incomplete_tool_identity_rejected(self):
        e, expected, _ = self.key_and_entry()
        for change in [dict(source_sha256='missing'), dict(name=''), dict(seed=True)]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                rotation_key(e, precision=F(1, 10**6), tool_identity=dict(expected['spec']['tool'], **change))


class PreflightControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = Path(__file__).resolve().parents[3]/'scripts/tracks/algorithm_codesign/preflight_parent_allocation.py'
        spec = importlib.util.spec_from_file_location('parent_metadata_preflight', cls.path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def fixture(self, root):
        import hashlib
        p = root/'source.txt'; p.write_bytes(b'off-domain mock source\n')
        manifest = {'science_execution_authorized': False, 'files': [
            {'path': 'source.txt', 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}]}
        plan = {'science_execution_authorized': False, 'authorization_commit': None, 'scientific_choices': {}}
        return plan, manifest

    def test_zero_compute_preflight_reports_pending_choices(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, manifest = self.fixture(Path(temp))
            result = self.module.inspect_metadata(plan, manifest, temp)
            self.assertFalse(result['science_execution_allowed'])
            self.assertEqual(result['scientific_calls'], 0)
            self.assertEqual(len(result['pending_scientific_decisions']), 7)

    def test_hash_tamper_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, manifest = self.fixture(Path(temp))
            (Path(temp)/'source.txt').write_text('changed')
            with self.assertRaises(ValueError):
                self.module.inspect_metadata(plan, manifest, temp)

    def test_metadata_cannot_authorize_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, manifest = self.fixture(Path(temp))
            for change in [dict(science_execution_authorized=True), dict(authorization_commit='unapproved')]:
                with self.subTest(change=change), self.assertRaises(ValueError):
                    self.module.inspect_metadata(dict(plan, **change), manifest, temp)

    def test_manifest_path_escape_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, manifest = self.fixture(Path(temp))
            for path in ['/tmp/source.txt', '../source.txt']:
                manifest['files'][0]['path'] = path
                with self.subTest(path=path), self.assertRaises(ValueError):
                    self.module.inspect_metadata(plan, manifest, temp)

    def test_execute_flag_rejected_before_any_work(self):
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run([sys.executable, '-B', str(self.path), '--execute',
                                     '--plan', 'unused', '--manifest', 'unused'],
                                    cwd=temp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(list(Path(temp).iterdir()), [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
