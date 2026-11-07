"""New splice delta only; invented rotations, no H4/RNG/synthesis/compile."""
from fractions import Fraction as F
import math
import unittest
from unittest.mock import patch

import numpy as np

from trottertracks.algorithm_codesign import same_position_residual as a
from trottertracks.algorithm_codesign.g7_generator import CanonicalGenerator
from trottertracks.algorithm_codesign.parent_allocation import event_statistics
from trottertracks.algorithm_codesign.parent_allocation_native import operator_target

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1., -1.])


def pauli(word):
    out = np.ones((1, 1), dtype=complex)
    for letter in word:
        out = np.kron(out, dict(I=I, X=X, Y=Y, Z=Z)[letter])
    return out


def rotate(word, angle):
    t = math.pi*float(angle.pi) + float(angle.rad)
    p = pauli(word)
    return math.cos(t/2)*np.eye(len(p))-1j*math.sin(t/2)*p


def time_matrix(steps):
    first = steps[0].operation
    size = len(pauli(first.generator)) if isinstance(first, a.RotationSite) else len(first)
    out = np.eye(size, dtype=complex)
    for step in steps:
        op = step.operation
        mat = rotate(op.generator, op.angle) if isinstance(op, a.RotationSite) else op
        out = mat @ out
    return out


def word_for(replacement, instructions=None):
    s = replacement.original
    if instructions is None:
        instructions = (rotate(s.generator, replacement.residual),)
    return a.ResidualWord(s.site_id, s.generator, replacement.residual, instructions)


class PlacementDeltaTests(unittest.TestCase):
    def setUp(self):
        self.sites = (
            a.RotationSite('d0', 'Z', a.Angle(rad=F(2, 5)), 'D_DIAGONAL'),
            a.RotationSite('basis', 'X', a.Angle(pi=F(1, 2)), 'FIXED'),
            a.RotationSite('r0', 'Y', a.Angle(rad=F(-3, 8)), 'R_DIAGONAL'),
            a.RotationSite('d1', 'Z', a.Angle(rad=F(-1, 7)), 'D_DIAGONAL'))
        self.coarse = {s.site_id: a.Angle() for s in self.sites if s.role != 'FIXED'}

    def plan(self, arm='S2'):
        return a.plan_sequence(self.sites, self.coarse, arm=arm)

    def test_three_arms_common_table_and_fixed_sites(self):
        self.assertEqual(self.plan('S0'), self.sites)
        self.assertEqual([isinstance(p, a.Replacement) for p in self.plan('S1')], [1, 0, 1, 1])
        self.assertEqual([isinstance(p, a.Replacement) for p in self.plan('S2')], [1, 0, 0, 1])

    def test_exact_symbolic_angle_sign_and_no_periodic_phase_loss(self):
        sites = (a.RotationSite('phase', 'I', a.Angle(pi=2, rad=F(-1, 5)), 'D_DIAGONAL'),)
        p, = a.plan_sequence(sites, {'phase': a.Angle(pi=2)}, arm='S2')
        self.assertEqual(p.residual, a.Angle(rad=F(-1, 5)))
        steps = a.splice_realization((p,), {'phase': word_for(p)})
        np.testing.assert_allclose(time_matrix(steps), rotate('I', sites[0].angle), atol=2e-15, rtol=0)
        # Dropping a 2*pi rotation changes the strict operator by a sign.
        self.assertGreater(np.max(abs(time_matrix(steps)-rotate('I', p.residual))), 1)

    def test_noncommuting_neighbors_stay_in_place(self):
        plan = self.plan()
        words = {p.original.site_id: word_for(p) for p in plan if isinstance(p, a.Replacement)}
        steps = a.splice_realization(plan, words)
        self.assertEqual([s.site_id for s in steps], ['d0', 'd0', 'basis', 'r0', 'd1', 'd1'])
        target = time_matrix(a.splice_realization(self.plan('S0'), {}))
        np.testing.assert_allclose(time_matrix(steps), target, atol=2e-15, rtol=0)
        moved = tuple(s for s in steps if s.part != 'residual') + tuple(s for s in steps if s.part == 'residual')
        self.assertGreater(np.max(abs(time_matrix(moved)-target)), .1)

    def test_zero_residual_needs_no_provider_and_no_extra_word(self):
        p = a.plan_sequence((self.sites[0],), {'d0': self.sites[0].angle}, arm='S2')
        self.assertEqual(len(a.splice_realization(p, {})), 1)
        with self.assertRaises(ValueError):
            a.splice_realization(p, {'d0': word_for(p[0])})

    def test_realization_site_generator_angle_and_scope_reject(self):
        p, = a.plan_sequence((self.sites[0],), {'d0': a.Angle()}, arm='S2')
        for kwargs in (dict(site_id='other'), dict(generator='X'), dict(angle=a.Angle(rad=1))):
            data = dict(site_id='d0', generator='Z', angle=p.residual, time_instructions=('toy',))
            data.update(kwargs)
            with self.assertRaises(ValueError):
                a.splice_realization((p,), {'d0': a.ResidualWord(**data)})
        for kwargs in (dict(semantics='CHANNEL'), dict(phase_retained=False), dict(time_instructions=())):
            data = dict(site_id='d0', generator='Z', angle=p.residual, time_instructions=('toy',))
            data.update(kwargs)
            with self.assertRaises(ValueError):
                a.ResidualWord(**data)
        with self.assertRaises(ValueError):
            a.splice_realization((p,), {})

    def test_duplicate_register_role_and_incomplete_coarse_table_reject(self):
        for sites in ((self.sites[0], self.sites[0]), (self.sites[0], a.RotationSite('bad', 'ZZ', a.Angle(), 'FIXED'))):
            with self.assertRaises(ValueError):
                a.plan_sequence(sites, self.coarse, arm='S2')
        for table in ({}, dict(self.coarse, basis=a.Angle()), dict(self.coarse, d0=.2)):
            with self.assertRaises(ValueError):
                a.plan_sequence(self.sites, table, arm='S2')
        with self.assertRaises(ValueError):
            a.RotationSite('d', 'Z', a.Angle(), 'MIXED_D_R')
        with self.assertRaises(TypeError):
            a.Angle(rad=.1)

    def test_canonical_finite_operator_mean_inserted_without_exact_claim(self):
        # Existing ordinary degree-3 generator, one invented involution.
        # Enumerate its two order branches only; no RNG/sample calls.
        p, = a.plan_sequence((a.RotationSite('toy', 'Y', a.Angle(rad=F(-1, 3)), 'D_DIAGONAL'),),
                            {'toy': a.Angle(rad=F(-1, 6))}, arm='S2')
        x = F(1, 12)  # abs(residual)/2, signed R_A(theta) convention
        with patch.object(CanonicalGenerator, 'sample', side_effect=AssertionError('no RNG')):
            gen = CanonicalGenerator((F(1),), x, 3, 'ordinary', root_bits=96, probability_bits=80)
            events = [dict(gen.event(i, (0,)*g.degree, 0), rotation_sign=-1) for i, g in enumerate(gen.groups)]
            event_statistics(events)  # existing coefficient/proposal/weight check
            mean = np.zeros((2, 2), complex)
            for event in events:
                op = operator_target(event, identity=lambda: I, multiply=lambda l, r: l@r,
                    involution=lambda _: Y,
                    rotation=lambda _, ratio, sign: math.cos(math.atan(float(ratio)))*I-1j*sign*math.sin(math.atan(float(ratio)))*Y,
                    scale=lambda matrix, value: matrix*value)
                step = a.splice_realization((p,), {'toy': word_for(p, (op,))})
                mean += float(event['proposal']*event['weight'])*time_matrix(step)
        poly = sum((1j*float(x))**k/math.factorial(k)*np.linalg.matrix_power(Y, k) for k in range(4))
        np.testing.assert_allclose(mean, poly@rotate('Y', p.coarse), atol=3e-15, rtol=0)
        exact = rotate('Y', p.original.angle)
        self.assertGreater(np.max(abs(mean-exact)), 1e-7)

    def test_joint_control_and_identity_phase_preserved_in_splice(self):
        # Ordinary diag(I,RZ(theta)) lowered to two commuting joint Paulis.
        theta = F(2, 5)
        sites = (a.RotationSite('i', 'IZ', a.Angle(rad=theta/2), 'D_DIAGONAL'),
                 a.RotationSite('z', 'ZZ', a.Angle(rad=-theta/2), 'D_DIAGONAL'))
        plan = a.plan_sequence(sites, {s.site_id: a.Angle() for s in sites}, arm='S2')
        mat = time_matrix(a.splice_realization(plan, {p.original.site_id: word_for(p) for p in plan}))
        target = np.zeros((4, 4), complex)
        target[:2, :2] = I
        target[2:, 2:] = rotate('Z', a.Angle(rad=theta))
        np.testing.assert_allclose(mat, target, atol=2e-15, rtol=0)
        # Controlled system identity is relative phase, not removable joint phase.
        phases = (a.RotationSite('ii', 'II', a.Angle(rad=theta/2), 'D_DIAGONAL'),
                  a.RotationSite('zi', 'ZI', a.Angle(rad=-theta/2), 'D_DIAGONAL'))
        pp = a.plan_sequence(phases, {s.site_id: a.Angle() for s in phases}, arm='S2')
        result = time_matrix(a.splice_realization(pp, {p.original.site_id: word_for(p) for p in pp}))
        expected = np.diag([1, 1, np.exp(-.5j*float(theta)), np.exp(-.5j*float(theta))])
        np.testing.assert_allclose(result, expected, atol=2e-15, rtol=0)

    def test_no_synthesis_sampler_or_scientific_provider_called(self):
        with patch.object(CanonicalGenerator, '__init__', side_effect=AssertionError('no generator')):
            plan = self.plan()
            words = {p.original.site_id: a.ResidualWord(p.original.site_id, p.original.generator,
                     p.residual, (('opaque-existing-provider-word',),)) for p in plan if isinstance(p, a.Replacement)}
            steps = a.splice_realization(plan, words)
            self.assertEqual(len(steps), 6)


if __name__ == '__main__':
    unittest.main()
