"""Independent exact formal controls; no production imports or science fixtures.

Q labels below are free involution symbols, not matrices, Pauli Hamiltonians,
circuits, or a registered RTE input. Analytic proofs are in the linked report.
"""
from fractions import Fraction as F
import unittest


def reduce_word(word):
    out = []
    for label in word:
        if out and out[-1] == label:
            out.pop()
        else:
            out.append(label)
    return tuple(out)


def plus(*polynomials):
    out = {}
    for p in polynomials:
        for key, value in p.items():
            out[key] = out.get(key, F(0)) + value
    return {k: v for k, v in out.items() if v}


def scale(poly, value):
    return {k: v * value for k, v in poly.items() if v * value}


def product(left, right):
    out = {}
    for (word, imaginary), a in left.items():
        for (other, imag2), b in right.items():
            key = reduce_word(word + other), (imaginary + imag2) % 2
            sign = -1 if imaginary + imag2 == 2 else 1
            out[key] = out.get(key, F(0)) + sign * a * b
    return {k: v for k, v in out.items() if v}


def adjoint(poly):
    return {(tuple(reversed(word)), imag): (-v if imag else v)
            for (word, imag), v in poly.items()}


def event(z, b, c, child, word, sigma=1):
    z, b, c = F(z), F(b), F(c)
    if c <= 0 or z < 0 or b < 0 or c * c != z * z + b * b:
        raise ValueError('nonzero exact Pythagorean control required')
    if len(word) % 2 or sigma not in (-1, 1):
        raise ValueError('even parent and signed time required')
    rotation = plus({((), 0): z / c}, {((child,), 1): -sigma * b / c})
    return scale(product(rotation, {(word, 0): F(1)}), (-1) ** (len(word) // 2))


def digital_statistics(coefficients, proposal):
    if len(coefficients) != len(proposal) or sum(proposal) != 1 or min(proposal) <= 0:
        raise ValueError('positive exact proposal required')
    weights = [c / q for c, q in zip(coefficients, proposal)]
    return weights, sum(q * w * w for q, w in zip(proposal, weights)), max(weights)


class FormalControls(unittest.TestCase):
    def test_mean_for_both_time_signs_and_parent_phases(self):
        for word in [(), (7, 8), (7, 8, 7, 8)]:
            for sigma in (-1, 1):
                actual = plus(scale(event(4, 3, 5, 0, word, sigma), 5),
                              scale(event(3, 4, 5, 1, word, sigma), 5))
                expected = scale({(word, 0): F(7),
                                  ((0,) + word, 1): -sigma * F(3),
                                  ((1,) + word, 1): -sigma * F(4)},
                                 (-1) ** (len(word) // 2))
                self.assertEqual(actual, expected)

    def test_each_branch_is_formally_unitary(self):
        for word in [(), (7, 8), (7, 8, 7, 8)]:
            for sigma in (-1, 1):
                u = event(4, 3, 5, 0, word, sigma)
                self.assertEqual(product(u, adjoint(u)), {((), 0): F(1)})
                self.assertEqual(product(adjoint(u), u), {((), 0): F(1)})

    def test_noncommuting_order_is_observable(self):
        correct = event(4, 3, 5, 0, (7, 8))
        wrong = scale(product({((7, 8), 0): F(1)}, event(4, 3, 5, 0, ())), -1)
        self.assertNotEqual(correct, wrong)

    def test_controlled_phase_cannot_be_dropped(self):
        controlled_branch = event(4, 3, 5, 0, (7, 8))
        missing_phase = scale(controlled_branch, -1)
        self.assertNotEqual(controlled_branch, missing_phase)
        # Ancilla-0 block remains identity in both implementations.
        self.assertNotEqual(({((), 0): F(1)}, controlled_branch),
                            ({((), 0): F(1)}, missing_phase))

    def test_pure_child_endpoint_retains_imaginary_phase(self):
        for sigma in (-1, 1):
            self.assertEqual(event(0, 3, 3, 0, (), sigma), {((0,), 1): -sigma * F(1)})
            self.assertNotEqual(event(0, 3, 3, 0, (), sigma), {((0,), 0): F(1)})

    def test_pure_parent_endpoint(self):
        self.assertEqual(event(3, 0, 3, 0, (7, 8)), {((7, 8), 0): F(-1)})

    def test_zero_event_is_not_normalized(self):
        with self.assertRaises(ValueError):
            event(0, 0, 0, 0, ())

    def test_original_common_angle_is_recovered(self):
        branches = [event(F(4, 3), 1, F(5, 3), 0, ()),
                    event(F(8, 3), 2, F(10, 3), 1, ())]
        self.assertEqual(plus(scale(branches[0], F(5, 3)),
                              scale(branches[1], F(10, 3))),
                         {((), 0): F(4), ((0,), 1): F(-1), ((1,), 1): F(-2)})
        self.assertEqual(F(5, 3) + F(10, 3), 5)

    def test_norm_cross_identity(self):
        # Expand both sides as commutative polynomials in four indeterminates.
        def mul(p, q):
            out = {}
            for key, a in p.items():
                for other, b in q.items():
                    term = tuple(sorted(key + other))
                    out[term] = out.get(term, F(0)) + a*b
            return {k: v for k, v in out.items() if v}
        zi, zj, bi, bj = [{(i,): F(1)} for i in range(4)]
        c1_squared = plus(mul(zi, zi), mul(bi, bi))
        c2_squared = plus(mul(zj, zj), mul(bj, bj))
        inner = plus(mul(zi, zj), mul(bi, bj))
        determinant = plus(mul(zi, bj), scale(mul(zj, bi), -1))
        self.assertEqual(plus(mul(c1_squared, c2_squared),
                              scale(mul(inner, inner), -1),
                              scale(mul(determinant, determinant), -1)), {})
        # Independent exact unequal-slope case, no square-root numerics.
        z1, b1, c1, z2, b2, c2 = map(F, (4, 3, 5, 3, 4, 5))
        self.assertEqual((c1*c2)**2-(z1*z2+b1*b2)**2, (z1*b2-z2*b1)**2)
        self.assertGreater((c1+c2)**2-((z1+z2)**2+(b1+b2)**2), 0)

    def test_fixed_cost_gradient_and_norm_stationarity(self):
        a, d = F(4), F(5)
        b = (F(1), F(2)); c = (F(5, 3), F(10, 3)); v = (F(1), F(-1))
        self.assertEqual(sum(a/d * vi for vi in v), 0)
        self.assertLess(a/d * sum(si*vi for si, vi in zip((F(1), F(3)), v)), 0)
        self.assertGreater(sum(bi*bi*vi*vi/(ci**3) for bi, ci, vi in zip(b, c, v)), 0)

    def test_equal_cost_kkt_reduces_to_original(self):
        lam, C, root = F(4), F(25), F(3)
        self.assertEqual(root*root, C-lam*lam)
        z = [bi*lam/root for bi in (F(1), F(2))]
        self.assertEqual(sum(z), 4)
        self.assertEqual(z, [F(4, 3), F(8, 3)])

    def test_digital_weight_mean_and_moment(self):
        c = [F(3, 2), F(5, 4)]; q = [F(1, 4), F(3, 4)]
        w, moment, bound = digital_statistics(c, q)
        self.assertEqual([qi*wi for qi, wi in zip(q, w)], c)
        self.assertEqual(moment, sum(ci*ci/qi for ci, qi in zip(c, q)))
        self.assertEqual(bound, 6)

    def test_zero_proposal_rejected(self):
        with self.assertRaises(ValueError):
            digital_statistics([F(1), F(1)], [F(0), F(1)])


if __name__ == '__main__':
    unittest.main(verbosity=2)
