"""Artificial rational identities and CDFs only; no repository numeric loaders."""
from fractions import Fraction as Q
import itertools
import math
import unittest

import numpy as np


I = ((Q(1), Q(0)), (Q(0), Q(1)))
ZERO = ((Q(0), Q(0)), (Q(0), Q(0)))


def mul(a, b):
    return tuple(tuple(sum(a[i][k]*b[k][j] for k in range(2)) for j in range(2)) for i in range(2))


def add(a, b):
    return tuple(tuple(a[i][j]+b[i][j] for j in range(2)) for i in range(2))


def scale(a, value):
    return tuple(tuple(x*value for x in row) for row in a)


def power(a, n):
    result = I
    for _ in range(n):
        result = mul(result, a)
    return result


def norm1(a):
    return max(sum(abs(a[i][j]) for i in range(2)) for j in range(2))


def subtract(a, b):
    return add(a, scale(b, -1))


def grid_masses(cdf, grid):
    cumulative = [Q(0)]
    for c in cdf:
        value = Q(float(c))*grid
        cumulative.append(Q(-(-value.numerator//value.denominator), grid))
    return [b-a for a, b in zip(cumulative, cumulative[1:])]


class StaticIdentities(unittest.TestCase):
    def test_finite_pairing_all_degrees_and_signed_time(self):
        # w*cos and w*sin identities supply the coefficients, without sqrt rounding.
        for k in [0, 2, 4]:
            for tau in [Q(3, 4), Q(-3, 4), Q(0)]:
                paired = {}
                for n in range(0, k+1, 2):
                    phase = (-1)**(n//2)
                    paired[n] = (phase*tau**n/math.factorial(n), Q(0))
                    paired[n+1] = (Q(0), -phase*tau**(n+1)/math.factorial(n+1))
                direct = {}
                cycle = [(1, 0), (0, -1), (-1, 0), (0, 1)]
                for degree in range(k+2):
                    weight = tau**degree/math.factorial(degree)
                    direct[degree] = tuple(weight*x for x in cycle[degree % 4])
                self.assertEqual(paired, direct)

    def test_rotation_sign_with_rational_pythagorean_point(self):
        tau, cosine, sine, b = Q(3, 4), Q(4, 5), Q(3, 5), Q(5, 4)
        self.assertEqual(cosine*cosine+sine*sine, 1)
        self.assertEqual(b*cosine, 1)
        self.assertEqual(-b*sine, -tau)  # exp(-i theta P), not exp(+i theta P).

    def test_independent_noncommuting_signed_component_products(self):
        x = ((Q(0), Q(1)), (Q(1), Q(0)))
        minus_z = ((Q(-1), Q(0)), (Q(0), Q(1)))
        components, probabilities = [x, minus_z], [Q(1, 3), Q(2, 3)]
        self.assertNotEqual(mul(x, minus_z), mul(minus_z, x))
        a = add(scale(x, probabilities[0]), scale(minus_z, probabilities[1]))
        for n in [1, 2, 3]:
            average = ZERO
            for indices in itertools.product(range(2), repeat=n):
                product, probability = I, Q(1)
                for index in indices:
                    product = mul(components[index], product)
                    probability *= probabilities[index]
                average = add(average, scale(product, probability))
            self.assertEqual(average, power(a, n))

    def test_inner_outer_normalizer_once(self):
        polynomial, b, r, q = Q(3, 4), Q(5, 4), 2, 3
        tail = polynomial**r/b**r
        self.assertEqual(tail**q, polynomial**(r*q)/b**(r*q))
        self.assertNotEqual(tail, polynomial**r/b)
        self.assertNotEqual(tail, polynomial**r/b**(2*r))

    def test_noncommuting_power_telescoping(self):
        a = ((Q(1), Q(1, 4)), (Q(0), Q(1, 2)))
        b = ((Q(1, 2), Q(0)), (Q(1, 3), Q(1)))
        self.assertNotEqual(mul(a, b), mul(b, a))
        q = 5
        total = ZERO
        eta, ka, kb = norm1(subtract(a, b)), norm1(a), norm1(b)
        bound = Q(0)
        for j in range(q):
            total = add(total, mul(mul(power(a, q-1-j), subtract(a, b)), power(b, j)))
            bound += eta*ka**(q-1-j)*kb**j
        self.assertEqual(total, subtract(power(a, q), power(b, q)))
        self.assertLessEqual(norm1(total), bound)

    def test_contraction_bound_and_invalid_unconditional_q_eta(self):
        a = ((Q(0), Q(1)), (Q(1), Q(0)))
        eta, q = Q(1, 100), 5
        b = add(a, ((eta, Q(0)), (Q(0), Q(0))))
        self.assertEqual(norm1(a), 1)
        self.assertLessEqual(norm1(subtract(power(a, q), power(b, q))), (1+eta)**q-1)
        a, b = scale(I, 2), scale(I, 2+eta)
        self.assertGreater(norm1(subtract(power(a, 2), power(b, 2))), 2*eta)

    def test_half_tail_product_error_order(self):
        r = ((Q(1), Q(1, 3)), (Q(0), Q(1)))
        t = ((Q(1, 2), Q(0)), (Q(1, 4), Q(1)))
        f = ((Q(0), Q(1)), (Q(1), Q(0)))
        r0, t0, f0 = I, I, I
        actual = subtract(mul(mul(r, t), f), mul(mul(r0, t0), f0))
        expanded = add(add(mul(mul(subtract(r, r0), t), f), mul(mul(r0, subtract(t, t0)), f)), mul(mul(r0, t0), subtract(f, f0)))
        self.assertEqual(actual, expanded)

    def test_grid_bucket_counts_include_right_boundary(self):
        for cdf in [[.25, .5, 1.], [0., .5, .5, 1.]]:
            exact = grid_masses(cdf, 16)
            counts = [0]*len(cdf)
            for k in range(16):
                index = next(j for j, c in enumerate(cdf) if Q(k, 16) < Q(c))
                counts[index] += 1
            self.assertEqual(exact, [Q(n, 16) for n in counts])
            self.assertEqual(sum(exact), 1)

    def test_numpy_cdf_normalization_is_distinct_from_raw_weights(self):
        p = np.array([.1, .2, .7])  # Artificial probabilities only; no RNG.
        cdf = p.cumsum()
        cdf /= cdf[-1]
        masses = grid_masses(cdf, 16)
        self.assertEqual(sum(masses), 1)
        self.assertNotEqual(masses, [Q(float(x)) for x in p])

    def test_total_variation_bounded_by_cdf_differences(self):
        original = [Q(1, 3), Q(1, 6), Q(1, 2)]
        alternative = grid_masses([float(Q(1, 3)), .5, 1.], 16)
        tv = sum(abs(a-b) for a, b in zip(original, alternative))/2
        cdf_bound = sum(abs(sum(original[:j])-sum(alternative[:j])) for j in [1, 2])
        self.assertLessEqual(tv, cdf_bound)
        self.assertLessEqual(tv, Q(2, 16))

    def test_budget_is_strict_and_requires_unknown_bridges(self):
        def sufficient(lo, hi, saved, bridge, shot, epsilon):
            if bridge is None:
                return None
            return max(abs(lo-saved), abs(hi-saved))+bridge+shot < epsilon
        self.assertTrue(sufficient(Q(0), Q(1, 10), Q(0), Q(0), Q(1, 10), Q(1, 3)))
        self.assertFalse(sufficient(Q(0), Q(1, 10), Q(0), Q(0), Q(1, 10), Q(1, 5)))
        self.assertIsNone(sufficient(Q(0), Q(0), Q(0), None, Q(1, 10), Q(1, 3)))


if __name__ == '__main__':
    unittest.main()
