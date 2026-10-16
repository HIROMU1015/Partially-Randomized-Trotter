"""Unexecuted rational enclosure kernels. No libm, BLAS or scientific imports.

Coverage of the *saved* radii remains conditional. These enclosures do not
certify the pilot, a platform, a physical experiment, or this implementation.
"""
from fractions import Fraction as F


class BoundLimit(RuntimeError):
    pass


class Bounds:
    def __init__(self, bits, resources):
        self.bits = bits
        self.resources = resources
        self.tol = F(1, 1 << bits)
        self.steps = 0

    def check(self, *values):
        self.steps += 1
        if self.steps > self.resources['max_bound_steps']:
            raise BoundLimit('max_bound_steps')
        for x in values:
            if max(x.numerator.bit_length(), x.denominator.bit_length()) > self.resources['max_rational_bits']:
                raise BoundLimit('max_rational_bits')
        return values[-1] if values else None

    def atan_small(self, x):
        # Alternating series, 0 < x < 1. Next signed term brackets the sum.
        if not 0 < x < 1:
            raise ValueError('atan domain')
        total, power = F(0), x
        for n in range(self.resources['max_series_terms_per_call']):
            total += (-1 if n % 2 else 1) * power / (2*n + 1)
            power *= x*x
            next_term = (-1 if (n+1) % 2 else 1) * power / (2*n + 3)
            self.check(total, power, next_term)
            if abs(next_term) <= self.tol:
                return min(total, total + next_term), max(total, total + next_term)
        raise BoundLimit('atan series terms')

    def pi(self):
        a, b = self.atan_small(F(1, 5)), self.atan_small(F(1, 239))
        return self.check(16*a[0]-4*b[1]), self.check(16*a[1]-4*b[0])

    def sin_positive(self, x):
        # Used only with 0 < x < pi/2 < 2. Terms decrease from the start.
        if not 0 < x < 2:
            raise ValueError('sin domain')
        total, term = F(0), x
        for n in range(self.resources['max_series_terms_per_call']):
            total += term
            term *= -x*x / ((2*n+2)*(2*n+3))
            self.check(total, term)
            if abs(term) <= self.tol * abs(total):
                return min(total, total+term), max(total, total+term)
        raise BoundLimit('sin series terms')

    def _log_unit(self, x):
        # x in [1,2], z <= 1/3; exact positive-series tail enclosure.
        if not 1 <= x <= 2:
            raise ValueError('log unit domain')
        z = (x-1)/(x+1)
        total, power = F(0), z
        for n in range(self.resources['max_series_terms_per_call']):
            total += 2*power/(2*n+1)
            power *= z*z
            remainder = 2*power/((2*n+3)*(1-z*z))
            self.check(total, power, remainder)
            if remainder <= self.tol * max(total, F(1)):
                return total, total+remainder
        raise BoundLimit('log series terms')

    def log_above_one(self, x):
        if x <= 1:
            raise ValueError('log domain')
        # Exact power-of-two range reduction; loops are resource-counted.
        y, n = x, 0
        while y >= 2:
            y /= 2
            n += 1
            self.check(y)
        a, b = self._log_unit(y), self._log_unit(F(2))
        return self.check(a[0]+n*b[0]), self.check(a[1]+n*b[1])


def rational_text(value):
    return str(value.numerator) + '/' + str(value.denominator)


def interval_text(pair):
    return [rational_text(x) for x in pair]


def ceil_fraction(value):
    return -(-value.numerator // value.denominator)
