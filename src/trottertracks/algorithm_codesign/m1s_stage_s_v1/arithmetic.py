"""Exact rational/outward dyadic arithmetic copied selectively from frozen V1.

Origin: 677dd6957e0e1913b70ca5f999458835ae8a2e23, bounds.py.
Only generic kernels are reused, never the old scientific evaluator/guards.
This does not certify saved engineering radii or a platform.
"""
from fractions import Fraction as F


class BoundLimit(RuntimeError):
    pass


class Bounds:
    def __init__(self, bits, resources):
        self.bits = bits
        self.resources = resources
        self.scale = 1 << (bits + 8)
        self.tol = F(1, self.scale)
        self.steps = 0


    def check(self, *values):
        self.steps += 1
        if self.steps > self.resources['max_bound_steps']:
            raise BoundLimit('max_bound_steps')
        for x in values:
            if max(x.numerator.bit_length(), x.denominator.bit_length()) > self.resources['max_rational_bits']:
                raise BoundLimit('max_rational_bits')
        return values[-1] if values else None


    def pack(self, pair):
        """Outward dyadic rounding controls denominator growth, never narrows."""
        lo, hi = pair
        self.check(lo, hi)
        if lo > hi:
            raise ValueError('reversed interval')
        return (F((lo*self.scale).__floor__(), self.scale),
                F((hi*self.scale).__ceil__(), self.scale))


    def sin_positive(self, x):
        # Used only with 0 < x < pi/2 < 2. Terms decrease from the start.
        if x == 0:
            return F(0), F(0)
        if not 0 < x < 2:
            raise ValueError('sin domain')
        total, term = F(0), x
        for n in range(self.resources['max_series_terms_per_call']):
            total += term
            term *= -x*x / ((2*n+2)*(2*n+3))
            self.check(total, term)
            if abs(term) <= self.tol * abs(total):
                return self.pack((min(total, total+term), max(total, total+term)))
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
                return self.pack((total, total+remainder))
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
        return self.pack((a[0]+n*b[0], a[1]+n*b[1]))


def rational_text(value):
    return str(value.numerator) + '/' + str(value.denominator)


def interval_text(pair):
    return [rational_text(x) for x in pair]
