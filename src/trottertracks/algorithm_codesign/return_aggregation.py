"""G6 formal, rational local queries. No quantum matrices, costs, RNG or synthesis.

Only Q_i**2=I is used. Words are operator-left-to-right, not circuit time.
This module never enumerates the reduced-word universe or computes its normalizer.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from math import factorial, isqrt


def reduce_word(word):
    stack = []
    for i in word:
        if stack and stack[-1] == i:
            stack.pop()
        else:
            stack.append(i)
    return tuple(stack)


def convolution(a, b, m):
    out = [F(0)] * (m + 1)
    for i, ai in enumerate(a):
        if ai:
            for j in range(min(len(b) - 1, m - i) + 1):
                if b[j]:
                    out[i + j] += ai * b[j]
    return tuple(out)


@dataclass(frozen=True)
class Interval:
    lo: F
    hi: F

    def __post_init__(self):
        if self.lo > self.hi:
            raise ValueError('reversed interval')

    @property
    def midpoint(self):
        return (self.lo + self.hi) / 2


def root_interval(value, bits):
    value = F(value)
    if value < 0 or not isinstance(bits, int) or bits < 1:
        raise ValueError('nonnegative radicand and positive bits required')
    scale = 1 << bits
    k = isqrt((value.numerator * scale * scale) // value.denominator)
    lo = F(k, scale)
    return Interval(lo, lo if lo * lo == value else F(k + 1, scale))


def dyadic_distribution(intervals, bits, eta):
    """Fixed-bit largest remainders; certify relative error, else fail closed.

    Caller supplies enclosures of a normalized positive distribution. No random
    draws or adaptive precision are performed. The extra interval checks are
    sufficient conditions; broad input intervals can legitimately be rejected.
    """
    eta = F(eta)
    if not intervals or not 0 < eta < 1 or bits < 1:
        raise ValueError('invalid dyadic request')
    if any(z.lo <= 0 for z in intervals):
        raise ValueError('positive support required')
    if not sum(z.lo for z in intervals) <= 1 <= sum(z.hi for z in intervals):
        raise ValueError('intervals do not enclose a normalized law')
    mids = [z.midpoint for z in intervals]
    total = sum(mids)
    scale = 1 << bits
    exact = [v * scale / total for v in mids]
    counts = [v.numerator // v.denominator for v in exact]
    deficit = scale - sum(counts)
    order = sorted(range(len(counts)), key=lambda i: (-(exact[i] - counts[i]), i))
    for i in order[:deficit]:
        counts[i] += 1
    law = tuple(F(c, scale) for c in counts)
    if any(q < (1 - eta) * z.hi or q > (1 + eta) * z.lo
           for q, z in zip(law, intervals)):
        raise ValueError('fixed bits/interval precision insufficient')
    return law


def dyadic_bernoulli(interval, bits, eta):
    """A downward dyadic threshold; never draws an acceptance coin."""
    eta = F(eta)
    if not 0 < interval.lo <= interval.hi <= 1 or not 0 < eta < 1 or bits < 1:
        raise ValueError('invalid acceptance interval')
    scale = 1 << bits
    q = F((interval.lo * scale).numerator // (interval.lo * scale).denominator, scale)
    if q < (1 - eta) * interval.hi:
        raise ValueError('fixed precision insufficient for acceptance')
    return q


def dyadic_index(law, bits, bit_integer):
    """Deterministic mapping of exactly H bits to an index; no sampler execution."""
    scale = 1 << bits
    if not isinstance(bit_integer, int) or not 0 <= bit_integer < scale or sum(law) != 1:
        raise ValueError('invalid bit string or law')
    counts = [q * scale for q in law]
    if any(c.denominator != 1 or c < 0 for c in counts):
        raise ValueError('law not dyadic at requested bits')
    end = 0
    for i, c in enumerate(counts):
        end += int(c)
        if bit_integer < end:
            return i
    raise AssertionError('normalized law must cover bit string')


@dataclass(frozen=True)
class Parent:
    word: tuple
    a: F
    children: tuple
    masses: tuple

    @property
    def s(self):
        return sum(self.masses, F(0))

    @property
    def d2(self):
        return self.a * self.a + self.s * self.s

    @property
    def angle_tangent(self):
        return self.s / self.a  # symbolic ratio only, no new native angle

    def child_law(self):
        return tuple(a / self.s for a in self.masses) if self.s else ()


class ReturnKernel:
    """Truncated first-passage/return series; O(L*m^2) rational preprocessing."""
    def __init__(self, probabilities, x, m):
        self.p = tuple(F(v) for v in probabilities)
        self.x = F(x)
        self.m = m
        if not self.p or any(p <= 0 for p in self.p) or sum(self.p) != 1:
            raise ValueError('positive normalized rational p required; remove zero labels first')
        if not isinstance(m, int) or m < 1 or m % 2 != 1 or not 0 <= self.x <= 1:
            raise ValueError('positive odd m and 0<=x<=1 required')
        self.chi = sum(p * p for p in self.p)
        fs = [[F(0)] * (m + 1) for _ in self.p]
        total = [F(0)] * (m + 1)
        for n in range(1, m + 1):
            for i, p in enumerate(self.p):
                fs[i][n] = (p if n == 1 else F(0)) + sum(
                    fs[i][b] * (total[n - 1 - b] - p * fs[i][n - 1 - b])
                    for b in range(1, n - 1))
            total[n] = sum(p * fs[i][n] for i, p in enumerate(self.p))
        gs = [F(0)] * (m + 1)
        gs[0] = F(1)
        for n in range(1, m + 1):
            gs[n] = sum(total[a] * gs[n - 1 - a] for a in range(1, n))
        self.first_passage = tuple(tuple(f) for f in fs)
        self.returns = tuple(gs)

    def _word(self, word):
        word = tuple(word)
        if any(not isinstance(i, int) or not 0 <= i < len(self.p) for i in word):
            raise ValueError('invalid label')
        if len(word) > self.m or reduce_word(word) != word:
            raise ValueError('query must be a reduced word of length <=m')
        return word

    def word_series(self, word):
        word = self._word(word)
        series = self.returns
        for i in word:
            series = convolution(series, self.first_passage[i], self.m)
        return series

    def _mass(self, series, length):
        return sum((-1) ** ((n - length) // 2) * self.x ** n / factorial(n) * series[n]
                   for n in range(length, self.m + 1, 2))

    def mass(self, word):
        word = self._word(word)
        return self._mass(self.word_series(word), len(word))

    def raw_mass(self, word):
        value = F(1)
        for i in self._word(word):
            value *= self.p[i]
        return value

    def t(self, degree):
        return self.x ** degree / factorial(degree)

    def parent(self, word):
        word = self._word(word)
        if len(word) % 2 or len(word) > self.m - 1:
            raise ValueError('parent must have even length <=m-1')
        series = self.word_series(word)
        children = tuple(i for i in range(len(self.p)) if not word or i != word[0])
        masses = tuple(self._mass(convolution(series, self.first_passage[i], self.m),
                                  len(word) + 1) for i in children)
        return Parent(word, self._mass(series, len(word)), children, masses)

    def envelope2(self, parent):
        l = len(parent.word)
        return (self.t(l) ** 2 + self.t(l + 1) ** 2) * self.raw_mass(parent.word) ** 2

    def order_intervals(self, root_bits):
        roots = tuple(root_interval(self.t(l) ** 2 + self.t(l + 1) ** 2, root_bits)
                      for l in range(0, self.m, 2))
        total = Interval(sum(z.lo for z in roots), sum(z.hi for z in roots))
        return tuple(Interval(z.lo / total.hi, z.hi / total.lo) for z in roots)

    def digital_parent(self, word, root_bits, probability_bits, eta, weight_eta):
        """One local packet, conditional on a reduced parent being proposed.

        Zero raw proposals must be discarded before calling this method. An
        algebraic ideal weight and its finite rational approximation are distinct.
        A circuit/controlled rotation is NOT produced or validated here.
        """
        if self.x == 0:
            raise ValueError('x=0 is the exact identity fast path, not this proposal')
        par = self.parent(word)
        root = root_interval(par.d2, root_bits)
        midpoint = root.midpoint
        if root.lo <= 0 or max(midpoint - root.lo, root.hi - midpoint) > F(weight_eta) * root.lo:
            raise ValueError('fixed weight precision insufficient')
        ai = root_interval(par.d2 / self.envelope2(par), root_bits)
        accept = dyadic_bernoulli(Interval(ai.lo, min(F(1), ai.hi)), probability_bits, eta)
        law = dyadic_distribution(tuple(Interval(q, q) for q in par.child_law()),
                                  probability_bits, eta) if par.s else ()
        return {'parent': par, 'd_interval': root, 'd_midpoint': midpoint,
                'acceptance': accept, 'child_law': law,
                'phase_i_power': len(par.word) % 4,
                'angle_tangent': par.angle_tangent,
                'circuit_word_time_order': tuple(reversed(par.word))}


def rational_event_weight(packet, child_index, dyadic_order_mass, dyadic_label_masses):
    """Finite rational coefficient divided by this event's dyadic proposal mass."""
    par = packet['parent']
    proposal = F(dyadic_order_mass) * packet['acceptance']
    for i in par.word:
        proposal *= dyadic_label_masses[i]
    if par.s:
        target = packet['d_midpoint'] * par.masses[child_index] / par.s
        proposal *= packet['child_law'][child_index]
    else:
        if child_index is not None:
            raise ValueError('no child for pure parent')
        target = packet['d_midpoint']
    if proposal <= 0:
        raise ValueError('support lost')
    return target / proposal
