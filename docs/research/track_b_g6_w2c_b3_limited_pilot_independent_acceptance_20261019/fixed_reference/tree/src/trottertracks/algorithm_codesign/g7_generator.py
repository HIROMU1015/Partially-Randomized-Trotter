"""G7 streaming/local generators. No enumeration, cost table, synthesis or signal.

Event coefficients refer to the finite rational approximation, not exact irrational
normalizers. Independent uniform bits give the stated proposal; deterministic
bitstream traces are interface/CPU diagnostics only.
"""
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction as F
import hashlib
from math import factorial

from .return_aggregation import (ReturnKernel, Interval, root_interval,
    dyadic_distribution, dyadic_index, reduce_word, rational_event_weight)

ARMS = ('ordinary', 'partial_return_tail', 'closed_P3_tail', 'full_return')


def raw_reduced_masses(p, m):
    p = tuple(map(F, p))
    v = p
    masses = [F(1), F(1)]
    for l in range(2, m + 1):
        total = sum(v)
        v = tuple(pi * (total - vi) for pi, vi in zip(p, v))
        masses.append(sum(v))
    return tuple(masses[:m + 1])


def sum_roots(squares, bits):
    roots = [root_interval(v, bits) for v in squares]
    return Interval(sum(z.lo for z in roots), sum(z.hi for z in roots))


def upper_normalizer(kernel, bits):
    root = root_interval(kernel.parent(()).d2, bits)
    reduced = raw_reduced_masses(kernel.p, kernel.m)
    tail = [root_interval(kernel.t(l) ** 2 + kernel.t(l + 1) ** 2, bits)
            for l in range(2, kernel.m, 2)]
    return Interval(root.lo + sum(z.lo * reduced[l] for z, l in zip(tail, range(2, kernel.m, 2))),
                    root.hi + sum(z.hi * reduced[l] for z, l in zip(tail, range(2, kernel.m, 2))))


def quartic_coefficients(p):
    chi, mu3, mu4 = (sum(F(v) ** k for v in p) for k in (2, 3, 4))
    # Independent square-root series arithmetic: sqrt(1+c2*x²+c4*x⁴).
    def root(c2, c4):
        return c2 / 2, c4 / 2 - c2 * c2 / 8
    oq, oc = root(F(1), F(0))
    rq, rc = root(1 - chi, chi * chi / 4 - chi / 3)
    nq, nc = root(1 - chi, chi * chi / 4 - (2 * chi - mu3) / 3)
    ordinary = (oq + F(1, 2), oc + F(1, 36))
    partial = (rq + (1 - chi) / 2, rc + (1 - chi) / 36)
    full = (nq + (1 - chi) / 2, nc + sum(F(v) * (1 - F(v)) ** 3 for v in p) / 36)
    return {'ordinary': ordinary, 'partial': partial, 'full': full,
            'partial_minus_full_x4': partial[1] - full[1]}


class DeterministicBits:
    """Fixed SHA256-counter bit interface; no sampling accuracy inference."""
    def __init__(self, seed='G7-development-bitstream-v1'):
        self.seed, self.counter, self.consumed_bits = seed.encode(), 0, 0

    def bits(self, n):
        value, available = 0, 0
        while available < n:
            digest = hashlib.sha256(self.seed + self.counter.to_bytes(8, 'big')).digest()
            self.counter += 1
            value = (value << 256) | int.from_bytes(digest, 'big')
            available += 256
        self.consumed_bits += n
        return value >> (available - n)


def _law(values, bits, eta):
    return dyadic_distribution(tuple(Interval(v, v) for v in values), bits, eta)


def _event(word, child, ratio, phase, alpha, proposal):
    if proposal <= 0 or alpha <= 0:
        raise ValueError('positive event support required')
    reduced = reduce_word(word)
    calls = Counter(reduced)
    calls[child] += 2  # compute and actual-adjoint uncompute
    return {'word': reduced, 'raw_word': tuple(word), 'child': child,
            'ratio': F(ratio), 'phase_i_power': phase % 4,
            'coefficient': alpha, 'proposal': proposal, 'weight': alpha / proposal,
            'provider_calls': dict(calls), 'helper': 1,
            'event_unitary': 'i^phase exp(-i*sigma*atan(ratio)*Q_child) Q(word)',
            'circuit_time_word': tuple(reversed(reduced))}


@dataclass(frozen=True)
class Group:
    mode: str
    degree: int
    square: F
    ratio: F
    first: int = -1


class CanonicalGenerator:
    def __init__(self, p, x, m, arm, root_bits=256, probability_bits=160,
                 eta=F(1, 10 ** 12), rho=F(1, 10 ** 12)):
        if arm not in ARMS[:-1] or m < 3 or m % 2 != 1:
            raise ValueError('registered canonical arm and odd degree >=3 required')
        self.p, self.x, self.m, self.arm = tuple(map(F, p)), F(x), m, arm
        # Validate mathematical input without a Green precomputation.
        if not self.p or min(self.p) <= 0 or sum(self.p) != 1 or not 0 < self.x <= 1:
            raise ValueError('invalid short-step input')
        self.K, self.H, self.eta, self.rho = root_bits, probability_bits, F(eta), F(rho)
        self.label_law = _law(self.p, self.H, self.eta)
        self.chi = sum(pi * pi for pi in self.p)
        self.groups = []
        if arm == 'ordinary':
            self._ordinary(0)
        else:
            a0 = 1 - self.chi * self.x ** 2 / 2
            if arm == 'partial_return_tail':
                b0 = self.x - self.chi * self.x ** 3 / 6
                self.groups.append(Group('root_iid', 0, a0 * a0 + b0 * b0, b0 / a0))
                if self.chi < 1:
                    self.groups.append(Group('distinct', 2,
                        (1 - self.chi) ** 2 * (self.t(2) ** 2 + self.t(3) ** 2), self.x / 3))
            else:
                mu3 = sum(pi ** 3 for pi in self.p)
                b0 = self.x - (2 * self.chi - mu3) * self.x ** 3 / 6
                self.root_child = tuple(pi * (self.x - (2 * self.chi - pi * pi) * self.x ** 3 / 6) / b0
                                        for pi in self.p)
                self.groups.append(Group('root_collected', 0, a0 * a0 + b0 * b0, b0 / a0))
                for j, pj in enumerate(self.p):
                    if pj < 1:
                        ratio = self.x * (1 - pj) / 3
                        square = self.t(2) ** 2 * pj * pj * (1 - pj) ** 2 * (1 + ratio * ratio)
                        self.groups.append(Group('fixed_first', 2, square, ratio, j))
            self._ordinary(4)
        self.roots = tuple(root_interval(g.square, self.K) for g in self.groups)
        self.B = Interval(sum(z.lo for z in self.roots), sum(z.hi for z in self.roots))
        for z in self.roots:
            if z.lo <= 0 or z.hi - z.lo > 2 * self.rho * z.lo:
                raise ValueError('fixed coefficient precision insufficient')
        self.group_law = dyadic_distribution(tuple(Interval(z.lo / self.B.hi, z.hi / self.B.lo)
                                                   for z in self.roots), self.H, self.eta)

    def t(self, l):
        return self.x ** l / factorial(l)

    def _ordinary(self, start):
        for l in range(start, self.m, 2):
            self.groups.append(Group('iid', l, self.t(l) ** 2 + self.t(l + 1) ** 2,
                                     self.x / (l + 1)))

    def conditional(self, excluded):
        ids = tuple(i for i in range(len(self.p)) if i != excluded)
        exact = tuple(self.p[i] / (1 - self.p[excluded]) for i in ids)
        return ids, exact, _law(exact, self.H, self.eta)

    def first_distribution(self):
        exact = tuple(pi * (1 - pi) / (1 - self.chi) for pi in self.p)
        return exact, _law(exact, self.H, self.eta)

    def event(self, group_index, word, child):
        g = self.groups[group_index]
        word = tuple(word)
        if len(word) != g.degree or child not in range(len(self.p)):
            raise ValueError('event inconsistent with group')
        ideal, proposal = F(1), self.group_law[group_index]
        if g.mode == 'distinct':
            if word[0] == word[1]:
                raise ValueError('distinct word required')
            exact, law = self.first_distribution()
            ideal *= exact[word[0]]; proposal *= law[word[0]]
            ids, exact, law = self.conditional(word[0])
            index = ids.index(word[1]); ideal *= exact[index]; proposal *= law[index]
        elif g.mode == 'fixed_first':
            if word[0] != g.first or word[1] == g.first or child == g.first:
                raise ValueError('closed P3 conditional label mismatch')
            ids, exact, law = self.conditional(g.first)
            index = ids.index(word[1]); ideal *= exact[index]; proposal *= law[index]
        elif g.mode == 'iid':
            for i in word:
                ideal *= self.p[i]; proposal *= self.label_law[i]
        if g.mode == 'fixed_first':
            ids, exact, law = self.conditional(g.first)
            index = ids.index(child); ideal *= exact[index]; proposal *= law[index]
        elif g.mode == 'root_collected':
            law = _law(self.root_child, self.H, self.eta)
            ideal *= self.root_child[child]; proposal *= law[child]
        else:
            ideal *= self.p[child]; proposal *= self.label_law[child]
        return _event(word, child, g.ratio, g.degree, self.roots[group_index].midpoint * ideal, proposal)

    def sample(self, bits):
        def draw(law):
            return dyadic_index(law, self.H, bits.bits(self.H))
        index = draw(self.group_law); g = self.groups[index]
        if g.mode == 'iid':
            word = tuple(draw(self.label_law) for _ in range(g.degree))
        elif g.mode == 'distinct':
            _, first_law = self.first_distribution(); first = draw(first_law)
            ids, _, law = self.conditional(first); word = (first, ids[draw(law)])
        elif g.mode == 'fixed_first':
            ids, _, law = self.conditional(g.first); word = (g.first, ids[draw(law)])
        else:
            word = ()
        if g.mode == 'fixed_first':
            ids, _, law = self.conditional(g.first); child = ids[draw(law)]
        elif g.mode == 'root_collected':
            child = draw(_law(self.root_child, self.H, self.eta))
        else:
            child = draw(self.label_law)
        return self.event(index, word, child)


class FullReturnGenerator:
    def __init__(self, p, x, m, root_bits=256, probability_bits=160,
                 eta=F(1, 10 ** 12), rho=F(1, 10 ** 12)):
        self.kernel = ReturnKernel(p, x, m)
        self.p, self.x, self.m, self.arm = self.kernel.p, self.kernel.x, m, 'full_return'
        self.K, self.H, self.eta, self.rho = root_bits, probability_bits, F(eta), F(rho)
        self.order = dyadic_distribution(self.kernel.order_intervals(self.K), self.H, self.eta)
        self.labels = _law(self.p, self.H, self.eta)
        self.B = sum_roots([self.kernel.t(l) ** 2 + self.kernel.t(l + 1) ** 2
                            for l in range(0, m, 2)], self.K)  # ordinary envelope, NOT B_new
        self.U = upper_normalizer(self.kernel, self.K)

    def packet(self, word):
        return self.kernel.digital_parent(word, self.K, self.H, self.eta, self.rho)

    def event(self, word, child):
        packet = self.packet(word)
        par = packet['parent']; index = par.children.index(child)
        weight = rational_event_weight(packet, index, self.order[len(word) // 2], self.labels)
        alpha = packet['d_midpoint'] * par.masses[index] / par.s
        return _event(word, child, par.angle_tangent, len(word), alpha, alpha / weight)

    def sample(self, bits):
        l = 2 * dyadic_index(self.order, self.H, bits.bits(self.H))
        word = tuple(dyadic_index(self.labels, self.H, bits.bits(self.H)) for _ in range(l))
        if reduce_word(word) != word:
            return None  # pre-quantum zero, never remapped to its reduced word
        packet = self.packet(word)
        if F(bits.bits(self.H), 1 << self.H) >= packet['acceptance']:
            return None
        par = packet['parent']
        child = par.children[dyadic_index(packet['child_law'], self.H, bits.bits(self.H))]
        # Avoid a second local coefficient query; no growing parent cache.
        index = par.children.index(child)
        weight = rational_event_weight(packet, index, self.order[l // 2], self.labels)
        alpha = packet['d_midpoint'] * par.masses[index] / par.s
        return _event(word, child, par.angle_tangent, l, alpha, alpha / weight)


def make_generator(p, x, m, arm, **kwargs):
    return FullReturnGenerator(p, x, m, **kwargs) if arm == 'full_return' else CanonicalGenerator(p, x, m, arm, **kwargs)


def log640_upper(terms=96):
    def log_from_z(z):
        partial = 2 * sum(z ** (2 * k + 1) / (2 * k + 1) for k in range(terms))
        tail = 2 * z ** (2 * terms + 1) / ((2 * terms + 1) * (1 - z * z))
        return partial + tail
    return 9 * log_from_z(F(1, 3)) + log_from_z(F(1, 9))


def budget(generator, epsilon_axis=F(1, 200), primitive_error=F(1, 10 ** 6)):
    eta, rho, m = generator.eta, generator.rho, generator.m
    kappa = (1 + rho) ** 2 / (1 - eta) ** (m + 2)
    B = generator.B.hi
    m2_upper = kappa * B * (generator.U.hi if generator.arm == 'full_return' else B)
    W = (1 + rho) * B / (1 - eta) ** (m + 2)
    # B_arm<=exp(x)<3, two Rz primitives, same conservative factor-2 bias for all.
    common_bias = 2 * (3 * rho + 3 * (1 + rho) * 2 * primitive_error)
    remaining = epsilon_axis - common_bias
    if remaining <= 0:
        raise ValueError('no common statistical margin')
    continuous = log640_upper() * (2 * m2_upper / remaining ** 2 + 4 * W / (3 * remaining))
    N = (continuous.numerator + continuous.denominator - 1) // continuous.denominator
    return {'N_per_axis': N, 'm2_upper': m2_upper, 'range_upper': W,
            'common_bias_upper': common_bias, 'remaining': remaining,
            'alpha_axis': F(1, 320), 'alpha_familywise_16_axes': F(1, 20),
            'known_B_envelope': generator.B, 'U': getattr(generator, 'U', None),
            'uses_exact_signal': False, 'uses_reference_B_new_or_m2': False,
            'quantum_call_hard_cap_per_axis': N}
