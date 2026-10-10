"""Closed P5 plus ordinary tail, without a word/event table or native costs."""
from fractions import Fraction as F
from .return_aggregation import Interval, dyadic_distribution, dyadic_index
from .g7_generator import CanonicalGenerator
from .g9_p5 import P5Closed


class ClosedP5Tail:
    def __init__(self, p, x, m=7, **kwargs):
        if m < 5 or m % 2 != 1:
            raise ValueError('odd m>=5 required')
        self.prefix = P5Closed(p, x, **kwargs)
        self.tail = CanonicalGenerator(p, x, m, 'ordinary', **kwargs)
        self.p, self.x, self.m = self.prefix.p, self.prefix.x, m
        self.arm = 'closed_P5_tail'
        self.K, self.H = self.prefix.K, self.prefix.H
        self.eta, self.rho = self.prefix.eta, self.prefix.rho
        self.tail_indices = tuple(i for i, g in enumerate(self.tail.groups) if g.degree >= 6)
        self.roots = self.prefix.roots + tuple(self.tail.roots[i] for i in self.tail_indices)
        self.B = Interval(sum(z.lo for z in self.roots), sum(z.hi for z in self.roots))
        self.group_law = dyadic_distribution(
            tuple(Interval(z.lo/self.B.hi, z.hi/self.B.lo) for z in self.roots),
            self.H, self.eta)

    def event(self, index, word, child):
        if index < len(self.prefix.groups):
            e = self.prefix.event(index, word, child)
            old_group = self.prefix.group_law[index]
        else:
            ti = self.tail_indices[index - len(self.prefix.groups)]
            e = self.tail.event(ti, word, child)
            old_group = self.tail.group_law[ti]
        e['proposal'] *= self.group_law[index] / old_group
        e['weight'] = e['coefficient'] / e['proposal']
        return e

    def sample(self, bits):
        draw = lambda law: dyadic_index(law, self.H, bits.bits(self.H))
        index = draw(self.group_law)
        if index < len(self.prefix.groups):
            g = self.prefix
            mode, j, k, _ = g.groups[index]
            if mode == 'root':
                word = ()
            elif mode == 'two':
                word = (j, k)
            else:
                word = (j,)
                for remaining in (2, 1, 0):
                    prev = word[-1]
                    ids = tuple(i for i in range(len(self.p)) if i != prev)
                    values = tuple(self.p[i]*g.completion(i, remaining)
                                   / g.completion(prev, remaining+1) for i in ids)
                    word += (ids[draw(g.law(values))],)
            _, s, b = g.parent_coefficients(word)
            ids = tuple(i for i, v in enumerate(b) if v)
            child = ids[draw(g.law(tuple(b[i]/s for i in ids)))]
        else:
            ti = self.tail_indices[index - len(self.prefix.groups)]
            word = tuple(draw(self.tail.label_law) for _ in range(self.tail.groups[ti].degree))
            child = draw(self.tail.label_law)
        return self.event(index, word, child)


def arm_names(m):
    if m not in (3, 5, 7):
        raise ValueError('registered degrees are 3/5/7')
    arms = ['ordinary', 'partial_return_tail', 'closed_P3_tail', 'full_return']
    if m >= 5:
        arms.append('closed_P5_full' if m == 5 else 'closed_P5_tail')
    return arms + ['matched_CTS']


def generators(p, x, m):
    """Production construction only. No CTS collection/reference table input."""
    from .g7_generator import make_generator
    gs = [make_generator(p, x, m, a) for a in arm_names(m)[:4]]
    if m == 5:
        gs.append(P5Closed(p, x))
    elif m == 7:
        gs.append(ClosedP5Tail(p, x, m))
    return gs


def static_support_bounds(L=3):
    """Combinatorial overestimates, independent of registered coefficient values."""
    rows = []
    for m in (3, 5, 7):
        ordinary = L*sum(L**l for l in range(0, m, 2))
        parents = 1 + sum(L*(L-1)**(l-1) for l in range(2, m, 2))
        full = L + (parents-1)*(L-1)
        prefix5 = L + L*(L-1)**2 + L*(L-1)**4
        # Three-qubit Pauli table has at most 2*4^3 coefficient slots.
        bounds = {'ordinary': ordinary, 'partial_return_tail': ordinary,
                  'closed_P3_tail': ordinary, 'full_return': full, 'matched_CTS': 128}
        if m >= 5:
            bounds['closed_P5_full' if m == 5 else 'closed_P5_tail'] = (
                prefix5 + L*sum(L**l for l in range(6, m, 2)))
        rows.extend({'m': m, 'arm': a, 'event_upper': bounds[a]} for a in arm_names(m))
    return rows
