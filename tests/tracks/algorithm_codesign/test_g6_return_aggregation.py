"""Independent off-domain formal checks. Enumeration is an oracle in this file only."""
from collections import defaultdict
from fractions import Fraction as F
from itertools import product
from math import factorial
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
from trottertracks.algorithm_codesign.return_aggregation import (
    ReturnKernel, Interval, root_interval, dyadic_distribution, dyadic_bernoulli,
    dyadic_index, reduce_word, rational_event_weight)

COUNTS = defaultdict(int)
CASES = (((F(3, 7), F(4, 7)), F(2, 5), 1),
         ((F(3, 7), F(4, 7)), F(2, 5), 3),
         ((F(3, 7), F(4, 7)), F(2, 5), 7),
         ((F(1, 5), F(3, 10), F(1, 2)), F(5, 7), 5),
         ((F(1, 5), F(3, 10), F(1, 2)), F(1), 7),
         ((F(1),), F(4, 5), 9))


def independent_reduce(raw):
    # Delete the first adjacent pair repeatedly, rather than the production stack.
    word = list(raw)
    while True:
        j = next((j for j in range(len(word) - 1) if word[j] == word[j + 1]), None)
        if j is None:
            return tuple(word)
        del word[j:j + 2]


def raw_distribution(p, m):
    tables = []
    for n in range(m + 1):
        table = defaultdict(F)
        for raw in product(range(len(p)), repeat=n):
            COUNTS['raw_formal_words'] += 1
            if COUNTS['raw_formal_words'] > 50000:
                raise RuntimeError('G6 formal enumeration cap')
            weight = F(1)
            for i in raw:
                weight *= p[i]
            table[independent_reduce(raw)] += weight
        tables.append(table)
    return tables


class G6Tests(unittest.TestCase):
    def test_general_series_insertion_positivity_parent_envelope(self):
        for p, x, m in CASES:
            kernel = ReturnKernel(p, x, m)
            tables = raw_distribution(p, m)
            words = set().union(*(set(t) for t in tables))
            for u in words:
                series = kernel.word_series(u)
                l = len(u)
                for n in range(m + 1):
                    self.assertEqual(series[n], tables[n][u])
                    COUNTS['generating_function_coefficients'] += 1
                a = sum((-1) ** ((n - l) // 2) * x ** n / factorial(n) * tables[n][u]
                        for n in range(l, m + 1, 2))
                self.assertEqual(kernel.mass(u), a)
                baseline = kernel.t(l) * kernel.raw_mass(u)
                self.assertGreaterEqual(a, baseline * (1 - kernel.chi * x * x / (l + 2)))
                self.assertLessEqual(a, baseline)
                self.assertGreater(a, 0)
                COUNTS['positive_aggregate_words'] += 1
                for n in range(l, m - 1, 2):
                    self.assertLessEqual(tables[n + 2][u], (n + 1) * kernel.chi * tables[n][u])
                    COUNTS['insertion_bounds'] += 1
                if l % 2 == 0:
                    par = kernel.parent(u)
                    self.assertEqual(par.a, a)
                    self.assertEqual(par.masses, tuple(kernel.mass((i,) + u) for i in par.children))
                    self.assertLessEqual(par.d2, kernel.envelope2(par))
                    self.assertGreaterEqual(par.d2 / kernel.envelope2(par), F(1, 8))
                    COUNTS['parent_envelopes'] += 1
            for sigma in (-1, 1):
                paired = defaultdict(lambda: [F(0), F(0)])
                target = defaultdict(lambda: [F(0), F(0)])
                def add(table, word, power, mass):
                    axis, sign = ((0, 1), (1, 1), (0, -1), (1, -1))[power % 4]
                    table[word][axis] += sign * mass
                for n, table in enumerate(tables):
                    for u, probability in table.items():
                        add(target, u, -sigma * n, x ** n / factorial(n) * probability)
                for u in words:
                    if len(u) % 2 == 0:
                        par = kernel.parent(u)
                        add(paired, u, -sigma * len(u), par.a)
                        for i, mass in zip(par.children, par.masses):
                            add(paired, (i,) + u, -sigma * (len(u) + 1), mass)
                self.assertEqual(dict(paired), dict(target))
                COUNTS['signed_formal_mean_checks'] += 1

    def test_m3_return_formula(self):
        k = ReturnKernel((F(3, 7), F(4, 7)), F(2, 5), 3)
        self.assertEqual(k.mass(()), 1 - k.chi * k.x ** 2 / 2)
        for i, p in enumerate(k.p):
            self.assertEqual(k.mass((i,)), p * (k.x - k.x ** 3 * (2 * k.chi - p * p) / 6))
        par = k.parent((0, 1))
        self.assertEqual(par.angle_tangent, k.x * (1 - k.p[0]) / 3)

    def test_zero_time_identity(self):
        k = ReturnKernel((F(3, 7), F(4, 7)), 0, 5)
        self.assertEqual(k.mass(()), 1)
        self.assertEqual(k.mass((0,)), 0)
        with self.assertRaises(ValueError):
            k.digital_parent((), 128, 64, F(1, 100), F(1, 100))

    def test_single_label_recurrent_walk(self):
        k = ReturnKernel((1,), F(4, 5), 9)
        self.assertEqual(k.first_passage[0], (F(0), F(1)) + (F(0),) * 8)
        self.assertEqual(k.returns, tuple(F(1 - n % 2) for n in range(10)))

    def test_last_label_is_not_enough_state(self):
        self.assertEqual(reduce_word((0, 1, 1, 0)), ())
        self.assertEqual(reduce_word((2, 1, 1, 0)), (2, 0))

    def test_odd_word_need_not_be_involution(self):
        self.assertEqual(reduce_word((0, 1, 2) * 2), (0, 1, 2, 0, 1, 2))

    def test_reducing_raw_proposal_would_double_count(self):
        p = (F(3, 7), F(4, 7))
        self.assertEqual(raw_distribution(p, 2)[2][()], sum(v * v for v in p))
        # The intended empty-parent proposal has only degree zero; reducing a
        # degree-two draw into it adds mass not present in its envelope.
        self.assertGreater(sum(v * v for v in p), 0)

    def test_accept_only_average_changes_mean(self):
        accepted = F(3, 5)
        conditional_mean = F(2, 7)
        self.assertNotEqual(conditional_mean, accepted * conditional_mean)

    def test_insertion_overcount_is_not_equality(self):
        p = (F(3, 7), F(4, 7))
        tables = raw_distribution(p, 4)
        self.assertLess(tables[4][()], 3 * sum(v * v for v in p) * tables[2][()])

    def test_invalid_domains(self):
        for p, x, m in (((1, 1), 1, 3), ((0, 1), 1, 3), ((1,), F(6, 5), 3),
                        ((1,), -1, 3), ((1,), 1, 2), ((), 1, 1)):
            with self.assertRaises(ValueError):
                ReturnKernel(p, x, m)

    def test_invalid_queries(self):
        k = ReturnKernel((F(3, 7), F(4, 7)), F(2, 5), 3)
        for word in ((0, 0), (2,), (0, 1, 0, 1)):
            with self.assertRaises(ValueError):
                k.mass(word)
        with self.assertRaises(ValueError):
            k.parent((0,))

    def test_root_integer_enclosure(self):
        for value in (F(0), F(1), F(7, 13), F(1, 10 ** 20), F(2)):
            z = root_interval(value, 160)
            self.assertLessEqual(z.lo ** 2, value)
            self.assertGreaterEqual(z.hi ** 2, value)
            self.assertLessEqual(z.hi - z.lo, F(1, 2 ** 160))

    def test_dyadic_law_support_and_exact_bit_map(self):
        law = dyadic_distribution((Interval(F(1, 3), F(1, 3)), Interval(F(2, 3), F(2, 3))),
                                   5, F(1, 10))
        self.assertEqual(sum(law), 1)
        counts = [0, 0]
        for b in range(32):
            counts[dyadic_index(law, 5, b)] += 1
        self.assertEqual(tuple(F(v, 32) for v in counts), law)

    def test_insufficient_precision_fails_closed(self):
        with self.assertRaises(ValueError):
            dyadic_distribution((Interval(F(1, 1000), F(1, 1000)),
                                  Interval(F(999, 1000), F(999, 1000))), 3, F(1, 100))
        with self.assertRaises(ValueError):
            dyadic_bernoulli(Interval(F(1, 100), F(1, 100)), 3, F(1, 100))

    def test_digital_packet_support_weights_and_l1_budget(self):
        p, x, m = (F(1, 5), F(3, 10), F(1, 2)), F(5, 7), 5
        k = ReturnKernel(p, x, m)
        eta, weight_eta = F(1, 1000), F(1, 1000000)
        order = dyadic_distribution(k.order_intervals(256), 160, eta)
        labels = dyadic_distribution(tuple(Interval(v, v) for v in p), 160, eta)
        b_upper = sum(root_interval(k.t(l) ** 2 + k.t(l + 1) ** 2, 256).hi
                      for l in range(0, m, 2))
        coefficient_error, proposal_sum = F(0), F(0)
        for l in range(0, m, 2):
            for word in product(range(len(p)), repeat=l):
                if independent_reduce(word) != word:
                    continue  # zero trial, never replaced by its reduced word
                packet = k.digital_parent(word, 256, 160, eta, weight_eta)
                par = packet['parent']
                z = packet['d_interval']
                coefficient_error += max(abs(packet['d_midpoint'] - z.lo),
                                         abs(packet['d_midpoint'] - z.hi))
                parent_probability = order[l // 2] * packet['acceptance']
                for i in word:
                    parent_probability *= labels[i]
                proposal_sum += parent_probability
                for j in range(len(par.children)) if par.s else (None,):
                    w = rational_event_weight(packet, j, order[l // 2], labels)
                    self.assertLessEqual(w, (1 + weight_eta) * b_upper / (1 - eta) ** (m + 2))
                    self.assertGreater(w, 0)
                    q = parent_probability * (packet['child_law'][j] if par.s else 1)
                    target = packet['d_midpoint'] * (par.masses[j] / par.s if par.s else 1)
                    self.assertEqual(q * w, target)
                    COUNTS['digital_event_weight_checks'] += 1
                self.assertEqual(packet['circuit_word_time_order'], tuple(reversed(word)))
                self.assertEqual(packet['phase_i_power'], l % 4)
        self.assertLessEqual(coefficient_error, weight_eta * b_upper)
        self.assertGreater(proposal_sum, 0)
        self.assertLessEqual(proposal_sum, 1)
        COUNTS['digital_global_l1_checks'] += 1

    def test_no_enumeration_or_science_import_in_prototype(self):
        path = Path(__file__).resolve().parents[3] / 'src/trottertracks/algorithm_codesign/return_aggregation.py'
        text = path.read_text()
        for forbidden in ('itertools', 'numpy', 'pygridsynth', 'trotterlib.', 'subprocess', 'random.'):
            self.assertNotIn(forbidden, text)


if __name__ == '__main__':
    unittest.main()
