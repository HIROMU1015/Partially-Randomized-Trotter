"""Pure artificial count/probability lemmas. No H4 operands, circuits or RNG.

Run with python3 <this file>. The existing DF integer counter is AST-extracted
without importing its Qiskit module. Its validation helper is a valid-input
identity stub; this test checks counting, not argument validation.
"""
import ast
from fractions import Fraction as Q
from itertools import product
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[3]


class ArtificialDesignChecks(unittest.TestCase):
    def test_law_weighted_additive_upper(self):
        # Synthetic two-component law, orders 0/2/4, one event. No circuits.
        probabilities = (Q(1, 3), Q(2, 3))
        orders = {0: Q(1, 2), 2: Q(1, 3), 4: Q(1, 6)}
        rotation, signed_product = (4, 10), (3, 7)
        expectation = Q(0)
        for k, pk in orders.items():
            for js in product(range(2), repeat=k + 1):
                p = pk
                for j in js:
                    p *= probabilities[j]
                upper = rotation[js[-1]] + sum(signed_product[j] for j in js[:-1])
                self.assertLessEqual(upper, max(rotation) + 4 * max(signed_product))
                expectation += p * upper
        mean_k = sum(k * p for k, p in orders.items())
        mean_v = sum(p * v for p, v in zip(probabilities, rotation))
        mean_w = sum(p * w for p, w in zip(probabilities, signed_product))
        self.assertEqual(expectation, mean_v + mean_k * mean_w)

    def test_existing_df_counter_on_artificial_descriptors(self):
        source = ROOT / 'src/trotterlib/df_rte_qiskit.py'
        function = next(n for n in ast.parse(source.read_text()).body
                        if isinstance(n, ast.FunctionDef)
                        and n.name == 'estimate_df_rte_structural_size_upper_bound')
        module = ast.parse('from __future__ import annotations\n')
        module.body.append(function)
        namespace = {'require_integer_count': lambda value, **kwargs: value}
        exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), namespace)
        counter = namespace[function.name]
        specs = [SimpleNamespace(basis_change_operations=(1, 2), diagonal_pauli_support=(0,)),
                 SimpleNamespace(basis_change_operations=(1,), diagonal_pauli_support=(0, 1)),
                 SimpleNamespace(basis_change_operations=(), diagonal_pauli_support=())]
        bound = counter(specs, maximum_taylor_order=4, event_count=2, controlled=True)
        # Enumerate only toy descriptor choices; no Qiskit build or compile.
        for js in product(range(3), repeat=5):
            size = 0
            for index, j in enumerate(js):
                spec = specs[j]
                if spec.diagonal_pauli_support:
                    size += 2 * len(spec.basis_change_operations)
                    size += len(spec.diagonal_pauli_support) if index < 4 else 1
            self.assertLessEqual(2 * size + 1, bound)

    def test_cdf_lower_including_unseen_cheap_and_expensive_costs(self):
        # A supplied deterministic CDF band, not a test of probabilistic coverage.
        empirical = {4: Q(1)}
        truths = ({0: Q(1, 4), 4: Q(3, 4)},
                  {0: Q(1, 4), 4: Q(13, 20), 100: Q(1, 10)})
        epsilon = Q(1, 4)
        for truth in truths:
            mean = sum(x * p for x, p in truth.items())
            for threshold in (0, 1, 4, 5, 100, 101):
                f = sum(p for x, p in truth.items() if x < threshold)
                fhat = sum(p for x, p in empirical.items() if x < threshold)
                self.assertLessEqual(abs(fhat - f), epsilon)
                lower = threshold * max(Q(0), 1 - fhat - epsilon)
                self.assertLessEqual(lower, mean)
            self.assertLessEqual(min(empirical) * (1 - epsilon), mean)

    def test_weighted_bounds_do_not_require_axis_independence(self):
        # Simultaneous component bounds suffice even for fully coupled prices.
        upper_p, lower_d = (5, 6), (10, 12)
        weights_p, weights_d = (3, 2), (1, 4)
        worst_p = sum(n * u for n, u in zip(weights_p, upper_p))
        best_d = sum(n * l for n, l in zip(weights_d, lower_d))
        for cp in ((0, 0), upper_p, (2, 6)):
            for cd in (lower_d, (100, 100), (10, 40)):
                self.assertLessEqual(sum(n*c for n, c in zip(weights_p, cp)), worst_p)
                self.assertGreaterEqual(sum(n*c for n, c in zip(weights_d, cd)), best_d)


if __name__ == '__main__':
    unittest.main(verbosity=2)
