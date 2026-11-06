"""Deterministic invented finite laws only; no science files, RNG or compiler.

Run: python3 -B docs/research/o2_a1_n3_decision_expected_rz_20261106/test_artificial_guarantees.py
Reuses the existing pure CV accountant and bounded-price radius. No new runner.
"""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import importlib.util
import math
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from trottertracks.algorithm_codesign import a1_control_variates as cv

# Avoid trotterlib.__init__ and its quantum dependencies.
spec = importlib.util.spec_from_file_location(
    "bounded_price_helper", ROOT / "src/trotterlib/h8_rare_tail_selection.py")
helper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helper
spec.loader.exec_module(helper)


def mean(rows):
    return sum((p * x for p, x in rows), F(0))


def variance(rows):
    mu = mean(rows)
    return sum((p * (x - mu)**2 for p, x in rows), F(0))


class ArtificialGuarantees(unittest.TestCase):
    def test_total_variance_and_saturated_strata_cancellation(self):
        masses = (F(1, 4), F(1, 4), F(1, 4), F(1, 8), F(1, 8))
        patterns = ((1, 0, 0), (0, 0, 0), (0, 0, 0), (0, 1, 0), (0, 1, 1))
        plus_probs = (F(1, 4), F(1, 2), F(3, 4), F(1), F(0))
        slopes = (F(1, 3), F(-1, 2), F(2, 5))
        nu = tuple(sum(p * g[j] for p, g in zip(masses, patterns)) for j in range(3))
        all_values, conditional_means = [], []
        within = F(0)
        raw_total, corrected_total = F(0), F(0)
        for p, g, prob in zip(masses, patterns, plus_probs):
            shift = sum(c * (v - n) for c, v, n in zip(slopes, g, nu))
            raw = ((prob, F(1)), (1 - prob, F(-1)))
            all_values.extend((p * q, x - shift) for q, x in raw)
            conditional_means.append((p, mean(raw) - shift))
            within += p * variance(raw)
            raw_total += p * mean(raw)
            corrected_total += p * (mean(raw) - shift)
        self.assertEqual(variance(all_values), within + variance(conditional_means))
        self.assertEqual(raw_total, corrected_total)

    def test_adaptive_N_uses_future_price_not_correlated_pilot_product(self):
        histories = ((F(1, 2), 1, 1), (F(1, 2), 3, 9))
        future_prices = ((F(1, 2), 1), (F(1, 2), 9))
        true_future_cost = sum(p * q * n * c for p, n, _ in histories for q, c in future_prices)
        wrong_same_history = sum(p * n * pilot for p, n, pilot in histories)
        product_of_means = sum(p * n for p, n, _ in histories) * mean(future_prices)
        self.assertEqual(true_future_cost, product_of_means)
        self.assertNotEqual(true_future_cost, wrong_same_history)

    def test_all_stages_and_cache_repeated_path_are_charged(self):
        records = []
        for axis in ("X", "Y"):
            scope = cv.Scope("toy", "invented", "toy-full", "m0", axis, F(1))
            # Two train records represent train/pilot bookkeeping, never two
            # charges for the same trial. Repeated path price remains chargeable.
            for i, (stage, price) in enumerate((("train", 7), ("train", 7), ("cal", 11), ("prod", 13))):
                records.append(cv.TrialCost(scope, f"toy-{axis}-{stage}", str(i), stage,
                    cv.FULL_SCOPE, (None, F(0), F(price), None, None), F(0), F(0), 0))
        result = cv.account_costs(records)
        self.assertEqual(result["quantum_total"][2], 2 * (7 + 7 + 11 + 13))
        self.assertEqual(result["quantum_by_stage"]["train"][2], 2 * (7 + 7))
        self.assertIsNone(result["quantum_total"][0])
        with self.assertRaises(ValueError):
            cv.account_costs(records + records[:1])

    def test_observed_max_is_rejected_as_support_proof(self):
        with self.assertRaisesRegex(ValueError, "ALL_SUPPORT_BOUND_REQUIRED"):
            helper.bounded_simultaneous_radius(bound=10, n=8, family_size=1,
                alpha=.2, support_proof="OBSERVED_MAXIMUM")

    def test_hoeffding_coverage_by_exhaustive_toy_samples(self):
        # Exhaust all 256 samples from a toy two-price law, without RNG.
        n, alpha, bound = 8, .2, 10
        radius = helper.bounded_simultaneous_radius(bound=bound, n=n,
            family_size=1, alpha=alpha, support_proof="PROVEN_ALL_SUPPORT_BOUND")
        failures = sum(abs(sum(xs) / n - 5) > radius for xs in product((0, 10), repeat=n))
        self.assertLessEqual(F(failures, 2**n), F(1, 5))
        self.assertTrue(math.isfinite(radius))

    def test_shared_price_interval_signed_difference_cancels_common_work(self):
        lo, hi = (F(2), F(4)), (F(6), F(9))
        w_a, w_b = (F(7), F(3)), (F(7), F(5))
        d = tuple(a - b for a, b in zip(w_a, w_b))
        lower = sum(x * (l if x >= 0 else u) for x, l, u in zip(d, lo, hi))
        upper = sum(x * (u if x >= 0 else l) for x, l, u in zip(d, lo, hi))
        separate_lower = sum(x * l for x, l in zip(w_a, lo)) - sum(x * u for x, u in zip(w_b, hi))
        separate_upper = sum(x * u for x, u in zip(w_a, hi)) - sum(x * l for x, l in zip(w_b, lo))
        for mu in product(*zip(lo, hi)):
            actual = sum(x * c for x, c in zip(d, mu))
            self.assertLessEqual(lower, actual)
            self.assertLessEqual(actual, upper)
        self.assertLess(upper - lower, separate_upper - separate_lower)

    def test_outer_price_envelope_keeps_failed_histories_and_pre_cost(self):
        # Third history is a stopped attempt: sunk cost remains, quota is zero.
        histories = ((F(1, 4), F(6), (1, 2)), (F(1, 4), F(9), (3, 1)),
                     (F(1, 2), F(4), (0, 0)))
        lo, mu, hi = (F(2), F(4)), (F(3), F(5)), (F(4), F(6))
        expectation = lambda prices: sum(p * (a + sum(n * c for n, c in zip(ns, prices)))
                                         for p, a, ns in histories)
        self.assertLessEqual(expectation(lo), expectation(mu))
        self.assertLessEqual(expectation(mu), expectation(hi))
        survivor = sum(p * (a + sum(n * c for n, c in zip(ns, mu)))
                       for p, a, ns in histories[:2]) / F(1, 2)
        self.assertNotEqual(survivor, expectation(mu))

    def test_stopped_production_bound_and_cheap_abort_counterexample(self):
        # Ideal completion and actual prefix are coupled on the same outcomes.
        rows = ((F(3, 4), 20, 20, False), (F(1, 4), 30, 7, True))
        planned = sum(p * c for p, c, _, _ in rows)
        actual = sum(p * c for p, _, c, _ in rows)
        abort = sum(p for p, _, _, failed in rows if failed)
        max_production_saving = 24
        self.assertLessEqual(planned - max_production_saving * abort, actual)
        self.assertLessEqual(actual, planned)
        # A zero-work abort has lower cost than a successful service without
        # providing that service. Price comparison alone cannot certify it.
        self.assertLess(0, actual)
        self.assertLess(0, 1 - abort)

    def test_no_quantum_or_scientific_dependencies_loaded(self):
        forbidden = ("qiskit", "numpy", "scipy", "mpmath", "trotterlib")
        self.assertFalse([name for name in sys.modules
                          if any(name == x or name.startswith(x + ".") for x in forbidden)])

    def test_engine_bit_coupling_controls_discontinuous_cost_policy(self):
        def expected_cost(prob):
            return sum((prob**sum(bits)) * ((1 - prob)**(2 - sum(bits)))
                       * (100 if any(bits) else 0) for bits in product((0, 1), repeat=2))
        engine, ideal = F(1, 4), F(1, 2)
        difference = abs(expected_cost(engine) - expected_cost(ideal))
        raw_mean_bound = 2 * abs(engine - ideal)
        history_mismatch_bound = min(F(1), 2 * raw_mean_bound / 2)
        self.assertLessEqual(difference, 100 * history_mismatch_bound)
        self.assertGreater(difference, raw_mean_bound)


if __name__ == "__main__":
    unittest.main(verbosity=2)
