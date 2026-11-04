"""Artificial 2x2 fixtures only: no saved numeric artifacts, RNG, or H4 modules."""
import ast
import itertools
import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src/trotterlib"))
import rte  # Standalone NumPy/stdlib module; avoids trotterlib.__init__.

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1., -1.]).astype(complex)


def functions(path, names):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"math": math, "np": np}
    exec(compile(module, "<unchanged-pure-functions>", "exec"), namespace)
    return namespace


def literal_order_mean(components, operators, distribution, probabilities=None):
    total = math.fsum(c.probability for c in components)
    a = sum((c.probability / total) * operators[c.component_id] for c in components)
    mean = np.zeros_like(a)
    probabilities = distribution.order_probabilities if probabilities is None else probabilities
    for n, p in zip(distribution.orders, probabilities):
        angle = math.atan(distribution.dimensionless_step_time / (n + 1))
        mean += p * (-1)**(n // 2) * (
            math.cos(angle)*I - 1j*math.sin(angle)*a) @ np.linalg.matrix_power(a, n)
    return mean


def disk_phase_bound(radius, error):
    if not math.isfinite(radius) or not math.isfinite(error) or not 0 <= error < radius:
        raise ValueError("Need finite radius>error>=0; otherwise no phase guarantee")
    return math.asin(error / radius)


class DesignIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tail = rte.normalize_involutory_tail("synthetic-xz", (
            rte.InvolutoryTailTerm("x", .7, X), rte.InvolutoryTailTerm("z", -.3, Z)))
        self.ops = {c.component_id: op for c, op in zip(self.tail.components, self.tail.operators)}

    def test_full_small_support_matches_literal_law_and_taylor(self):
        for tau in (.17, -.17):
            d = rte.finite_rte_distribution(tau, 2)
            events = rte.enumerate_rte_events(self.tail.components, d)
            enumerated = rte.exact_enumerated_event_mean_operator(events, self.ops)
            literal = literal_order_mean(self.tail.components, self.ops, d)
            reference = rte.finite_taylor_operator(self.tail.normalized_hamiltonian, tau, 2)
            np.testing.assert_allclose(enumerated, literal, atol=1e-13, rtol=0)
            np.testing.assert_allclose(literal, reference/d.exact_finite_distribution, atol=1e-13, rtol=0)

    def test_order_probability_mutation_is_visible_to_reference(self):
        d = rte.finite_rte_distribution(.4, 2)
        canonical = literal_order_mean(self.tail.components, self.ops, d)
        mutated = literal_order_mean(self.tail.components, self.ops, d, probabilities=(1., 0.))
        self.assertGreater(np.linalg.norm(mutated-canonical), 1e-4)

    def test_fresh_independent_occurrences_compose_in_actual_application_order(self):
        legacy = functions("src/trotterlib/rpe_hadamard_failure_validation.py",
                           ("_apply_event_to_state", "_trajectory_signal"))
        d = rte.finite_rte_distribution(.23, 0)
        events = rte.enumerate_rte_events(self.tail.components, d)
        f = math.cos(.31)*I - 1j*math.sin(.31)*X
        reverse = math.cos(.19)*I - 1j*math.sin(.19)*Z
        psi = np.array([1., 1j])/math.sqrt(2.)
        mean = literal_order_mean(self.tail.components, self.ops, d)
        direct = 0j
        for trajectory in itertools.product(events, repeat=4):
            weight = math.prod(e.event_probability for e in trajectory)
            direct += weight*legacy['_trajectory_signal'](initial_state=psi, forward=f,
                reverse=reverse, events=trajectory, operators=self.ops, q_m=2, rte_steps=2)
        step = reverse @ np.linalg.matrix_power(mean, 2) @ f
        expected = np.vdot(psi, np.linalg.matrix_power(step, 2) @ psi)
        self.assertAlmostEqual(abs(direct-expected), 0., places=13)

    def test_disk_geometry_handles_phase_wrap(self):
        z = .6*np.exp(3.12j)
        bound = disk_phase_bound(abs(z), .05)
        for direction in (0., .7, 2.9, -2.2):
            perturbed = z + .05*np.exp(1j*direction)
            error = abs(np.angle(perturbed*np.conj(z)))
            self.assertLessEqual(error, bound + 1e-14)

    def test_zero_radius_nonfinite_and_origin_intersection_are_unresolved(self):
        for radius, error in ((0., 0.), (.1, .1), (.1, .2), (math.nan, .1), (1., math.inf)):
            with self.assertRaises(ValueError): disk_phase_bound(radius, error)

    def test_no_discrepancy_and_equivalence_are_different_interval_events(self):
        def classify(difference, half_width, tolerance):
            if abs(difference)+half_width <= tolerance: return "WITHIN_TOLERANCE"
            if abs(difference)-half_width > tolerance: return "OUTSIDE_TOLERANCE"
            return "UNDETERMINED"
        self.assertEqual(classify(0., .2, .1), "UNDETERMINED")
        self.assertEqual(classify(.05, .01, .1), "WITHIN_TOLERANCE")
        self.assertEqual(classify(.2, .01, .1), "OUTSIDE_TOLERANCE")

    def test_branch_bound_requires_strict_margin(self):
        legacy = functions("src/trotterlib/rpe_four_round_phase_validation.py",
                           ("_wrap_phase", "_phase_distance", "reconstruct_rpe_phase"))
        b = math.pi/4
        safe = legacy['reconstruct_rpe_phase']([b, -b], [1, 2])['rounds'][-1]
        self.assertLess(safe['distance_from_previous'], safe['next_smallest_branch_distance'])
        b = math.pi/3
        tie = legacy['reconstruct_rpe_phase']([b, -b], [1, 2])['rounds'][-1]
        self.assertAlmostEqual(tie['distance_from_previous'], tie['next_smallest_branch_distance'])

    def test_marginal_binary_outcome_has_one_statistical_level(self):
        weights = (.25, .75)
        conditional_means = (-.6, .2)
        marginal_mean = sum(p*m for p, m in zip(weights, conditional_means))
        marginal_plus = sum(p*(1+m)/2 for p, m in zip(weights, conditional_means))
        self.assertAlmostEqual(marginal_plus, (1+marginal_mean)/2)


if __name__ == "__main__":
    unittest.main()
