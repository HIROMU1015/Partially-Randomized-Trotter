"""Stdlib artificial regression tests; no molecular inputs or science imports.

Execute the actual accounting definitions, with only preparation/cost objects
mocked. Reuse the unchanged assertions from test_rpe_certification_boundary.py.
The dependency-heavy pytest suite is not required or claimed to have run.
"""
from __future__ import annotations

import ast
from dataclasses import asdict
import math
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
BEFORE = "3fd8fca966e6b5a6a972aa71de68b31a0129f5ab"
SOURCE = "src/trotterlib/rpe_resource_accounting.py"


class ArtificialPreparation:
    product_formula = "2nd"
    is_deterministic_only = True
    exact_rte_lambda_r = 0.0
    threshold_operator_error_bound = 0.0
    hamiltonian_hash = partition_hash = preparation_hash = "ARTIFICIAL_ONLY"
    ld = 1
    coefficient_atol = 0.0


class ArtificialCost(SimpleNamespace):
    pass


class ArtificialCompiler(SimpleNamespace):
    pass


def load_accounting(source: str, name: str) -> ModuleType:
    """Retain all real definitions/stdlib imports; replace relative imports."""
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and not node.level:
            if node.module not in {"__future__", "dataclasses", "fractions", "typing"}:
                raise AssertionError("UNINSPECTED_IMPORT")
        if isinstance(node, ast.Import):
            if any(alias.name != "math" for alias in node.names):
                raise AssertionError("UNINSPECTED_IMPORT")
    tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.level)]
    module = ModuleType(name)
    sys.modules[name] = module
    def integer(value, *, name, minimum=0):
        if type(value) is not int or value < minimum:
            raise ValueError(name)
        return value
    module.__dict__.update(
        DFPartialS2Preparation=ArtificialPreparation,
        CircuitCost=ArtificialCost,
        require_integer_count=integer,
    )
    exec(compile(tree, "<accounting-artificial-fixture>", "exec"), module.__dict__)
    return module


def candidate(module, *, stat=.2, alpha=.01, provider=None, empirical=False):
    return module.evaluate_rpe_round_candidate(
        ArtificialPreparation(), module.RPERoundSpecification(0, .1),
        module.RPEErrorAllocation(0, 0, stat, alpha, alpha),
        module.RPEPFErrorModel(0, "ARTIFICIAL_ONLY", not empirical),
        beta_rpe=.4, rte_steps_per_occurrence=0, finite_taylor_order=0,
        cost_metric="rz_count", cost_provider=provider,
        hadamard_sampling_policy=module.RPEHadamardSamplingPolicy(
            "fresh_iid_per_hadamard_shot", True,
        ),
    )


class StaticProvider:
    """Synthetic constants; no circuit construction or compilation."""
    def __init__(self, module):
        self.module = module
        self.calls = 0
        self.cost = ArtificialCost(
            rz_count=7.0, rz_depth=7.0, cx_count=2.0, cx_depth=2.0,
            total_depth=9.0, circuit_size=10.0,
            compiler=ArtificialCompiler(label="ARTIFICIAL_ONLY"),
        )

    def evaluate(self, request):
        self.calls += 1
        return self.module.RPERoundCompiledCost(
            self.cost, self.cost, None, None, "ARTIFICIAL_ONLY", None,
            cost_model_fingerprint="ARTIFICIAL_ONLY",
        )


def old_source():
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "show", BEFORE + ":" + SOURCE], text=True,
    )


class ShotUnderflowRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = load_accounting(old_source(), "_b03_old_accounting")
        cls.after = load_accounting((ROOT / SOURCE).read_text(), "_b03_new_accounting")

    def test_direct_underflow_is_explicit_overflow(self):
        with self.assertRaises(ZeroDivisionError):
            self.before._hoeffding_shots(1e-200, .025)
        with self.assertRaisesRegex(OverflowError, "coordinate square underflowed"):
            self.after._hoeffding_shots(1e-200, .025)

    def test_tiny_beta_is_infeasible_and_skips_cost_provider(self):
        with self.assertRaises(ZeroDivisionError):
            candidate(self.before, stat=1e-200)
        for empirical in (False, True):
            with self.subTest(empirical=empirical):
                provider = StaticProvider(self.after)
                result = candidate(self.after, stat=1e-200, provider=provider, empirical=empirical)
                self.assertFalse(result.feasible)
                self.assertIn("shot_count_overflow", result.infeasibility_reasons)
                self.assertEqual(result.guarantee_status, "not_certified")
                self.assertIsNone(result.cosine_shots)
                self.assertIsNone(result.sine_shots)
                self.assertIsNone(result.round_total_cost)
                self.assertEqual(provider.calls, 0)

    def test_square_underflow_boundary_and_positive_subnormals(self):
        edge = math.ldexp(math.sqrt(.5), -537)
        values = [math.ldexp(1., -538), math.nextafter(edge, 0.), edge,
                  math.nextafter(edge, math.inf), math.ldexp(1., -537)]
        saw_zero = saw_positive = False
        for epsilon in values:
            with self.subTest(epsilon=epsilon.hex()):
                squared = epsilon * epsilon
                saw_zero |= squared == 0.
                saw_positive |= squared > 0.
                message = "coordinate square underflowed" if squared == 0. else "shot count overflowed"
                with self.assertRaisesRegex(OverflowError, message):
                    self.after._hoeffding_shots(epsilon, .025)
        self.assertTrue(saw_zero and saw_positive)

    def test_normal_integer_shots_are_bit_for_bit_unchanged(self):
        for epsilon in (1., .5, .1, 1e-3, 1e-40, 1e-100):
            for alpha in (.5, .025, 1e-12):
                with self.subTest(epsilon=epsilon, alpha=alpha):
                    old = self.before._hoeffding_shots(epsilon, alpha)
                    new = self.after._hoeffding_shots(epsilon, alpha)
                    self.assertEqual(new, old)
                    self.assertIs(type(new), int)
                    self.assertGreater(new, 0)

    def test_ceil_at_and_around_integer_boundary(self):
        alpha = 2. * math.exp(-1.)
        self.assertEqual(self.after._hoeffding_shots(1., alpha), 2)
        for a in (math.nextafter(alpha, 0.), alpha, math.nextafter(alpha, 1.)):
            for epsilon in (math.nextafter(1., 0.), 1., math.nextafter(1., math.inf)):
                with self.subTest(epsilon=epsilon, alpha=a):
                    self.assertEqual(self.after._hoeffding_shots(epsilon, a),
                                     self.before._hoeffding_shots(epsilon, a))

    def test_nonfinite_zero_and_negative_coordinates_still_rejected(self):
        for epsilon in (math.nan, math.inf, -math.inf, 0., -0., -.1):
            with self.subTest(epsilon=epsilon):
                with self.assertRaises(ValueError):
                    self.after._hoeffding_shots(epsilon, .025)

    def test_positive_square_but_shot_overflow_stays_explicit(self):
        self.assertGreater(1e-160 * 1e-160, 0.)
        for module in (self.before, self.after):
            with self.assertRaisesRegex(OverflowError, "shot count overflowed"):
                module._hoeffding_shots(1e-160, .025)

    def test_normal_candidate_and_cost_outputs_are_unchanged(self):
        for stat in (.01, .2, .35, 1e-100):
            for alpha in (.01, .025):
                for empirical in (False, True):
                    with self.subTest(stat=stat, alpha=alpha, empirical=empirical):
                        before_provider = StaticProvider(self.before)
                        after_provider = StaticProvider(self.after)
                        old = candidate(self.before, stat=stat, alpha=alpha, provider=before_provider, empirical=empirical)
                        new = candidate(self.after, stat=stat, alpha=alpha, provider=after_provider, empirical=empirical)
                        self.assertEqual(asdict(new), asdict(old))
                        self.assertEqual(after_provider.calls, before_provider.calls)

    def test_unmodified_existing_certification_assertions(self):
        """Reuse actual test bodies; substitute only artificial prep/cost types."""
        path = ROOT / "tests/test_rpe_certification_boundary.py"
        tree = ast.parse(path.read_text())
        nodes = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name != "_deterministic_preparation":
                node.decorator_list = []
                nodes.append(node)
        class Approx:
            def __init__(self, expected): self.expected = expected
            def __eq__(self, actual): return math.isclose(actual, self.expected, rel_tol=1e-6, abs_tol=1e-12)
        ns = dict(self.after.__dict__)
        ns.update(_deterministic_preparation=ArtificialPreparation,
                  CompilerSettings=ArtificialCompiler, CircuitCost=ArtificialCost,
                  pytest=SimpleNamespace(approx=Approx))
        ns["FRESH_IID_POLICY"] = self.after.RPEHadamardSamplingPolicy("fresh_iid_per_hadamard_shot", True)
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), "<existing-certification-assertions>", "exec"), ns)
        cases = [
            ("test_rpe_certification_requires_strict_pi_over_three_margin", (math.nextafter(math.pi/3, 0.), "certified")),
            ("test_rpe_certification_requires_strict_pi_over_three_margin", (math.pi/3, "not_certified")),
            ("test_rpe_certification_requires_strict_pi_over_three_margin", (math.nextafter(math.pi/3, math.inf), "not_certified")),
            ("test_pi_over_three_dominates_empirical_screening_status", ()),
            ("test_certification_rechecks_phase_budget_without_feasibility_tolerance", ()),
            ("test_certification_uses_accurate_phase_budget_summation", ()),
            ("test_summary_propagates_branch_status_and_conditional_energy_bound", (math.nextafter(math.pi/3, 0.), "certified")),
            ("test_summary_propagates_branch_status_and_conditional_energy_bound", (math.pi/3, "not_certified")),
            ("test_summary_union_certification_does_not_use_feasibility_tolerance", ()),
            ("test_summary_union_bound_sums_axis_allocations_directly", ()),
        ]
        for name, args in cases:
            with self.subTest(existing=name, parameters=args):
                ns[name](*args)


if __name__ == "__main__":
    unittest.main(verbosity=2)
