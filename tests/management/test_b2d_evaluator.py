"""Focused stdlib mock tests; do not instantiate the dormant DenseBackend."""
import ast
from dataclasses import replace
from fractions import Fraction as F
import importlib.abc
import hashlib
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
FORBIDDEN = {"numpy", "scipy", "qiskit", "openfermion", "pyscf", "numba", "trotterlib", "cupy", "torch"}


class Barrier(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in FORBIDDEN:
            raise AssertionError("Forbidden scientific import: " + fullname)
        return None


guard = Barrier()
sys.meta_path.insert(0, guard)
try:
    from trottertracks.b2d_adapter.adapter import (
        DContext, DRun, ScalarTerm, TailOccurrence, actual_adjoint,
    )
    from trottertracks.b2d_evaluator import _baselines
    from trottertracks.b2d_evaluator.plans import ARMS, make_arm_plan
    from trottertracks.b2d_evaluator.evaluation import (
        Settings, ExecutionApproval, build_graph, evaluate,
    )
    from trottertracks.b2d_evaluator.dense_backend import DenseBackend
finally:
    sys.meta_path.remove(guard)


def settings():
    # Synthetic labels only. Numeric test conditions are not scientific adoption.
    return Settings(
        source_commit="MOCK_SOURCE", fixture_definition_commit="MOCK_FIXTURE_COMMIT",
        fixture_definition_blob="MOCK_FIXTURE_BLOB", hamiltonian_id="MOCK_H",
        sector_id="MOCK_N1", state_id="MOCK_STATE", input_data_sha256="MOCK_ARRAYS",
        context=DContext("MOCK_PREP", "MOCK_TAIL", ("one", "df-a", "df-b"), 2,
                         (ScalarTerm("constant", F(17, 100)),)),
        deltas=(F(9, 100), F(-9, 100)), q_values=(2, 3), n_d_values=(1, 2),
        r_rte=2, cutoff=2, seed=41, truncation_tolerance=1.0,
        coefficient_atol=F(0), identity_policy="extract_identity_phase",
        finite_bit_policy="analytic_moments_no_sampling", precision="complex128",
        numerical_atol=1e-12, radius_floor=0.1, branch_margin=0.1,
        decision_policy="descriptive_only_no_scientific_go",
    )


class MockBackend:
    is_mock = True

    def __init__(self):
        self.calls = []

    def perform(self, kind, args, params):
        self.calls.append((kind, args, params))
        return {"mock_kind": kind, "mock_call": len(self.calls)}


def mock_approval(graph):
    return ExecutionApproval(
        "MOCK_APPROVAL_NOT_USER_AUTHORIZATION", "MOCK_NOT_A_GRANT",
        graph.settings.source_commit, graph.fingerprint,
        tuple(graph.call_counts.items()), "MOCK_NO_OUTPUT_WRITTEN", "MOCK_NO_RETRY",
        mock_only=True,
    )


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        barrier = Barrier()
        sys.meta_path.insert(0, barrier)
        self.addCleanup(sys.meta_path.remove, barrier)

    def test_five_families_and_fixed_coefficient_sources(self):
        s = settings()
        tail = TailOccurrence(s.context, "t", s.deltas[0], object(), object(), object())
        plans = [make_arm_plan(a, s.context, s.deltas[0], None if a == "A0" else 1, tail) for a in ARMS]
        self.assertEqual(len({p.family for p in plans}), 5)
        self.assertTrue(all(p.tail is tail for p in plans))
        self.assertTrue(all(p.coefficient_source == "20d64c2fea1ed2d2188777df4f3d962ee94df486" for p in plans[1:]))
        self.assertEqual(plans[3].family, "published_Morales_Y8m10b")

    def test_a0_preserves_forward_and_reverse_sweeps(self):
        s = settings()
        t = TailOccurrence(s.context, "t", F(9, 100), "payload", "phase", "normalization")
        p = make_arm_plan("A0", s.context, t.evolution_time, None, t)
        self.assertEqual(tuple(x.block_id for x in p.before_tail), s.context.block_ids)
        self.assertEqual(tuple(x.block_id for x in p.after_tail), s.context.block_ids[::-1])
        self.assertTrue(all(x.evolution_time == F(9, 200) for x in p.before_tail))
        self.assertNotEqual(p.before_tail, p.after_tail)
        self.assertEqual(p.mathematical_operations, p.append_operations[::-1])

    def test_a4_uses_unchanged_adapter_and_actual_inverse(self):
        s = settings()
        t = TailOccurrence(s.context, "t", F(9, 100), object(), object(), object())
        p = make_arm_plan("A4", s.context, t.evolution_time, 2, t)
        self.assertEqual(p.before_tail, DRun(s.context, F(9, 400), 2).append_operations())
        self.assertEqual(p.before_tail, p.after_tail)
        self.assertIs(p.append_operations[len(p.before_tail)], t)
        self.assertEqual(actual_adjoint(actual_adjoint(p.before_tail)), p.before_tail)
        with self.assertRaises(TypeError):
            actual_adjoint((t,))

    def test_signed_times_and_same_direction_for_all_nested_arms(self):
        s = settings()
        for arm in ARMS[1:]:
            t = TailOccurrence(s.context, "t", F(9, 100), object(), object(), object())
            negative = replace(t, evolution_time=-t.evolution_time)
            a = make_arm_plan(arm, s.context, t.evolution_time, 2, t)
            b = make_arm_plan(arm, s.context, negative.evolution_time, 2, negative)
            self.assertEqual(a.before_tail, a.after_tail)
            self.assertEqual(tuple(x.evolution_time for x in b.before_tail),
                             tuple(-x.evolution_time for x in a.before_tail))

    def test_mismatched_tail_or_unsupported_arm_is_rejected(self):
        s = settings()
        t = TailOccurrence(s.context, "t", F(9, 100), object(), object(), object())
        for arm, n, bad_tail in (("A5", 1, t), ("A0", 1, t), ("A4", 0, t),
                                  ("A4", 1, replace(t, evolution_time=F(1)))):
            with self.subTest(arm=arm, n=n), self.assertRaises(ValueError):
                make_arm_plan(arm, s.context, F(9, 100), n, bad_tail)

    def test_grid_fairness_and_input_identity_are_shared(self):
        s = settings()
        graph = build_graph(s)
        self.assertEqual(len(graph.rows), 36)
        self.assertEqual(sum(r["arm"] == "A0" for r in graph.rows), 4)
        for arm in ARMS[1:]:
            self.assertEqual(sum(r["arm"] == arm for r in graph.rows), 8)
        self.assertEqual({r["input_fingerprint"] for r in graph.rows}, {s.fingerprint})
        self.assertEqual({r["tail_identity"] for r in graph.rows}, {s.context.tail_id})

    def test_shared_reference_calls_are_cached(self):
        graph = build_graph(settings())
        self.assertEqual(graph.call_counts["rte_moments"], 2)
        self.assertEqual(graph.call_counts["validate_input"], 1)
        self.assertEqual(graph.call_counts["a4_reference_check"], 4)
        physical = [n for n in graph.nodes if n.kind == "exponential" and n.parameters[0] == "exact_physical"]
        self.assertEqual(len(physical), 4)
        self.assertEqual(len(graph.nodes), len(set(graph.nodes)))

    def test_unresolved_conditions_are_rejected(self):
        s = settings()
        for field, value in (("state_id", ""), ("numerical_atol", None), ("radius_floor", None),
                             ("branch_margin", None), ("n_d_values", ()), ("deltas", (F(0),)),
                             ("cutoff", 1), ("q_values", (True,)), ("precision", "unspecified"),
                             ("decision_policy", "GO")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                build_graph(replace(s, **{field: value}))

    def test_no_approval_rejected_before_factory_is_touched(self):
        graph = build_graph(settings())
        def forbidden(_):
            self.fail("An unapproved backend must not be constructed.")
        with self.assertRaises(PermissionError):
            evaluate(graph, forbidden)

    def test_wrong_source_grid_calls_or_grant_rejected(self):
        graph = build_graph(settings())
        approved = mock_approval(graph)
        for field, value in (("source_commit", "wrong"), ("graph_fingerprint", "wrong"),
                             ("approved_call_counts", ()), ("grant_reference", "")):
            with self.subTest(field=field), self.assertRaises(PermissionError):
                evaluate(graph, lambda _: self.fail("Factory must not be touched"), replace(approved, **{field: value}))

    def test_mock_schema_and_budget_cover_all_nodes(self):
        graph = build_graph(settings())
        backend = MockBackend()
        result = evaluate(graph, lambda _: backend, mock_approval(graph))
        self.assertEqual(len(backend.calls), len(graph.nodes))
        self.assertEqual(result["mode"], "MOCK_ONLY")
        self.assertEqual(result["decision"], "DESCRIPTIVE_ONLY_NO_SCIENTIFIC_GO")
        self.assertEqual(len(result["rows"]), 36)
        self.assertEqual(result["stop"], "MANDATORY_STOP_NO_AUTOMATIC_EXTENSION")
        self.assertIsNone(next(r for r in result["rows"] if r["arm"] == "A0")["outputs"]["D_half_operator_error"])

    def test_scalar_phase_and_normalization_have_explicit_nodes(self):
        graph = build_graph(settings())
        self.assertIn("physical_scalar_phase", graph.call_counts)
        self.assertIn("normalization", graph.call_counts)
        self.assertIn("rescale_raw", graph.call_counts)
        for r in graph.rows:
            self.assertIn("finite_RTE_error", r["outputs"])
            self.assertIn("normalization_consistency", r["outputs"])

    def test_baseline_function_asts_match_fetched_fixed_source(self):
        # SHA256s from the actually fetched fixed upstream function ASTs.
        tree = ast.parse(Path(_baselines.__file__).read_text())
        hashes = {n.name: hashlib.sha256(ast.dump(n, include_attributes=False).encode()).hexdigest()
                  for n in tree.body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(hashes, {
            "trotter_2nd_list": "23609ba7c378044df6935947081144e011e4fd5a9cb52d9250e7abbd57c96a25",
            "yoshida_4th_list": "bab025846ae7f99c6d1d3143b7e358331631e3126f83fe3bff30a96ffeb227da",
            "morales_2025_y8m10b_list": "bb9ed2bb2466aea68928c2106b09fe702971237bbda16d21fa23fe64b51f4618",
        })

    def test_dormant_backend_import_has_no_scientific_imports(self):
        self.assertFalse(FORBIDDEN.intersection(name.split(".")[0] for name in sys.modules))


if __name__ == "__main__":
    unittest.main()
