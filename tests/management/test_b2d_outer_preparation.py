"""Metadata/opaque mocks only. Never materialize a Hamiltonian or operator."""
from dataclasses import replace
from fractions import Fraction as F
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "src"), str(Path(__file__).resolve().parent)]
from test_b2d_evaluator import Barrier, FORBIDDEN, MockBackend, settings

guard = Barrier()
sys.meta_path.insert(0, guard)
try:
    from trottertracks.b2d_adapter.adapter import TailOccurrence, actual_adjoint
    from trottertracks.b2d_adapter._published import symmetric_s2_sequence
    from trottertracks.b2d_evaluator._baselines import yoshida_4th_list
    from trottertracks.b2d_evaluator.plans import make_arm_plan
    from trottertracks.b2d_evaluator.outer import (
        METHODS, DStage, OuterCase, RTEStage, mathematical_stages, make_outer_plan,
        fuse_d_stages, normalization_product, build_outer_graph, evaluate_outer,
    )
    from trottertracks.b2d_evaluator.outer_backend import OuterDenseBackend
    from trottertracks.b2d_evaluator.errors import Stage1Failure
finally:
    sys.meta_path.remove(guard)


def case(method="outer-S2", **kwargs):
    return OuterCase(method, kwargs.pop("total_time", F(9, 50)),
        kwargs.pop("macro_steps", 2), kwargs.pop("inner_arm", "A2"), kwargs.pop("n_d", 1),
        **kwargs)


def plan(c=None, **kwargs):
    return make_outer_plan(settings().context, c or case(), r_rte=2, cutoff=2,
        truncation_tolerance=1.0, seed=41, independent_occurrences=True, **kwargs)


def cases():
    return tuple(case(m, refinement=2) if m == "refined-S2" else case(m) for m in METHODS)


class OuterTests(unittest.TestCase):
    def setUp(self):
        guard = Barrier()
        sys.meta_path.insert(0, guard)
        self.addCleanup(sys.meta_path.remove, guard)

    def test_yoshida_reuses_fixed_generator_with_negative_middle_R(self):
        expected = tuple(map(F, symmetric_s2_sequence(yoshida_4th_list())))
        stages = mathematical_stages("outer-Yoshida4")
        self.assertEqual(tuple(v for k, v in stages if k == "R"), expected)
        self.assertEqual(tuple(k for k, v in stages), tuple("DRDDRDDRD"))
        self.assertLess(expected[1], 0)
        self.assertAlmostEqual(float(sum(expected)), 1)
        p = plan(case("outer-Yoshida4", macro_steps=1))
        rs = [s for s in p.body if isinstance(s, TailOccurrence)]
        self.assertEqual(len(rs), 3)
        self.assertLess(rs[1].evolution_time, 0)

    def test_saba_known_coefficients_positive_and_two_R_occurrences(self):
        stages = mathematical_stages("SABA2")
        self.assertEqual(tuple(k for k, v in stages), tuple("DRDRD"))
        self.assertAlmostEqual(float(stages[0][1]), (1 - 1 / math.sqrt(3)) / 2)
        self.assertAlmostEqual(float(stages[2][1]), 1 / math.sqrt(3))
        self.assertTrue(all(v > 0 for k, v in stages))
        for kind in ("D", "R"):
            self.assertEqual(sum(v for k, v in stages if k == kind), 1)
        self.assertEqual(plan(case("SABA2", macro_steps=1)).cost_metadata["R_occurrences"], 2)

    def test_outer_s2_matches_old_A2_A3_half_words(self):
        s = settings()
        tail = TailOccurrence(s.context, "MOCK", F(9, 100), "opaque", "phase", "N")
        for family in ("A2", "A3"):
            for n in (1, 2):
                p = plan(case(inner_arm=family, n_d=n), fuse_boundaries=False)
                old = make_arm_plan(family, s.context, F(9, 100), n, tail)
                self.assertEqual(p.body[0].append_operations(), old.before_tail)
                self.assertEqual(p.body[2].append_operations(), old.after_tail)

    def test_refined_s2_preserves_T_without_implicit_refinement(self):
        a, b = plan(), plan(case("refined-S2", refinement=3))
        self.assertEqual(b.case.step_count, 6)
        self.assertEqual(a.case.total_time, b.case.total_time)
        for p in (a, b):
            self.assertEqual(p.cost_metadata["D_time"], p.case.total_time)
            self.assertEqual(p.cost_metadata["R_time"], p.case.total_time)
        self.assertEqual(b.cost_metadata["R_occurrences"], 3 * a.cost_metadata["R_occurrences"])
        for bad in (None, 1, True, 1.5):
            with self.subTest(bad=bad), self.assertRaises((ValueError, TypeError)):
                case("refined-S2", refinement=bad)

    def test_math_append_order_and_signed_time_reversal(self):
        for c in cases():
            positive = plan(c, fuse_boundaries=False)
            negative = plan(replace(c, total_time=-c.total_time), fuse_boundaries=False)
            self.assertEqual(positive.mathematical_operations, positive.body[::-1])
            for x, y in zip(positive.body, negative.body, strict=True):
                tx = x.duration if isinstance(x, DStage) else x.evolution_time
                ty = y.duration if isinstance(y, DStage) else y.evolution_time
                self.assertEqual(tx, -ty)

    def test_legal_D_fusion_is_literal_kernel_concatenation(self):
        a = DStage(settings().context, "A2", F(3, 20), 1)
        b = replace(a, repetitions=2)
        merged, = fuse_d_stages((a, b))
        self.assertEqual(merged.tau, a.tau)
        self.assertEqual(merged.repetitions, 3)
        self.assertEqual(merged.append_operations(), a.append_operations() + b.append_operations())
        for other in (replace(a, tau=F(1, 10)), replace(a, inner_arm="A3"),
                      replace(a, context=replace(a.context, preparation_id="different"))):
            self.assertEqual(fuse_d_stages((a, other)), (a, other))

    def test_R_is_a_barrier_and_cannot_be_an_actual_inverse(self):
        d, r, _ = plan(case(macro_steps=1)).body
        self.assertEqual(fuse_d_stages((d, r, d)), (d, r, d))
        with self.assertRaises(TypeError):
            actual_adjoint((r,))
        with self.assertRaises(TypeError):
            fuse_d_stages((d, "pretend_inverse", d))

    def test_heterogeneous_yoshida_D_durations_are_not_added_into_one_PF(self):
        p = plan(case("outer-Yoshida4", macro_steps=1))
        # Six D approximations remain: changing tau would change the actual word.
        self.assertEqual(p.cost_metadata["D_runs"], 6)
        self.assertEqual(p.cost_metadata["negative_R_stages"], 1)
        self.assertGreater(p.cost_metadata["R_absolute_time"], abs(p.case.total_time))

    def test_each_occurrence_binds_its_actual_signed_config_and_unique_seed(self):
        p = plan(case("outer-Yoshida4"))
        tails = [s for s in p.body if isinstance(s, TailOccurrence)]
        received = []
        def factory(tail, **kwargs):
            received.append((tail, kwargs))
            return "config", "distribution"
        for t in tails:
            self.assertEqual(t.payload.make_config("MOCK_TAIL", factory), ("config", "distribution"))
            self.assertEqual(t.payload.duration, t.evolution_time)
        self.assertEqual(len({t.occurrence_id for t in tails}), len(tails))
        self.assertEqual(len({kw["seed"] for _, kw in received}), len(tails))
        self.assertEqual([kw["evolution_time"] for _, kw in received],
                         [float(t.evolution_time) for t in tails])
        self.assertTrue(all(kw["rte_steps"] == kw["finite_taylor_order"] == 2 for _, kw in received))

    def test_normalization_is_product_of_actual_occurrence_values(self):
        value = normalization_product([NS(normalization_product=n) for n in (2, 3, 5)])
        self.assertEqual(value["normalization_product"], 30)
        self.assertEqual(value["attenuation"], 1 / 30)
        self.assertEqual(value["occurrence_normalizations"], (2, 3, 5))
        for values in ([], [0], [-1], [float("inf")], [1e300, 1e300]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                normalization_product([NS(normalization_product=n) for n in values])

    def test_correlated_mean_product_is_rejected(self):
        for independence in (False, None, "independent"):
            with self.assertRaises(ValueError):
                make_outer_plan(settings().context, case(), r_rte=2, cutoff=2,
                    truncation_tolerance=1, seed=41, independent_occurrences=independence)

    def test_scalar_event_phase_and_control_zero_metadata_are_retained(self):
        context = replace(settings().context, controlled=True, ancilla_qubit=2)
        p = make_outer_plan(context, case("SABA2"), r_rte=2, cutoff=2,
            truncation_tolerance=1, seed=41, independent_occurrences=True)
        for phase in p.scalar_phases:
            self.assertEqual(phase.evolution_time, p.case.total_time)
            self.assertEqual(phase.destination, "relative_ancilla_phase")
            self.assertEqual(phase.ancilla_qubit, 2)
        for t in p.body:
            if isinstance(t, TailOccurrence):
                self.assertIs(t.context, context)
                self.assertIn("control_zero_branch", t.event_phase_metadata)
                self.assertIn("Nj", t.normalization_metadata)

    def test_stage_specific_RTE_backend_glue_uses_existing_callbacks_only(self):
        b = OuterDenseBackend.__new__(OuterDenseBackend)  # NO scientific constructor.
        b.validated = True
        b.i = NS(symbolic_tail="OPAQUE_SYMBOLIC", normalized_tail="OPAQUE_TAIL")
        calls = []
        b.execution = NS(science_call=lambda kind, fn, **kw: fn())
        def config(tail, **kwargs):
            calls.append(kwargs)
            return kwargs, "opaque-distribution"
        b.make_rte_config = config
        b.finite_rte_operator_moments = lambda tail, config: NS(
            corrected_operator=("corrected", config["evolution_time"]),
            attenuated_event_mean_operator=("raw", config["evolution_time"]),
            normalization_product=2 if config["evolution_time"] > 0 else 3)
        pos = b.perform("outer_rte_moments", (), (RTEStage(F(1, 10), 2, 2, 1, 41),))
        neg = b.perform("outer_rte_moments", (), (RTEStage(F(-2, 10), 3, 4, 1, 42),))
        self.assertNotEqual(pos.corrected_operator, pos.attenuated_event_mean_operator)
        self.assertEqual([x["evolution_time"] for x in calls], [0.1, -0.2])
        self.assertEqual(b.perform("outer_normalization", (pos, neg), ())["normalization_product"], 6)
        b.validated = False
        with self.assertRaises(ValueError):
            b.perform("outer_rte_moments", (), (RTEStage(F(1, 10), 2, 2, 1, 41),))

    def test_independent_inner_reference_glue_matches_the_literal_math_word(self):
        b = OuterDenseBackend.__new__(OuterDenseBackend)
        b.validated = True; b.s = settings()
        b._check_word = lambda actual, word, code: tuple(word)
        for arm in ("A2", "A3"):
            d = DStage(b.s.context, arm, F(-1, 9), 2)
            word = b.perform("outer_d_check", ("OPAQUE_ACTUAL",), (d,))
            self.assertEqual(word, tuple((op.block_id, op.evolution_time)
                                        for op in reversed(d.append_operations())))

    def test_DAG_raw_corrected_order_and_normalization_dependencies(self):
        g = build_outer_graph(settings(), (case("outer-Yoshida4"),), independent_occurrences=True)
        row = g.rows[0]
        norm = g.nodes[row["outputs"]["normalization"]]
        self.assertEqual(len(norm.dependencies), 6)
        signal = g.nodes[row["outputs"]["signals"]]
        for index, field in ((2, "corrected_operator"), (3, "attenuated_event_mean_operator")):
            phase = g.nodes[signal.dependencies[index]]
            self.assertEqual(phase.parameters, (F(9, 50),))
            word = g.nodes[phase.dependencies[0]]
            tails = [g.nodes[i] for i in word.dependencies if g.nodes[i].kind == "moment_field"]
            self.assertEqual(tuple(n.parameters[0] for n in tails), (field,) * 6)
            self.assertEqual(tuple(n.dependencies[0] for n in tails), norm.dependencies)
        self.assertNotIn("inverse", g.call_counts)

    def test_shared_references_complete_mock_and_all_gates_first(self):
        g = build_outer_graph(settings(), cases(), independent_occurrences=True)
        b = MockBackend()
        result = evaluate_outer(g, lambda _: b)
        kinds = [c[0] for c in b.calls]
        self.assertLess(max(i for i, k in enumerate(kinds) if k in
            ("outer_order_check", "outer_d_check", "outer_diagnostics", "outer_rescale_check")), kinds.index("operator_error"))
        self.assertEqual(len(b.calls), len(g.nodes))
        self.assertEqual(len(g.nodes), len(set(g.nodes)))
        self.assertEqual(sum(n.kind == "exponential" and n.parameters[0] == "exact_physical"
                             for n in g.nodes), 1)
        self.assertEqual(result["schema"], "b2d_outer_splitting_v1")
        self.assertEqual(result["mode"], "MOCK_ONLY")
        self.assertEqual(result["stop"], "MANDATORY_STOP_NO_AUTOMATIC_EXTENSION")

    def test_correctness_failure_never_starts_comparison_or_retries(self):
        g = build_outer_graph(settings(), cases(), independent_occurrences=True)
        for fail_kind in ("outer_order_check", "outer_rescale_check"):
            b = MockBackend()
            original = b.perform
            def perform(kind, args, params):
                value = original(kind, args, params)
                if kind == fail_kind:
                    raise Stage1Failure("MOCK_CORRECTNESS_FAILURE")
                return value
            b.perform = perform
            with self.subTest(fail_kind=fail_kind), self.assertRaises(Stage1Failure):
                evaluate_outer(g, lambda _: b)
            kinds = [x[0] for x in b.calls]
            self.assertNotIn("operator_error", kinds)
            self.assertNotIn("signals", kinds)
            self.assertEqual(kinds.count(fail_kind), 1)

    def test_unresolved_conditions_and_non_mock_backend_are_rejected(self):
        for s, cs in ((replace(settings(), source_commit=""), cases()),
                      (settings(), (case(total_time=F(1)),)), (settings(), ())):
            with self.assertRaises(ValueError):
                build_outer_graph(s, cs, independent_occurrences=True)
        g = build_outer_graph(settings(), cases(), independent_occurrences=True)
        b = MockBackend(); b.is_mock = False
        with self.assertRaises(ValueError):
            evaluate_outer(g, lambda _: b)
        self.assertFalse(FORBIDDEN.intersection(n.split(".")[0] for n in sys.modules))

    def test_old_science_scope_cannot_be_reused_for_new_outer_graph(self):
        from trottertracks.b2d_evaluator.scope import registry
        g = build_outer_graph(settings(), cases(), independent_occurrences=True)
        with self.assertRaises(PermissionError):
            registry(g)


if __name__ == "__main__":
    unittest.main()
