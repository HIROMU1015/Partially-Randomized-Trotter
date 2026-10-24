"""Only labels, rational metadata and mocks; no scientific imports or gates.

Run exactly this file with Python -S. No pytest collection or repository
scientific __init__ is needed. The import barrier also covers lazy imports.
"""
from pathlib import Path
import ast
from dataclasses import replace
from fractions import Fraction as F
import hashlib
import importlib.abc
import sys
from types import SimpleNamespace
import unittest

FORBIDDEN = {"numpy", "scipy", "qiskit", "openfermion", "pyscf", "numba", "torch", "cupy", "trotterlib"}


class NoScientificImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in FORBIDDEN:
            raise AssertionError("Scientific import outside mock scope: " + fullname)
        return None


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
_before_adapter_import = set(sys.modules)
_import_guard = NoScientificImports()
sys.meta_path.insert(0, _import_guard)
try:
    from trottertracks.b2d_adapter import _published as published
    from trottertracks.b2d_adapter.adapter import (
        BlockExponential, DContext, DRun, PhaseOperation, ScalarTerm, TailOccurrence,
        actual_adjoint, context_from_preparation, fuse_adjacent_d_runs,
        make_b2d_plan, published_math_sequences,
    )
finally:
    # Do not poison unrelated test modules during future broad collection.
    sys.meta_path.remove(_import_guard)
_adapter_imports = set(sys.modules) - _before_adapter_import


# AST hashes obtained from the fetched fixed upstream files, not from
# numerical evaluation of coefficients or any Hamiltonian/operator.
UPSTREAM_FUNCTION_AST_SHA256 = {
    "symmetric_s2_sequence": "2b8febbb47191abff84477c6020e92ad70296e39e9b9f556e16d412f3e6f7e10",
    "inverse_s2_sequence": "9c576a4d62ec9290b304ae1085bb949117fd4dcdb7713e1b908febdf881ff50f",
    "_iter_s2_steps": "6e2a500323be4d931554cfb02eb20e6ee14c1e8060b362935c95d496ef257250",
    "morales_2025_yp8m8_kernel_list": "dea97aa1df8c71f905d4496aeb5a37ebf5ef0b8f63ae1dff8babde83d5b7c4c0",
    "morales_2025_yp8m8_processor_gamma_list": "41d82c1e78b881c510878624559f5f6aa0d7ca38a6c7c619216090f38aa1a4bc",
    "morales_2025_yp8m8_processor_sequence": "c7a7d717581095a438ff3912f89ebe0faf8453651d34064bf366298752a395b0",
}


def context(*, controlled=False):
    return DContext(
        "mock-preparation", "mock-tail", ("one-body", "fragment-a", "fragment-b"), 3,
        (ScalarTerm("constant", F(2, 7)), ScalarTerm("tail-identity", F(-1, 9))),
        controlled=controlled, ancilla_qubit=3 if controlled else None,
    )


def tail(ctx, identity="tail-0", time=F(3, 5)):
    # The three opaque objects are never evaluated, resampled, or converted.
    return TailOccurrence(ctx, identity, time, object(), object(), object())


def manual_s2(blocks, time, controlled=False, ancilla=None):
    # Explicit mock reference, independent of the upstream loop/generator.
    a, b, c = blocks
    return (
        BlockExponential(a, time / 2, controlled, ancilla),
        BlockExponential(b, time / 2, controlled, ancilla),
        BlockExponential(c, time, controlled, ancilla),
        BlockExponential(b, time / 2, controlled, ancilla),
        BlockExponential(a, time / 2, controlled, ancilla),
    )


def cancel_formal_inverse_pairs(operations):
    """Free-word reduction of mock labels, not matrix multiplication."""
    stack = []
    for op in operations:
        if isinstance(op, BlockExponential) and stack and isinstance(stack[-1], BlockExponential):
            inverse = replace(stack[-1], evolution_time=-stack[-1].evolution_time)
            if op == inverse:
                stack.pop()
                continue
        stack.append(op)
    return tuple(stack)


class B2DAdapterTests(unittest.TestCase):
    def setUp(self):
        self.modules_before_test = set(sys.modules)
        guard = NoScientificImports()
        sys.meta_path.insert(0, guard)
        self.addCleanup(sys.meta_path.remove, guard)

    def test_upstream_functions_and_coefficients_are_unmodified(self):
        tree = ast.parse(Path(published.__file__).read_text())
        hashes = {
            node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
            for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        self.assertEqual(hashes, UPSTREAM_FUNCTION_AST_SHA256)
        self.assertEqual(published.UPSTREAM_COMMIT, "20d64c2fea1ed2d2188777df4f3d962ee94df486")
        self.assertEqual(published.PRODUCT_FORMULA_BLOB, "70d68691a1dcdedea5447aac4b8bbfa51141d4cb")

    def test_kernel_and_processor_use_published_order(self):
        kernel, processor = published_math_sequences()
        weights = tuple(map(F, published.morales_2025_yp8m8_kernel_list()))
        gamma = tuple(map(F, published.morales_2025_yp8m8_processor_gamma_list()))
        self.assertEqual(kernel, weights[:0:-1] + weights[:1] + weights[1:])
        self.assertEqual(processor, gamma[::-1] + tuple(-x for x in gamma[::-1]))
        self.assertEqual((len(kernel), len(processor)), (17, 20))  # published SOURCE lengths

    def test_math_product_is_reversed_for_append(self):
        ctx = context(controlled=True)
        run = DRun(ctx, F(1, 11), 2)
        expected = tuple(
            op
            for weight in reversed(run.mathematical_s2_weights())
            for op in manual_s2(ctx.block_ids, run.tau * weight, True, 3)
        )
        self.assertEqual(run.append_operations(), expected)

    def test_processor_uses_actual_adjoint_and_noncommuting_order_labels(self):
        ctx = context()
        run = DRun(ctx, F(1, 13), 1)
        _, processor = published_math_sequences()
        p_append = tuple(op for w in reversed(processor) for op in manual_s2(ctx.block_ids, run.tau * w))
        operations = run.append_operations()
        self.assertEqual(operations[:len(p_append)], actual_adjoint(p_append))
        self.assertEqual(operations[-len(p_append):], p_append)
        self.assertEqual(actual_adjoint(actual_adjoint(operations)), operations)
        self.assertEqual(cancel_formal_inverse_pairs(operations + actual_adjoint(operations)), ())

    def test_two_d_halves_have_the_same_direction(self):
        ctx = context()
        t = tail(ctx)
        plan = make_b2d_plan(ctx, step_time=F(3, 5), n_d=2, tail_occurrences=[t])
        left, opaque, right = plan.body
        self.assertIs(opaque, t)
        self.assertEqual(left.nominal_time, F(3, 10))
        self.assertEqual(left.append_operations(), right.append_operations())
        self.assertNotEqual(left.append_operations(), actual_adjoint(right.append_operations()))

    def test_tail_is_a_barrier_even_with_identical_d_processors(self):
        ctx = context()
        t = tail(ctx)
        run = DRun(ctx, F(1, 5), 1)
        self.assertEqual(fuse_adjacent_d_runs((run, t, run)), (run, t, run))
        with self.assertRaises(TypeError):
            actual_adjoint((t,))

    def test_adjacent_fusion_retains_tau_and_formal_word(self):
        ctx = context()
        tails = (tail(ctx, "tail-0"), tail(ctx, "tail-1"))
        raw = make_b2d_plan(ctx, step_time=F(3, 5), n_d=2, tail_occurrences=tails)
        fused = make_b2d_plan(ctx, step_time=F(3, 5), n_d=2, tail_occurrences=tails, fuse_boundaries=True)
        self.assertEqual(len(fused.body), 5)
        middle = fused.body[2]
        self.assertEqual((middle.tau, middle.kernel_repetitions), (F(3, 20), 4))
        self.assertEqual(cancel_formal_inverse_pairs(tuple(raw.iter_append_operations())),
                         cancel_formal_inverse_pairs(tuple(fused.iter_append_operations())))
        self.assertIs(fused.body[1], tails[0])
        self.assertIs(fused.body[3], tails[1])

    def test_incompatible_contexts_and_times_are_not_fused(self):
        ctx = context()
        first = DRun(ctx, F(1, 10), 2)
        alternatives = (
            replace(first, tau=F(1, 5)),
            replace(first, context=replace(ctx, preparation_id="different-preparation")),
            replace(first, context=replace(ctx, tail_id="different-tail")),
            replace(first, context=replace(ctx, block_ids=ctx.block_ids[::-1])),
            replace(first, context=context(controlled=True)),
            replace(first, context=replace(ctx, scalar_terms=(ScalarTerm("constant", 1),))),
        )
        for second in alternatives:
            with self.subTest(second=second):
                self.assertEqual(fuse_adjacent_d_runs((first, second)), (first, second))

    def test_negative_time_and_fractional_coefficients_are_retained(self):
        ctx = context(controlled=True)
        plus = make_b2d_plan(ctx, step_time=F(3, 5), n_d=3, tail_occurrences=[tail(ctx)])
        minus_tail = tail(ctx, time=F(-3, 5))
        minus = make_b2d_plan(ctx, step_time=F(-3, 5), n_d=3, tail_occurrences=[minus_tail])
        for positive, negative in zip(plus.body[0].append_operations(), minus.body[0].append_operations(), strict=True):
            self.assertEqual(positive.block_id, negative.block_id)
            self.assertEqual(positive.evolution_time, -negative.evolution_time)
        self.assertIs(minus.body[1], minus_tail)
        self.assertEqual(minus.scalar_phases[0].phase_argument, -plus.scalar_phases[0].phase_argument)

    def test_scalar_and_event_phase_and_normalization_survive(self):
        ctx = context(controlled=True)
        t = tail(ctx)
        plan = make_b2d_plan(ctx, step_time=F(3, 5), n_d=2, tail_occurrences=[t, tail(ctx, "tail-1")])
        phases = plan.scalar_phases
        self.assertEqual(len(phases), len(ctx.scalar_terms))
        self.assertEqual(phases[0].evolution_time, F(6, 5))
        self.assertEqual(phases[0].phase_argument, -F(6, 5) * ctx.scalar_terms[0].coefficient)
        self.assertTrue(all(p.destination == "relative_ancilla_phase" and p.ancilla_qubit == 3 for p in phases))
        operations = tuple(plan.iter_append_operations())
        self.assertEqual(operations[:len(phases)], phases)
        bound = next(op for op in operations if isinstance(op, TailOccurrence))
        self.assertIs(bound.payload, t.payload)
        self.assertIs(bound.event_phase_metadata, t.event_phase_metadata)
        self.assertIs(bound.normalization_metadata, t.normalization_metadata)
        plain = replace(phases[0], controlled=False, ancilla_qubit=None)
        self.assertEqual(plain.destination, "global_phase")

    def test_cost_metadata_distinguishes_repetitions_and_pairs(self):
        ctx = context()
        tails = [tail(ctx, "t0"), tail(ctx, "t1")]
        raw = make_b2d_plan(ctx, step_time=F(3, 5), n_d=3, tail_occurrences=tails)
        fused = make_b2d_plan(ctx, step_time=F(3, 5), n_d=3, tail_occurrences=tails, fuse_boundaries=True)
        self.assertEqual((raw.structural_cost.kernel_repetitions, raw.structural_cost.processor_pair_count), (12, 4))
        self.assertEqual((fused.structural_cost.kernel_repetitions, fused.structural_cost.processor_pair_count), (12, 3))
        self.assertEqual(fused.structural_cost.payment_unit, "per_contiguous_D_run_per_execution")
        self.assertEqual(fused.structural_cost.price_status, "NO_NATIVE_COST_OR_SHOT_ESTIMATE")

    def test_metadata_bridge_reads_existing_preparation_without_building(self):
        prep = SimpleNamespace(
            preparation_hash="prepared-id", tail_extraction=SimpleNamespace(tail_hash="tail-id"),
            deterministic_block_order=("one-body", "df-7"), num_system_qubits=2,
            constant_coefficient=F(2, 7), extracted_identity_coefficient=F(-1, 9),
        )
        ctx = context_from_preparation(prep, controlled=True, ancilla_qubit=2)
        self.assertEqual(ctx.block_ids, prep.deterministic_block_order)
        self.assertEqual((ctx.preparation_id, ctx.tail_id), ("prepared-id", "tail-id"))
        self.assertEqual(tuple(t.coefficient for t in ctx.scalar_terms), (F(2, 7), F(-1, 9)))

    def test_mismatched_or_missing_tail_bindings_are_rejected(self):
        ctx = context()
        t = tail(ctx)
        for tails in ([], [t, t], [replace(t, evolution_time=F(1, 2))], [replace(t, context=replace(ctx, tail_id="wrong"))]):
            with self.subTest(tails=tails), self.assertRaises(ValueError):
                make_b2d_plan(ctx, step_time=F(3, 5), n_d=1, tail_occurrences=tails)
        for field in ("payload", "event_phase_metadata", "normalization_metadata"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(t, **{field: None})

    def test_invalid_counts_registers_and_nonfinite_times_are_rejected(self):
        ctx = context()
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count), self.assertRaises((ValueError, TypeError)):
                DRun(ctx, F(1, 10), count)
        for time in (float("inf"), float("nan"), True, "0.1"):
            with self.subTest(time=time), self.assertRaises((ValueError, TypeError)):
                DRun(ctx, time, 1)
        with self.assertRaises(ValueError):
            replace(ctx, block_ids=("duplicate", "duplicate"))
        with self.assertRaises(ValueError):
            replace(ctx, controlled=True, ancilla_qubit=1)
        with self.assertRaises(ValueError):
            replace(ctx, ancilla_qubit=3)

    def test_no_scientific_modules_loaded(self):
        added = _adapter_imports | (set(sys.modules) - self.modules_before_test)
        self.assertFalse(FORBIDDEN.intersection(name.split(".")[0] for name in added))


if __name__ == "__main__":
    # The authorized standalone command must start without scientific modules.
    assert not FORBIDDEN.intersection(name.split(".")[0] for name in sys.modules)
    unittest.main()
