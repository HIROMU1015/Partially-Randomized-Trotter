"""Artificial exact arithmetic only; no quantum libraries or scientific data."""
import copy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from itertools import permutations
import unittest
from unittest.mock import patch

import fixtures as fx
from oracle import old
import prototype as p


class O1Tests(unittest.TestCase):
    def test_all_structures_states_and_ordered_supports(self):
        count = 0
        for k in (1, 2, 3):
            kinds = ["identity", "diagonal", "tiny", "sparse_complex", "dense"]
            kinds += ["hadamard"] if k == 1 else ["controlled"]
            if k == 2:
                kinds.append("xx_block")
            for kind in kinds:
                m = fx.matrix(kind, k)
                prepared = p.prepare_matrix(m)
                for qs in permutations(range(4), k):
                    for sk in ("sparse", "general", "dyadic", "cancellation"):
                        v = fx.state(4, sk)
                        observed = p.apply_o1(v, prepared, qs)
                        expected = old.apply_exact(v, m, qs)
                        self.assertEqual(expected, observed, (kind, qs, sk))
                        cache = p.IndexCache()
                        combined = p.apply_o1_o2(v, prepared, qs, cache)
                        self.assertEqual(expected, combined, (kind, qs, sk, "O2 cold"))
                        self.assertEqual(expected, p.apply_o1_o2(v, prepared, qs, cache))
                        for ancilla in range(4):
                            if old.norm_squared(expected):
                                a = old.born_probability(observed, ancilla)
                                b = old.born_probability(expected, ancilla)
                                self.assertEqual(a, b)
                                self.assertEqual(old.threshold_count(a), old.threshold_count(b))
                        count += 1
        self.assertEqual(count, 1008)

    def test_each_gate_of_fixed_benchmark_sequence_and_nonmutation(self):
        for fixture in fx.benchmark_fixtures():
            before = copy.deepcopy(fixture)
            a = b = c = fixture.initial
            prepared = tuple(map(p.prepare_matrix, fixture.matrices))
            cache = p.IndexCache()
            for mi, qs in fixture.gates:
                a = old.apply_exact(a, fixture.matrices[mi], qs)
                b = p.apply_o1(b, prepared[mi], qs)
                c = p.apply_o1_o2(c, prepared[mi], qs, cache)
                self.assertEqual(a, b)
                self.assertEqual(a, c)
                pa = old.born_probability(a, fixture.ancilla)
                pb = old.born_probability(b, fixture.ancilla)
                self.assertEqual(pa, pb)
                self.assertEqual(pa, old.born_probability(c, fixture.ancilla))
                ta, tb = old.threshold_count(pa), old.threshold_count(pb)
                self.assertEqual(ta, tb)
                self.assertEqual(ta, old.threshold_count(old.born_probability(c, fixture.ancilla)))
                self.assertEqual([int(w >= ta) for w in fx.FIXED_WORDS],
                                 [int(w >= tb) for w in fx.FIXED_WORDS])
            self.assertEqual(fixture, before)

    def test_tiny_entries_zero_rows_and_exact_types(self):
        m = fx.matrix("tiny", 1)
        r = p.prepare_matrix(m)
        self.assertEqual(r.nnz, 3)
        v = ((F(0), F(0)), (F(1), F(0)))
        self.assertEqual(p.apply_o1(v, r, (0,))[0], m[0][1])
        zero = p.prepare_matrix(((fx.Z, fx.Z), (fx.Z, fx.Z)))
        self.assertEqual(p.apply_o1(v, zero, (0,)), [fx.Z, fx.Z])
        with self.assertRaises(ValueError):
            old.born_probability(p.apply_o1(v, zero, (0,)), 0)
        for bad in (1.0, 1, complex(1)):
            with self.assertRaises(ValueError):
                p.prepare_matrix((((bad, F(0)),),))

    def test_invalid_shapes_support_and_state_rejected_without_fallback(self):
        for m in ((), ((fx.ONE, fx.Z),), tuple((fx.ONE,) * 3 for _ in range(3))):
            with self.assertRaises(ValueError):
                p.prepare_matrix(m)
        r = p.prepare_matrix(fx.matrix("identity", 1))
        for v, qs in (((), (0,)), ((fx.ONE,) * 3, (0,)),
                      ((fx.ONE,) * 4, (0, 0)), ((fx.ONE,) * 4, (-1,)),
                      ((fx.ONE,) * 4, (2,)), ((fx.ONE,) * 4, (True,)),
                      ((fx.ONE,) * 4, (0, 1)), (((F(1), 0.0),) * 4, (0,))):
            with patch.object(old, "apply_exact", side_effect=AssertionError("no validation bypass")):
                with self.assertRaises(ValueError):
                    p.apply_o1(v, r, qs)
                with self.assertRaises(ValueError):
                    p.apply_o1_o2(v, r, qs, p.IndexCache())

    def test_prepared_data_are_detached_and_immutable(self):
        m = [list(row) for row in fx.matrix("diagonal", 1)]
        r = p.prepare_matrix(m)
        m[0][0] = fx.Z
        self.assertNotEqual(r.matrix[0][0], fx.Z)
        with self.assertRaises(FrozenInstanceError):
            r.nnz = 0
        with self.assertRaises(TypeError):
            r.rows[0][0] = (1, fx.Z)

    def test_general_dense_fallback_reuses_original_kernel(self):
        r = p.prepare_matrix(fx.matrix("dense", 2))
        v = fx.state(3)
        with patch.object(old, "apply_exact", wraps=old.apply_exact) as call:
            self.assertEqual(p.apply_o1(v, r, (2, 0)), old.apply_exact(v, r.matrix, (2, 0)))
            self.assertEqual(call.call_count, 2)

    def test_born_scale_zero_norm_and_grid_boundary_words(self):
        v = ((F(3), F(0)), fx.Z, (F(0), F(4)), fx.Z)
        self.assertEqual(old.born_probability(v, 1), F(9, 25))
        self.assertEqual(old.born_probability(tuple((7*x, 7*y) for x, y in v), 1), F(9, 25))
        with self.assertRaises(ValueError):
            old.born_probability((fx.Z,) * 4, 1)
        M = 1 << 53
        for k in (1, M // 2, M - 1):
            for epsilon in (-F(1, M*M), F(0), F(1, M*M)):
                q = F(k, M) + epsilon
                t = old.threshold_count(q)
                self.assertEqual(t, k + int(epsilon > 0))
                for w in fx.FIXED_WORDS:
                    self.assertEqual(int(w >= t), int(F(w, M) >= q))
                for w in (t - 1, t, t + 1):
                    if 0 <= w < M:
                        self.assertEqual(int(w >= t), int(F(w, M) >= q))
        self.assertEqual(old.threshold_count(F(0)), 0)
        self.assertEqual(old.threshold_count(F(1)), M)
        for q in (F(-1, 10), F(11, 10)):
            with self.assertRaises(ValueError):
                old.threshold_count(q)


class O2Tests(unittest.TestCase):
    def test_plan_matches_original_indices_and_updates_every_amplitude_once(self):
        for n in range(0, 5):
            for k in range(n + 1):
                for qs in permutations(range(n), k):
                    plan = p.build_index_plan(1 << n, qs)
                    mask = sum(1 << q for q in qs)
                    original = tuple(tuple(base | sum(((j >> i) & 1) << q for i, q in enumerate(qs))
                                           for j in range(1 << k))
                                     for base in range(1 << n) if not base & mask)
                    self.assertEqual(plan.blocks, original)
                    self.assertEqual(sorted(x for b in plan.blocks for x in b), list(range(1 << n)))

    def test_cache_order_dimension_cold_warm_eviction_regeneration(self):
        cache = p.IndexCache(2)
        first = cache.get(8, (2, 0))
        self.assertIs(first, cache.get(8, (2, 0)))
        reverse = cache.get(8, (0, 2))
        self.assertNotEqual(first.blocks, reverse.blocks)
        self.assertEqual(cache.get(16, (2, 0)).dimension, 16)
        self.assertEqual(cache.get(8, (2, 0)), first)
        self.assertEqual(cache.info(), dict(entries=2, hits=1, misses=4, evictions=2, stored_indices=24))
        with self.assertRaises(FrozenInstanceError):
            first.dimension = 16
        with self.assertRaises(TypeError):
            first.blocks[0][0] = 3
        for dimension, qs in ((0, ()), (3, ()), (8, (0, 0)), (8, (3,)), (8, (-1,)), (8, (0.0,))):
            with self.assertRaises(ValueError):
                cache.get(dimension, qs)

    def test_evicted_rebuilt_cache_preserves_state(self):
        cache = p.IndexCache(1)
        r = p.prepare_matrix(fx.matrix("sparse_complex", 2))
        v = fx.state(4)
        a = p.apply_o1_o2(v, r, (3, 0), cache)
        cache.get(16, (1, 2))
        self.assertEqual(a, p.apply_o1_o2(v, r, (3, 0), cache))

    def test_zero_qubit_scalar_and_input_matrix_unchanged(self):
        scalar = (((F(2, 3), F(-1, 5)),),)
        v = fx.state(2)
        r = p.prepare_matrix(scalar)
        for method in (p.apply_o1, lambda v, r, qs: p.apply_o1_o2(v, r, qs, p.IndexCache())):
            self.assertEqual(method(v, r, ()), old.apply_exact(v, scalar, ()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
