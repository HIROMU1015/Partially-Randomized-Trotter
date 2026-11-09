"""Technical integer/phase truth tables only; no quantum simulation or compile.

The mock sink follows one classical bit string and an exact symbolic phase.
It never constructs a statevector, matrix, molecular input, signal or price.
"""
from __future__ import annotations

from fractions import Fraction as F
import importlib.util
import itertools
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from trottertracks import df_threshold_reflection as th


def occupations(n):
    return itertools.product((0, 1), repeat=n)


class BitPhaseSink:
    def __init__(self, bits):
        self.bits = list(bits)
        self.pi = F(0)
        self.radians = F(0)
        self.basis_calls = []

    def x(self, target, controls):
        if target in controls or len(set(controls)) != len(controls):
            raise AssertionError("invalid reversible gate")
        if all(self.bits[i] for i in controls):
            self.bits[target] ^= 1

    def phase(self, angle, controls):
        if all(self.bits[i] for i in controls):
            self.pi = (self.pi + angle.pi) % 2
            self.radians += angle.radians

    def basis(self, basis_id, basis_hash, inverse, system):
        # An exact fermionic orbital swap: |11> has a minus sign.
        if (basis_id, basis_hash) != ("swap", "mock-swap"):
            raise AssertionError("unbound or wrong frame")
        self.basis_calls.append((basis_id, basis_hash, inverse))
        a, b = system[:2]
        self.pi = (self.pi + self.bits[a] * self.bits[b]) % 2
        self.bits[a], self.bits[b] = self.bits[b], self.bits[a]


def execute(operations, sink):
    for op in operations:
        if op.kind == "x":
            sink.x(op.target, op.controls)
        elif op.kind == "phase":
            sink.phase(op.angle, op.controls)
        else:
            sink.basis(op.basis_id, op.basis_hash, op.inverse, op.system)


class GridTests(unittest.TestCase):
    def test_stored_binary_float_is_exact_not_decimal_rounding(self):
        p = th.OccupationPolynomial(2, 0.1, (((0,), 0.3), ((0, 0), F(-1, 7)), ((1,), F(1, 13))))
        g = th.ThresholdGrid.from_polynomial(p)
        self.assertEqual(p.constant, F.from_float(0.1))
        self.assertNotEqual(p.constant, F(1, 10))
        for bits in occupations(2):
            self.assertEqual(g.value(bits), p.value(bits))
            self.assertTrue(0 <= g.value_integer(bits) <= g.extent)

    def test_number_cdf_full_ij_and_diagonal_n_squared_convention(self):
        p = th.OccupationPolynomial.from_number_cdf(((F(1, 2), F(1, 3)), (F(2, 3), F(-1, 4))), constant=F(2, 5))
        g = th.ThresholdGrid.from_polynomial(p)
        for a, b in occupations(2):
            self.assertEqual(g.value((a, b)), F(2, 5) + F(a, 2) + a*b - F(b, 4))

    def test_exact_reflection_identity_without_operator_matrices(self):
        p = th.OccupationPolynomial(3, F(-1, 4), (((0,), F(1, 2)), ((1,), F(-1, 4)), ((0, 2), F(1, 4))))
        g = th.ThresholdGrid.from_polynomial(p)
        for bits in occupations(3):
            z = g.value_integer(bits)
            signs = [2 * int(z >= j) - 1 for j in range(1, g.extent + 1)]
            self.assertTrue(all(s in (-1, 1) and s*s == 1 for s in signs))
            self.assertEqual(p.value(bits), g.center + g.spacing * sum(signs) / 2)

    def test_rank_one_signed_square_and_df_block_fields(self):
        for lam in (F(3, 8), F(-3, 8)):
            eta = (F(1, 2), F(-1), F(3, 2))
            g = th.ThresholdGrid.from_df_block(SimpleNamespace(eta=eta, lam=lam), constant=F(5, 7))
            self.assertIsNotNone(g.charge_weights)
            self.assertEqual(g.terms, ())
            for bits in occupations(3):
                q = sum(v*b for v, b in zip(eta, bits))
                self.assertEqual(g.value(bits), F(5, 7) + lam*q*q)
                self.assertTrue(0 <= g.value_integer(bits) <= g.extent)

    def test_scalar_and_zero_coefficient_cancellation(self):
        p = th.OccupationPolynomial(2, F(-2, 3), (((0,), F(1, 5)), ((0,), F(-1, 5))))
        g = th.ThresholdGrid.from_polynomial(p)
        self.assertEqual(g.extent, 0)
        self.assertEqual(g.center, F(-2, 3))
        self.assertEqual(th.ThresholdGrid.from_df_square((1, 2), 0).extent, 0)

    def test_huge_grid_is_not_materialized(self):
        g = th.ThresholdGrid.from_polynomial(th.OccupationPolynomial(2, terms=(((0,), F(1)), ((1,), F(1, 2**300)))))
        self.assertGreater(g.extent, 2**300)
        self.assertEqual(len(g.terms), 2)
        tail = th.ThresholdTail((th.ThresholdFragment("huge", g),), "extract_identity_phase")
        atom = th.ThresholdAtom(tail.fragments[0], g.extent)
        self.assertEqual(tail.probability(atom), F(1, g.extent))

    def test_invalid_input_is_rejected_without_coefficient_truncation(self):
        for value in (float("nan"), float("inf"), complex(1, 1), True):
            with self.assertRaises((TypeError, ValueError, OverflowError)):
                th.ThresholdGrid.from_df_square((value,), 1)
        with self.assertRaises(ValueError):
            th.OccupationPolynomial(2, terms=(((0, 1, 2), F(1)),))
        with self.assertRaises(ValueError):
            th.ThresholdGrid(1, F(0), F(1), 1, 0, (((0,), -1),))
        with self.assertRaises(ValueError):
            th.ThresholdGrid.from_df_square((1,), 1).value((2,))

    def test_rank_one_full_identity_mass_equals_full_pauli_mass(self):
        # Exact finite Boolean Fourier identity, not a performance benchmark.
        for eta in ((1,), (1, 2), (1, -2), (1, -2, 3), (0, -1, 2)):
            for lam in (F(3, 7), F(-3, 7)):
                g = th.ThresholdGrid.from_df_square(eta, lam)
                n = len(eta)
                coefficients = [sum((g.value(bits) * (-1)**sum(b*m for b, m in zip(bits, mask))
                                     for bits in occupations(n)), F(0)) / 2**n for mask in occupations(n)]
                full_pauli_mass = sum(abs(c) for c in coefficients)
                tail = th.ThresholdTail((th.ThresholdFragment("square", g),))
                self.assertEqual(tail.mass, full_pauli_mass)


class ArithmeticTests(unittest.TestCase):
    def test_modular_add_on_every_register_value_and_active_inactive_controls(self):
        for width in range(1, 6):
            register = tuple(range(width))
            for increment in (-9, -3, -1, 0, 1, 2, 7):
                operations = th._add_constant(register, increment, (width, width+1))
                for initial in range(1 << width):
                    for controls in occupations(2):
                        sink = BitPhaseSink([(initial >> i) & 1 for i in register] + list(controls))
                        execute(operations, sink)
                        observed = sum(sink.bits[i] << i for i in register)
                        self.assertEqual(observed, (initial + increment*math.prod(controls)) % (1 << width))
                        self.assertEqual(sink.bits[width:], list(controls))
                        execute(reversed(operations), sink)
                        self.assertEqual(sum(sink.bits[i] << i for i in register), initial)

    def test_comparator_restores_z_and_xors_both_initial_flag_values(self):
        for width in range(1, 6):
            register, flag = tuple(range(width)), width
            for threshold in range(1, 1 << width):
                operations = th._compare(register, flag, threshold)
                for z in range(1 << width):
                    for initial_flag in (0, 1):
                        original = [(z >> i) & 1 for i in register] + [initial_flag]
                        sink = BitPhaseSink(original)
                        execute(operations, sink)
                        self.assertEqual(sink.bits[:width], original[:width])
                        self.assertEqual(sink.bits[flag], initial_flag ^ int(z >= threshold))
                        execute(reversed(operations), sink)
                        self.assertEqual(sink.bits, original)

    def check_plan(self, atom, *, angle=None, controlled=True):
        plan = th.reflection_plan(atom, angle=angle, controlled=controlled)
        for bits in occupations(len(plan.system)):
            for control_bit in ((0, 1) if controlled else (1,)):
                initial = list(bits) + [0]*(plan.num_qubits-len(bits))
                if controlled:
                    initial[plan.control] = control_bit
                sink = BitPhaseSink(initial)
                plan.emit(sink)
                self.assertEqual(sink.bits, initial, "all arithmetic workspace must return to zero")
                logical = tuple(reversed(bits[:2])) + bits[2:] if atom.fragment.basis_id else bits
                sign = 2*int(atom.fragment.grid.value_integer(logical) >= atom.threshold)-1
                self.assertEqual(sink.pi, F(control_bit * int(sign == -1)) if angle is None else F(0))
                self.assertEqual(sink.radians, F(0) if angle is None else -th.exact(angle)*sign*control_bit)
                if angle is None:
                    plan.emit(sink)
                    self.assertEqual((sink.pi, sink.radians), (0, 0), "R dagger = R and R squared = I")
        return plan

    def test_general_quadratic_reflection_and_rotation_absolute_controlled_phase(self):
        grid = th.ThresholdGrid.from_polynomial(th.OccupationPolynomial(3, terms=(((0,), F(2)), ((1,), F(-1)), ((0, 2), F(1)))))
        fragment = th.ThresholdFragment("general", grid)
        for j in (1, grid.extent):
            for controlled in (False, True):
                self.check_plan(th.ThresholdAtom(fragment, j), controlled=controlled)
                self.check_plan(th.ThresholdAtom(fragment, j), angle=F(-2, 7), controlled=controlled)

    def test_signed_charge_square_reflection_and_rotation_uncompute(self):
        for lam in (F(1, 3), F(-2, 5)):
            grid = th.ThresholdGrid.from_df_square((1, -2, 3), lam)
            fragment = th.ThresholdFragment("square", grid)
            for j in (1, grid.extent//2, grid.extent):
                self.check_plan(th.ThresholdAtom(fragment, j))
                self.check_plan(th.ThresholdAtom(fragment, j), angle=F(3, 11))

    def test_existing_frame_provider_receives_actual_inverse_and_hash(self):
        grid = th.ThresholdGrid.from_df_square((1, 2), 1)
        atom = th.ThresholdAtom(th.ThresholdFragment("rotated", grid, "swap", "mock-swap"), 2)
        plan = self.check_plan(atom, angle=F(1, 3))
        sink = BitPhaseSink([0]*plan.num_qubits)
        plan.emit(sink)
        self.assertEqual(sink.basis_calls, [("swap", "mock-swap", True), ("swap", "mock-swap", False)])

    def test_omitting_product_global_phase_is_detected(self):
        g = th.ThresholdGrid.from_df_square((1,), 1)
        plan = th.reflection_plan(th.ThresholdAtom(th.ThresholdFragment("f", g), 1))
        initial = [0]*plan.num_qubits
        initial[plan.control] = 1
        sink = BitPhaseSink(initial)
        # Deliberately discard the phase on the control alone.
        broken = [op for op in plan.instructions if not (op.kind == "phase" and op.controls == (plan.control,))]
        execute(broken, sink)
        self.assertNotEqual(sink.pi, F(1))

    def test_scalar_identity_sign_and_rotation_have_no_arithmetic_workspace(self):
        for sign in (-1, 1):
            for angle in (None, F(1, 7)):
                plan = th.reflection_plan(th.ThresholdAtom(None, identity_sign=sign), num_modes=2, angle=angle)
                self.assertEqual((plan.value, plan.charge, plan.flag), ((), (), None))
                for bit in (0, 1):
                    sink = BitPhaseSink([1, 0, bit])
                    plan.emit(sink)
                    self.assertEqual(sink.pi, F(bit*int(sign == -1)) if angle is None else 0)
                    self.assertEqual(sink.radians, 0 if angle is None else -angle*sign*bit)

    def test_structural_counts_include_workspace_comparison_uncompute_and_basis(self):
        g = th.ThresholdGrid.from_df_square((1, -2), 1)
        plan = th.reflection_plan(th.ThresholdAtom(th.ThresholdFragment("f", g, "swap", "mock-swap"), 1))
        counts = plan.structural_counts()
        self.assertEqual(counts["basis_calls"], 2)
        self.assertEqual(counts["phase_operations"], 2)
        self.assertEqual(plan.num_qubits, 2+g.value_width+g.charge_width+2)
        self.assertTrue(counts["x_by_control_arity"])


class TailTests(unittest.TestCase):
    def test_rejection_uses_all_bits_and_does_not_modulo_bias(self):
        draws = iter((7, 6, 4))
        calls = []
        self.assertEqual(th.uniform_below(5, lambda n: (calls.append(n), next(draws))[1]), 4)
        self.assertEqual(calls, [3, 3, 3])
        self.assertEqual(th.uniform_below(1, lambda _: self.fail("no random bits needed")), 0)
        with self.assertRaises(ValueError):
            th.uniform_below(3, lambda n: 1 << n)

    def test_exact_group_then_threshold_probability_and_scalar_policies(self):
        fragment = th.ThresholdFragment("f", th.ThresholdGrid.from_df_square((1,), F(2)))
        faithful = th.ThresholdTail((fragment,))
        extracted = th.ThresholdTail((fragment,), "extract_identity_phase")
        self.assertEqual((faithful.mass, extracted.mass, extracted.deterministic_scalar), (2, 1, 1))
        reflection = faithful.draw_atom(lambda _: 0)
        identity = faithful.draw_atom(lambda _: 1)
        self.assertEqual(reflection, th.ThresholdAtom(fragment, 1))
        self.assertEqual(identity, faithful.identity_atom())
        self.assertEqual(faithful.probability(reflection), F(1, 2))
        self.assertEqual(faithful.probability(identity), F(1, 2))
        self.assertEqual(extracted.probability(reflection), F(1))
        with self.assertRaises(ValueError):
            extracted.probability(identity)

    def test_scalar_only_and_exact_identity_collection_across_fragments(self):
        a = th.ThresholdFragment("a", th.ThresholdGrid.from_df_square((1,), 0, constant=F(2, 3)))
        b = th.ThresholdFragment("b", th.ThresholdGrid.from_df_square((1,), 0, constant=F(-2, 3)))
        tail = th.ThresholdTail((a, b))
        self.assertEqual(tail.mass, 0)
        with self.assertRaises(ValueError):
            tail.draw_atom(lambda _: self.fail("must stop before randomness"))
        negative = th.ThresholdTail((b,))
        self.assertEqual(negative.draw_atom(lambda _: self.fail("one group")), th.ThresholdAtom(None, identity_sign=-1))

    def test_complete_hierarchical_mock_law_has_the_documented_atom_masses(self):
        f = th.ThresholdFragment("f", th.ThresholdGrid.from_df_square((1, 0), 2))
        g = th.ThresholdFragment("g", th.ThresholdGrid.from_df_square((1, -2), F(3, 2)))
        tail = th.ThresholdTail((f, g))
        law = {}
        # Group integer weights 1,3,4. Enumerate accepted integers, not RNG runs.
        for group_draw in range(8):
            thresholds = range(4) if 1 <= group_draw < 4 else (0,)
            for threshold_draw in thresholds:
                draws = iter((group_draw, threshold_draw))
                atom = tail.draw_atom(lambda _: next(draws))
                law[atom] = law.get(atom, F(0)) + F(1, 8*len(thresholds))
        self.assertEqual(sum(law.values()), 1)
        for atom, probability in law.items():
            self.assertEqual(probability, tail.probability(atom))

    def test_scalar_extraction_preserves_h_but_changes_finite_polynomial(self):
        # Exact Boolean scalar algebra: P1(-i*s*f) vs e^-isc P1(-i*s*(f-c)).
        # Compare the polynomial coefficients before any exponential evaluation.
        g = th.ThresholdGrid.from_df_square((1,), F(2))
        for bits in occupations(1):
            self.assertEqual(g.value(bits), g.center + g.spacing*(2*g.value_integer(bits)-g.extent)/2)
        faithful = th.ThresholdTail((th.ThresholdFragment("f", g),))
        extracted = th.ThresholdTail(faithful.fragments, "extract_identity_phase")
        self.assertNotEqual(faithful.mass, extracted.mass)
        # At occupation 0: f=0, f-c=-1. The unshifted P1 has no time term;
        # the shifted polynomial has one. Exact scalar evolution is separate.
        self.assertEqual(g.value((0,)), 0)
        self.assertNotEqual(g.value((0,))-extracted.deterministic_scalar, 0)


class RTEBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load only the unchanged RTE file, avoiding scientific package __init__.
        path = Path(__file__).resolve().parents[1] / "src/trotterlib/rte.py"
        name = "_threshold_test_rte_contract"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        cls.api = module

    def fixture(self):
        f = th.ThresholdFragment("f", th.ThresholdGrid.from_df_square((1,), 2))
        tail = th.ThresholdTail((f,))
        atom, identity = th.ThresholdAtom(f, 1), tail.identity_atom()
        return tail, atom, identity

    def test_existing_event_schema_and_factory_agree_on_complete_mock_dictionary(self):
        tail, atom, identity = self.fixture()
        # Deliberately a mock order distribution, not a scientific RTE evaluation.
        dist = SimpleNamespace(orders=(0, 2), unnormalized_order_weights=(3.0, 1.0),
                               exact_finite_distribution=4.0, dimensionless_step_time=-0.5)
        components = [self.api.RTEComponent(a.component_id, 0.5, 1.0, 1, is_identity=a.fragment is None,
                                          df_fragment_id="f" if a.fragment else None) for a in (atom, identity)]
        with patch.object(self.api.np, "eye", side_effect=AssertionError("no dense operator")), \
             patch.object(self.api.np, "load", side_effect=AssertionError("no saved input")):
            bound = th.bind_rte_event(tail, (atom, identity, atom), dist, 1, rte_api=self.api)
            reference = self.api._make_event((0, 1, 0), components, dist, 1)
        self.assertEqual(bound.event, reference)
        self.assertEqual(bound.conditional_atom_probability, F(1, 8))
        self.assertEqual((bound.event_probability_roundoff, bound.event_coefficient_roundoff), (0, 0))
        self.assertIsNone(bound.event.application_sequence[-1].diagonal_pauli_support)

    def test_canonical_order_law_is_delegated_with_roundoff_record(self):
        fragment = th.ThresholdFragment("f", th.ThresholdGrid.from_df_square((1,), F(1, 3)))
        tail = th.ThresholdTail((fragment,))
        factory = unittest.mock.Mock(return_value="canonical-order-object")
        params = tail.order_parameters(F(2, 7), 2, rte_api=SimpleNamespace(finite_rte_distribution=factory))
        self.assertEqual(params.distribution, "canonical-order-object")
        self.assertEqual(params.exact_dimensionless_time, F(2, 21))
        self.assertEqual(params.time_conversion_error, abs(F.from_float(float(F(2, 21)))-F(2, 21)))
        factory.assert_called_once_with(float(F(2, 21)), 2)

    def test_real_canonical_zero_time_interface_requires_no_scientific_evaluation(self):
        tail, atom, _ = self.fixture()
        params = tail.order_parameters(0, 0, rte_api=self.api)
        self.assertEqual(params.distribution.exact_finite_distribution, 1)
        bound = th.bind_rte_event(tail, (atom,), params.distribution, 0, rte_api=self.api)
        self.assertEqual(bound.event.rotation_angle, 0)

    def test_event_emitter_preserves_phase_product_order_and_shared_control(self):
        g = th.ThresholdGrid.from_df_square((1, -2), 1)
        fragment = th.ThresholdFragment("f", g, "swap", "mock-swap")
        tail = th.ThresholdTail((fragment,))
        a, b, identity = th.ThresholdAtom(fragment, 1), th.ThresholdAtom(fragment, g.extent), tail.identity_atom()
        dist = SimpleNamespace(orders=(2,), unnormalized_order_weights=(1.0,), exact_finite_distribution=1.0,
                               dimensionless_step_time=0.5)
        bound = th.bind_rte_event(tail, (a, identity, b), dist, 0, rte_api=self.api)
        plans = [th.reflection_plan(x, num_modes=2, angle=bound.event.rotation_angle if i == 2 else None)
                 for i, x in enumerate((identity, b, a))]
        size = max(p.num_qubits for p in plans)
        for bits in occupations(2):
            for control in (0, 1):
                initial = list(bits)+[0]*(size-2)
                initial[-1] = control
                sink = BitPhaseSink(initial)
                self.assertEqual(th.emit_rte_event(bound, sink, num_modes=2), size)
                logical = tuple(reversed(bits))
                r_a = 2*int(g.value_integer(logical) >= a.threshold)-1
                r_b = 2*int(g.value_integer(logical) >= b.threshold)-1
                self.assertEqual(sink.bits, initial)
                self.assertEqual(sink.pi, F(control*(int(r_b == -1)+1)) % 2)
                self.assertEqual(sink.radians, -F.from_float(bound.event.rotation_angle)*r_a*control)

    def test_consecutive_thresholds_share_value_and_gaussian_without_changing_phase(self):
        g = th.ThresholdGrid.from_df_square((1, -2), 1)
        fragment = th.ThresholdFragment("f", g, "swap", "mock-swap")
        tail = th.ThresholdTail((fragment,))
        a, b = th.ThresholdAtom(fragment, 1), th.ThresholdAtom(fragment, g.extent)
        dist = SimpleNamespace(orders=(2,), unnormalized_order_weights=(1.0,),
                               exact_finite_distribution=1.0, dimensionless_step_time=-0.25)
        bound = th.bind_rte_event(tail, (a, b, a), dist, 0, rte_api=self.api)
        shared = th.rte_event_plans(bound, num_modes=2)
        self.assertEqual(len(shared), 1)
        self.assertEqual(shared[0].structural_counts()["basis_calls"], 2)
        naive = [th.reflection_plan(atom, angle=bound.event.rotation_angle if i == 2 else None)
                 for i, atom in enumerate((b, a, a))]
        self.assertLess(len(shared[0].instructions), sum(len(p.instructions) for p in naive))
        for bits in occupations(2):
            for control in (0, 1):
                initial = list(bits)+[0]*(shared[0].num_qubits-2)
                initial[-1] = control
                left, right = BitPhaseSink(initial), BitPhaseSink(initial)
                shared[0].emit(left)
                for plan in naive:
                    plan.emit(right)
                self.assertEqual((left.bits, left.pi, left.radians), (right.bits, right.pi, right.radians))

    def test_negative_identity_product_sign_is_applied_exactly_once(self):
        g = th.ThresholdGrid.from_df_square((1,), -2)
        fragment = th.ThresholdFragment("negative", g)
        tail = th.ThresholdTail((fragment,))
        atom, identity = th.ThresholdAtom(fragment, 1), tail.identity_atom()
        dist = SimpleNamespace(orders=(2,), unnormalized_order_weights=(1.0,),
                               exact_finite_distribution=1.0, dimensionless_step_time=0.25)
        bound = th.bind_rte_event(tail, (atom, identity, atom), dist, 0, rte_api=self.api)
        self.assertEqual(bound.event.product_sign_phase, -1)
        plans = th.rte_event_plans(bound, num_modes=1)
        size = max(p.num_qubits for p in plans)
        for bit in (0, 1):
            initial = [bit]+[0]*(size-1)
            initial[-1] = 1
            sink = BitPhaseSink(initial)
            th.emit_rte_event(bound, sink, num_modes=1)
            sign = 2*int(g.value_integer((bit,)) >= 1)-1
            # Negative identity and the order-two paired phase cancel.
            self.assertEqual(sink.pi, int(sign == -1))
            self.assertEqual(sink.radians, -F.from_float(bound.event.rotation_angle)*sign)
            self.assertEqual(sink.bits, initial)

    def test_float_interface_refuses_silent_underflow(self):
        fragment = th.ThresholdFragment("f", th.ThresholdGrid.from_df_square((1,), F(1, 2**2000)))
        with self.assertRaisesRegex(ValueError, "underflow"):
            th.ThresholdTail((fragment,)).order_parameters(1, 0, rte_api=SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
