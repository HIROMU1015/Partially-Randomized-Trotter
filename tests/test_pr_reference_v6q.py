"""Artificial scalar/duck-circuit fixtures only: no Qiskit, H4, build or sim."""
import copy
from decimal import Decimal
from fractions import Fraction as F
import importlib.util
import math
from pathlib import Path
import struct
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

PATH = Path(__file__).resolve().parents[1] / 'src/trottertracks/pr_reference_v6q.py'
spec = importlib.util.spec_from_file_location('v6q_prepared', PATH)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def instruction(name, params=(), wires=None):
    width = 2 if name in ('rzz', 'xx_plus_yy') else 1
    return NS(operation=NS(name=name, params=list(params), num_qubits=width, condition=None),
              qubits=tuple(range(width)) if wires is None else tuple(wires), clbits=())


def lane(phase=0.0, *ops):
    return NS(global_phase=phase, data=list(ops), num_qubits=2)


def independent_exact(f, g):
    """Integer common-grid oracle, independent of provider's Fraction walk."""
    terms = []
    for qc in (f, g):
        numerator, denominator = qc.global_phase.as_integer_ratio()
        terms.append((numerator, denominator))
        for inst in qc.data:
            if inst.operation.name in ('rz', 'rzz'):
                n, d = inst.operation.params[0].as_integer_ratio()
                terms.append((-n, 2*d))
    denominator = max(d for n, d in terms)
    assert all(denominator % d == 0 for n, d in terms)
    return F(sum(n*(denominator//d) for n, d in terms), denominator)


class ExactBinaryPhaseTests(unittest.TestCase):
    def test_large_signed_cancellation(self):
        f = lane(2.0**60, instruction('rz', (-2.0,)))
        g = lane(0.0, instruction('rz', (2.0**61,)))
        self.assertEqual(independent_exact(f, g), 1)
        self.assertEqual(p.exact_binary_phase(f, g), 1.0)

    def test_zero_and_signed_zero(self):
        f = lane(1.25, instruction('rzz', (2.5,)))
        g = lane(-0.0, instruction('rz', (0.0,)))
        self.assertEqual(independent_exact(f, g), 0)
        self.assertEqual(struct.pack('>d', p.exact_binary_phase(f, g)), struct.pack('>d', 0.0))

    def test_normalized_globals_and_all_supported_families(self):
        f = lane(1.25, instruction('rz', (.5,)), instruction('xx_plus_yy', (.3, 0)),
                 instruction('p', (.7,)), instruction('z'), instruction('id'))
        g = lane(5.0, instruction('rzz', (.25,)))
        expected = independent_exact(f, g)
        self.assertEqual(p.exact_binary_phase(f, g), float(expected))

    def test_actual_binary_half_not_decimal_reinterpretation(self):
        f, g = lane(0.0, instruction('rz', (.1,))), lane()
        target = independent_exact(f, g)
        self.assertNotEqual(target, F(-1, 20))
        self.assertEqual(F.from_float(p.exact_binary_phase(f, g)), target)

    def test_two_lane_rounding_would_lose_the_residual(self):
        f = lane(1.0, instruction('rz', (-2.0**-53,)))
        g = lane(0.0, instruction('rz', (2.0,)))
        self.assertEqual(independent_exact(f, g), F(2)**-54)
        separate = float(independent_exact(f, lane())) + float(independent_exact(g, lane()))
        self.assertEqual(separate, 0.0)
        self.assertEqual(p.exact_binary_phase(f, g), 2.0**-54)

    def test_combined_fraction_is_rounded_once(self):
        f, g = lane(1.0, instruction('rz', (.2,))), lane(.5)
        with patch.object(p, '_round_once', wraps=p._round_once) as rounding:
            p.exact_binary_phase(f, g)
        rounding.assert_called_once_with(independent_exact(f, g))
        self.assertIsInstance(rounding.call_args.args[0], F)

    def test_wire_gate_order_and_inputs_are_unchanged(self):
        f = lane(.25, instruction('rzz', (.5,), (1, 0)), instruction('rz', (.25,), (1,)))
        g = lane(.125, instruction('xx_plus_yy', (.7, .2), (1, 0)))
        before = copy.deepcopy((f, g))
        operation_ids = [id(x.operation) for c in (f, g) for x in c.data]
        p.exact_binary_phase(f, g)
        self.assertEqual((f, g), before)
        self.assertEqual([id(x.operation) for c in (f, g) for x in c.data], operation_ids)

    def test_unsupported_gate_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unqualified'):
            p.exact_binary_phase(lane(0.0, instruction('rx', (.1,))), lane())

    def test_nonfinite_and_unsupported_parameters_rejected(self):
        for value in (float('nan'), float('inf'), -float('inf'), Decimal('.1'),
                      F(1, 10), '.1', .1j, object(), True):
            with self.subTest(value=repr(value)), self.assertRaises((TypeError, ValueError)):
                p.exact_binary_phase(lane(0.0, instruction('rz', (value,))), lane())

    def test_integer_conversion_must_be_exact(self):
        self.assertEqual(p.exact_binary_phase(lane(0, instruction('rz', (2,))), lane(0)), -1.)
        with self.assertRaisesRegex(ValueError, 'would round'):
            p.exact_binary_phase(lane(2**53+1), lane())

    def test_float_subclass_uses_stored_binary_value(self):
        class MisleadingFloat(float):
            def __float__(self):
                return 99.0
        self.assertEqual(p.exact_binary_phase(lane(MisleadingFloat(.5)), lane()), .5)

    def test_shape_and_classical_condition_rejected(self):
        for defect in ('params', 'width', 'condition', 'clbits'):
            op = instruction('rz', (.5,))
            if defect == 'params': op.operation.params = []
            elif defect == 'width': op.operation.num_qubits = 2
            elif defect == 'condition': op.operation.condition = ('c', 1)
            else: op.clbits = ('c',)
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                p.exact_binary_phase(lane(0.0, op), lane())

    def test_overflow_and_nonzero_subnormal_output_rejected(self):
        with self.assertRaises(OverflowError):
            p.exact_binary_phase(lane(sys.float_info.max), lane(sys.float_info.max))
        with self.assertRaisesRegex(ArithmeticError, 'subnormal'):
            p.exact_binary_phase(lane(0.0, instruction('rz', (sys.float_info.min,))), lane())

    def test_known_positive_and_negative_ties_are_even(self):
        a = 1.0
        b = math.nextafter(a, math.inf)
        c = math.nextafter(b, math.inf)
        for target, expected in (((F.from_float(a)+F.from_float(b))/2, a),
                                 ((F.from_float(b)+F.from_float(c))/2, c)):
            for sign in (1, -1):
                with self.subTest(sign=sign, expected=expected):
                    self.assertEqual(p._round_once(sign*target), sign*expected)

    def test_midpoint_verifier_rejects_a_bad_runtime_result(self):
        target = F(1) + 3*F(2)**-54
        with self.assertRaisesRegex(ArithmeticError, 'nearest-even'):
            p._verify_nearest_even(target, 1.0)
        p._verify_nearest_even(target, math.nextafter(1.0, math.inf))

    def test_legacy_is_default_for_all_rounds(self):
        f, g = lane(.5), lane(.25)
        old = NS(vacuum_phase=Mock(side_effect=lambda c: c.global_phase))
        with patch.object(p, '_legacy_module', return_value=old), \
                patch.object(p, 'exact_binary_phase', side_effect=AssertionError('Unexpected opt-in')):
            for m in range(18):
                self.assertEqual(p.phase_for_round(f, g, round_m=m), .75)
        self.assertEqual(old.vacuum_phase.call_args_list[0].args, (f,))
        self.assertEqual(old.vacuum_phase.call_args_list[1].args, (g,))

    def test_opt_in_restricted_to_m17(self):
        for m in (None, 0, 16, 18, '17', True):
            with self.subTest(m=m), self.assertRaises(ValueError):
                p.phase_for_round(lane(), lane(), round_m=m, mode='exact_binary_m17')
        self.assertEqual(p.phase_for_round(lane(.5), lane(.25), round_m=17,
                                         mode='exact_binary_m17'), .75)

    def test_reference_wrapper_delegates_unchanged_gadget(self):
        f, g = lane(.5), lane(.25)
        old = NS(reference_core=Mock(return_value='mock-core'))
        with patch.object(p, '_legacy_module', return_value=old):
            self.assertEqual(p.reference_for_round(f, g, round_m=17,
                             mode='exact_binary_m17'), 'mock-core')
        old.reference_core.assert_called_once_with(f, g, .75)

    def test_lane_width_mismatch_rejected(self):
        f, g = lane(), lane()
        g.num_qubits = 3
        with self.assertRaises(ValueError):
            p.exact_binary_phase(f, g)

    def test_import_needs_no_quantum_or_scientific_data(self):
        before = set(sys.modules)
        again = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(again)
        added = set(sys.modules) - before
        self.assertFalse(any(x.startswith(('qiskit', 'numpy', 'trotterlib')) for x in added))


if __name__ == '__main__':
    unittest.main()
