"""Artificial Fraction/sign/reuse checks only; never calculates H4 references."""
from copy import deepcopy
from fractions import Fraction
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('ld1_reference', Path(__file__).with_name('evaluate.py'))
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)


class Checks(unittest.TestCase):
    def test_saved_native_axis_alias_is_exact_and_swaps_rejected(self):
        row = dict(m=1, axis='X', q=2, R_outer=4)
        s = dict(m=1, axis='cosine', q_PF=2, R_outer=4)
        E.structural_identity(s, row)
        s['axis'] = 'sine'
        with self.assertRaisesRegex(ValueError, 'STRUCTURAL_IDENTITY'):
            E.structural_identity(s, row)

    def test_fractional_aggregate_and_negative_boundary(self):
        # Rounding each per-shot price would give 0; exact total is 1/3+2/3=1.
        rows = [dict(weighted_term=E.enc(Fraction(1, 3))), dict(weighted_term=E.enc(Fraction(2, 3)))]
        total, bounds = E.aggregate(rows, [('higher', 2), ('lower', 0)])
        self.assertEqual(total, 1)
        self.assertEqual([E.frac(r['D_PF']) for r in bounds], [1, -1])
        self.assertTrue(all(r['Delta'] is None and r['rank_status'].startswith('UNRESOLVED') for r in bounds))

    def test_fixed_predict_function_reused_with_signed_coefficients(self):
        predict = E.predictor()
        model = dict(features=['one', 'q', 'qR'], coefficients={
            'one': E.enc(Fraction(-1, 3)), 'q': E.enc(Fraction(1, 2)), 'qR': E.enc(Fraction(1, 6))})
        self.assertEqual(predict(model, dict(one=1, q=2, qR=4)), Fraction(4, 3))
        with self.assertRaisesRegex(ValueError, 'Missing feature'):
            predict(model, dict(one=1, q=2))

    def test_m4_reuses_full_forecast_without_double_residual_or_observation(self):
        row = dict(m=4, axis='X', physical_id='TOY', q=16, R_outer=32, structural={'toy': 'source'})
        s = dict(mean_S=1.25)
        p = dict(m=4, axis='X', physical_id='TOY', q=16, R_outer=32,
                 ideal_structural_source=row['structural'],
                 ideal_structural=dict(binary64_hex=(1.25).hex(), exact_binary64_value=E.enc(Fraction(5, 4))),
                 residual_prediction=E.enc(Fraction(1, 2)), predicted_RZ=E.enc(Fraction(7, 4)))
        model = dict(axis='X', m4_prediction=p)
        def forbidden(*a):
            raise AssertionError('No sampler, fit, predictor or observation needed')
        structural, residual, price, role = E.price(row, s, model, {}, forbidden)
        self.assertEqual((structural, residual, price), (Fraction(5, 4), Fraction(1, 2), Fraction(7, 4)))
        self.assertEqual(role, 'FROZEN_M4_FULL_PREDICTION')
        bad = deepcopy(model); bad['m4_prediction']['predicted_RZ'] = E.enc(Fraction(9, 4))
        with self.assertRaisesRegex(ValueError, 'M4_FROZEN_DECOMPOSITION'):
            E.price(row, s, bad, {}, forbidden)

    def test_axis_source_and_negative_reference_rejected(self):
        row = dict(m=1, axis='Y', physical_id='TOY', q=2, R_outer=4)
        with self.assertRaisesRegex(ValueError, 'MODEL_AXIS'):
            E.price(row, dict(mean_S=1.25), dict(axis='X'), {}, lambda *a: 0)
        with self.assertRaisesRegex(ValueError, 'NEGATIVE_REFERENCE_PRICE'):
            E.price(row, dict(mean_S=1.25), dict(axis='Y'), {}, lambda *a: Fraction(-2))

    def test_inputs_roundtrip_and_fixed_scope(self):
        import json
        data = json.loads(json.dumps(E.inputs()))
        self.assertEqual(data, E.inputs())
        data['shots_per_axis'][0] = 0
        # Source-reference mismatch is rejected before any saved real data read.
        with patch.object(E, 'selected', side_effect=AssertionError('Must reject before reads')):
            with self.assertRaisesRegex(ValueError, 'FIXED_INPUT_REFERENCE_MISMATCH'):
                E.check_inputs(data)


if __name__ == '__main__':
    unittest.main()
