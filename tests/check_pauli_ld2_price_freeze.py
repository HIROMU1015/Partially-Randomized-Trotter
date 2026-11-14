"""C2 glue tests: saved metadata only, synthetic prices for every fit."""
from copy import deepcopy
from fractions import Fraction
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('c2_freeze', ROOT / 'scripts/freeze_pauli_ld2_prices.py')
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)


class FreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared = F.prepare()  # No real-data fit/prediction.

    def synthetic(self):
        prepared = deepcopy(self.prepared)
        for axis, coeff in (('X', (3, 7, 2)), ('Y', (11, 13, 5))):
            for r in prepared['groups'][axis]:
                q = r['features']['pf_step_count']['numerator']
                s = r['features']['event_count']['numerator']
                r['rz_count'] = dict(numerator=coeff[0]+coeff[1]*q+coeff[2]*s, denominator=1)
        return prepared

    def test_fixed_receipt_mapping_and_read_boundary(self):
        p = self.prepared
        self.assertEqual([len(p['groups'][a]) for a in ('X', 'Y')], [5, 5])
        self.assertEqual(sum(len(i['individual_receipts']) for i in p['inputs']), 18)
        self.assertEqual(p['design_check']['identifiability']['rank'], 3)
        self.assertEqual(p['groups']['X'][4]['source']['step_seeds'], [[
            p['groups']['X'][0]['source']['step_seeds'][0][0],
            p['groups']['X'][0]['source']['step_seeds'][1][0]]])
        self.assertTrue(all(path.startswith(F.BASE + '/') or path in (F.META, F.TEMPLATES)
                            for _, path in F.READS))

    def test_synthetic_two_axis_fit_and_four_predictions(self):
        p = self.synthetic()
        out = F.calibrate(p, dict(commit='0'*40, synthetic=True))
        for axis, coeff in (('X', (3, 7, 2)), ('Y', (11, 13, 5))):
            model = out['models'][axis]
            self.assertEqual(model['features'], F.FEATURES)
            self.assertEqual([Fraction(**v) for v in model['coefficients'].values()], list(coeff))
            for pred in [r for r in out['predictions'] if r['axis'] == axis]:
                self.assertEqual(Fraction(**pred['predicted_RZ']),
                                 coeff[0]+coeff[1]*pred['q']+coeff[2]*pred['q']*pred['R'])
                self.assertIsNone(pred['observed_RZ'])
                self.assertEqual(pred['n_paths'], 2 if pred['m'] == 5 else 1)
        self.assertEqual((len(out['models']), len(out['predictions'])), (2, 4))
        self.assertIsNone(out['holdout_tolerance'])
        self.assertFalse(out['law_expectation'])

    def test_pooling_or_duplicate_cell_is_rejected(self):
        p = self.synthetic(); D = F.design_api()
        for records in (p['groups']['X'] + p['groups']['Y'],
                        p['groups']['X'] + [p['groups']['X'][0]]):
            with self.assertRaises(ValueError):
                D.fit_identified(ROOT, records, plan=p['plan'], axis='X')

    def test_wrong_axis_and_feature_are_rejected(self):
        for field in ('axis', 'features'):
            p = self.synthetic(); records = p['groups']['X']
            if field == 'axis':
                records[0]['axis'] = 'Y'
            else:
                records[0]['features']['event_count']['numerator'] += 1
            with self.assertRaises(ValueError):
                F.design_api().fit_identified(ROOT, records, plan=p['plan'], axis='X')

    def test_exact_d2_metadata_gate_on_synthetic_model(self):
        p = self.synthetic(); out = F.calibrate(p, dict(synthetic=True))
        reference = F.ref('0'*40, 'synthetic.json')
        self.assertEqual(F.check_d2_consumer(p, reference, frozen=out)['status'],
                         'FROZEN_METADATA_CHECKED_NOT_MODEL_QUALIFICATION')
        bad = deepcopy(out); bad['models']['Y']['domain']['axis'] = 'X'
        with self.assertRaises(ValueError):
            F.check_d2_consumer(p, reference, frozen=bad)
        bad = deepcopy(out); bad['models']['X']['calibration'][0]['source']['step_seeds'] = [[0]]
        with self.assertRaises(ValueError):
            F.check_d2_consumer(p, reference, frozen=bad)


if __name__ == '__main__':
    unittest.main(verbosity=2)
