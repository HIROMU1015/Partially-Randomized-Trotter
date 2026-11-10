"""Artificial S1 checks only; no real H4 data are loaded by this test file."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import h4_df10_price_v2 as v2


def pairs(method, q, bases, cohort):
    return [dict(method=method, q=q, T=.1*q, cohort=cohort,
                 candidate_id='synthetic', path_index=i, basis_groups=b,
                 order_2_count=0 if method == 'PR' else None,
                 baseline=[100, 100], actual=[100+2*q+3*(b or 0), 100-q+4*(b or 0)])
            for i, b in enumerate(bases)]


class S1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fit = staticmethod(v2.fixed_fit(ROOT))

    def test_multiresponse_coefficients_and_rank_failure(self):
        training = pairs('PR', 2, [0, 1, 2], 'SYNTHETIC_TRAIN')
        c = v2.calibrate(training, 'MqB', self.fit)
        np.testing.assert_allclose(c['coefficients'], [[2, -1], [3, 4]], atol=1e-12)
        row = v2.evaluate(pairs('PR', 4, [3], 'SYNTHETIC_CHECK'), 'KNOWN_V1_USED', 'MqB', c)
        np.testing.assert_allclose([r['prediction_rz'] for r in row], [117, 108], atol=1e-12)
        failed = v2.calibrate(pairs('PR', 2, [1, 1], 'SYNTHETIC'), 'MqB', self.fit)
        self.assertEqual(failed['status'], 'EVALUATION_UNAVAILABLE')
        self.assertEqual(failed['rank'], 1)

    def test_campaign_only_three_training_fits_and_no_known_price_leakage(self):
        training = pairs('PF', 2, [None], 'SYNTHETIC_TRAIN') + pairs('PR', 2, [0, 1, 2], 'SYNTHETIC_TRAIN')
        known = pairs('PF', 4, [None], 'SYNTHETIC_CHECK') + pairs('PR', 2, [2, 3], 'SYNTHETIC_CHECK')
        calls = []
        def fit(x, y, q):
            calls.append(len(x))
            return self.fit(x, y, q)
        result, rows = v2.campaign(training, known, fit)
        self.assertEqual(calls, [1, 3, 3])
        altered = deepcopy(known)
        for p in altered:
            p['actual'] = [9999, 8888]
        again, rows2 = v2.campaign(training, altered, self.fit)
        self.assertEqual(result['calibrations'], again['calibrations'])
        self.assertEqual([r['prediction_rz'] for r in rows], [r['prediction_rz'] for r in rows2])
        self.assertTrue(all(s['n_paths'] <= 3 for s in result['summaries']))

    def test_axis_pairing_is_by_identity_and_rejects_bad_binding(self):
        rows = []
        for i in (1, 0):
            for axis, value in [('sine', 17+i), ('cosine', 31+i)]:
                rows.append(dict(method='PR', cohort='SYNTHETIC', candidate_id='synthetic',
                    q=2, T=.2, path_index=i, axis=axis, compiled_rz=value,
                    prediction=dict(model=v2.V1_MODEL, structural_rz=10,
                                    features={'tail_basis_groups': i+1})))
        self.assertEqual([p['actual'] for p in v2.pair_rows(rows)], [[31, 17], [32, 18]])
        for bad in (rows[:-1], rows+[rows[0]]):
            with self.assertRaises(ValueError):
                v2.pair_rows(bad)
        bad = deepcopy(rows)
        bad[0]['prediction']['features']['tail_basis_groups'] = 99
        with self.assertRaises(ValueError):
            v2.pair_rows(bad)

    def test_zero_price_negative_and_nonfinite_diagnostics(self):
        p = pairs('PF', 2, [None], 'SYNTHETIC')
        p[0]['actual'] = [0, 1]
        c = dict(status='FITTED', coefficients=[[-100, -100]])
        rows = v2.evaluate(p, 'KNOWN_V1_USED', 'Mq', c)
        self.assertEqual(rows[0]['prediction_rz'], -100)
        self.assertIsNone(rows[0]['relative_error'])
        self.assertEqual(rows[0]['diagnostic'], 'NEGATIVE_PRICE')
        self.assertEqual(v2.summarize(rows)[0]['negative_predictions'], 1)
        c['coefficients'] = [[float('nan'), float('nan')]]
        rows = v2.evaluate(p, 'KNOWN_V1_USED', 'Mq', c)
        self.assertIsNone(v2.summarize(rows)[0]['metrics'])


if __name__ == '__main__':
    unittest.main()
