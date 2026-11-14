"""Synthetic prediction checks and fixed metadata only; never real m9 observations."""
from copy import deepcopy
from fractions import Fraction
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('m9_freeze',ROOT/'scripts/freeze_pf_m9_predictions.py')
F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(F.V,'predict',side_effect=AssertionError('Real prediction is separate execution')):
            cls.rows=F.targets(ROOT)

    def synthetic(self):
        models={}
        for i,r in enumerate(self.rows):
            models[r['model_group_id']]=dict(features=['one','q'],
                coefficients=dict(one=dict(numerator=i+1,denominator=7),q=dict(numerator=13,denominator=3)))
        return models

    def test_fixed_twelve_points_match_source_plan_and_coefficients(self):
        self.assertEqual(len(self.rows),12)
        self.assertEqual(len({(r['physical_id'],r['axis']) for r in self.rows}),12)
        self.assertEqual({r['m'] for r in self.rows},{9});self.assertEqual({r['T'] for r in self.rows},{51.2})
        self.assertEqual({r['q'] for r in self.rows},{224,96,32})
        self.assertTrue(all(r['coefficient_source']['commit']=='12372fa6b53c878dc5dd124e036572ba0cd82061' for r in self.rows))

    def test_exact_synthetic_predictions_no_fit_or_judgement(self):
        with patch.object(F.V,'fit',side_effect=AssertionError('No refit')),\
             patch.object(F.V,'validate',side_effect=AssertionError('No holdout judgement')),\
             patch.object(F.V,'predict',wraps=F.V.predict) as call:
            result=F.predict(self.rows,self.synthetic(),dict(held_physical_ids=[]),'ART_SOURCE')
        self.assertEqual(call.call_count,12)
        for i,r in enumerate(result['rows']):
            self.assertEqual(Fraction(**r['predicted_RZ']),Fraction(i+1,7)+Fraction(13,3)*r['q'])
            self.assertIsNone(r['observed_RZ']);self.assertIsNone(r['decision'])
            self.assertFalse(r['qualified_for_candidate'])
        self.assertEqual(result['fit_calls'],0);self.assertIsNone(result['G']);self.assertIsNone(result['campaign_pass'])

    def test_unknown_independence_holds_both_axes_without_prediction(self):
        pid=self.rows[0]['physical_id']
        with patch.object(F.V,'predict',wraps=F.V.predict) as call:
            result=F.predict(self.rows,self.synthetic(),dict(held_physical_ids=[pid]),'ART_SOURCE')
        self.assertEqual(call.call_count,10);self.assertEqual(result['held_predictions'],2)
        self.assertTrue(all(r['predicted_RZ'] is None for r in result['rows'] if r['physical_id']==pid))

    def test_all_unresolved_has_no_prediction_calls(self):
        with patch.object(F.V,'predict',side_effect=AssertionError('No prediction before resolving independence')):
            result=F.predict(self.rows,self.synthetic(),dict(held_physical_ids=[r['physical_id'] for r in self.rows]),'ART_SOURCE')
        self.assertEqual(result['prediction_calls'],0);self.assertEqual(result['held_predictions'],12)

    def test_scoped_policy_observed_denominator_and_exact_threshold(self):
        p=F.POLICY;self.assertEqual(p['rounds'],[9]);self.assertEqual(p['axes'],['X','Y'])
        self.assertEqual(p['value'],dict(numerator=1,denominator=50));self.assertEqual(p['scale'],'observed')
        self.assertEqual(p['campaign_success'],'ALL_12_INDIVIDUAL_PASS')
        self.assertEqual(p['required_individual_results'],12)
        self.assertFalse(p['automatic_refit']);self.assertFalse(p['other_round_qualification'])
        # Artificial boundary only: exact inclusive comparison, no real observed prices.
        self.assertTrue(Fraction(2,100)<=Fraction(1,50));self.assertFalse(Fraction(201,10000)<=Fraction(1,50))

    def test_wrong_physical_source_or_frozen_group_rejected(self):
        original=F.G.evidence
        for change in ('physical','q','model_axis','actual_width'):
            def supplied(root,ref):
                data=original(root,ref)
                if ref==F.ALIGN:
                    row=data['rows'][36]
                    if change=='physical':row['physical_id']='OTHER'
                    if change=='q':row['physical_candidate']['q']=225
                    if change=='actual_width':row['physical_candidate']['delta']=.3
                if ref==F.MODELS and change=='model_axis':data['models']['FULL_PF|2nd|16/7|X']['domain']['axis']='Y'
                return data
            with self.subTest(change=change),patch.object(F.G,'evidence',side_effect=supplied),self.assertRaises(ValueError):
                F.targets(ROOT)

    def test_prior_evidence_files_not_rewritten(self):
        for sha,path in [('12372fa6b53c878dc5dd124e036572ba0cd82061',F.MODELS['path']),
                         ('3b1301738507d612694bf865d904b9376e038f69','docs/research/pf_m3_saved_price_admission_20261115/ADMISSION.json')]:
            self.assertEqual(F.C.blob(ROOT,sha,path),F.C.blob(ROOT,'HEAD',path))


if __name__=='__main__':unittest.main(verbosity=2)
