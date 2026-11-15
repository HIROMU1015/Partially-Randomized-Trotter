"""Synthetic exact prices and pinned identity metadata; no real 60-price evaluation."""
from copy import deepcopy
from fractions import Fraction
import unittest
from unittest.mock import patch

import freeze_pf_closure_prices as C


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(C.V,'predict',side_effect=AssertionError('No real prediction in metadata tests')):
            cls.rows,cls.models,cls.assumptions=C.targets(C.ROOT)

    def artificial_models(self):
        return {key:dict(features=['one','q'],coefficients=dict(one=dict(numerator=i+1,denominator=1),
                q=dict(numerator=3,denominator=1)),status='ARTIFICIAL_TEST_MODEL')
                for i,key in enumerate(sorted(self.models))}

    def test_fixed_60_axes_12_groups_and_original_locators(self):
        self.assertEqual(len(self.rows),60)
        self.assertEqual(len({C.key(r) for r in self.rows}),60)
        self.assertEqual(len({r['configuration_id'] for r in self.rows}),6)
        self.assertEqual({r['m'] for r in self.rows},set(C.ROUNDS))
        self.assertTrue(all(r['frozen_model']['commit']==C.FROZEN['commit'] for r in self.rows))

    def test_96_direct_metadata_includes_four_original_saved_m5_axes(self):
        inventory,paths=C.direct_inventory(C.ROOT)
        self.assertEqual(len(inventory),96)
        saved=[r for r in inventory.values() if r.get('original_cohort')=='EXPLORATION']
        self.assertEqual(len(saved),4)
        self.assertEqual({r['physical_id'] for r in saved},{'0cd1527c80f7b9841a83','13d1cd4869951cdf1e2c'})
        self.assertFalse(set(inventory)&{C.key(r) for r in self.rows})

    def test_synthetic_exact_integer_prices_keep_xy_and_conditional_semantics(self):
        models=self.artificial_models()
        with patch.object(C.V,'fit',side_effect=AssertionError('No fit')),patch.object(C.V,'validate',side_effect=AssertionError('No validation')):
            out=C.calculate(self.rows,models,self.assumptions,'ARTIFICIAL_SOURCE')
        self.assertEqual(out['prediction_calls'],60)
        for row in out['rows']:
            model=models[row['model_group_id']]
            self.assertEqual(row['compiled_RZ'],model['coefficients']['one']['numerator']+3*row['q'])
            self.assertEqual(row['mean']['rz_count'],row['compiled_RZ'])
            self.assertEqual(row['classification'],'MODEL_PREDICTION')
            self.assertFalse(row['direct_observation']);self.assertTrue(row['conditional_use_adopted'])
            self.assertFalse(row['model']['qualified_for_candidate']);self.assertFalse(row['two_percent_guarantee'])
            self.assertIsNone(row['error_bound'])
        x=next(r for r in out['rows'] if r['axis']=='X')
        y=next(r for r in out['rows'] if r['axis']=='Y' and r['physical_id']==x['physical_id'])
        self.assertNotEqual(x['compiled_RZ'],y['compiled_RZ'])
        self.assertIsNone(out['N']);self.assertIsNone(out['G'])

    def test_missing_duplicate_and_scope_change_do_not_predict(self):
        for change in ('missing','duplicate','wrong_round'):
            rows=deepcopy(self.rows)
            if change=='missing':rows.pop()
            elif change=='duplicate':rows[1]=deepcopy(rows[0])
            else:rows[0]['m']=9
            with self.subTest(change=change),patch.object(C.V,'predict',side_effect=AssertionError('No prediction')),self.assertRaises(ValueError):
                C.calculate(rows,self.artificial_models(),[], 'ARTIFICIAL_SOURCE')

    def test_fractional_or_negative_rz_is_rejected_without_rounding(self):
        with patch.object(C.V,'predict',return_value=Fraction(1,2)),self.assertRaisesRegex(ValueError,'integer'):
            C.calculate(self.rows,self.artificial_models(),[],'ARTIFICIAL_SOURCE')
        with patch.object(C.V,'predict',return_value=Fraction(-1)),self.assertRaisesRegex(ValueError,'integer'):
            C.calculate(self.rows,self.artificial_models(),[],'ARTIFICIAL_SOURCE')

    def test_frozen_axis_or_actual_width_mismatch_rejected(self):
        original=C.G.evidence
        for condition in ('model_axis','physical_width'):
            def read(root,ref):
                data=original(root,ref)
                if ref==C.FROZEN and condition=='model_axis':
                    data['models'][self.rows[0]['model_group_id']]['domain']['axis']='OTHER'
                if ref==self.rows[0]['alignment'] and condition=='physical_width':
                    data['physical_candidate']['delta']=123
                return data
            with self.subTest(condition=condition),patch.object(C.G,'evidence',side_effect=read),self.assertRaises(ValueError):
                C.targets(C.ROOT)


if __name__=='__main__':unittest.main(verbosity=2)
