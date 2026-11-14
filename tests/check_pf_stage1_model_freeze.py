"""Synthetic calibration; real fixed inputs are checked as metadata only."""
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('stage1_freeze',ROOT/'scripts/freeze_pf_stage1_models.py')
F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)


def artificial():
    groups={}
    for route in ('FULL_PF','TRUNCATED_PF'):
        for formula,width in (('2nd','16/7'),('4th','16/3'),('4th(m5_best)','16')):
            for axis in ('X','Y'):
                group=[route,formula,width,axis];key='|'.join(group);n=len(groups)+1
                records=[F.V.record(cell_id='%s:%s:%s:m%d'%(*group[:3],m),role='CALIBRATION',route=route,axis=axis,
                    features=dict(one=1,q=q),rz_count=Fraction(n,7)+Fraction(13*n,3)*q,
                    kind='DIRECT_COMPILED_DETERMINISTIC',source=dict(fixture='ARTIFICIAL',cell=m,group=key),
                    implementation=dict(lowering_id='ART_NATIVE'),profile=dict(fixture='ART_PROFILE'))
                    for m,q in zip((4,5),F.EXPECTED_Q[formula])]
                groups[key]=dict(model_group=group,records=records,inputs=[dict(fixture='ART_LOCATOR',m=m) for m in (4,5)],
                    C_inputs_source=dict(fixture='ART_CAL',group=key),holdout_cell_source=dict(fixture='ART_PLAN',group=key),
                    holdout_cell=dict(configuration_id='ART_'+route+formula,m=6,axis=axis,
                        physical_candidate=dict(id='ART_'+key,q=F.EXPECTED_Q[formula][2]),
                        features=dict(one=1,q=F.EXPECTED_Q[formula][2])))
    return dict(groups=groups,blocked={},comparison_policy=dict(fixture='ART_POLICY',holdout=dict(scale='observed',value='1/50')))


class Synthetic(unittest.TestCase):
    def test_exact_fit_and_m6_predictions_twelve_separate_models(self):
        p=artificial()
        with patch.object(F.V,'fit',wraps=F.V.fit) as fit,patch.object(F.V,'predict',wraps=F.V.predict) as predict,\
             patch.object(F.V,'validate',side_effect=AssertionError('Holdout forbidden')):
            result=F.calibrate(p,'ART_SOURCE')
        self.assertEqual((fit.call_count,predict.call_count),(12,12))
        for n,(key,m) in enumerate(sorted(result['models'].items()),1):
            g=p['groups'][key];index=list(p['groups']).index(key)+1
            self.assertEqual(Fraction(**m['coefficients']['one']),Fraction(index,7))
            self.assertEqual(Fraction(**m['coefficients']['q']),Fraction(13*index,3))
            self.assertEqual(Fraction(**m['m6_prediction']['predicted_RZ']),
                             Fraction(index,7)+Fraction(13*index,3)*g['holdout_cell']['features']['q'])
            self.assertEqual(m['model_group_id'],key)
            self.assertEqual(m['status'],'CALIBRATED_UNVALIDATED')
            self.assertIsNone(m['m6_prediction']['observed_RZ'])
            self.assertIsNone(m['m6_prediction']['holdout_pass'])
            self.assertFalse(m['qualified_for_candidate'])
            self.assertIsNone(m['extrapolation_scope'])
        self.assertIsNone(result['campaign_pass']);self.assertIsNone(result['G'])

    def test_wrong_axis_formula_q_or_role_rejected(self):
        for field in ('axis','role','q','cell_id','source_profile'):
            g=next(iter(artificial()['groups'].values()));records=deepcopy(g['records'])
            if field=='q':records[1]['features']['q']=dict(numerator=15,denominator=1)
            elif field=='source_profile':records[1]['profile']=dict(fixture='FOREIGN')
            elif field=='cell_id':records[1]['cell_id']='FULL_PF:4th:16/3:m5'
            else:records[1][field]='Y' if field=='axis' else 'HOLDOUT'
            with self.subTest(field=field),self.assertRaises(ValueError):F.check_records(records,g['model_group'])

    def test_blocked_group_not_filled_from_other_axis(self):
        p=artificial();key=next(iter(p['groups']));del p['groups'][key];p['blocked'][key]=dict(reason='missing fixed record')
        result=F.calibrate(p,'ART_SOURCE')
        self.assertEqual(result['fit_calls'],11);self.assertNotIn(key,result['models'])
        self.assertEqual(result['status'],'PARTIAL_FREEZE_UNRESOLVED_INPUT_GROUPS')

    def test_holdout_path_refused_without_read(self):
        with patch.object(F.G,'evidence',side_effect=AssertionError('Must not open')):
            with self.assertRaises(ValueError):F.read(ROOT,dict(commit='a'*40,path='docs/holdout/run01/PRICES.json'))

    def test_unmodified_D_consumer_accepts_synthetic_frozen_schema(self):
        p=artificial();frozen=F.calibrate(p,'ART_SOURCE');raw=json.dumps(frozen).encode()
        original=F.M.C.blob
        def supplied(root,commit,path):
            return raw if commit=='a'*40 else original(root,commit,path)
        with patch.object(F.M.C,'blob',side_effect=supplied):
            received=F.d_consumer(ROOT,dict(commit='a'*40,path='ART_FROZEN.json'),p['groups'])
        self.assertEqual(len(received['models']),12)
        altered=deepcopy(frozen);first=next(iter(altered['models'].values()));first['status']='HOLDOUT_WITHIN_DECLARED_TOLERANCE'
        raw=json.dumps(altered).encode()
        with patch.object(F.M.C,'blob',side_effect=supplied),self.assertRaises(ValueError):
            F.d_consumer(ROOT,dict(commit='a'*40,path='ART_FROZEN.json'),p['groups'])


class FixedMetadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(F.V,'fit',side_effect=AssertionError('Real fit is separate execution')),\
             patch.object(F.V,'predict',side_effect=AssertionError('No real prediction in tests')):
            cls.data=F.prepare(ROOT)

    def test_24_accepted_prices_and_original_saved_cohort(self):
        self.assertEqual(self.data['blocked'],{})
        self.assertEqual(len(self.data['groups']),12)
        inputs=[i for g in self.data['groups'].values() for i in g['inputs']]
        self.assertEqual(len(inputs),24)
        saved=[i for i in inputs if i['origin']=='C_ADMITTED_SAVED_RECEIPT_NOT_RECOMPILED']
        self.assertEqual(len(saved),4)
        self.assertTrue(all(i['cohort']=='EXPLORATION' and i['m']==4 for i in saved))
        self.assertTrue(all(i['original_price_source']['commit']=='155529e20a8fac39e00786c40b57830b71203c4f'
                            and i['original_price_source']['key']==i['physical_candidate']['id']
                            and len(i['original_price_source']['original_trial']['sha256'])==64 for i in saved))

    def test_holdout_plan_is_identity_only_and_policy_fixed(self):
        for g in self.data['groups'].values():
            h=g['holdout_cell'];self.assertEqual(h['m'],6);self.assertNotIn('observed_RZ',h)
            self.assertEqual(h['physical_candidate']['q'],F.EXPECTED_Q[g['model_group'][1]][2])
        h=self.data['comparison_policy']['holdout']
        self.assertEqual(h['value'],dict(numerator=1,denominator=50));self.assertEqual(h['scale'],'observed')
        self.assertEqual(h['success'],'ALL_12_PASS')


if __name__=='__main__':unittest.main(verbosity=2)
