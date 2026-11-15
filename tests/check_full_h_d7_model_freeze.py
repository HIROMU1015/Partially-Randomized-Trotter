"""D7 synthetic prices and original metadata gate; no real fits in tests."""
from copy import deepcopy
from fractions import Fraction
import json
from unittest.mock import patch
import unittest

import freeze_full_h_d7_models as F


def artificial():
    data=json.loads((F.ROOT/F.BASE/'INPUTS.json').read_text());D,api=F.metadata_api(F.ROOT)
    plan=data['plan'];M=D.c_model(F.ROOT);records=[]
    for cell in plan['cells']:
        if cell['role']!='CALIBRATION':continue
        for axis in ('X','Y'):
            coeff=(10,2,3) if axis=='X' else (20,3,-1)
            features=D.features(cell,plan['feature_map']);values=list(features.values())
            y=sum(c*v for c,v in zip(coeff,values))
            r=M.record(cell_id=cell['cell_id'],role='CALIBRATION',route=cell['route'],axis=axis,features=features,rz_count=y,
                kind=cell['kind'],source=dict(commit='ARTIFICIAL_ONLY',cohort=cell['cohort'],step_seeds=api['seed_table'](plan,cell['cell_id'])),
                implementation=cell['implementation'],profile=cell['profile'],n_paths=cell['n_paths'],law=cell['law'])
            records.append(dict(status='ACCEPTED_CALIBRATION_RECORD',record=r,locator=dict(fixture=True,cell=cell['cell_id'],axis=axis)))
    return dict(plan=plan,calibration=records,holdouts=data['holdouts'],paired=data['paired'])


class Tests(unittest.TestCase):
    def test_exact_overdetermined_two_axis_fit_and_four_predictions(self):
        d=F.calculate(F.ROOT,artificial(),'ARTIFICIAL_ONLY')
        self.assertEqual((d['fit_calls'],d['prediction_calls']),(2,4))
        for axis,values in (('X',(10,2,3)),('Y',(20,3,-1))):
            model=d['models'][axis]
            self.assertEqual(model['method'],'EXACT_NORMAL_EQUATION_LEAST_SQUARES')
            self.assertEqual([Fraction(v['numerator'],v['denominator']) for v in model['coefficients'].values()],list(values))
            self.assertEqual(model['rank'],3);self.assertFalse(model['qualified_for_candidate'])
        self.assertEqual({(p['cell_id'],p['axis']) for p in d['predictions']},{(c,a) for c in ('M5_ROUND','M6_ROUND') for a in ('X','Y')})

    def test_missing_axis_record_keeps_that_fit_held(self):
        data=artificial();next(r for r in data['calibration'] if r['record']['axis']=='Y')['status']='HELD'
        d=F.calculate(F.ROOT,data,'ARTIFICIAL_ONLY')
        self.assertEqual(d['held_axes'],['Y']);self.assertEqual((d['fit_calls'],d['prediction_calls']),(1,2))

    def test_original_D1_metadata_gate_reads_one_model_per_axis(self):
        data=artificial();d=F.calculate(F.ROOT,data,'ARTIFICIAL_ONLY');_,api=F.metadata_api(F.ROOT)
        ref=dict(commit='f'*40,path='ARTIFICIAL_MODELS.json');original=api['P1'].blob
        with patch.object(api['P1'],'blob',side_effect=lambda c,p:json.dumps(d).encode() if (c,p)==(ref['commit'],ref['path']) else original(c,p)):
            result=api['frozen_model_gate'](data['plan'],ref)
            self.assertEqual(result['status'],'FROZEN_METADATA_CHECKED_NOT_MODEL_QUALIFICATION')
            d['models']['X']['features']=['wrong']
            with self.assertRaisesRegex(ValueError,'domain/features'):api['frozen_model_gate'](data['plan'],ref)

    def test_path_counts_do_not_create_fit_weights(self):
        D,_=F.metadata_api(F.ROOT);M=D.c_model(F.ROOT);data=artificial()
        rows=[r['record'] for r in data['calibration'] if r['record']['axis']=='X'];rows[-1]['rz_count']['numerator']+=7
        kwargs=dict(features=list(D.FEATURE_MAPS['FULL_H_GROUP_V1'][0]),design_source=data['plan']['design_source'])
        a=M.fit(rows,**kwargs);other=deepcopy(rows)
        for r in other:r['n_paths']=99
        b=M.fit(other,**kwargs);self.assertEqual(a['coefficients'],b['coefficients'])

    def test_holdout_observations_are_rejected_by_evidence_reader(self):
        with self.assertRaisesRegex(ValueError,'HOLDOUT_OBSERVATION'):
            F.checked_evidence(F.ROOT,dict(commit='mock',path='run/M5_ROUND/PRICES.json'))


if __name__=='__main__':unittest.main()
