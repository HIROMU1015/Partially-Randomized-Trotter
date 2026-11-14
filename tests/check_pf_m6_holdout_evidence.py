"""Artificial exact-comparison and identity tests. No H4 inputs, observed prices or fit."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pf_m6_holdout_evidence as H

def fixture():
    rows=[];models={}
    for route in H.ROUTES:
        for formula,width,q in [('2nd','16/7',28),('4th','16/3',12),('4th(m5_best)','16',4)]:
            for axis in ('X','Y'):
                group=[route,formula,width,axis];key=H.S.key(group)
                row=dict(model_group=group,m=6,formula=formula,width_over_h=width,route=route,axis=axis,
                    configuration_id='ARTIFICIAL_'+route+formula,physical_candidate=dict(id='TOY_'+route+formula,q=q,T=6.4,delta=6.4/q),
                    features=dict(one=1,q=q));rows.append(row)
                models[key]=dict(model_group_id=key,model_group=group,formula='C(q)=beta_one+beta_q*q',status='CALIBRATED_UNVALIDATED',
                    qualified_for_candidate=False,domain=dict(route=route,axis=axis),m6_prediction=dict(
                    configuration_id=row['configuration_id'],physical_candidate=deepcopy(row['physical_candidate']),axis=axis,q=q,
                    features=row['features'],m=6,observed_RZ=None,holdout_pass=None,independent_holdout_status='NOT_EVALUATED',
                    cell_source=dict(H.S.P7_CELL_PLAN,selector=[len(rows)-1]),predicted_RZ=dict(numerator=102,denominator=1)))
    return rows,models

class Tests(unittest.TestCase):
    def test_exact_boundary_observed_denominator_and_FAIL(self):
        row=dict(predicted_RZ=dict(numerator=102,denominator=1))
        r=H.judge_row(row,dict(observed_RZ=100));self.assertEqual((r['decision'],r['relative_error']),('PASS',dict(numerator=1,denominator=50)))
        row['predicted_RZ']=dict(numerator=103,denominator=1)
        self.assertEqual(H.judge_row(row,dict(observed_RZ=100))['decision'],'FAIL')
    def test_rounding_cannot_turn_FAIL_into_PASS(self):
        row=dict(predicted_RZ=dict(numerator=10200000000000000001,denominator=100000000000000000))
        self.assertEqual(float(H.exact(row['predicted_RZ'])),102.)
        self.assertEqual(H.judge_row(row,dict(observed_RZ=100))['decision'],'FAIL')
    def test_missing_invalid_or_incompatible_receipt_is_UNKNOWN(self):
        row=dict(predicted_RZ=dict(numerator=100,denominator=1),axis='X')
        self.assertEqual(H.judge_row(row)['decision'],'UNKNOWN')
        for value in (0,-1,100.,True,float('inf')):
            self.assertEqual(H.judge_row(row,dict(observed_RZ=value))['decision'],'UNKNOWN')
        self.assertEqual(H.judge_row(row,dict(observed_RZ=100),problem='foreign source')['decision'],'UNKNOWN')
    def test_twelve_identities_and_axis_q_width_source_rejection(self):
        rows,models=fixture();self.assertEqual(len(H.validate_predictions(models,rows)),12)
        key=next(iter(models))
        for field,value in [('axis','Y'),('q',29),('configuration_id','FOREIGN')]:
            bad=deepcopy(models);bad[key]['m6_prediction'][field]=value
            with self.assertRaises(ValueError):H.validate_predictions(bad,rows)
        bad=deepcopy(rows);bad[0]['physical_candidate']['delta']=.25
        with self.assertRaises(ValueError):H.validate_predictions(models,bad)
        bad=deepcopy(models);bad.pop(key)
        with self.assertRaises(ValueError):H.validate_predictions(bad,rows)
        bad=deepcopy(models);bad[key]['formula']='2nd'
        with self.assertRaises(ValueError):H.validate_predictions(bad,rows)

if __name__=='__main__':unittest.main(verbosity=2)
