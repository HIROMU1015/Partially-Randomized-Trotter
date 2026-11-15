"""Artificial F1 prices; no physical prediction, fit, compile or Q3 evaluation."""
from copy import deepcopy
from fractions import Fraction
from unittest.mock import Mock
import unittest

import run_full_h_f1_diagnostic as F


def summary_rows():
    return [dict(m=m,axis=a,physical_id=f'mock{m}',N=57 if m==6 else 1,
                 C_ref=F.exact(1 if a=='X' else 2),status='MODEL_SCENARIO_ONLY',reason=None,
                 reused_prediction={'artificial':a} if m==6 else None)
            for m in range(13) for a in ('X','Y')]


def axis_fixture(m=0,axis='X'):
    p=dict(q=2,R_outer=3,s=6,family='FULL_H_PURE_RTE',L_D=0,K=2,id='artificial')
    r=dict(m=m,axis=axis,physical_candidate=p,q_grouping=2,R_outer=3,event_positions=6,
        original_configuration_id='mock',total_configuration_id='mock_total',original_candidate_id='old',
        total_candidate_id='new',allocation_id='fixture',N=57,T_exact={'supplied':'mock'},
        frozen_model={'mock':'model'},admission_source={'mock':'admission'},B_join={'mock':'B'})
    model=dict(model='PRICE_LINEAR_FEATURE_MODEL_V1',feature_map='FULL_H_GROUP_V1',
        features=['one','event_count','outer_boundary_count'],domain=dict(route='FULL_H_RTE',axis=axis),
        coefficients={k:F.exact(v) for k,v in zip(['one','event_count','outer_boundary_count'],[-10,1,1])})
    D,_=F.D7.metadata_api(F.ROOT);M=D.c_model(F.ROOT)
    return r,model,D.features,M.predict


class Tests(unittest.TestCase):
    def test_valid_total_and_both_boundary_signs_are_exact(self):
        rows=summary_rows();result=F.summarize(rows,[dict(label='low',RZ=200),dict(label='high',RZ=300)])
        # 12 artificial rounds contribute 3, and m6 contributes 57*(1+2).
        self.assertEqual(result['T_ref'],F.exact(207))
        self.assertEqual([r['D_PF'] for r in result['full_total_comparisons']],[F.exact(-7),F.exact(93)])
        self.assertIsNone(result['G']);self.assertIsNone(result['winner'])

    def test_invalid_axis_nulls_whole_total_but_keeps_m6(self):
        rows=summary_rows();rows[0].update(C_ref=F.exact(-1),status='OUT_OF_DOMAIN_NEGATIVE_MODEL_REFERENCE',reason='negative')
        result=F.summarize(rows,[dict(label='low',RZ=100),dict(label='high',RZ=200)])
        self.assertIsNone(result['T_ref'])
        self.assertTrue(all(r['D_PF'] is None and r['Delta_lower_cost_condition'] is None for r in result['full_total_comparisons']))
        self.assertEqual(result['m6']['model_partial_RZ'],F.exact(171))
        self.assertEqual([r['PF_minus_m6_reference'] for r in result['m6']['comparisons']],[F.exact(-71),F.exact(29)])
        self.assertFalse(result['m6']['is_mother_mean_lower_bound'])

    def test_m6_invalid_means_no_partial_or_replacement(self):
        rows=summary_rows();next(r for r in rows if (r['m'],r['axis'])==(6,'Y'))['status']='HELD'
        result=F.summarize(rows,[dict(label='artificial',RZ=100)])
        self.assertIsNone(result['m6']['model_partial_RZ']);self.assertIsNone(result['T_ref'])

    def test_negative_exact_output_is_retained(self):
        row,model,features,predict=axis_fixture();out=F.evaluate_axis(row,model,[],features,predict)
        self.assertEqual(out['C_ref'],F.exact(-3))
        self.assertEqual(out['status'],'OUT_OF_DOMAIN_NEGATIVE_MODEL_REFERENCE')
        self.assertIsNone(out['E_C']);self.assertFalse(out['qualified_for_candidate'])

    def test_m5_m6_reuse_never_calls_predict(self):
        for m in (5,6):
            row,model,features,_=axis_fixture(m);f=features(dict(q=2,R=3),'FULL_H_GROUP_V1')
            pred=dict(m=m,axis='X',physical=deepcopy(row['physical_candidate']),
                features={k:F.exact(v) for k,v in f.items()},predicted_RZ=F.exact(Fraction(101,2)))
            forbidden=Mock(side_effect=AssertionError('Do not recalculate saved forecasts'))
            out=F.evaluate_axis(row,model,[pred],features,forbidden)
            forbidden.assert_not_called();self.assertEqual(out['C_ref'],pred['predicted_RZ'])

    def test_identity_units_and_duplicate_axis_rejected(self):
        row,model,features,predict=axis_fixture();model['domain']['axis']='Y'
        with self.assertRaisesRegex(ValueError,'Axis'):F.evaluate_axis(row,model,[],features,predict)
        row,model,features,predict=axis_fixture();row['physical_candidate']['s']=7;row['event_positions']=7
        with self.assertRaisesRegex(ValueError,'event count'):F.evaluate_axis(row,model,[],features,predict)
        rows=summary_rows();rows[-1]=deepcopy(rows[0])
        with self.assertRaisesRegex(ValueError,'Unique'):F.summarize(rows,[])


if __name__=='__main__':unittest.main()
