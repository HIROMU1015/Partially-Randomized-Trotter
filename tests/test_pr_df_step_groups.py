"""Artificial same-step DF acquisition/model scope; no Hamiltonian or compile."""
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_df_step_group_test',ROOT/'scripts/run_pr_df_candidate_prices.py')
D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)


def config():
    rows=[]
    for m in range(13):
        q=1 if m==0 else 2**(m-1);step=Fraction(1,8)if m==0 else Fraction(1,4)
        rows.append(dict(m=m,T=D.enc(Fraction(1,8)*2**m),q_PF=q,R_outer=1,
            actual_delta_pf=D.enc(step),candidate=dict(id='TOY'+str(m)),physical_candidate=dict(id='TOY'+str(m)),
            structure=dict(signed_weights=[1.],allocation=[1]),law_binding=dict(model='ARTIFICIAL')))
    return dict(id='ARTIFICIAL_CONFIG',spec=dict(method='PR',split={},formula='2nd',finite_K=2,
        L_D=3,binding=dict(system='ARTIFICIAL'),law=dict(model='ARTIFICIAL'),profile={}),rounds=rows)


def quote(c,reuse=()):
    return dict(configuration_id=c['id'],rows=[dict(m=r['m'],axis=a,q=r['q_PF'],R=r['R_outer'],physical_id=r['candidate']['id'],
        unit_price_RZ=D.enc(10)if r['m']in reuse else None,source=dict(commit='FIXED_TOY',path='price.json')if r['m']in reuse else None)
        for r in c['rounds']for a in('X','Y')])


class StepGroups(unittest.TestCase):
    def test_missing_groups_reuse_early_price_and_share_one_later_step(self):
        c=config();r=D.design_groups(c,quote(c,reuse=[0]))
        self.assertEqual(len(r['groups']),1)
        g=r['groups'][0];self.assertEqual(g['covered_rounds'],list(range(1,13)))
        self.assertEqual(g['actual_step'],D.enc(Fraction(1,4)))
        self.assertEqual(g['design']['calibration_rounds'],[1,7])
        self.assertEqual(g['design']['holdout_round'],4)
        self.assertTrue(all(v['action']=='REUSE_PRICE'for v in r['round_actions']if v['m']==0))
        self.assertEqual(D.design_groups(c,quote(c,reuse=range(13)))['groups'],[])

    def test_single_early_step_has_direct_finite_point_instead_of_widened_model(self):
        r=D.design_groups(config())
        self.assertEqual(len(r['groups']),2)
        early=r['groups'][0]
        self.assertEqual(early['covered_rounds'],[0]);self.assertEqual(early['design']['mode'],'DIRECT_FINITE_COHORT')
        self.assertIsNone(early['design']['holdout_round'])

    def test_bad_quote_does_not_merge_other_physical_width(self):
        c=config();q=quote(c);q['rows'][3]['q']+=1
        with self.assertRaisesRegex(ValueError,'PHYSICAL_SCHEDULE'):D.design_groups(c,q)
        q=quote(c);q['configuration_id']='OTHER'
        with self.assertRaisesRegex(ValueError,'CONFIGURATION'):D.design_groups(c,q)

    def test_partial_affine_freeze_writes_only_covered_actual_step(self):
        c=config();covered=list(range(1,13));selected=D.design(c,covered)
        cal=[dict(m=m,axis=a,q=c['rounds'][m]['q_PF'],R=1,mean_C=D.enc(7+3*c['rounds'][m]['q_PF']))
             for a in('X','Y')for m in selected['calibration_rounds']]
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='.df-step-test-')as temp:
            inputs=dict(configuration=dict(toy=True),output_directory=D.rel(temp),covered_rounds=covered,model_design=selected)
            with patch.object(D,'read',return_value=c),patch.object(D,'means',return_value=cal),patch.object(D,'runtime',side_effect=AssertionError('No scientific worker')):
                D.freeze(inputs,'TOY_RAW','TOY_SOURCE')
            prices=json.loads((Path(temp)/'AXIS_PRICES.json').read_text())
            self.assertEqual(prices['positive_finite_axes'],24)
            self.assertEqual({r['m']for r in prices['rows']},set(covered))
            self.assertTrue(all(D.dec(r['C_ref'])==7+3*c['rounds'][r['m']]['q_PF']for r in prices['rows']))

    def test_direct_freeze_has_no_coefficients_or_extrapolated_rounds(self):
        c=config();cal=[dict(m=0,axis=a,q=1,R=1,mean_C=D.enc(Fraction(101,2)))for a in('X','Y')]
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='.df-step-test-')as temp:
            inputs=dict(configuration=dict(toy=True),output_directory=D.rel(temp),covered_rounds=[0],model_design=D.design(c,[0]))
            with patch.object(D,'read',return_value=c),patch.object(D,'means',return_value=cal):D.freeze(inputs,'TOY_RAW','TOY_SOURCE')
            prices=json.loads((Path(temp)/'AXIS_PRICES.json').read_text());model=json.loads((Path(temp)/'FROZEN_MODELS.json').read_text())
            self.assertEqual(len(prices['rows']),2)
            self.assertTrue(model['models']['X']['direct']);self.assertEqual(model['models']['X']['coefficients'],{})
            self.assertTrue(all(r['m']==0 and D.dec(r['C_ref'])==Fraction(101,2)for r in prices['rows']))

    def test_same_physical_requests_repeat_same_group_without_new_science(self):
        c=config();a=D.design_groups(c,quote(c));b=D.design_groups(deepcopy(c),quote(c))
        self.assertEqual(a,b)
        self.assertFalse(a['execution_performed'])


if __name__=='__main__':unittest.main(verbosity=2)
