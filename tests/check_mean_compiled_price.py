"""Artificial events/C1 Accounting and mock B records only; no compilation."""
from copy import deepcopy
import itertools
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from trottertracks import mean_compiled_price as m
from trottertracks.h4_df10_price_v1 import Accounting
from trotterlib.rte_connected_cluster_cost_validation import _sequence_form
from check_r_variable_compiled_price import toy_task,mock_delegate
from check_pf_pr_rpe_pipeline import fake_shot

STATES=(None,'a','b')
ANGLES={'a':.2,'b':.7}


def event_kernel(probabilities):
    # Synthetic block Ginv X G. C1 literal Z-fold charges one RZ for a
    # different basis at the next X, zero for an adjacent inverse pair.
    return m.MeanKernel.from_events(STATES,('a','b'),probabilities,
        lambda active,token:(token,float(active!=token)))


def token_kernel(token):
    return event_kernel((1.,0.) if token=='a' else (0.,1.))


def source_count(tokens,axis='X'):
    a=Accounting();a.gate('h',(8,),role='wrapper')
    for token in tokens:
        ops=[dict(name='p',qubits=[0],parameters=[str(ANGLES[token])])]
        a.basis(ops,True,'artificial_basis');a.gate('x',(0,));a.basis(ops,False,'artificial_basis')
    a.gate('p',(8,),(.23,),'scalar_once')
    if axis=='Y':a.gate('sdg',(8,),role='wrapper')
    a.gate('h',(8,),role='wrapper');a.gate('measure',(8,),role='measurement')
    return a.finish()['structural_rz']


def program(q=2,axis='X'):
    prefix=m.MeanKernel(STATES,np.eye(3),np.full(3,source_count([],axis)))
    identity=m.MeanKernel.identity(STATES)
    specs=[(+1.,2,'a','b',(.25,.75)),(-.5,1,'b','a',(.75,.25))]
    pieces=[m.MeanPiece(w,r,token_kernel(l),event_kernel(probs),token_kernel(t),{'fixture':'SYNTHETIC_SIGNED_LAW'})
            for w,r,l,t,probs in specs]
    return specs,pieces,prefix,identity


def enumerate_program(q,specs,axis):
    slots=[(l,t,r,probs) for _ in range(q) for _,r,l,t,probs in specs]
    E=sum(r for _,_,r,_ in slots);total=0.
    for events in itertools.product(('a','b'),repeat=E):
        seq=[];probability=1.;start=0
        for left,right,r,probs in slots:
            block=events[start:start+r];start+=r;seq.extend([left,*block,right])
            probability*=math.prod(probs[0 if x=='a' else 1] for x in block)
        total+=probability*source_count(seq,axis)
    return total


def supplied_records(task):
    records=mock_delegate(task,task['candidates'])
    for r in records:
        if r['mean']['rz_count'] is None:continue
        c=r['identity']['candidate']
        r['law_binding']=dict(definition={'model':'MOCK_FINITE_LAW','source':{'fixture':'SYNTHETIC'}},
                              candidate_id=c['id'],R_outer=c['R_outer'],K=c['K'],q_PF=c['q'])
    return records


class Tests(unittest.TestCase):
    def test_short_C1_structural_basis_and_literal_Z_cancellation(self):
        base=source_count([])
        self.assertEqual(source_count(['a','a','a'])-base,2)
        self.assertEqual(source_count(['a','b','a'])-base,4)
        for tokens in itertools.product(('a','b'),repeat=4):
            kernel=m.MeanKernel.identity(STATES)
            for token in tokens:kernel=kernel.then(token_kernel(token))
            mean,_=kernel.expected([1,0,0],[0,1,1])
            self.assertEqual(mean,source_count(tokens)-base)

    def test_full_wrapper_signed_pieces_and_outer_boundaries_match_enumeration(self):
        for axis in ('X','Y'):
            specs,pieces,prefix,suffix=program(axis=axis)
            result=m.full_wrapper_mean(2,pieces,prefix,suffix,[1,0,0],[0,1,1])
            self.assertAlmostEqual(result['mean'],enumerate_program(2,specs,axis))
            self.assertFalse(result['compiled_price_qualified'])

    def test_reset_at_PF_RTE_boundary_would_change_the_answer(self):
        same=token_kernel('a')
        uninterrupted=same.then(same).expected([1,0,0],[0,1,1])[0]
        closed_twice=2*same.expected([1,0,0],[0,1,1])[0]
        self.assertEqual(uninterrupted,2)
        self.assertEqual(closed_twice,4)

    def test_binary_repeat_matches_sequential_for_short_program(self):
        kernel=event_kernel((.25,.75))
        for r in (0,1,2,7,13):
            direct=m.MeanKernel.identity(STATES)
            for _ in range(r):direct=direct.then(kernel)
            fast=kernel.repeat(r)
            np.testing.assert_allclose(fast.transition,direct.transition,atol=1e-13,rtol=0)
            np.testing.assert_allclose(fast.reward,direct.reward,atol=1e-13,rtol=0)

    def test_eleven_signed_pieces_large_q_R_use_fixed_memory_shapes(self):
        identity=m.MeanKernel.identity(STATES);before=token_kernel('a');after=token_kernel('b')
        pieces=[m.MeanPiece(-.5 if j==5 else .1,2**16,before,
            event_kernel((.75,.25) if j==5 else (.25,.75)),after,{'fixture':'SYNTHETIC_11'}) for j in range(11)]
        result=m.full_wrapper_mean(2**20,pieces,identity,identity,[1,0,0],[0,1,1])
        self.assertEqual(result['state_count'],3);self.assertEqual(result['pieces'],11)
        self.assertFalse(result['event_sequence_materialized'])
        self.assertLess(result['kernel_compositions'],500)
        self.assertEqual(before.transition.shape,(3,3));self.assertEqual(before.reward.shape,(3,))
        self.assertIsNone(result['numerical_roundoff_bound'])

    def test_invalid_probability_or_nonclosed_state_fails_without_fallback(self):
        with self.assertRaises(ValueError):event_kernel((.5,.6))
        with self.assertRaises(ValueError):m.MeanKernel.from_events(STATES,['x'],[1.],lambda s,e:('OUTSIDE',1))
        with self.assertRaises(ValueError):event_kernel((.5,.5)).expected([1,0,0],None)
        with self.assertRaises(ValueError):event_kernel((.5,.5)).repeat(-1)
        with self.assertRaises(ValueError):m.MeanKernel(STATES,np.eye(3),[0,float('nan'),0])

    def test_reused_connected_forms_match_full_artificial_enumeration(self):
        probs={0:.75,2:.25};forms=[_sequence_form(pattern) for pattern in itertools.product((0,2),repeat=4)]
        keys={key for f in forms for key in f}
        coefficients={k:(.1 if k.startswith('k1') else -.03 if k.startswith('k2') else .02)*(1+k.count('2')) for k in keys}
        actual=0.
        for pattern,form in zip(itertools.product((0,2),repeat=4),forms):
            actual+=math.prod(probs[x] for x in pattern)*math.fsum(v*coefficients[k] for k,v in form.items())
        result=m.local_residual_mean(4,probs,coefficients,target='C_MINUS_S')
        self.assertAlmostEqual(result['internal_residual_mean'],actual)
        self.assertIsNone(result['full_wrapper_mean'])
        self.assertIsNone(result['boundary_residual'])
        huge=m.local_residual_mean(10**12,probs,coefficients,target='C_MINUS_S')
        self.assertEqual(len(huge['form']),14)

    def test_raw_compiled_clusters_missing_coefficients_and_double_count_rejected(self):
        with self.assertRaises(ValueError):m.local_residual_mean(4,{0:.5,2:.5},{},target='COMPILED_C')
        with self.assertRaises(ValueError):m.local_residual_mean(4,{0:.5,2:.5},{},target='C_MINUS_S')
        term=dict(target='C_MINUS_S',mode='C1_AFFINE',mean=2,coverage='FULL_WRAPPER')
        with self.assertRaises(ValueError):m.combine_mean(10,residual_terms=[term,term])
        with self.assertRaises(ValueError):m.combine_mean(-1,residual_terms=[term])
        self.assertEqual(m.combine_mean(10,residual_terms=[term])['mean'],12)
        term.update(mode='LOCAL_C_MINUS_S',coverage='RTE_INTERNAL_ONLY')
        self.assertIsNone(m.combine_mean(10,residual_terms=[term])['mean'])
        term.update(coverage='FULL_WRAPPER',mean=-20)
        with self.assertRaises(ValueError):m.combine_mean(10,residual_terms=[term])

    def test_provider_is_passive_and_keeps_R_and_axis_identity(self):
        task=toy_task();saved=supplied_records(task);provider=m.MeanPriceProvider(saved)
        with patch.object(m,'full_wrapper_mean',side_effect=AssertionError('No model evaluation in provider')):
            result=provider(task,task['candidates'])
        self.assertEqual(len(result),4)
        self.assertEqual([r['price_status'] for r in result],['SAVED_DIRECT_FINITE_COHORT']*2+['MISSING']*2)
        self.assertEqual([r['mean']['rz_count'] for r in result],[7,9,None,None])
        self.assertEqual(result[0]['identity'],saved[0]['identity'])
        self.assertIsNone(result[2]['uncertainty']['joint_covariance'])
        self.assertTrue(all(r['normalization_is_price_multiplier'] is False for r in result))

    def test_B_run_missing_accepts_one_response_per_axis_without_G_or_retry(self):
        task=toy_task();provider=m.MeanPriceProvider(supplied_records(task))
        shots=[fake_shot(task,r) for r in task['candidates']]
        with patch.object(m.p.accounting,'aggregate_supplied_axes',side_effect=AssertionError('No G')):
            result=m.p.run_missing(task,shots,[],c_provider=provider)
            self.assertEqual(result['status'],'BATCH_PREPARED_NO_TOTALS')
            self.assertIsNone(result['totals']);self.assertEqual(len(result['prices']),4)
            never=lambda *_:self.fail('Missing/unqualified lookup must not retry automatically')
            again=m.p.run_missing(task,shots,result['prices'],c_provider=never)
            self.assertIsNone(again['totals'])

    def test_multiple_cohorts_require_selection_and_are_not_pooled(self):
        task=toy_task(Rs=(4,));first=supplied_records(task);second=deepcopy(first)
        for r in second:r['cohort']='MOCK_SECOND';r['mean']['rz_count']+=1
        result=m.MeanPriceProvider(first+second)(task,task['candidates'])
        self.assertTrue(all(r['price_status']=='SELECTION_REQUIRED' for r in result))
        self.assertTrue(all(r['mean']['rz_count'] is None for r in result))
        choices={(r['identity']['candidate']['id'],r['axis']):r['cohort'] for r in second}
        selected=m.MeanPriceProvider(first+second,selections=choices)(task,task['candidates'])
        self.assertEqual([r['mean']['rz_count'] for r in selected],[8,10])

    def test_H6_signed_recipe_metadata_and_outside_prefix_are_kept(self):
        task=toy_task('H6',(16,32),q=2);result=m.MeanPriceProvider()(task,task['candidates'])
        self.assertTrue(all(len(r['mean_price_scope']['allocation'])==11 for r in result))
        self.assertTrue(all(r['price_status']=='MISSING' for r in result))
        bad=deepcopy(task);c=bad['candidates'][0]['candidate'];c['prefix']=5;c['id']=m.p.candidate_id(c)
        bad['candidates'][0]['structure']=m.p.structure(c)
        result=m.MeanPriceProvider()(bad,[bad['candidates'][0]])
        self.assertTrue(all(r['price_status']=='OUTSIDE_DOMAIN' for r in result))

    def test_external_RMSE_kappa_schedule_is_echoed_not_recomputed(self):
        task=toy_task(Rs=(4,));saved=supplied_records(task)
        ctx=dict(schedule_id='MOCK_RMSE_M3',source={'fixture':'SYNTHETIC'},M=3,kappa_schedule=[1,3,5,9])
        result=m.MeanPriceProvider(saved,schedule_context=ctx)(task,task['candidates'])
        self.assertEqual(result[0]['schedule_mapping']['context'],ctx)
        self.assertEqual(result[0]['schedule_mapping']['status'],'ECHOED_NOT_SCIENTIFICALLY_VALIDATED')
        self.assertEqual(result[0]['mean'],saved[0]['mean'])
        ctx['kappa_schedule'][1]=99
        self.assertEqual(result[0]['schedule_mapping']['context']['kappa_schedule'][1],3)

    def test_model_stays_unqualified_and_finite_cohort_is_not_all_law(self):
        task=toy_task(Rs=(4,));saved=supplied_records(task)
        for r in saved:r.update(classification='MODEL_PREDICTION',model=dict(qualified_for_candidate=False,
            domain=deepcopy(r['identity']['candidate']),binding=deepcopy(task['binding']),compiler=task['profile']['compiler']))
        result=m.MeanPriceProvider(saved)(task,task['candidates'])
        self.assertTrue(all(r['price_status']=='UNQUALIFIED_MODEL' for r in result))
        for r in saved:r['expectation']='EXPLICIT_CONDITIONAL_MODEL'
        self.assertTrue(all(r['model']['qualified_for_candidate'] is False for r in m.MeanPriceProvider(saved)(task,task['candidates'])))
        for r in saved:r['model']['qualified_for_candidate']=True
        with self.assertRaises(ValueError):m.MeanPriceProvider(saved)(task,task['candidates'])
        saved=supplied_records(task);saved[0]['expectation']='ALL_LAW_EXPECTATION'
        with self.assertRaises(ValueError):m.MeanPriceProvider(saved)(task,task['candidates'])

    def test_foreign_context_failed_status_and_missing_law_are_safe(self):
        task=toy_task(Rs=(4,));saved=supplied_records(task)
        other=deepcopy(saved);other[0]['identity']['binding']['state']='FOREIGN'
        result=m.MeanPriceProvider(other)(task,task['candidates'])
        self.assertIsNone(result[0]['mean']['rz_count'])
        saved[0].pop('law_binding')
        with self.assertRaises(ValueError):m.MeanPriceProvider(saved)(task,task['candidates'])
        failed=deepcopy(supplied_records(task)[0]);failed.update(classification='UNAVAILABLE',price_status='FAILED',reason='MOCK_BACKEND_FAILURE',mean={'rz_count':None})
        result=m.MeanPriceProvider([failed])(task,task['candidates'])
        self.assertEqual(result[0]['price_status'],'FAILED')
        self.assertIsNone(result[0]['mean']['rz_count'])


if __name__=='__main__':unittest.main(verbosity=2)
