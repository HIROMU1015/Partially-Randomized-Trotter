"""Invented H/state/lambda only; exercise B IDs and existing finite-mean core."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import expm

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import df_head_pauli_tail_p1 as A
from trottertracks import pr_rpe_rmse as rmse


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value

F=module('P1_B_fixture','tests/check_pr_rpe_signal_plan.py')
G=module('P1_operator_fixture','tests/check_df_head_pauli_tail_p0.py')
SOURCE=dict(fixture='ARTIFICIAL_ONLY')


def setup(prefix=1,M=2):
    b=A.b_api(ROOT);_,original,_,_=F.fixture(prefix=prefix,M=M,kappa='2')
    original=deepcopy(original);original['spec']['binding']['rank']=3
    item=dict(source=SOURCE,dictionary_sha256='ARTIFICIAL_DICTIONARY_'+str(prefix),
        payload=dict(prefix=prefix,lambda_P=.7,saved_DF_scalar=0.,mapping='ARTIFICIAL_MODE0_MSB'))
    config=A.generate_configuration(b,original,item)
    ham,sector,full_state,state,meta=G.fixture()
    car=A.p0.car_api(ROOT)(ham,SimpleNamespace(basis_indices=np.arange(8)),full_state,meta['saved_tail_descriptors'])
    tail_scalar=meta['saved_tail_descriptors'][str(prefix)]['scalar']
    # This is synthetic descriptor construction, never a real-data reanalysis.
    item['payload']['saved_DF_scalar']=tail_scalar
    config=A.generate_configuration(b,original,item)
    cache={};contexts=[]
    for indices,psi in ((sector.basis_indices,state),(np.arange(8),full_state)):
        tail={prefix:dict(scalar=tail_scalar,lambda_R=.7,source=config['spec']['split']['lambda_source'])}
        ctx=A.provider.make_context([a[np.ix_(indices,indices)] for a in car.matrices],psi,ham.constant,
            tail,binding=config['spec']['binding'],source=SOURCE)
        ctx.tails[prefix]['dictionary_identity']=A.dictionary_identity(item)
        ctx.normalizers=cache;ctx.normalizer_budget=5
        contexts.append(ctx)
    def ref(ctx,job):
        H=sum(ctx.matrices)+ctx.constant*np.eye(len(ctx.state));t=job['physical_context']['physical_candidate']['T']
        z=complex(np.vdot(ctx.state,expm(-1j*t*H)@ctx.state));v=dict(real=z.real,imag=z.imag)
        return dict(binding=ctx.binding,T=t,source=SOURCE,evidence=dict(reference=v,independent_reference=v,signal_difference=0.))
    return b,original,item,config,contexts,ref


class P1Tests(unittest.TestCase):
    def setUp(self):
        for guard in (patch.object(np,'load',side_effect=AssertionError('No saved arrays')),
            patch.object(rmse,'nominal_axis_shots',side_effect=AssertionError('No N'))):
            guard.start();self.addCleanup(guard.stop)

    def test_B_dictionary_identity_preserves_kappa_and_never_reuses_DF_ID(self):
        b,original,item,c,_,_=setup(M=12)
        self.assertEqual(c['spec']['kappa']['value'],original['spec']['kappa']['value'])
        for k in ('h','M','formula','delta_pf','finite_K','binding','profile','rmse_policy'):
            self.assertEqual(c['spec'][k],original['spec'][k])
        self.assertNotEqual(c['id'],original['id']);self.assertEqual(len(c['rounds']),13)
        self.assertTrue(all(a['candidate']['id']!=b['candidate']['id'] for a,b in zip(c['rounds'],original['rounds'])))
        changed=deepcopy(item);changed['dictionary_sha256']='FOREIGN_DICTIONARY'
        self.assertNotEqual(c['id'],A.generate_configuration(b,original,changed)['id'])
        b.validate(c)

    def test_fixed_R_control_is_not_a_changed_common_kappa(self):
        b,original,item,c,_,_=setup(M=6)
        control=A.generate_configuration(b,original,item,fixed_R=4)
        self.assertEqual(control['spec']['kappa'],c['spec']['kappa'])
        self.assertFalse(control['rounds'][0]['mapping']['kappa_applied'])
        self.assertEqual(control['rounds'][0]['R_outer'],4)
        self.assertEqual([r['R_outer'] for r in control['rounds'][1:]],[r['R_outer'] for r in c['rounds'][1:]])
        self.assertNotEqual(c['id'],control['id']);b.validate(control)
        item=deepcopy(item);item['payload']['prefix']=2
        original['spec']['L_D']=2
        with self.assertRaises(ValueError):A.generate_configuration(b,original,item,fixed_R=4)

    def test_full_sector_means_independent_raw_and_one_normalizer(self):
        b,_,_,c,contexts,ref=setup();job=A.job(c,0,SOURCE,b)
        from trotterlib import rte
        actual=rte.finite_rte_distribution
        with patch.object(rte,'finite_rte_distribution',wraps=actual) as calls:
            small=A.provider.evaluate(contexts[0],job,ref(contexts[0],job),root=ROOT)
            full=A.provider.evaluate(contexts[1],job,ref(contexts[1],job),root=ROOT)
            self.assertEqual(calls.call_count,1)
        for v in (small,full):
            self.assertEqual(v['status'],'LOCAL_FINITE_MEAN_NOT_RMSE_QUALIFIED')
            self.assertTrue(v['raw_corrected_independent_propagation']);self.assertIsNone(v['N_X'])
            self.assertEqual(v['candidate']['id'],job['B_candidate_id'])
        out=A.compare_means(small,full,kind='ARTIFICIAL_FULL_SECTOR')
        self.assertEqual(out['status'],'LOCAL_COMPARISON_GATE_PASS')

    def test_foreign_dictionary_law_lambda_and_old_DF_ID_rejected(self):
        b,_,_,c,contexts,ref=setup();job=A.job(c,0,SOURCE,b);ctx=contexts[0]
        for changed in ('dictionary','id','lambda_source'):
            bad=deepcopy(job)
            if changed=='dictionary':bad['physical_context']['dictionary_identity']['dictionary_sha256']='BAD'
            elif changed=='id':bad['physical_context']['physical_candidate']['id']=A.common.candidate_id(bad['physical_context']['physical_candidate'])
            else:bad['physical_context']['split']['lambda_source']={'foreign':'SOURCE'}
            with self.assertRaises(ValueError):A.provider.evaluate(ctx,bad,ref(ctx,job),root=ROOT)

    def test_different_R_is_not_compared_and_DF_raw_difference_is_expected(self):
        b,_,_,c,contexts,ref=setup();job=A.job(c,0,SOURCE,b);ctx=contexts[0]
        small=A.provider.evaluate(ctx,job,ref(ctx,job),root=ROOT)
        different=deepcopy(small);different['candidate']['R_outer']+=1
        out=A.compare_means(small,different,kind='ARTIFICIAL_DIFFERENT_R',strict=False)
        self.assertIsNone(out['numerical_comparison']);self.assertNotIn('signal_differences',out)
        with self.assertRaises(ValueError):A.compare_means(small,different,kind='STRICT_DIFFERENT_R')
        raw=deepcopy(small);raw['signal']['raw']['real']+=.1;raw['B_candidate_id']='ARTIFICIAL_DF_ID'
        out=A.compare_means(small,raw,kind='ARTIFICIAL_FIXED_R_DF',strict=False)
        self.assertEqual(out['status'],'LOCAL_COMPARISON_GATE_PASS')
        with self.assertRaises(ValueError):A.compare_means(small,raw,kind='FOREIGN_STRICT')

    def test_normalizer_budget_failure_preserves_diagnostic(self):
        b,_,_,c,contexts,ref=setup();job=A.job(c,0,SOURCE,b);ctx=contexts[0]
        ctx.normalizer_budget=0
        out=A.provider.evaluate(ctx,job,ref(ctx,job),root=ROOT)
        self.assertEqual(out['status'],'NUMERICAL_FAILURE')
        self.assertIn('scope exhausted',out['error']['message'])
        self.assertFalse(out['automatic_retry']);self.assertIsNone(out['signal'])


if __name__=='__main__':unittest.main()
