"""Two-level D2 backend/connection tests. No saved molecular arrays are read."""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
from scipy.linalg import expm
import h46_new2_core as old
from trottertracks import h6_df19_rpe_d2 as d2


def toy():
    a=np.array([[.2,.1+.03j],[.1-.03j,-.15]],complex)
    b=np.diag([.07,-.03]).astype(complex)
    tail=np.array([[.03,.02j],[-.02j,-.03]],complex)
    identity=.04
    ctx=SimpleNamespace(state=np.array([1.,1j])/math.sqrt(2),
        ham=SimpleNamespace(constant=.17),matrices=[a,b,tail+identity*np.eye(2)],
        tails={1:dict(matrix=tail,scalar=identity,lambda_R=.5)},half_steps={})
    ctx.full=sum(ctx.matrices)+ctx.ham.constant*np.eye(2)
    return ctx


def candidate(method='PR',T=.7,q=2):
    return dict(system='TOY',method=method,T=T,q=q,delta=T/q,formula='4th(m5_best)',
        prefix=1 if method=='PR' else 2,R_outer=32 if method=='PR' else None,
        K=2 if method=='PR' else None,id='TOY')


def ready(ctx,c):
    for key in d2.half_step_keys([dict(candidate=c)]):
        key,value=d2.half_step(ctx,key);ctx.half_steps[key]=value
    return d2.zrecord(np.vdot(ctx.state,expm(-1j*c['T']*ctx.full)@ctx.state))


class D2Tests(unittest.TestCase):
    def test_noncommuting_signed_m5_matches_existing_spectral_finite_mean(self):
        ctx=toy();c=candidate();ref=ready(ctx,c)
        actual=d2.evolve(ctx,c,ref)
        with patch.object(old,'weights',d2.schedule.weights):
            expected=old.compare_prefix(ctx.state,[old.spectrum(a) for a in ctx.matrices[:2]],
                old.spectrum(ctx.tails[1]['matrix']),.21,.5,c['formula'],2,T=.7,
                total_steps=32,cutoffs=(2,))['finite_rows'][0]
        for key,want in [('corrected',expected['corrected_signal']),('raw',expected['raw_signal'])]:
            z=actual['signal'][key]
            self.assertLess(abs(complex(z['real'],z['imag'])-want),2e-13)
        self.assertAlmostEqual(actual['signal']['log_B'],expected['log_B'],places=14)
        self.assertIn(True,[w<0 for w in actual['signed_weights']])
        self.assertEqual(sum(actual['allocation']),32)
        self.assertEqual(actual['event_count'],64)

    def test_scalar_applied_once_at_total_time(self):
        ctx=toy();c=candidate();ref=ready(ctx,c)
        a=d2.evolve(ctx,c,ref);ctx.ham.constant+=.31
        b=d2.evolve(ctx,c,ref)
        for key in ('corrected','raw','exact_tail'):
            x=a['signal'][key];y=b['signal'][key]
            self.assertLess(abs(complex(y['real'],y['imag'])-
                np.exp(-.31j*c['T'])*complex(x['real'],x['imag'])),2e-13)
        self.assertEqual(a['signal']['log_B'],b['signal']['log_B'])

    def test_full_pf_matches_independent_ordered_matrix_product(self):
        ctx=toy();c=candidate('FULL',T=.3,q=1);ref=ready(ctx,c)
        actual=d2.evolve(ctx,c,ref)
        v=ctx.state.copy()
        for w in d2.schedule.weights(c['formula']):
            for matrix in (*ctx.matrices,*reversed(ctx.matrices)):
                v=expm(-.5j*c['T']*w*matrix)@v
        z=np.vdot(ctx.state,v*np.exp(-1j*ctx.ham.constant*c['T']))
        self.assertLess(abs(complex(actual['signal']['corrected']['real'],
            actual['signal']['corrected']['imag'])-z),1e-13)
        self.assertEqual(actual['signal']['corrected'],actual['signal']['raw'])
        self.assertEqual(actual['signal']['log_B'],0.)

    def test_exponential_backend_never_calls_eigensolver(self):
        ctx=toy();c=candidate()
        with patch('scipy.linalg.eigh',side_effect=AssertionError('eigh forbidden')), \
             patch('numpy.linalg.eigh',side_effect=AssertionError('eigh forbidden')):
            ref=ready(ctx,c);pair=d2.reference_pair(ctx,c['T']);out=d2.evolve(ctx,c,ref)
        self.assertLess(pair['signal_difference'],1e-12)
        self.assertLess(out['signal']['allowance_evidence']['raw_corrected_vector_closure'],1e-12)

    def test_shared_half_step_cache_and_invalid_allocation(self):
        pf=candidate('FULL');pr=candidate()
        self.assertEqual(d2.half_step_keys([dict(candidate=pf),dict(candidate=pr)]),
                         d2.half_step_keys([dict(candidate=pf)]))
        ctx=toy();pr['R_outer']=4
        with self.assertRaises(ValueError):d2.evolve(ctx,pr,dict(real=1.,imag=0.))

    def test_plan_rejects_q_confusion_or_foreign_binding(self):
        plan=json.loads((ROOT/d2.PLAN_PATH).read_text())
        d2.validate_plan(plan)
        bad=deepcopy(plan);bad['rows'][-1]['candidate']['q']=32
        with self.assertRaises(ValueError):d2.validate_plan(bad)
        bad=deepcopy(plan);bad['scientific_binding']['state']='different'
        with self.assertRaises(ValueError):d2.validate_plan(bad)

    def test_binding_allowance_and_unknown_actual_lane_reuse_D1(self):
        plan=json.loads((ROOT/d2.PLAN_PATH).read_text())
        for index in (0,1,4,5,6,7):
            row=plan['rows'][index];c=row['candidate']
            sig=dict(reference=dict(real=1.,imag=0.),corrected=dict(real=.999,imag=.0001),
                raw=dict(real=.999,imag=.0001),total_signed_bias=dict(real=-.001,imag=.0001),
                log_B=0.,allowance_evidence=dict(maximum_intermediate_norm=1.01,
                    raw_corrected_vector_closure=1e-14))
            record=dict(candidate=c,signal=sig,origin='SYNTHETIC_ONLY',signed_weights=[1.],allocation=[32],event_count=32)
            output=d2.analyze(record,row,plan,state_energy=0.,state_residual=1e-12,
                reference_evidence=dict(signal_difference=2e-12 if c['T']!=.4 else None))
            self.assertEqual(output['q_RPE'],row['q_RPE'])
            self.assertEqual(output['q_PF'],c['q'])
            self.assertEqual(output['ideal_numerical_pilot']['status'],'CONDITIONAL_ROUND_PASS')
            self.assertIsNone(output['actual_implementation']['N_X'])
            self.assertIsNone(output['compiled_price']);self.assertIsNone(output['G'])
            self.assertGreaterEqual(output['error_disk']['preserved_u_X'],1e-9)
        missing=deepcopy(record);missing['candidate']=dict(c,q=32)
        with self.assertRaises(ValueError):d2.analyze(missing,row,plan,state_energy=0.,state_residual=0.,reference_evidence={})
        with self.assertRaises(ValueError):d2.analyze(record,row,plan,state_energy=0.,state_residual=0.,reference_evidence={})


if __name__=='__main__':unittest.main()
