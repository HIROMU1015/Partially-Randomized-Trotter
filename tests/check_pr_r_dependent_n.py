"""Artificial matrices and mock N(T,R) only. No molecular state/signal loads."""
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from trottertracks import pr_r_dependent_n as model
from trottertracks import pf_pr_rpe_pipeline as p
from check_pf_pr_rpe_pipeline import fake_shot


def kernel():
    import run_h6_df19_rpe_d2 as io
    from trottertracks import h6_df19_rpe_d2 as action
    return io,action


def fake(task,row):
    n=fake_shot(task,row);n.update(diagnostics=dict(synthetic_only=True));return n


def pair():return dict(reference=dict(real=1.,imag=0.),independent_reference=dict(real=1.,imag=0.),signal_difference=0.,certified=False)


class RTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.bases={s:model.saved_task(ROOT,s) for s in ('H4','H6')}

    def task(self,s='H4',mapping=None):
        return model.with_round_R(self.bases[s],mapping or {12:8 if s=='H4' else 32},dict(path='ARTIFICIAL_ONLY'))['task']

    def test_metadata_and_import_never_run_numeric_backend(self):
        code="import sys;from pathlib import Path;from trottertracks import pr_r_dependent_n as m;b=m.saved_task(Path.cwd(),'H6');m.with_round_R(b,{12:32},{'path':'MOCK'});assert not any(x in sys.modules for x in ('numpy','scipy','qiskit'))"
        env=__import__('os').environ.copy();env['PYTHONPATH']='src:scripts'
        subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,env=env,check=True)
        with patch.object(model.scalar,'analyze_bound_point',side_effect=AssertionError('No numeric call')):
            for system in self.bases:model.validate_task(self.task(system),self.bases[system])

    def test_R_ID_recipe_only_changes_and_q_variables_remain_distinct(self):
        for s,mapping in [('H4',{0:2,7:8,12:16}),('H6',{4:11,8:22,12:32})]:
            task=self.task(s,mapping);model.validate_task(task,self.bases[s])
            for row,old in zip(task['candidates'],self.bases[s]['candidates']):
                if row['candidate']['method']!='PR' or row['m'] not in mapping:self.assertEqual(row,old);continue
                self.assertNotEqual(row['candidate']['id'],old['candidate']['id'])
                a=deepcopy(row['candidate']);a.update(R_outer=old['candidate']['R_outer'],id=old['candidate']['id']);self.assertEqual(a,old['candidate'])
                self.assertEqual(sum(row['structure']['allocation']),mapping[row['m']]);self.assertEqual(row['structure']['event_count'],row['q_PF']*mapping[row['m']])
            if s=='H6':
                row=task['candidates'][-1];self.assertEqual((row['q_RPE'],row['q_PF']),(4096,256));self.assertEqual(task['candidates'][:8],self.bases[s]['candidates'][:8])

    def test_invalid_R_early_H6_and_foreign_base_or_policy_are_rejected(self):
        for s,mp in [('H4',{0:0}),('H4',{0:True}),('H4',{0:2.5}),('H4',{13:8}),('H6',{3:16}),('H6',{4:10})]:
            with self.assertRaises(ValueError):self.task(s,mp)
        for change in (lambda t:t['configuration'].update(alpha_axis=.025),lambda t:t['binding'].update(state='OTHER'),lambda t:t['profile'].update(control='OTHER'),lambda t:t['candidates'][-1]['candidate'].update(prefix=2)):
            task=self.task();change(task)
            with self.assertRaises(ValueError):model.validate_task(task,self.bases['H4'])
        foreign=deepcopy(self.bases['H4']);foreign['binding']['state']='OTHER'
        with self.assertRaises(ValueError):model.with_round_R(foreign,{12:8},{})

    def test_H6_allocation_signed_order_minimum_and_whole_outer_total(self):
        for R in (11,16,22,32):
            row=self.task('H6',{12:R})['candidates'][-1];st=row['structure']
            self.assertEqual(len(st['allocation']),11);self.assertEqual(sum(st['allocation']),R);self.assertGreaterEqual(min(st['allocation']),1)
            self.assertTrue(any(w<0 for w in st['signed_weights']));self.assertEqual(st['signed_weights'],self.bases['H6']['candidates'][-1]['structure']['signed_weights'])
            if R==16:self.assertEqual(st['allocation'],[1,2,1,2,1,2,1,2,1,2,1])
            if R==11:self.assertEqual(st['allocation'],[1]*11)

    def test_exact_cache_reuse_excludes_old_R_and_preserves_saved_FAIL(self):
        base=self.bases['H4'];shots=[fake(base,r) for r in base['candidates']];task=self.task()
        reused,old=model.partition_saved(task,shots);self.assertEqual((len(reused),len(old)),(25,1))
        reused[0]['plan'].update(status='UNQUALIFIED_ROUND',N_X=None,N_Y=None)
        provider=Mock(side_effect=AssertionError('No retry'))
        self.assertEqual(p.run_missing(task,reused,[],n_provider=provider)['status'],'STOP_UNQUALIFIED_N');provider.assert_not_called()
        bad=deepcopy(shots);bad[0]['identity']['binding']['state']='OTHER'
        with self.assertRaises(ValueError):model.partition_saved(task,bad)

    def test_artificial_S2_and_signed_m5_R_dependence_matches_old_spectral_route(self):
        kernel()
        from check_h6_df19_rpe_d2 import toy,ready,old
        from trottertracks import h6_df19_rpe_d2 as action
        for formula,Rs in [('2nd',(4,8,16)),('4th(m5_best)',(11,16,22))]:
            logs=[];corrected=[]
            for R in Rs:
                ctx=toy();c=dict(system='TOY',method='PR',T=.7,q=2,delta=.35,formula=formula,prefix=1,R_outer=R,K=2,id='TOY')
                ref=ready(ctx,c);got=action.evolve(ctx,c,ref)
                with patch.object(old,'weights',action.schedule.weights):
                    want=old.compare_prefix(ctx.state,[old.spectrum(a) for a in ctx.matrices[:2]],old.spectrum(ctx.tails[1]['matrix']),.21,.5,formula,2,T=.7,total_steps=R,cutoffs=(2,))['finite_rows'][0]
                for field,target in [('corrected',want['corrected_signal']),('raw',want['raw_signal'])]:
                    z=got['signal'][field];self.assertLess(abs(complex(z['real'],z['imag'])-target),2e-13)
                logs.append(got['signal']['log_B']);corrected.append(got['signal']['corrected'])
                self.assertEqual(got['event_count'],2*R);self.assertLess(got['signal']['allowance_evidence']['raw_corrected_vector_closure'],1e-9)
            self.assertGreater(logs[0],logs[-1]);self.assertNotEqual(corrected[0],corrected[-1])

    def test_tiny_tau_normalizer_rounding_is_unqualified_before_propagation(self):
        io,action=kernel();row=self.task('H4',{12:10**12})['candidates'][-1]
        ctx=SimpleNamespace(tails={1:dict(lambda_R=.5)})
        with self.assertRaisesRegex(ValueError,'normalizer precision'):model.check_normalizers(ctx,row,action)

    def test_artificial_scalar_uses_R_stage_count_unknown_actual_and_integer_check(self):
        task=self.task('H4',{0:8,1:16});outputs=[]
        for row in (task['candidates'][1],task['candidates'][3]):
            st=row['structure'];sig=dict(reference=dict(real=1.,imag=0.),corrected=dict(real=.999,imag=.0001),raw=dict(real=.999,imag=.0001),total_signed_bias=dict(real=-.001,imag=.0001),log_B=0.,allowance_evidence=dict(maximum_intermediate_norm=1.,exact_tail_norm_error=0.,raw_corrected_vector_closure=0.))
            rec=dict(candidate=row['candidate'],signal=sig,**{k:st[k] for k in ('signed_weights','allocation','event_count')})
            n=model.analyze(task,row,rec,pair(),dict(state_energy=0.,state_residual=0.),dict(path='TOY'));outputs.append(n)
            self.assertEqual(n['plan']['status'],'CONDITIONAL_ROUND_PASS');self.assertIsNone(n['actual']['N_X'])
            self.assertFalse(n['diagnostics']['allowance']['certified']);self.assertIn('integer_checks',n['diagnostics'])
            self.assertEqual(n['diagnostics']['allowance']['stages'],row['q_PF']*5+2*st['event_count'])
            rec['signal']['allowance_evidence']['maximum_intermediate_norm']=float('nan')
            with self.assertRaises(ValueError):model.analyze(task,row,rec,pair(),dict(state_energy=0.,state_residual=0.),{})
        self.assertGreater(outputs[1]['diagnostics']['allowance']['stages'],outputs[0]['diagnostics']['allowance']['stages'])

    def test_artificial_H6_scalar_binding_and_R_stage_count(self):
        task=self.task('H6',{4:22});row=task['candidates'][9];st=row['structure']
        sig=dict(reference=dict(real=1.,imag=0.),corrected=dict(real=.999,imag=.0001),raw=dict(real=.999,imag=.0001),total_signed_bias=dict(real=-.001,imag=.0001),log_B=0.,allowance_evidence=dict(maximum_intermediate_norm=1.,exact_tail_norm_error=0.,raw_corrected_vector_closure=0.))
        record=dict(candidate=row['candidate'],signal=sig,**{k:st[k] for k in ('signed_weights','allocation','event_count')})
        n=model.analyze(task,row,record,pair(),dict(state_energy=0.,state_residual=0.),dict(path='TOY'))
        self.assertEqual(n['plan']['binding']['rank'],19);self.assertEqual(n['plan']['status'],'CONDITIONAL_ROUND_PASS')
        self.assertIsNone(n['actual']['N_X']);self.assertEqual(n['diagnostics']['allowance']['stages'],231)

    def test_mock_multi_round_batch_connects_run_missing_without_prices_or_G(self):
        io,action=kernel()
        for system,mapping in [('H4',{10:8,11:16,12:32}),('H6',{10:22,11:32,12:44})]:
            task=self.task(system,mapping);base=self.bases[system];saved=[fake(base,r) for r in base['candidates']];reuse,_=model.partition_saved(task,saved)
            ctx=SimpleNamespace(scientific_binding=task['binding'],half_steps={});factory=Mock(return_value=(ctx,{}))
            provider=model.RBatch(task,base,dict(path='MOCK_ONLY'),factory)
            with patch.object(action,'reference_pair',return_value=pair()) as refs,patch.object(action,'half_step',side_effect=lambda ctx,k:(k,None)), \
                 patch.object(io,'worker_resources',side_effect=lambda ctx,n:dict(half_step_workers=n)),patch.object(model,'check_normalizers'), \
                 patch.object(action,'evolve',return_value={}) as means,patch.object(model,'analyze',side_effect=lambda t,r,*a,**kw:fake(t,r)), \
                 patch.object(p,'evaluate',side_effect=AssertionError('No G')):
                out=p.run_missing(task,reuse,[],n_provider=provider,c_provider=None)
            factory.assert_called_once();self.assertEqual(refs.call_count,3);self.assertEqual(means.call_count,3)
            self.assertEqual(out['plan']['missing_N'],[]);self.assertEqual(out['prices'],[]);self.assertIsNone(out['totals'])
            self.assertEqual(provider.last_run['PR_exact_tail_controls_completed'],3)

    def test_supplied_reference_binding_gate_and_candidate_failure_never_retry(self):
        io,action=kernel();task=self.task();row=task['candidates'][-1];ctx=SimpleNamespace(scientific_binding=task['binding'],half_steps={})
        factory=Mock(return_value=(ctx,{}));ref=Mock(return_value=dict(binding={'state':'OTHER'},T=row['candidate']['T'],source={'path':'MOCK'},evidence=pair()))
        provider=model.RBatch(task,self.bases['H4'],dict(path='MOCK'),factory,ref)
        with self.assertRaises(ValueError):provider(task,[row])
        self.assertEqual(provider.last_run['status'],'STOP_CONTEXT_OR_REFERENCE');ref.assert_called_once()
        inconsistent=pair();inconsistent['independent_reference']['imag']=.5
        ref.return_value=dict(binding=task['binding'],T=row['candidate']['T'],source={'path':'MOCK'},evidence=inconsistent)
        with self.assertRaisesRegex(ValueError,'Observed reference pair gate'):provider(task,[row])
        self.assertEqual(provider.last_run['references'][row['candidate']['T']],inconsistent)
        ref.return_value=dict(binding=task['binding'],T=row['candidate']['T'],source={'path':'MOCK'},evidence=pair())
        with patch.object(action,'half_step',side_effect=lambda c,k:(k,None)),patch.object(io,'worker_resources',return_value=dict(half_step_workers=1)), \
             patch.object(model,'check_normalizers'),patch.object(action,'evolve',side_effect=ValueError('TOY_GATE')) as means:
            out=provider(task,[row]);means.assert_called_once()
        self.assertIsNone(out[0]['plan']['N_X']);self.assertEqual(provider.last_run['status'],'STOP_UNQUALIFIED_N')
        with patch.object(action,'reference_pair',side_effect=AssertionError('Reuse bound saved pair')),              patch.object(action,'half_step',side_effect=lambda c,k:(k,None)),patch.object(io,'worker_resources',return_value=dict(half_step_workers=1)),              patch.object(model,'check_normalizers'),patch.object(action,'evolve',return_value={}),patch.object(model,'analyze',side_effect=lambda t,r,*a,**kw:fake(t,r)):
            passed=provider(task,[row])
        self.assertEqual(passed[0]['plan']['status'],'CONDITIONAL_ROUND_PASS')
        self.assertEqual(passed[0]['diagnostics']['reference_origin']['kind'],'SUPPLIED_BOUND_REFERENCE')
        with self.assertRaises(ValueError):provider(task,[task['candidates'][0]])
        with self.assertRaises(ValueError):provider(task,[task['candidates'][1]])

    def test_historical_sources_and_artifacts_are_unchanged(self):
        paths=['src/trottertracks/h4_df10_rpe_batch_provider.py','src/trottertracks/h6_rpe_batch_provider.py','src/trottertracks/h6_df19_rpe_d2.py',
          'src/trottertracks/h4_df10_rpe_d1.py','src/trottertracks/pf_pr_rpe_pipeline.py','scripts/verify_h4_df10_rpe_shot_integers.py','src/trotterlib/rte.py']
        for path in paths:self.assertEqual((ROOT/path).read_bytes(),subprocess.check_output(['git','show',model.BASELINE+':'+path],cwd=ROOT))
        for path in ['docs/research/h4_df10_rpe_batch_results_20261110/N_HANDOFF.json','docs/research/h6_df19_rpe_m8_12_results_20261110/run01/result.json']:
            self.assertEqual(subprocess.check_output(['git','rev-parse','HEAD:'+path],cwd=ROOT),subprocess.check_output(['git','rev-parse',model.BASELINE+':'+path],cwd=ROOT))


if __name__=='__main__':unittest.main()
