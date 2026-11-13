"""Artificial matrices/signals + passive saved metadata; no real molecular run."""
import ast
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from trottertracks import h4_pr_rmse_completion as B
from trottertracks import pr_rpe_power_mean as P
from trottertracks import h6_df19_rpe_d2 as direct
from check_h4_pr_m1_rmse import packet_fixture
A=B.A;p=B.p
MOCK=dict(commit='SYNTHETIC_ONLY',path='synthetic/run01/result.json',fixture='SYNTHETIC_ONLY_NO_MOLECULE')


def setup(commuting=False):
    config,schedule,old,prior,new=packet_fixture()
    req={r['identity']['m']:r for r in schedule['rows'] if p.arm(r['identity']['candidate'])=='PR'}
    z=np.diag([1.,-1.]).astype(complex);x=np.array([[0,1],[1,0]],complex)
    h0=.01*z;h1=.02*(z if commuting else x);tail=.03*z
    state=np.array([1.,0.],complex) if commuting else np.array([1.,.2j])/math.sqrt(1.04)
    ctx=SimpleNamespace(state=state,matrices=[h0,h1],tails={1:dict(matrix=tail,lambda_R=.5,scalar=.11)},
        ham=SimpleNamespace(constant=.07),full=h0+h1+tail+.18*np.eye(2),half_steps={},scientific_binding=schedule['design']['binding'])
    return config,schedule,old,prior,new,req,ctx


def pair(ctx,c):
    ev=direct.reference_pair(ctx,c['T'])
    return dict(binding=ctx.scientific_binding,T=c['T'],source=MOCK,evidence=ev)


def scalar_fixture(m=3):
    config,s,old,prior,new,req,ctx=setup(True);r=req[m];c=r['identity']['candidate'];ref=pair(ctx,c)
    sc=P.spectral_context(ctx,ROOT);rec=P.evolve(sc,dict(candidate=c,structure=r['identity']['structure']),ref['evidence']['reference'],root=ROOT)
    return r,rec,ref,dict(state_energy=.24,state_residual=0.)


class CompletionTests(unittest.TestCase):
    def test_power_matches_direct_noncommuting_and_signed_time(self):
        *_,req,ctx=setup(False);spectral=P.spectral_context(ctx,ROOT)
        for m in (0,2,4):
            c=req[m]['identity']['candidate'];ref=pair(ctx,c)['evidence']['reference'];row=dict(candidate=c,structure=p.structure(c))
            for sign in (1,-1):
                q=deepcopy(c);q['T']*=sign;q['delta']=q['T']/q['q'];q['id']=p.candidate_id(q)
                keys=direct.half_step_keys([dict(candidate=q)]);ctx.half_steps=dict(direct.half_step(ctx,k) for k in keys)
                a=direct.evolve(ctx,q,ref);b=P.evolve(spectral,dict(candidate=q,structure=p.structure(q)),ref,root=ROOT)
                for key in ('corrected','raw','exact_tail'):
                    zz=lambda t:complex(t['real'],t['imag'])
                    self.assertLess(abs(zz(a['signal'][key])-zz(b['signal'][key])),1e-10)
                self.assertEqual(a['signal']['log_B'],b['signal']['log_B'])
                self.assertEqual(b['backend']['tail_block_applications'],q['q'])

    def test_long_shape_no_event_loop_or_law_change(self):
        *_,req,ctx=setup(True);c=req[12]['identity']['candidate'];sc=P.spectral_context(ctx,ROOT)
        b=P.evolve(sc,dict(candidate=c,structure=p.structure(c)),pair(ctx,c)['evidence']['reference'],root=ROOT)
        self.assertEqual(b['backend']['logical_event_count'],33423360)
        self.assertEqual(b['backend']['tail_block_applications'],4096)
        self.assertIsNotNone(b['signal']['allowance_evidence']['exact_tail_norm_error'])
        self.assertTrue(b['backend']['raw_corrected_independent_propagation'])
        self.assertTrue(b['backend']['scalar_once'])

    def test_saved_norm_bound_keeps_missing_observation(self):
        req,rec,ref,state=scalar_fixture();rec['signal']['allowance_evidence'].pop('exact_tail_norm_error')
        proof=B._source_norm(req['identity']['candidate']['id']);prior=dict(u=1e-9,log_B_margin=1e-12,certified=False,source=MOCK)
        out=B.diagnose(req,rec,ref,state,MOCK,root=ROOT,norm_proof=proof,previous_allowance=prior)
        self.assertNotIn('exact_tail_norm_error',out['signal']['allowance_evidence'])
        self.assertIsNone(out['norm_evidence']['observed_error'])
        self.assertEqual(out['norm_evidence']['kind'],'SOURCE_GATE_LIMIT_NOT_OBSERVED')
        self.assertIsNone(out['actual_implementation']['law_error'])
        self.assertIsNone(out['conditional_geometry']['N_X'])

    def test_unknown_or_foreign_norm_proof_does_not_pass(self):
        req,rec,ref,state=scalar_fixture();rec['signal']['allowance_evidence']['exact_tail_norm_error']=None
        for proof in (None,dict(B._source_norm(req['identity']['candidate']['id']),upper_limit=2e-9)):
            with self.assertRaises(ValueError):B.diagnose(req,rec,ref,state,MOCK,root=ROOT,norm_proof=proof,previous_allowance=dict(u=1e-9,log_B_margin=1e-12,certified=False))
        with self.assertRaises(ValueError):B.check_norm({},B._source_norm(req['identity']['candidate']['id']),'MISSING_CANDIDATE')
        with self.assertRaises(ValueError):B.diagnose(req,rec,ref,state,MOCK,root=ROOT,norm_proof=B._source_norm(req['identity']['candidate']['id']))

    def test_observed_norm_failure_closure_and_old_Hoeffding_are_separate(self):
        from trottertracks import pr_r_dependent_n as old
        req,rec,ref,state=scalar_fixture(2)
        with patch.object(old,'analyze',side_effect=AssertionError('HP solver')),patch.object(p,'shot_record',side_effect=AssertionError('HP record')):
            out=B.diagnose(req,rec,ref,state,MOCK,root=ROOT)
        self.assertEqual(out['norm_evidence']['kind'],'OBSERVED_SCALAR')
        bad=deepcopy(rec);bad['signal']['allowance_evidence']['exact_tail_norm_error']=1e-4
        with self.assertRaises(ValueError):B.diagnose(req,bad,ref,state,MOCK,root=ROOT)
        bad=deepcopy(rec);bad['signal']['raw']['real']+=.01
        with self.assertRaises(ValueError):B.diagnose(req,bad,ref,state,MOCK,root=ROOT)

    def test_reference_anchor_preserves_original_and_all_new_paths(self):
        ref=dict(evidence=dict(reference=dict(real=1.,imag=1e-15),independent_reference=dict(real=1.,imag=-1e-15),signal_difference=2e-15),source=MOCK)
        signal=dict(reference=dict(real=1.,imag=0.))
        out=B.anchored_reference(signal,ref)
        self.assertEqual(out['evidence']['reference'],signal['reference'])
        self.assertEqual(out['new_pair_before_anchor'],ref['evidence'])
        self.assertEqual(ref['evidence']['reference']['imag'],1e-15)
        ref['evidence']['reference']['imag']=1e-3
        with self.assertRaises(ValueError):B.anchored_reference(signal,ref)

    def test_real_plan_calls_no_science_or_packet_recreation(self):
        with patch.object(B,'load_context',side_effect=AssertionError('array load')),patch.object(P,'evolve',side_effect=AssertionError('real mean')),patch.object(A,'nominal_axis_shots',side_effect=AssertionError('real N')),patch.object(A,'b_handoff',side_effect=AssertionError('old packet remake')):
            plan=B.plan(ROOT)
        self.assertEqual([r['m'] for r in plan['rows']],list(B.REMAINING))
        self.assertEqual([r['nominal_N_X'] for r in plan['preserved_nominals']],[64,60,38])
        self.assertEqual(plan['S1']['reference_actions'],4);self.assertEqual(plan['S2']['new_reference_actions'],0)
        self.assertEqual(plan['S2']['fast_means'],8);self.assertEqual(plan['S2']['direct_equivalence_means'],1)
        self.assertTrue(all(r['nominal_N_X'] is None for r in plan['rows']))
        self.assertIsNone(plan['S3']['global_RMSE']);self.assertEqual(plan['science_calls'],0)

    def test_default_CLI_passive_import_and_no_S3_execution(self):
        with tempfile.TemporaryDirectory(prefix='completion-metadata-') as tmp:
            r=subprocess.run([sys.executable,'-B','scripts/run_h4_pr_rmse_completion.py','--out',str(Path(tmp)/'plan.json')],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(json.loads((Path(tmp)/'plan.json').read_text())['science_calls'],0)
            r=subprocess.run([sys.executable,'-B','scripts/run_h4_pr_rmse_completion.py','--execute','--stage','S3','--out',str(Path(tmp)/'no')],cwd=ROOT,capture_output=True)
            self.assertNotEqual(r.returncode,0);self.assertFalse((Path(tmp)/'no').exists())
        env=__import__('os').environ.copy();env['PYTHONPATH']='src'
        subprocess.run([sys.executable,'-B','-c',"from trottertracks import h4_pr_rmse_completion;import sys;assert not any(k in sys.modules for k in ('numpy','scipy','qiskit'))"],cwd=ROOT,env=env,check=True)

    def test_preserved_results_and_old_numeric_SOURCE_bytes(self):
        for path in ('src/trottertracks/h4_pr_m1_rmse.py','src/trottertracks/pr_rpe_rmse.py','scripts/run_h4_pr_m1_rmse.py'):
            self.assertEqual((ROOT/path).read_bytes(),B.M.blob(ROOT,'ac873e7fcc42e937084fa153476cbd87565702b4',path))
        for path in (B.DIR+'run01/result.json',B.DIR+'run01/N_HANDOFF.json'):
            self.assertEqual((ROOT/path).read_bytes(),B.M.blob(ROOT,B.RELEASE,path))

    def test_old_policy_or_fake_sufficient_N_cannot_enter_future_packet(self):
        config,schedule,old,prior,new,requests,ctx=setup(True)
        record=deepcopy(next(r for r in A.evaluate(schedule)['records'] if r['request_id']==requests[2]['request_id']))
        record.update(status='UNQUALIFIED_MODEL_ASSUMPTIONS',nominal_N_X=17,nominal_N_Y=17)
        for change in ({'policy_id':'HOEFFDING'},{'sufficient_N_X':17},{'nominal_N_X':17.5},{'status':'QUALIFIED'}):
            bad=dict(deepcopy(record),**change)
            with self.subTest(change=change),self.assertRaises(ValueError):
                B.join_records(config,schedule,old,[bad],MOCK,root=ROOT)

    def test_unresolved_decoder_coordinates_remain_null_without_sampling(self):
        zero=A.decode_supplied_coordinates([(0.,0.)]*13,h=.1)
        tie=A.decode_supplied_coordinates([(1.,0.),(-1.,0.)],h=.1)
        self.assertEqual(zero['status'],'UNRESOLVED_ZERO_COORDINATES')
        self.assertEqual(tie['status'],'UNRESOLVED_BRANCH_TIE')
        self.assertIsNone(zero['energy_estimate_ha']);self.assertIsNone(tie['energy_estimate_ha'])

    def test_group_failure_has_no_retry_or_new_reference_fallback(self):
        c,s,old,prior,new,req,ctx=setup(True)
        planned=dict(H_state_binding=ctx.scientific_binding,rows=[dict(m=m,T=req[m]['identity']['candidate']['T']) for m in B.S2])
        def saved(root,row,binding):
            return dict(binding=binding,T=row['T'],source=MOCK,evidence=dict(reference=dict(real=1.,imag=0.),independent_reference=dict(real=1.,imag=0.),signal_difference=0.))
        original_blob=B.M.blob
        def source_blob(root,commit,path):
            return (ROOT/path).read_bytes() if commit=='0'*40 and path in (B.MODULE,B.RUNNER,'src/trottertracks/pr_rpe_power_mean.py') else original_blob(root,commit,path)
        with tempfile.TemporaryDirectory(prefix='completion-SYNTHETIC-failure-') as tmp,patch.object(B,'plan',return_value=planned),patch.object(B,'load_fixed',return_value=(c,old,s,req)),patch.object(B,'load_context',return_value=(ctx,dict(state_energy=.24,state_residual=0.))),patch.object(B,'_archive',return_value=({},[])),patch.object(B,'saved_reference',side_effect=saved),patch.object(B.M,'blob',side_effect=source_blob),patch.object(P,'evolve',side_effect=ValueError('SYNTHETIC_POWER_GATE')) as means,patch.object(direct,'reference_pair',side_effect=AssertionError('No reference fallback')):
            out=Path(tmp)/'run'
            with self.assertRaises(ValueError):B.execute(ROOT,out,'0'*40,'S2')
            result=json.loads((out/'result.json').read_text())
            self.assertEqual(means.call_count,8);self.assertEqual(len(result['candidate_errors']),8)
            self.assertEqual(sorted(result['fast_mean_rounds_attempted']),list(B.S2))
            self.assertFalse(result['automatic_retry']);self.assertEqual(result['new_reference_actions'],0)
            self.assertFalse((out/'N_HANDOFF.json').exists())

    def test_synthetic_group_runner_and_full_packet_preservation(self):
        c,s,old,prior,new,req,ctx=setup(True)
        previous=B.M.handoff(c,s,old,prior,new,MOCK,root=ROOT,evidence_domain='SYNTHETIC_ONLY')
        spectral=P.spectral_context(ctx,ROOT);references={m:pair(ctx,r['identity']['candidate']) for m,r in req.items()}
        rows=[];raw={};endpoints=[]
        for m in B.REMAINING:
            candidate=req[m]['identity']['candidate'];entry=dict(m=m,T=candidate['T'])
            if m in B.S1:
                record=P.evolve(spectral,dict(candidate=candidate,structure=req[m]['identity']['structure']),references[m]['evidence']['reference'],root=ROOT)
                record['signal']['allowance_evidence'].pop('exact_tail_norm_error')
                raw[candidate['id']]=record;endpoints.append(dict(candidate=candidate,epsilon=.0005,u=1e-9,log_B_margin=1e-12))
                entry.update(saved_mean=MOCK,norm_evidence=B._source_norm(candidate['id']),old_allowance_source=MOCK)
            rows.append(entry)
        planned=dict(H_state_binding=ctx.scientific_binding,rows=rows)
        def saved(_,r,b):return references[r['m']]
        original_blob=B.M.blob
        def source_blob(root,commit,path):
            return (ROOT/path).read_bytes() if commit=='0'*40 and path in (B.MODULE,B.RUNNER,'src/trottertracks/pr_rpe_power_mean.py') else original_blob(root,commit,path)
        def read(_,source):return {'model_records':prior} if source['commit']==B.V.S0 else {'model_record':new}
        with tempfile.TemporaryDirectory(prefix='completion-SYNTHETIC-') as tmp,patch.object(B,'plan',return_value=planned),patch.object(B,'load_fixed',return_value=(c,previous,s,req)),patch.object(B,'load_context',return_value=(ctx,dict(state_energy=.24,state_residual=0.))),patch.object(B,'_archive',return_value=(raw,endpoints)),patch.object(B,'saved_reference',side_effect=saved),patch.object(B.M,'blob',side_effect=source_blob),patch.object(p.SavedArtifacts,'read',read):
            result=B.execute(ROOT,Path(tmp)/'run','0'*40,'S1_S2')
            self.assertEqual(result['status'],'LOCAL_NOMINAL_BATCH_COMPLETE_GLOBAL_UNQUALIFIED')
            self.assertEqual(result['new_fast_means'],8);self.assertEqual(result['new_direct_validation_means'],1)
            self.assertEqual(result['new_reference_actions'],4)
            self.assertEqual(result['reference_actions_attempted'],4)
            self.assertEqual(result['direct_validations_attempted'],1)
            packet=json.loads((Path(tmp)/'run/N_HANDOFF.json').read_text())
            for m in (0,1,6):self.assertEqual(packet['rounds'][m]['nominal_N_X'],previous['rounds'][m]['nominal_N_X'])
            self.assertTrue(all(r['N_X'] is None for r in packet['rounds']))
            self.assertIsNone(packet['rmse_evidence']['rmse_bound_ha'])
            self.assertEqual(packet['diagnostic_scope']['newly_evaluated_rounds'],list(B.REMAINING))


if __name__=='__main__':unittest.main()
