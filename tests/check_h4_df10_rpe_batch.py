"""Artificial matrices/mock batches and static saved fixtures; no H4 numerics."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from trottertracks import h4_df10_rpe_batch_provider as h4
from trottertracks import pf_pr_rpe_pipeline as p
import run_h4_df10_rpe_batch as cli
from check_pf_pr_rpe_pipeline import fake_shot


def fake(task,row):
    n=fake_shot(task,row);n.update(diagnostics=dict(synthetic_only=True),origin='MOCK_ONLY');return n


def kernel():
    import run_h6_df19_rpe_d2 as io
    from trottertracks import h6_df19_rpe_d2 as action
    return io,action


def mock_batch(packet):
    io,action=kernel();factory=Mock(return_value=(SimpleNamespace(half_steps={}),{}))
    provider=h4.H4NBatch(packet,dict(path='SYNTHETIC_ONLY'),factory)
    return provider,factory,io,action


class H4BatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.packet=h4.prepare(ROOT)

    def ready(self):return deepcopy(self.packet)

    def test_static_prepare_import_has_no_numerical_calls(self):
        code="import sys;from pathlib import Path;from trottertracks.h4_df10_rpe_batch_provider import prepare;p=prepare(Path.cwd());assert not any(x in sys.modules for x in ('numpy','scipy','qiskit'));assert len(p['saved_N'])==2"
        env=__import__('os').environ.copy();env['PYTHONPATH']='src:scripts'
        subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,env=env,check=True)
        with patch.object(h4.scalar,'analyze_bound_point',side_effect=AssertionError('No saved reanalysis')), \
             patch.object(h4,'load_saved_context',side_effect=AssertionError('No arrays')), \
             patch.object(p,'evaluate',side_effect=AssertionError('No G')):
            ready=h4.prepare(ROOT);plan=cli.plan(ready)
        self.assertEqual([n['plan']['N_X'] for n in ready['saved_N']],[705,977])
        self.assertEqual(len(plan['rows']),26)
        self.assertEqual(sum(r['input_route']=='SAVED_SIGNAL_SCALAR' for r in plan['rows']),2)
        self.assertEqual(sum(r['input_route']=='NEW_REFERENCE_AND_FINITE_MEAN' for r in plan['rows']),22)
        self.assertNotIn('/private/',json.dumps(plan));self.assertIsNone(plan['total_resources'])

    def test_saved_m4_raw_allowance_reference_and_binding_are_available(self):
        ready=self.ready()
        for cid,point in ready['saved_points'].items():
            row=next(r for r in ready['task']['candidates'] if r['candidate']['id']==cid)
            h4.validate_saved_point(ready['task'],row,point)
            self.assertEqual(row['m'],4);self.assertEqual(point['saved_signal']['reference'],ready['context_metadata']['reference']['1.6'])
            self.assertTrue(point['saved_allowance']['empirical_not_certified'])
            self.assertEqual(ready['saved_origins'][cid]['allowance_epsilon'],.0005)
            self.assertNotIn('N',point['saved_finite_time_endpoint'])
            self.assertEqual(point['prices'],[])
        with patch.object(h4,'sha',return_value='FOREIGN'):
            with self.assertRaisesRegex(ValueError,'bytes differ'):h4.prepare(ROOT)

    def test_fixed_task_rejects_policy_binding_and_candidate_changes(self):
        ready=self.ready()
        for change in (lambda t:t['configuration'].update(alpha_axis=.025),
                lambda t:t['configuration'].update(beta_stat=.1),
                lambda t:t['binding'].update(state='FOREIGN'),
                lambda t:t['profile'].update(control='OTHER'),
                lambda t:t['source'].update(commit='FOREIGN'),
                lambda t:t['candidates'][0]['candidate'].update(method='FULL'),
                lambda t:t['candidates'][1]['candidate'].update(R_outer=16),
                lambda t:t['candidates'][0].update(q_PF=2)):
            task=deepcopy(ready['task']);change(task)
            with self.assertRaises(ValueError):h4.validate_task(task,ready['task'])

    def test_B0_is_prefix7_with_constant_once_and_keeps_B0_identity(self):
        io,action=kernel()
        import numpy as np
        from scipy.linalg import expm
        a=np.array([[.2,.1+.03j],[.1-.03j,-.15]],complex)
        matrices=[a]+[np.array([[.01*j,.002j],[-.002j,-.01*j]]) for j in range(1,11)]
        ctx=SimpleNamespace(state=np.array([1.,1j])/2**.5,ham=SimpleNamespace(constant=.17),matrices=matrices,half_steps={},tails={7:dict(scalar=.29)})
        ctx.full=sum(matrices)+.17*np.eye(2)
        for q in (1,16):
            c=deepcopy(self.packet['task']['candidates'][0]['candidate']);c.update(q=q,T=.1*q)
            for k in action.half_step_keys([dict(candidate=c)]):key,value=action.half_step(ctx,k);ctx.half_steps[key]=value
            with patch('numpy.linalg.eigh',side_effect=AssertionError('No eigensolve')):
                result=h4.evolve_h4(ctx,c,dict(real=1.,imag=0.))
            v=ctx.state.copy()
            for _ in range(q):
                for matrix in (*matrices[:8],*reversed(matrices[:8])):v=expm(-.05j*matrix)@v
            z=np.vdot(ctx.state,v*np.exp(-1j*.17*c['T']))
            actual=complex(result['signal']['corrected']['real'],result['signal']['corrected']['imag'])
            self.assertLess(abs(actual-z),2e-13);self.assertEqual(result['candidate'],c)
            self.assertEqual(result['signal']['log_B'],0.);self.assertEqual(result['event_count'],0)
            self.assertGreater(abs(z-np.vdot(ctx.state,expm(-1j*c['T']*ctx.full)@ctx.state)),1e-3)
            self.assertGreater(abs(actual-z*np.exp(-.29j*c['T'])),1e-3)

    def test_PR_S2_R4_raw_corrected_matches_independent_saved_algorithm_on_toy(self):
        kernel()
        from check_h6_df19_rpe_d2 import toy,ready,old
        from trottertracks import h6_df19_rpe_d2 as action
        import numpy as np
        for q in (1,16):
            ctx=toy();c=dict(system='TOY',method='PR',formula='2nd',prefix=1,R_outer=4,K=2,T=.1*q,q=q,delta=.1,id='TOY')
            ref=ready(ctx,c);got=h4.evolve_h4(ctx,c,ref)
            with patch.object(old,'weights',action.schedule.weights):
                want=old.compare_prefix(ctx.state,[old.spectrum(a) for a in ctx.matrices[:2]],old.spectrum(ctx.tails[1]['matrix']),.21,.5,'2nd',q,T=.1*q,total_steps=4,cutoffs=(2,))['finite_rows'][0]
            for field,target in [('corrected',want['corrected_signal']),('raw',want['raw_signal'])]:
                val=got['signal'][field];self.assertLess(abs(complex(val['real'],val['imag'])-target),2e-13)
            self.assertEqual(got['allocation'],[4]);self.assertEqual(got['event_count'],4*q)
            self.assertAlmostEqual(got['signal']['log_B'],want['log_B'],places=13)
            ctx.ham.constant+=.31;shift=h4.evolve_h4(ctx,c,ref)
            for field in ('corrected','raw','exact_tail'):
                a=got['signal'][field];b=shift['signal'][field]
                self.assertLess(abs(complex(b['real'],b['imag'])-np.exp(-.31j*c['T'])*complex(a['real'],a['imag'])),2e-13)

    def test_artificial_scalar_unknown_actual_and_T04_reference_requirement(self):
        ready=self.ready();task=ready['task'];row=task['candidates'][4]
        sig=dict(reference=dict(real=1.,imag=0.),corrected=dict(real=.999,imag=.0001),raw=dict(real=.999,imag=.0001),
            total_signed_bias=dict(real=-.001,imag=.0001),log_B=0.,allowance_evidence=dict(maximum_intermediate_norm=1.,raw_corrected_vector_closure=0.))
        record=dict(signal=sig,candidate=deepcopy(row['candidate']),**{k:deepcopy(row['structure'][k]) for k in ('signed_weights','allocation','event_count')})
        result=h4.analyze_fresh(task,row,record,dict(signal_difference=0.),dict(state_energy=0.,state_residual=0.),dict(path='TOY_ONLY'))
        self.assertEqual(result['plan']['status'],'CONDITIONAL_ROUND_PASS')
        self.assertIsNone(result['actual']['N_X']);self.assertFalse(result['diagnostics']['ground_state_certified'])
        self.assertFalse(result['diagnostics']['allowance']['certified'])
        with self.assertRaises(KeyError):h4.analyze_fresh(task,row,record,{},dict(state_energy=0.,state_residual=0.),{})
        with self.assertRaises(ValueError):h4.analyze_fresh(task,row,record,dict(signal_difference=1e-8),dict(state_energy=0.,state_residual=0.),{})
        wrong=deepcopy(record);wrong['allocation']=[16]
        with self.assertRaisesRegex(ValueError,'signed recipe differs'):h4.analyze_fresh(task,row,wrong,dict(signal_difference=0.),dict(state_energy=0.,state_residual=0.),{})
        bad=deepcopy(result);bad['identity']['binding']['state']='FOREIGN'
        with self.assertRaises(ValueError):p.plan(task,[bad],[])

    def test_full26_mock_batch_calls_missing24_once_and_keeps_C_interface(self):
        ready=self.ready();provider,factory,io,action=mock_batch(ready);task=ready['task']
        with patch.object(h4,'analyze_point',side_effect=lambda t,r,*a,**kw:fake(t,r)) as saved, \
             patch.object(h4,'analyze_fresh',side_effect=lambda t,r,*a,**kw:fake(t,r)) as fresh, \
             patch.object(action,'reference_pair',return_value=dict(reference=dict(real=1.,imag=0.))) as refs, \
             patch.object(action,'half_step',side_effect=lambda ctx,key:(key,'MOCK_MATRIX')) as halves, \
             patch.object(io,'worker_resources',side_effect=lambda ctx,n:dict(half_step_workers=n)) as workers, \
             patch.object(h4,'evolve_h4',return_value={}) as means, \
             patch.object(p,'evaluate',side_effect=AssertionError('No G')):
            result=p.run_missing(task,ready['saved_N'],[],n_provider=provider,c_provider=None)
        factory.assert_called_once();self.assertEqual(saved.call_count,2);self.assertEqual(fresh.call_count,22)
        self.assertEqual(refs.call_count,11);self.assertEqual(means.call_count,22);self.assertEqual(halves.call_count,8)
        self.assertEqual([c.args[1] for c in workers.call_args_list],[8,22])
        self.assertEqual(provider.last_run['PR_exact_tail_controls'],11)
        self.assertEqual(result['shots'][:2],ready['saved_N']);self.assertEqual(len(result['shots']),26)
        self.assertEqual(result['prices'],[]);self.assertIsNone(result['totals'])
        # Existing C record interface consumes the resulting same identity.
        row=task['candidates'][0]
        price=dict(**p.measurement(row,'X'),identity=p.identity(task,row),axis='X',classification='DIRECT_COMPILED_DETERMINISTIC',cohort='MOCK_ONLY',n_paths=1,
            mean=dict(rz_count=1.),source=dict(path='MOCK_ONLY'),expectation='DETERMINISTIC',model=None,uncertainty=None,normalization_is_price_multiplier=False,peak_qubits=9)
        self.assertEqual(p.plan(task,result['shots'],[price])['missing_N'],[])

    def test_saved_N_FAIL_stops_before_provider_and_saved_m3_direct_is_copy(self):
        ready=self.ready();provider=Mock(side_effect=AssertionError('No retry'))
        ready['saved_N'][0]['plan'].update(status='UNQUALIFIED_ROUND',N_X=None,N_Y=None)
        result=p.run_missing(ready['task'],ready['saved_N'],[],n_provider=provider)
        self.assertEqual(result['status'],'STOP_UNQUALIFIED_N');provider.assert_not_called()
        ready=self.ready();provider,factory,io,action=mock_batch(ready)
        rows=[r for r in ready['task']['candidates'] if r['m']==3]
        self.assertEqual(provider(ready['task'],rows),ready['saved_N']);factory.assert_not_called()

    def test_missing_saved_signal_or_raw_stops_without_fresh_fallback(self):
        for remove in ('point','raw','u'):
            ready=self.ready();cid=h4.M4_IDS[0]
            if remove=='point':ready['saved_points'].pop(cid)
            elif remove=='raw':ready['saved_points'][cid]['saved_signal'].pop('raw')
            else:ready['saved_points'][cid]['saved_allowance'].pop('u')
            provider,factory,io,action=mock_batch(ready)
            with patch.object(h4,'analyze_point',side_effect=AssertionError('Preflight first')):
                result=provider(ready['task'],[r for r in ready['task']['candidates'] if r['m']!=3])
            factory.assert_not_called();self.assertEqual(provider.last_run['status'],'STOP_SAVED_INPUT')
            self.assertTrue(all(n['plan']['N_X'] is None for n in result))

    def test_saved_signal_scalar_FAIL_is_preserved_without_replacement(self):
        ready=self.ready();provider,factory,io,action=mock_batch(ready)
        with patch.object(h4,'analyze_point',side_effect=lambda t,r,*a,**kw:h4.failed_record(t,r,{},'MOCK_SAVED_FAIL','UNQUALIFIED_ROUND')) as scalar:
            result=provider(ready['task'],[r for r in ready['task']['candidates'] if r['m']!=3])
        self.assertEqual(scalar.call_count,2);factory.assert_not_called()
        self.assertEqual(provider.last_run['status'],'STOP_UNQUALIFIED_SAVED_N')
        self.assertEqual(sum(n['plan']['status']=='UNQUALIFIED_ROUND' for n in result),2)
        self.assertTrue(all(n['plan']['N_X'] is None for n in result))

    def test_reference_error_stops_and_finite_mean_FAIL_is_not_retried(self):
        ready=self.ready();rows=ready['task']['candidates'][:2];provider,factory,io,action=mock_batch(ready)
        with patch.object(action,'reference_pair',side_effect=ValueError('MOCK_REFERENCE_GATE')) as refs, \
             patch.object(h4,'evolve_h4') as means:
            with self.assertRaisesRegex(ValueError,'MOCK_REFERENCE_GATE'):provider(ready['task'],rows)
            refs.assert_called_once();means.assert_not_called()
        self.assertEqual(provider.last_run['status'],'STOP_REFERENCE_GATE')
        with patch.object(action,'reference_pair',return_value=dict(reference={})), \
             patch.object(action,'half_step',side_effect=lambda ctx,key:(key,None)), \
             patch.object(io,'worker_resources',return_value=dict(half_step_workers=2)), \
             patch.object(h4,'evolve_h4',side_effect=ValueError('MOCK_MEAN_GATE')) as means:
            result=provider(ready['task'],rows)
        self.assertEqual(means.call_count,2);self.assertEqual(provider.last_run['status'],'STOP_UNQUALIFIED_N')
        self.assertTrue(all(n['plan']['status']=='NUMERICAL_GATE_FAILURE' for n in result))

    def test_mock_runner_persists_coverage_no_C_G_and_no_overwrite(self):
        ready=self.ready();provider=Mock(return_value=[fake(ready['task'],r) for r in ready['task']['candidates'] if r['m']!=3]);provider.last_run=dict(synthetic_only=True)
        with tempfile.TemporaryDirectory() as temp,patch.object(p,'evaluate',side_effect=AssertionError('No G')):
            out=Path(temp)/'MOCK';result=cli.execute(ready,provider,'SYNTHETIC_ONLY',out)
            provider.assert_called_once();self.assertEqual(len(provider.call_args.args[1]),24)
            self.assertEqual(result['status'],'H4_N_BATCH_EVALUATED');self.assertEqual(len(result['records']),26)
            coverage=json.loads((out/'COVERAGE.json').read_bytes());self.assertTrue(coverage['all_13_N_rounds_conditional_pass'])
            self.assertTrue(all(r['actual_N_X'] is None and r['C'] is None and r['G'] is None for r in coverage['rows']))
            with self.assertRaises(FileExistsError):cli.execute(ready,provider,'SYNTHETIC_ONLY',out)
            provider.assert_called_once()
            invalid_records=deepcopy(result['records']);cli.invalidate_new_records(ready,invalid_records)
            self.assertEqual(invalid_records[:2],ready['saved_N'])
            self.assertEqual(p.run_missing(ready['task'],invalid_records,[])['status'],'STOP_UNQUALIFIED_N')
            self.assertIn('computed_plan_before_source_failure',invalid_records[2]['diagnostics'])
            invalid=cli.coverage(ready,invalid_records,'MOCK',valid=False)
            self.assertEqual(invalid['qualified_rounds_by_arm'],dict(PF=[3],PR=[3]))
            incomplete=cli.coverage(ready,ready['saved_N'],'MOCK')
            self.assertEqual(incomplete['qualified_rounds_by_arm'],dict(PF=[3],PR=[3]))
            failure=Mock(side_effect=ValueError('MOCK_REFERENCE'));failure.last_run=dict(status='STOP_REFERENCE_GATE')
            failed=cli.execute(ready,failure,'MOCK',Path(temp)/'FAIL')
            self.assertEqual(failed['status'],'H4_STOP_TECHNICAL_OR_REFERENCE');failure.assert_called_once()
            self.assertEqual(failed['records'],ready['saved_N'])

    def test_common_C_and_H6_numerical_and_historical_artifacts_unchanged(self):
        paths=list(cli.SOURCE_FILES[3:])+['src/trottertracks/h4_df10_price_provider.py','src/trottertracks/h6_rpe_batch_provider.py',
            'docs/research/h6_df19_rpe_m8_12_results_20261110/run01/result.json','docs/research/h6_df19_rpe_m8_12_results_20261110/run01/COVERAGE.json']
        for path in paths:
            current=(ROOT/path).read_bytes() if (ROOT/path).is_file() else h4.blob(ROOT,'HEAD',path)
            self.assertEqual(current,h4.blob(ROOT,h4.BASELINE,path),path)
        d1=json.loads(h4.blob(ROOT,h4.N_CARRIER,'docs/research/h4_df10_rpe_d1_20261110/result.json'))
        for old,n in zip(d1['rows'],self.packet['saved_N']):
            self.assertEqual(n['plan'],old['ideal_numerical_pilot'])
            self.assertEqual(n['actual'],old['actual_implementation'])


if __name__=='__main__':unittest.main()
