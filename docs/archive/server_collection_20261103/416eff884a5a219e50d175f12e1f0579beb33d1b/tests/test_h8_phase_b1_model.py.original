"""Artificial B1 inputs only. No saved H8 costs, arrays, fit or residuals."""
from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import h8_phase_b1_validation as gate
import run_h8_phase_b1_validation as runner
import launch_h8_phase_b1_validation as launcher
m=gate.model
SOURCE='a'*40
INPUT='b'*64


def fixture():
    rows=[]
    for j,(method,axis) in enumerate(m.STRATA):
        a,b,d=100*(j+1),30*(j+1),7*(j+1)
        for p in (14,21):
            for q in ((1,8) if method=='B0' else (1,2,4,8)):
                rows.append(dict(method=method,prefix=p,q=q,axis=axis,formula='2nd',
                    role='HOLDOUT' if method=='B0' and (p,q)==m.HOLDOUT else 'TRAIN',
                    saved_RZ=a+b*q+d*p*q,control='symmetric_directional',
                    runtime_profile_sha256=m.PROFILE,compiler_hash=m.COMPILER,
                    origin_SOURCE=SOURCE if method=='B0' else m.B0I_SOURCE,
                    scalar_policy=m.SCALARS[method],cost_preparation_hash=m.PREPARATIONS[p],
                    fingerprint='c'*64,input_binding_sha256=INPUT,
                    evidence={k:True for k in m.REQUIRED}))
    return rows


class ModelTests(unittest.TestCase):
    def run_model(self,rows):
        calls=[];events=[]
        result=m.evaluate(rows,acquisition_source=SOURCE,input_binding=INPUT,
            take=calls.append,emit=lambda name,value:events.append((name,deepcopy(value))))
        return result,calls,events
    def test_four_independent_models_and_exact_calls(self):
        result,calls,_=self.run_model(fixture())
        self.assertEqual(result['status'],'COMPLETE_MANDATORY_STOP')
        for j,row in enumerate(result['frozen_model']['models']):
            self.assertEqual(row['coefficients'],list(map(str,(100*(j+1),30*(j+1),7*(j+1)))))
        self.assertEqual(calls.count('cost_fit'),4)
        for key,count in m.CAPS.items():self.assertEqual(calls.count(key),count)
        self.assertFalse(result['frozen_model']['B2_authorized'])
        self.assertFalse(result['frozen_model']['C_authorized'])
    def test_holdout_does_not_affect_fit(self):
        a,_,_=self.run_model(fixture());rows=fixture()
        next(r for r in rows if r['method']=='B0' and r['axis']=='cosine' and r['role']=='HOLDOUT')['saved_RZ']+=1
        b,_,_=self.run_model(rows)
        self.assertEqual(a['frozen_model']['models'][0]['coefficients'],b['frozen_model']['models'][0]['coefficients'])
    def test_input_order_does_not_change_model(self):
        self.assertEqual(self.run_model(fixture())[0],self.run_model(list(reversed(fixture())))[0])
    def test_exact_two_percent_fails(self):self.assertFalse(m.diagnostic(F(102),100)['PASS'])
    def test_just_below_two_percent_passes(self):self.assertTrue(m.diagnostic(F(101999,1000),100)['PASS'])
    def test_measured_denominator(self):self.assertTrue(m.diagnostic(F(100),102)['PASS'])
    def test_nonpositive_prediction(self):
        with self.assertRaisesRegex(ValueError,'NONPOSITIVE'):m.predict((F(-1),F(0),F(0)),14,1)
    def test_singular_design(self):
        with self.assertRaisesRegex(ValueError,'SINGULAR'):m.solve([dict(prefix=14,q=1,saved_RZ=x) for x in (1,2,3)])
    def test_rational_ols_normal_equation_residuals(self):
        rows=[dict(prefix=p,q=q,saved_RZ=100+30*q+7*p*q+(p==21 and q==8)) for p in (14,21) for q in (1,2,4,8)]
        coefficients=m.solve(rows)
        self.assertTrue(any(v.denominator!=1 for v in coefficients))
        for col in (lambda r:1,lambda r:r['q'],lambda r:r['prefix']*r['q']):
            self.assertEqual(sum((m.predict(coefficients,r['prefix'],r['q'])-r['saved_RZ'])*col(r) for r in rows),0)
    def test_first_failed_cosine_stops_before_other_fits(self):
        rows=fixture();next(r for r in rows if r['method']=='B0' and r['axis']=='cosine' and r['role']=='HOLDOUT')['saved_RZ']*=2
        calls=[];events=[]
        with self.assertRaisesRegex(ValueError,'AXIS_ERROR_GE_2_PERCENT:B0/cosine'):
            m.evaluate(rows,acquisition_source=SOURCE,input_binding=INPUT,take=calls.append,emit=lambda n,v:events.append(deepcopy(v)))
        self.assertEqual(calls.count('cost_fit'),1);self.assertFalse(events[-1]['checks'][-1]['PASS'])
        self.assertEqual(calls.count('refit'),0)
    def test_sine_failure_is_not_averaged_with_cosine(self):
        rows=fixture();next(r for r in rows if r['method']=='B0' and r['axis']=='sine' and r['role']=='HOLDOUT')['saved_RZ']*=2
        calls=[]
        with self.assertRaisesRegex(ValueError,'B0/sine'):
            m.evaluate(rows,acquisition_source=SOURCE,input_binding=INPUT,take=calls.append,emit=lambda *_:None)
        self.assertEqual(calls.count('cost_fit'),2)
    def test_b0i_bad_anchor_stops_without_refit(self):
        rows=fixture();next(r for r in rows if r['method']=='B0I' and r['axis']=='cosine' and (r['prefix'],r['q'])==(21,8))['saved_RZ']*=2
        calls=[]
        with self.assertRaisesRegex(ValueError,'B0I/cosine'):
            m.evaluate(rows,acquisition_source=SOURCE,input_binding=INPUT,take=calls.append,emit=lambda *_:None)
        self.assertEqual(calls.count('cost_fit'),3)
    def test_all_provenance_failures_precede_any_fit(self):
        changes={'origin_SOURCE':'d'*40,'runtime_profile_sha256':'d'*64,'compiler_hash':'d'*64,
            'cost_preparation_hash':'d'*64,'control':'ordinary','scalar_policy':'other',
            'input_binding_sha256':'d'*64,'formula':'4th','saved_RZ':True,'fingerprint':'',
            'prefix':True,'q':0,'role':'HOLDOUT'}
        for key,value in changes.items():
            with self.subTest(key=key):
                rows=fixture();rows[0][key]=value;calls=[]
                with self.assertRaises(ValueError):
                    m.evaluate(rows,acquisition_source=SOURCE,input_binding=INPUT,take=calls.append,emit=lambda *_:None)
                self.assertEqual(calls,[])
    def test_missing_duplicate_or_extra_axis(self):
        for rows in (fixture()[:-1],fixture()+fixture()[:1],fixture()[:-1]+fixture()[:1]):
            with self.assertRaises(ValueError):m.validate_rows(rows,SOURCE,INPUT)
    def test_missing_evidence(self):
        for key in m.REQUIRED:
            rows=fixture();rows[0]['evidence'][key]=False
            with self.assertRaisesRegex(ValueError,'PROVENANCE'):m.validate_rows(rows,SOURCE,INPUT)
    def test_old_b0_source_rejected(self):
        for source in ('c31ef424175c02379eafebe8fc6c7548c1d9aabe','cb8a56c6b2a772827275f9200b02638030f33f72'):
            with self.assertRaisesRegex(ValueError,'NEW_B0_SOURCE'):m.validate_rows(fixture(),source,INPUT)


class ReceiptAndGrantTests(unittest.TestCase):
    def setUp(self):
        self.terminal=dict(status='COMPLETE_MANDATORY_STOP',SOURCE=SOURCE,next_stage_authorized=False)
        self.launcher=dict(status='COMPLETE_MANDATORY_STOP',native_exit_code=0,first_stop=None,cleanup_errors=[],all_reaped=True,max_monitor_gap_seconds=1.)
        self.final=dict(status='PASS_MANDATORY_STOP',SOURCE=SOURCE,profile_sha256=m.PROFILE,next_stage_authorized=False)
    def test_complete_acquisition(self):gate.completed_acquisition(SOURCE,self.terminal,self.launcher,self.final)
    def test_stop_with_eight_axes_is_still_rejected(self):
        self.terminal.update(status='STOP',axes=[{}]*8)
        with self.assertRaisesRegex(ValueError,'ACQUISITION_NOT_COMPLETE'):gate.completed_acquisition(SOURCE,self.terminal,self.launcher,self.final)
    def test_every_cleanup_failure_rejected(self):
        for key,value in dict(status='STOP',native_exit_code=2,first_stop='THREADS',cleanup_errors=['live'],all_reaped=False,max_monitor_gap_seconds=5.01).items():
            with self.subTest(key=key):
                l=dict(self.launcher);l[key]=value
                with self.assertRaises(ValueError):gate.completed_acquisition(SOURCE,self.terminal,l,self.final)
    def test_final_gate_mismatch(self):
        self.final['profile_sha256']='d'*64
        with self.assertRaisesRegex(ValueError,'FINAL_GATE'):gate.completed_acquisition(SOURCE,self.terminal,self.launcher,self.final)
    def test_draft_rejected_before_fit_or_filesystem_mutation(self):
        with patch.object(gate.io,'read',return_value={}),patch.object(gate.io,'sha',return_value='e'*64),patch.object(gate,'verify_source') as source,patch.object(gate,'normalize') as normalize:
            with self.assertRaisesRegex(ValueError,'NEW_EXPLICIT_B1_GRANT'):runner.execute('cfg','grant','fresh')
            source.assert_not_called();normalize.assert_not_called()
    def test_invalid_launcher_never_forks_or_changes_affinity(self):
        with patch.object(gate.io,'read',return_value={}),patch.object(gate.io,'sha',return_value='e'*64),patch.object(launcher.observer,'supervise') as supervise,patch.object(os_module(),'sched_setaffinity') as affinity:
            with self.assertRaises(ValueError):launcher.launch('cfg','grant')
            supervise.assert_not_called();affinity.assert_not_called()
    def test_bytes_missing_mismatch_and_json_rejections(self):
        with tempfile.TemporaryDirectory(prefix='b1-fixture-',dir=Path.home()) as tmp:
            p=Path(tmp)/'input.json';p.write_text('{}')
            row=dict(path=str(p),bytes=2,sha256=gate.io.sha(p));items={k:dict(row) for k in gate.INPUTS}
            with patch.object(gate,'REGISTRY_SHA',row['sha256']):
                gate.verify_receipts(items)
                p.write_text('{ }')
                with self.assertRaisesRegex(ValueError,'BYTES'):gate.verify_receipts(items)
                p.unlink()
                with self.assertRaises(ValueError):gate.verify_receipts(items)
            for data in ('{"x":1,"x":2}','{"x":NaN}'):
                p.write_text(data)
                with self.assertRaises(ValueError):gate.io.read(p)
    def test_safe_default_has_no_real_receipt_access(self):
        for name in ('run_h8_phase_b1_validation.py','launch_h8_phase_b1_validation.py','h8_phase_b1_validation.py'):
            result=subprocess.run([sys.executable,'-P','-B',str(ROOT/'scripts'/name)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr);json.loads(result.stdout)
    def test_new_grant_binding_and_scope_fail_closed(self):
        base='/home/AbeHiromu/projects/h6-handoff-evidence/b1-artificial-only'
        config=dict(SOURCE=SOURCE,plan_sha256=m.digest(m.plan()),output=base+'/output',control=base+'/control',lock=base+'/lock')
        grant=dict(schema='H8_B1_NEW_FIT_GRANT_V1',valid=True,approved_by_user=True,runtime_authorization=True,
            approval_text='ARTIFICIAL FIXTURE NOT AUTHORIZATION',scope=m.SCOPE,SOURCE=SOURCE,
            config_sha256='d'*64,plan_sha256=config['plan_sha256'],resources=m.RESOURCES,call_caps=m.CAPS,
            one_shot=True,retry=False,resume=False,refit=False,additional_measurement=False,
            B2_authorized=False,C_authorized=False,next_stage_authorized=False,
            driver_cpu=0,observer_cpu=1,allowed_cpus=[0,1])
        self.assertEqual(gate.authorize(config,grant,config_sha='d'*64,affinity={0,1}),[0,1])
        changes=dict(valid=False,approved_by_user=False,runtime_authorization=False,scope='B2',SOURCE='c'*40,
            config_sha256='c'*64,plan_sha256='c'*64,resources={},call_caps={},retry=True,resume=True,
            refit=True,additional_measurement=True,B2_authorized=True,C_authorized=True,
            next_stage_authorized=True,allowed_cpus=[0,0],approval_text='')
        for key,value in changes.items():
            bad=deepcopy(grant);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):gate.authorize(config,bad,config_sha='d'*64,affinity={0,1})
    def test_complete_artificial_receipt_normalization_without_fit(self):
        plan=gate.acquisition.plan();runtime={str(i):'d'*64 for i in range(34273)}
        provider={str(i):'e'*64 for i in range(265)}
        old=dict(SOURCE=SOURCE,execution_sealed=True,provider_source=gate.acquisition.PROVIDER,
            expected_profile_digest=m.PROFILE,runtime_file_hashes=runtime,provider_hashes=provider,
            files=plan['binding']['inputs'],plan_sha256=m.digest(plan),resources=plan['resources'])
        grant=dict(schema='H8_B0_ANCHOR_NEW_SCOPE_GRANT_V1',approved_by_user=True,runtime_authorization=True,
            approval_text='ARTIFICIAL ONLY',SOURCE=SOURCE,scope=gate.acquisition.SCOPE,
            config_sha256='d'*64,plan_sha256=m.digest(plan),resources=plan['resources'],
            call_caps=plan['call_caps'],one_shot=True,retry=False,resume=False,next_stage_authorized=False)
        terminal=dict(self.terminal,scope=gate.acquisition.SCOPE,axes=[],actual_calls=plan['call_caps'])
        for r in fixture():
            if r['method']!='B0':continue
            terminal['axes'].append(dict(case_id=f"B0_p{r['prefix']}_q{r['q']}_2nd",axis=r['axis'],role=r['role'],
                SOURCE=SOURCE,scope=gate.acquisition.SCOPE,compiler_hash=m.COMPILER,profile_sha256=m.PROFILE,
                control='symmetric_directional',scalar_policy=m.SCALARS['B0'],
                preparation_binding=plan['binding']['prefix_preparations'][str(r['prefix'])],native_exit_code=0,
                reaped=True,fingerprint_before='c'*64,fingerprint_after='c'*64,measured_RZ=r['saved_RZ'],max_control_error=0.))
        fresh=dict(source=SOURCE,config_sha256='d'*64,grant_sha256=m.digest(grant),**{k:True for k in (
            'profile_verified','source_input_bytes_verified','physical_cores_verified','resources_idle_verified',
            'protected_jobs_unchanged','quota_verified','external_RSS_monitor_ready','driver_bindings_verified')})
        values=dict(B0_config=old,B0_grant=grant,B0_terminal=terminal,B0_launcher_terminal=self.launcher,
            B0_final_gate=self.final,B0_fresh_gate=fresh,
            B0I_registry=dict(schema='H8_B1_ANCHOR_METADATA_V1',anchors=[r for r in fixture() if r['method']=='B0I']))
        config=dict(B0_acquisition_SOURCE=SOURCE,inputs={key:dict(path=key,sha256='d'*64) for key in gate.INPUTS})
        with patch.object(gate,'verify_receipts'),patch.object(gate.io,'read',side_effect=lambda p:values[p]),\
             patch.object(gate.acquisition,'RUNTIME_MAP_SHA',m.digest(runtime)),\
             patch.object(gate.acquisition,'PROVIDER_MAP_SHA',m.digest(provider)),patch.object(m,'solve') as solve:
            rows,binding=gate.normalize(config);self.assertEqual(len(rows),24);solve.assert_not_called()
            self.assertEqual(binding,m.digest(plan['binding']['inputs']))
            terminal['axes'][0]['preparation_binding']={}
            with self.assertRaisesRegex(ValueError,'PREPARATION_BINDING'):gate.normalize(config)
            solve.assert_not_called()


class ArtificialDriverTests(unittest.TestCase):
    def driver_fixture(self,folder):
        cpus=sorted(os_module().sched_getaffinity(0))[:2]
        self.assertEqual(len(cpus),2)
        config=dict(SOURCE=SOURCE,plan_sha256=m.digest(m.plan()),python=sys.executable,
            B0_acquisition_SOURCE=SOURCE,output=str(folder/'output'),control=str(folder/'control'),lock=str(folder/'lock'),inputs={})
        grant=dict(schema='H8_B1_NEW_FIT_GRANT_V1',valid=True,approved_by_user=True,runtime_authorization=True,
            approval_text='ARTIFICIAL FIXTURE ONLY',scope=m.SCOPE,SOURCE=SOURCE,
            config_sha256='',plan_sha256=config['plan_sha256'],resources=m.RESOURCES,call_caps=m.CAPS,
            one_shot=True,retry=False,resume=False,refit=False,additional_measurement=False,
            B2_authorized=False,C_authorized=False,next_stage_authorized=False,
            driver_cpu=cpus[0],observer_cpu=cpus[1],allowed_cpus=cpus)
        cfg=folder/'config.json';g=folder/'grant.json';fresh=folder/'fresh.json'
        gate.write(cfg,config);grant['config_sha256']=gate.io.sha(cfg);gate.write(g,grant)
        gate.write(fresh,dict(SOURCE=SOURCE,config_sha256=gate.io.sha(cfg),grant_sha256=m.digest(grant),
            monotonic_seconds=time.monotonic(),observer_identity=[123,456,789,0],allowed_cpus=cpus,
            resources_verified=True,quota_verified=True))
        return cfg,g,fresh
    def run_artificial_driver(self,failed):
        import contextlib
        with tempfile.TemporaryDirectory(prefix='b1-driver-',dir=Path.home()/'projects/h6-handoff-evidence') as tmp:
            folder=Path(tmp);cfg,grant,fresh=self.driver_fixture(folder);rows=fixture()
            if failed:next(r for r in rows if r['role']=='HOLDOUT' and r['axis']=='cosine')['saved_RZ']*=2
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(gate,'verify_source'))
                stack.enter_context(patch.object(gate,'verify_receipts'))
                stack.enter_context(patch.object(gate,'normalize',return_value=(rows,INPUT)))
                stack.enter_context(patch.object(runner.observer.owned,'identity',return_value=(123,456,789,0)))
                stack.enter_context(patch.object(runner.observer,'process_record',return_value=dict(zombie=False,threads=1,RSS_bytes=1,AS_bytes=1,CPU_seconds=0)))
                stack.enter_context(patch.object(runner.observer,'snapshot',return_value={}))
                stack.enter_context(patch.object(runner.observer,'resource_reason',return_value=None))
                stack.enter_context(patch.object(runner.resource,'setrlimit'))
                stack.enter_context(patch.object(runner.os,'sched_setaffinity'))
                for key in gate.io.THREAD_KEYS:stack.enter_context(patch.dict(runner.os.environ,{key:'1'}))
                if failed:
                    with self.assertRaisesRegex(ValueError,'AXIS_ERROR_GE_2_PERCENT'):runner.execute(cfg,grant,fresh)
                else:runner.execute(cfg,grant,fresh)
                terminal=gate.io.read(folder/'output/terminal.json')
                self.assertEqual(terminal['actual_calls']['cost_fit'],1 if failed else 4)
                self.assertEqual(terminal['status'],'STOP' if failed else 'COMPLETE_MANDATORY_STOP')
                if failed:self.assertIn('AXIS_ERROR_GE_2_PERCENT',terminal['first_stop_reason'])
                self.assertEqual((folder/'output/frozen_model.json').exists(),not failed)
                before=(folder/'output/terminal.json').read_bytes()
                with self.assertRaisesRegex(ValueError,'ONE_SHOT_ALREADY_SPENT'):runner.execute(cfg,grant,fresh)
                self.assertEqual(before,(folder/'output/terminal.json').read_bytes())
    def test_artificial_driver_success_and_second_execution_rejected(self):self.run_artificial_driver(False)
    def test_artificial_driver_preserves_first_STOP_and_no_retry(self):self.run_artificial_driver(True)


def os_module():
    import os
    return os


if __name__=='__main__':unittest.main()
