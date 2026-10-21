"""Artificial execution, boundaries, cohort identity and paired accounting."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import CodeType,SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h246_confirmation_contract as c
import h246_confirmation_runtime as rt
import h246_confirmation_evaluation as ev
import launch_h246_confirmation as launch


def toy_task(name='TOY',formula='2nd'):
    from test_h46_pr_design_pilot import toy_task as old
    t=old(name,formula);t['stage']='H4_CONFIRMATION';t['candidate']['system']='H4';return t

def options():return dict(runtime_caps=False,host_PSI_kill=False,process_start_method='fork',prewarm_readonly_contexts=True,worker_input_cache=True)

def fake_worker(task):
    runtime=rt.RUNTIME;cfg=runtime.runner_namespace['WORKER_CFG'];stop=runtime.runner_namespace['WORKER_STOP']
    if cfg.get('failure')==task['id']:stop.set();raise ValueError('ARTIFICIAL_FAILURE')
    if stop.is_set():raise ValueError('OWN_STOP')
    for kind,n in c.task_limits(task).items():
        for i in range(n):c.reserve(cfg,task,kind,i)
    out=Path(cfg['output'])/task['id'];out.mkdir()
    for axis in ('cosine','sine'):
        q=out/('quarantine_'+axis);q.mkdir()
        for name in ('wrapper.qpy','compiled.qpy','vectors.npz'):
            c.storage.write_binary(cfg,q/name,lambda f:f.write(b'ARTIFICIAL_OPAQUE_BYTES'))
    c.save(out/'storage_audit.json',c.storage.audit_task(cfg,task))
    for i in range(c.task_limits(task)['request_sampling_calls']):c.save(out/f'piece_law_{i}.json',{'artificial':True})
    c.save(out/'result.json',dict(status='UNIT_COST_PASS_NOT_ACCOUNTING',task=task,SOURCE=cfg['SOURCE'],
      plan_sha256=cfg['plan_sha256'],stage=cfg['stage'],axis_costs=[dict(axis=a,compiler_settings_hash=c.law.COMPILER_HASH,
      native_policy=c.law.NATIVE_POLICY) for a in ('cosine','sine')],shot_accounting=False,cost_ranking=False,
      storage_audit_sha256=c.sha(out/'storage_audit.json'),byte_restoration_PASS=True,native_scientific_PASS=True,
      conditional_sampling=False,cohort_pooling=False,seed_cohort=c.d.COHORT,events=dict(N2=0,N4=0,stratum='S0')))
    return dict(id=task['id'],result_sha256=c.sha(out/'result.json'),peak_RSS_KiB=1)

class ContractTests(unittest.TestCase):
    def test_fixed_counts_and_budget(self):
        for stage,n in zip(c.STAGES,([24,28],[25,9],[23,9])):
            tasks=c.tasks_for(stage);p=c.comparison(stage)
            self.assertEqual(p['recommended']['n_C'],n);self.assertEqual(len(tasks),sum(n))
            self.assertEqual(c.law.call_scope(tasks,'PR')['compile_axes'],2*sum(n))
            self.assertEqual(len({t['candidate']['id'] for t in tasks}),2)
        for stage in ('H4_AUXILIARY','H4_PR_EXPLORATION'):
            with self.assertRaises(ValueError):c.tasks_for(stage)
    def test_seed_namespace_and_collision(self):
        tasks=[toy_task()]
        regs=[c.law.registry(tasks,'a'*40,'ARTIFICIAL',cohort,[]) for cohort in ('DESIGN_PILOT','EXPLORATION','CONFIRMATION')]
        self.assertEqual(len({json.dumps(x) for x in regs}),3)
        value=next(iter(regs[2]['TOY'].values()))
        with self.assertRaisesRegex(ValueError,'COLLISION'):c.law.registry(tasks,'a'*40,'ARTIFICIAL','CONFIRMATION',[value])
    def test_draft_refusal_before_any_input(self):
        with patch.object(c,'read',side_effect=AssertionError('NO_INPUT_READ')):
            for cfg in ({},{'authorized':True,'grant':{'valid':False}},{'authorized':False,'grant':{'valid':True}}):
                with self.assertRaisesRegex(ValueError,'EXPLICIT_NEW_CONFIRMATION'):c.authorize(cfg)
    def test_grant_identity_and_forbidden_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=dict(authorized=True,stage='H4_CONFIRMATION',SOURCE='a'*40,cohort='CONFIRMATION',plan_sha256='b'*64,
              call_scope={'synthetic':1},analysis_scope=c.analysis_scope('H4_CONFIRMATION'),comparison={'synthetic':True},
              decisions=dict(c.decisions(),scientific_execution_adopted=True),storage=c.storage.storage_policy(),
              output='synthetic_output',ledger='synthetic_ledger',marker='synthetic_marker',supervisor={'synthetic':True},
              approval_receipt_path=str(Path(tmp)/'approval'))
            c.save(cfg['approval_receipt_path'],dict(execution_authorized=True,**{k:cfg[k] for k in
              ('SOURCE','stage','cohort','comparison','call_scope','analysis_scope','storage','decisions')}))
            cfg['grant']=dict(valid=True,grant_id='ARTIFICIAL',retry=False,next_phase_authorized=False,
              approval_reference=cfg['approval_receipt_path']+'#sha256='+c.sha(cfg['approval_receipt_path']),
              **{k:cfg[k] for k in ('SOURCE','stage','cohort','plan_sha256','call_scope','analysis_scope','output','ledger','marker','supervisor')})
            with patch.object(c,'verify'):
                c.authorize(cfg)
                for key,value in [('cohort','EXPLORATION'),('stage','H4_AUXILIARY'),('SOURCE','bad'),('call_scope',{}),
                  ('output','foreign'),('analysis_scope',{}),('comparison',{})]:
                    bad=copy.deepcopy(cfg);bad[key]=value
                    with self.subTest(key=key),self.assertRaises(ValueError):c.authorize(bad)
                for key in ('pooling','adaptive_top_up','candidate_reselection','new_shot_policy'):
                    bad=copy.deepcopy(cfg);bad['decisions'][key]=True
                    with self.subTest(key=key),self.assertRaises(ValueError):c.authorize(bad)
                for key in ('retry','next_phase_authorized'):
                    bad=copy.deepcopy(cfg);bad['grant'][key]=True
                    with self.assertRaises(ValueError):c.authorize(bad)
    def test_duplicate_reservation_no_refund(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=dict(tasks=[toy_task()],marker=str(Path(tmp)/'marker'),ledger=tmp,grant={},SOURCE='a'*40,stage='H4_CONFIRMATION')
            c.save(cfg['marker'],dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant={}))
            c.reserve(cfg,cfg['tasks'][0],'TASK')
            with self.assertRaises(FileExistsError):c.reserve(cfg,cfg['tasks'][0],'TASK')
    def test_cli_all_execution_modes_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'draft';p.write_text('{"authorized":false,"grant":{"valid":false}}')
            for mode in ('execute','supervise','audit','evaluate'):
                r=subprocess.run([sys.executable,'-B',str(c.ROOT/'scripts/run_h246_confirmation.py'),'--'+mode,'--config',str(p)],capture_output=True,text=True)
                self.assertEqual(r.returncode,2);self.assertIn('EXPLICIT_NEW_CONFIRMATION',r.stdout)
            self.assertEqual(list(Path(tmp).iterdir()),[p])

class RuntimeTests(unittest.TestCase):
    def tearDown(self):rt.RUNTIME=None
    def identical(self,a,b):
        self.assertEqual(a.co_code,b.co_code);self.assertEqual(a.co_names,b.co_names)
        for x,y in zip(a.co_consts,b.co_consts):
            if isinstance(x,CodeType):self.identical(x,y)
            else:self.assertEqual('CONFIRMATION' if x=='DESIGN_PILOT' else x,y)
    def test_numeric_bytecode_and_constants_unchanged(self):
        import h46_pr_pilot_native as original
        r=rt.runtime(dict(stage='H4_CONFIRMATION',signal_path='synthetic'))
        self.identical(original.run_task.__code__,r.native.run_task.__code__)
        self.identical(original.context.__code__,r.h46_context.__code__)
        self.assertIsNot(r.native_namespace,original.__dict__)
        self.assertEqual(r.native_namespace['c'].SIGNAL_RECORDS,'synthetic')
    def test_parallel_mock_success_and_failure(self):
        for failure in (None,'A'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/'run';tasks=[toy_task(x) for x in ('A','B','C')]
                cfg=dict(SOURCE='a'*40,plan_sha256='b'*64,stage='H4_CONFIRMATION',signal_path='synthetic',tasks=tasks,
                  run_root=str(root),output=str(root/'output'),ledger=str(root/'ledger'),marker=str(root/'CONSUMED.json'),
                  grant={},storage=c.storage.storage_policy(),runtime={'python':sys.executable},resources=options(),failure=failure)
                rt.RUNTIME=rt.runtime(cfg);r=rt.RUNTIME
                with patch.object(c,'authorize'),patch.object(c,'verify'),\
                  patch.object(r.native_namespace['c'],'authorize'),patch.object(r.native_namespace['c'],'verify'),\
                  patch.dict(r.runner_namespace,{'fresh':lambda cfg:{'workers':2}}),patch('h246_confirmation_runtime.context') as context:
                    result=r.runner.execute(cfg,target=fake_worker)
                self.assertEqual(result,1 if failure else 0);self.assertEqual(context.call_count,1)
                done=c.read(root/'completion_witness.json');self.assertFalse(done['retry']);self.assertTrue(done['mandatory_STOP'])
                self.assertTrue(all(w['exit']==0 and w['reaped'] for w in done['workers'].values()))
                if not failure:self.assertEqual(len(done['results']),3)
    def test_only_one_stage_per_process(self):
        rt.get_runtime(dict(stage='H4_CONFIRMATION',signal_path='synthetic'))
        with self.assertRaisesRegex(ValueError,'ONE_STAGE'):rt.get_runtime(dict(stage='H6_CONFIRMATION'))

class ControllerTests(unittest.TestCase):
    def test_parallel_local_failure_does_not_signal_healthy_job(self):
        for failed in (False,True):
            with self.subTest(failed=failed),tempfile.TemporaryDirectory() as tmp:
                dest=Path(tmp);paths=[];configs=[];processes=[]
                class Proc:
                    def __init__(self,code):self.code=code;self.reaped=False;self.signals=[]
                    def wait(self):self.reaped=True;return self.code
                    def send_signal(self,sig):self.signals.append(sig)
                for system in ('H2','H4','H6'):
                    cfg=dict(stage=system+'_CONFIRMATION',SOURCE='a'*40,plan_sha256='b'*64,runtime={'python':sys.executable},
                      tasks=[{}]*2,resources=dict(worker_RSS_proxy_bytes=1,raw_pair_proxy_bytes=1,final_growth_proxy_bytes=1,workload_proxy_seconds=1))
                    path=dest/(system+'.json');c.save(path,cfg);paths.append(path);configs.append(cfg)
                    processes.append(Proc(int(failed and system=='H4')))
                observation=dict(idle_CPU_equivalents=3,effective_available_RAM_bytes=100,disk_free_bytes=100)
                with patch.object(c,'authorize'),patch('run_h46_m5_signal.fresh_resources',return_value=observation),\
                  patch.object(launch.subprocess,'Popen',side_effect=processes),\
                  patch.object(launch.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='ARTIFICIAL',stderr='')):
                    code=launch.launch(paths,dest/'controller')
                self.assertEqual(code,int(failed));self.assertTrue(all(p.reaped for p in processes))
                self.assertTrue(all(not p.signals for p in processes))
                done=c.read(dest/'controller'/'CONTROLLER_FINISHED.json')
                self.assertTrue(done['mandatory_STOP']);self.assertFalse(done['fixed_comparison_evaluation_started'])
    def test_activation_refuses_unapproved_before_new_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp);cfg=dict(authorized=False,grant={'valid':False})
            dp=dest/'draft';ap=dest/'approval';c.save(dp,cfg);c.save(ap,{'execution_authorized':False})
            with patch.object(c,'verify'),self.assertRaisesRegex(ValueError,'EXPLICIT_USER_RECEIPT'):
                c.activate(dp,ap,dest/'new')
            self.assertFalse((dest/'new').exists());self.assertFalse(c.read(dp)['grant']['valid'])

class AllocationTests(unittest.TestCase):
    def configs(self):
        return [dict(stage=s+'_CONFIRMATION',tasks=[{}]*100,resources=dict(worker_RSS_proxy_bytes=rss,
          raw_pair_proxy_bytes=1,final_growth_proxy_bytes=10,workload_proxy_seconds=demand))
          for s,rss,demand in [('H2',2,1),('H4',2,2),('H6',5,8)]]
    def test_64_workers_not_arbitrarily_capped(self):
        out=launch.allocate(dict(idle_CPU_equivalents=64,effective_available_RAM_bytes=1000,disk_free_bytes=10000),self.configs())
        self.assertEqual(sum(out['workers'].values()),64);self.assertGreater(out['workers']['H6'],28)
    def test_ram_disk_respected(self):
        for ram,disk in [(30,10000),(1000,45)]:
            cfg=self.configs();out=launch.allocate(dict(idle_CPU_equivalents=64,effective_available_RAM_bytes=ram,disk_free_bytes=disk),cfg)
            used=sum(out['workers'][c['stage'][:2]]*c['resources']['worker_RSS_proxy_bytes'] for c in cfg if c['stage'][:2] in out['workers'])
            self.assertLessEqual(used,ram);self.assertLessEqual(out['peak_growth_proxy_bytes'],disk)
    def test_one_core_sequential_fallback(self):
        out=launch.allocate(dict(idle_CPU_equivalents=1,effective_available_RAM_bytes=1000,disk_free_bytes=1000),self.configs())
        self.assertEqual(sum(out['workers'].values()),1);self.assertTrue(out['sequential_fallback'])
    def test_no_disk_fails(self):
        with self.assertRaises(ValueError):launch.allocate(dict(idle_CPU_equivalents=64,effective_available_RAM_bytes=1000,disk_free_bytes=0),self.configs())

class EvaluationTests(unittest.TestCase):
    def test_pairing_covariance_included(self):
        r=ev.paired_statistics([(1,4),(3,2),(5,1)],10,20)
        self.assertTrue(r['independent_weighted_covariance_match']);self.assertLess(r['covariance'],0)
        self.assertAlmostEqual(r['mean'],230/3)
    def test_invalid_or_unpaired_rejected(self):
        for pairs in ([(1,2)],[(1,2,3),(2,3,4)],[(1,2),(2,None)]):
            with self.assertRaises(ValueError):ev.paired_statistics(pairs,1,1)
    def fixture(self):
        group=lambda key:dict(candidate={'id':key},cost_row={'shot':dict(N_cos=2,N_sin=3)})
        p=dict(groups=[group('L'),group('R')],recommended=dict(n_C=[2,2],z=2.4))
        obs={k:[dict(pair=[a,a+1],stratum='S0'),dict(pair=[a+1,a+2],stratum='S0')] for k,a in [('L',10),('R',8)]}
        return p,obs
    def test_no_unknown_cost_imputation(self):
        p,obs=self.fixture();r=ev.summarize(p,obs)
        self.assertEqual(r['point_status'],'REPRODUCED');self.assertFalse(r['population_guarantee'])
        self.assertIn('S4',r['groups'][0]['unobserved_strata']);self.assertIsNone(r['groups'][0]['unobserved_costs'])
    def test_reversed_keeps_fixed_candidates(self):
        p,obs=self.fixture();obs['L'],obs['R']=obs['R'],obs['L'];r=ev.summarize(p,obs)
        self.assertEqual(r['point_status'],'REVERSED');self.assertEqual(r['groups'][0]['candidate']['id'],'L')
    def test_pooling_or_topup_count_refused(self):
        p,obs=self.fixture();obs['L'].append(obs['L'][0])
        with self.assertRaises(ValueError):ev.summarize(p,obs)
    def test_partial_audit_rejected_before_comparison_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c.save(root/'end_audit.json',dict(status='STOP'))
            cfg=dict(run_root=tmp,stage='H4_CONFIRMATION')
            with patch.object(c,'authorize'),patch.object(c,'consumed'),self.assertRaises(ValueError):ev.evaluate(cfg)
            self.assertFalse((root/'COMPARISON_CONSUMED.json').exists())

@unittest.skipUnless(os.environ.get('M5_TEST_PROVIDER'),'synthetic native provider required')
class NativeSyntheticTests(unittest.TestCase):
    def test_native_three_formulas_and_audit(self):
        from test_h46_m5_cost_readiness import NativeToyTests
        sys.path.insert(0,str(Path(os.environ['M5_TEST_PROVIDER'])/'src'))
        import h46_m5_native as original
        toy=NativeToyTests.toy(self)
        for formula in ('2nd','4th','4th(m5_best)'):
            with self.subTest(formula=formula),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);(root/'output').mkdir();(root/'ledger').mkdir();t=toy_task(formula=formula)
                cfg=dict(SOURCE='a'*40,plan_sha256='b'*64,stage='H4_CONFIRMATION',tasks=[t],run_root=tmp,
                  output=str(root/'output'),ledger=str(root/'ledger'),marker=str(root/'CONSUMED.json'),grant={},
                  storage=c.storage.storage_policy(),runtime={'python':sys.executable},resources=options(),
                  systems={'H4':{'binding':{'synthetic':True}}},registry_path=str(root/'seeds.json'),sealed_files=[],signal_path=str(root/'signal.json'))
                c.save(cfg['marker'],dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant={}))
                c.save(cfg['registry_path'],c.law.registry([t],cfg['SOURCE'],'ARTIFICIAL','CONFIRMATION',[]))
                sig=original.signal(toy,dict(t['candidate'],policy=t['candidate']['r_policy']))
                c.save(cfg['signal_path'],dict(records=[dict(candidate=t['candidate'],signal=sig)]))
                for p in (cfg['registry_path'],cfg['signal_path']):cfg['sealed_files'].append(c.seal(p,'artificial_input'))
                r=rt.runtime(cfg);view=r.native_namespace['c']
                with patch.object(view,'authorize'),patch.object(view,'verify'),patch.dict(r.native_namespace,{'context':lambda *a:toy}):
                    r.native.run_task(cfg,t,threading.Event())
                result=c.read(root/'output'/t['id']/'result.json')
                self.assertEqual(result['seed_cohort'],'CONFIRMATION');self.assertTrue(all(g['PASS'] for g in result['checks']))
                audited=r.auditor.task_audit(cfg,t);self.assertEqual(audited['archives'],6)
                self.assertTrue(all(c.read(p)['cohort']=='CONFIRMATION' for p in (root/'output'/t['id']).glob('piece_law_*.json')))

if __name__=='__main__':unittest.main()
