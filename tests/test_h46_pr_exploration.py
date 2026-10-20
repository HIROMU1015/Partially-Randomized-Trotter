"""Design/authorization/process tests and synthetic four-mode native fixtures."""
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
import h46_pr_exploration_design as d
import h46_pr_exploration_contract as c
import h46_pr_exploration_runtime as r

def options():
    return dict(process_start_method='fork',prewarm_readonly_contexts=True,worker_input_cache=True,
                worker_allocation_policy='FRESH_JOINT_CPU_RAM_V1')

def task(system='H4',formula='2nd',name='TOY'):
    from test_h46_pr_design_pilot import toy_task
    t=toy_task(name,formula);t['stage']=system+'_PR_EXPLORATION';t['candidate']['system']=system
    return t

def fake_worker(t):
    rt=r.get_runtime();cfg=rt.runner_namespace['WORKER_CFG'];stop=rt.runner_namespace['WORKER_STOP']
    if cfg.get('failure')==t['id']:stop.set();raise ValueError('SYNTHETIC_FAILURE')
    if stop.is_set():raise ValueError('OWN_STOP')
    for kind,n in c.task_limits(t).items():
        for i in range(n):c.reserve(cfg,t,kind,i)
    out=Path(cfg['output'])/t['id'];out.mkdir()
    for a in ('cosine','sine'):
        q=out/('quarantine_'+a);q.mkdir()
        for name in ('wrapper.qpy','compiled.qpy','vectors.npz'):
            c.storage.write_binary(cfg,q/name,lambda f:f.write(b'ARTIFICIAL_ONLY'))
    c.save(out/'storage_audit.json',c.storage.audit_task(cfg,t))
    for i in range(c.task_limits(t)['request_sampling_calls']):c.save(out/f'piece_law_{i}.json',{'synthetic':True})
    c.save(out/'result.json',dict(status='UNIT_COST_PASS_NOT_ACCOUNTING',task=t,SOURCE=cfg['SOURCE'],
      plan_sha256=cfg['plan_sha256'],stage=cfg['stage'],axis_costs=[dict(axis=a,compiler_settings_hash=d.COMPILER_HASH,
      native_policy=d.NATIVE_POLICY) for a in ('cosine','sine')],shot_accounting=False,cost_ranking=False,
      storage_audit_sha256=c.sha(out/'storage_audit.json'),byte_restoration_PASS=True,native_scientific_PASS=True,
      conditional_sampling=False,cohort_pooling=False,seed_cohort=d.COHORT,events=dict(N2=0,N4=0,stratum='S0')))
    return dict(id=t['id'],result_sha256=c.sha(out/'result.json'),peak_RSS_KiB=1)

class DesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.proposal=d.proposal()
    def test_36_options_and_no_adoption(self):
        self.assertEqual(len(self.proposal['options']),36);self.assertIsNone(self.proposal['n_E_adopted'])
        self.assertFalse(self.proposal['scientific_execution_authorized'])
        self.assertTrue(all(not o['adopted'] for o in self.proposal['options']))
    def test_recommended_counts_and_budget(self):
        choice=d.selected_option(self.proposal,d.RECOMMENDED)
        self.assertEqual((choice['trajectories'],choice['compile_axes']),(1400,2800))
        self.assertEqual([choice['system_call_scope_proposed'][s]['trajectories'] for s in ('H4','H6')],[793,607])
        expected={'H4':{'2nd':[255,143,62,48],'4th':[54,33,33,33],'4th(m5_best)':[33]*4},
                  'H6':{'2nd':[167,110,33,33],'4th':[33]*4,'4th(m5_best)':[33]*4}}
        for s in expected:
            for f,ns in expected[s].items():
                groups=sorted([g for g in d.groups() if g['system']==s and g['formula']==f],key=lambda x:x['q'])
                self.assertEqual([choice['n_E_by_candidate_proposed'][g['id']] for g in groups],ns)
    def test_nominal_and_common_are_separate_options(self):
        self.assertEqual(d.selected_option(self.proposal,d.option_id(.02,4,'GROUP',2))['trajectories'],1007)
        self.assertEqual(d.selected_option(self.proposal,d.option_id(.02,4,'COMMON_4Q',2))['trajectories'],2020)
        self.assertEqual(d.selected_option(self.proposal,d.option_id(.02,4,'COMMON_4Q',33))['trajectories'],2300)
    def test_null_not_filled_by_floor(self):
        rows=d.groups();nom={g['id']:2 for g in rows};nom[rows[0]['id']]=None
        self.assertIsNone(d.allocate(rows,nom,'GROUP',33)[rows[0]['id']])
        common=d.allocate(rows,nom,'COMMON_4Q',33)
        self.assertTrue(all(common[g['id']] is None for g in rows if g['system']==rows[0]['system'] and g['formula']==rows[0]['formula']))
    def test_count_and_cohort_rejections(self):
        counts={g['id']:2 for g in d.groups() if g['system']=='H4'}
        for value in (None,True,0,1):
            bad=dict(counts);bad[next(iter(bad))]=value
            with self.assertRaises(ValueError):d.tasks_for(d.STAGES[0],bad)
        for stage in ('H4_DESIGN_PILOT','H4_PR','H6_CONFIRMATION'):
            with self.assertRaises(ValueError):d.tasks_for(stage,counts)
    def test_scientific_call_scope(self):
        t=task(formula='4th(m5_best)');t['candidate']['q']=8
        budget=d.call_scope([t],'PR')
        self.assertEqual((budget['request_sampling_calls'],budget['rte_events'],budget['compile_axes']),(88,256,2))
        for k in ('shot_calculation','cost_accounting','fit','ranking'):self.assertEqual(budget[k],0)
    def test_seed_independence_collision_no_retry(self):
        tasks=[task()];registers=[d.registry(tasks,'a'*40,'SYNTHETIC',cohort,[]) for cohort in ('DESIGN_PILOT','EXPLORATION','CONFIRMATION')]
        self.assertEqual(len({json.dumps(v) for v in registers}),3)
        first=next(iter(registers[1]['TOY'].values()))
        with self.assertRaisesRegex(ValueError,'COLLISION'):d.registry(tasks,'a'*40,'SYNTHETIC','EXPLORATION',[first])
    def test_draft_rejected_before_any_input_or_runtime(self):
        with patch.object(c,'read',side_effect=AssertionError('NO_INPUT_READ')):
            for cfg in ({},{'authorized':False,'grant':{'valid':True}},{'authorized':True,'grant':{'valid':False}}):
                with self.assertRaisesRegex(ValueError,'EXPLICIT_EXPLORATION'):c.authorize(cfg)
    def test_adopted_scope_guards_with_artificial_files(self):
        # Synthetic authorization fixture: no actual SOURCE/inputs or grant.
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp);root=dest/'future_run';t=task();scope=d.call_scope([t],'PR')
            decisions={k:True for k in ('sample_design_adopted','finite_sample_limit_accepted','unknown_rare_tail_accepted',
              'resources_storage_policy_accepted','unit_cost_acquisition_only_accepted')}
            decisions.update(formal_shot_policy_adopted=False,AD03='DEFERRED',pooling=False,adaptive_top_up=False,statistics_execution_authorized=False)
            supervisor={k:str(dest/n) for k,n in [('entered','SUPERVISOR_ENTERED.json'),('log','native_PRIVATE.log'),('exit','external_exit_witness_PRIVATE.json')]}
            lease=dest/'lease.json';c.save(lease,dict(SOURCE='a'*40,runtime_caps=False,adopted=True,
              peak_growth_proxy_bytes=100,workers={'H4':2,'H6':1}))
            cfg=dict(authorized=True,SOURCE='a'*40,stage=t['stage'],cohort=d.COHORT,tasks=[t],decisions=decisions,
              call_scope=scope,plan_sha256='b'*64,pipeline_identity='c'*64,run_root=str(root),output=str(root/'output'),
              ledger=str(root/'ledger'),marker=str(root/'CONSUMED.json'),storage=c.storage.storage_policy(),supervisor=supervisor,
              resources=dict(proxy_policy_adopted=True,full_output_capacity_adopted=True,worker_RSS_proxy_bytes=10,
                disk_proxy_bytes=100,inode_proxy=20,joint_admission_path=str(lease),joint_admission_sha256=c.sha(lease)),
              sealed_files=[{'path':str(lease)}],approval_receipt_path=str(dest/'approval.json'))
            c.save(cfg['approval_receipt_path'],dict(execution_authorized=True,**{k:cfg[k] for k in
              ('SOURCE','stage','cohort','decisions','call_scope','storage')}))
            cfg['plan']=dict(run_root=str(root),supervisor=supervisor)
            cfg['grant']=dict(valid=True,grant_id='SYNTHETIC_ONLY',retry=False,next_phase_authorized=False,
              storage_policy_sha256=c.digest(cfg['storage']),approval_reference=cfg['approval_receipt_path']+'#sha256='+c.sha(cfg['approval_receipt_path']),
              **{k:cfg[k] for k in ('SOURCE','stage','cohort','plan_sha256','pipeline_identity','call_scope','output','ledger','marker','supervisor')})
            with patch.object(c,'proposal_tasks',return_value=[t]),patch.object(c,'verify'):
                c.authorize(cfg)
                mutations=[('cohort','DESIGN_PILOT'),('stage','H4_PR'),('call_scope',{}),('output',str(dest/'foreign'))]
                for key,value in mutations:
                    bad=copy.deepcopy(cfg);bad[key]=value
                    with self.assertRaises(ValueError):c.authorize(bad)
                for key in ('formal_shot_policy_adopted','pooling','adaptive_top_up','statistics_execution_authorized'):
                    bad=copy.deepcopy(cfg);bad['decisions'][key]=True
                    with self.assertRaises(ValueError):c.authorize(bad)
                for key in ('sample_design_adopted','unknown_rare_tail_accepted'):
                    bad=copy.deepcopy(cfg);bad['decisions'][key]=False
                    with self.assertRaises(ValueError):c.authorize(bad)
                for key,value in [('retry',True),('next_phase_authorized',True),('storage_policy_sha256','bad'),('SOURCE','bad')]:
                    bad=copy.deepcopy(cfg);bad['grant'][key]=value
                    with self.assertRaises(ValueError):c.authorize(bad)
                self.assertFalse(root.exists())

class RuntimeTests(unittest.TestCase):
    def assert_code_identical(self,a,b):
        self.assertEqual(a.co_code,b.co_code);self.assertEqual(a.co_names,b.co_names)
        self.assertEqual(a.co_varnames,b.co_varnames);self.assertEqual(len(a.co_consts),len(b.co_consts))
        for x,y in zip(a.co_consts,b.co_consts):
            if isinstance(x,CodeType):self.assert_code_identical(x,y)
            else:self.assertEqual('EXPLORATION' if x=='DESIGN_PILOT' else x,y)
    def test_native_numeric_operations_unchanged(self):
        import h46_pr_pilot_native as old
        rt=r.get_runtime()
        for name in ('run_task','context','gate','atomic_binary'):
            self.assert_code_identical(getattr(old,name).__code__,getattr(rt.native,name).__code__)
        self.assertIsNot(rt.native_namespace,old.__dict__)
        self.assertIs(rt.native_namespace['c'],c);self.assertIsNot(rt.native_namespace['CACHE'],old.CACHE)
    def test_old_modules_not_mutated_and_optimization_both_systems(self):
        import run_h46_pr_design_pilot as old
        rt=r.get_runtime();self.assertIsNot(rt.runner.execute,old.execute)
        self.assertIs(rt.runner_namespace['fresh'],r.fresh)
        for stage in d.STAGES:self.assertTrue(r.enabled(dict(stage=stage,resources=options())))
        with self.assertRaises(ValueError):r.enabled(dict(stage='H6_DESIGN_PILOT',resources=options()))
    def test_joint_allocator_no_historical_cpu_cap(self):
        a=c.storage.allocation(48,1000,{'H4':100,'H6':100},{'H4':2,'H6':5},{'H4':1,'H6':3})
        self.assertEqual(sum(a.values()),48);self.assertGreater(a['H6'],28)
        with self.assertRaises(ValueError):c.storage.allocation(1,1000,{'H4':1,'H6':1},{'H4':2,'H6':5},{'H4':1,'H6':1})
    def test_duplicate_reservation_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            t=task();cfg=dict(tasks=[t],marker=str(Path(tmp)/'marker'),grant={},ledger=tmp,SOURCE='a'*40,stage=t['stage'])
            c.save(cfg['marker'],dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant={}))
            c.reserve(cfg,t,'TASK')
            with self.assertRaises(FileExistsError):c.reserve(cfg,t,'TASK')
    def test_parallel_synthetic_success_and_failure_reap(self):
        rt=r.get_runtime()
        for failure in (None,'A'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/'run';tasks=[task(name=x) for x in ('A','B','C')]
                cfg=dict(SOURCE='a'*40,plan_sha256='b'*64,stage=d.STAGES[0],tasks=tasks,run_root=str(root),
                  output=str(root/'output'),ledger=str(root/'ledger'),marker=str(root/'CONSUMED.json'),grant={},
                  storage=c.storage.storage_policy(),runtime={'python':sys.executable},resources=options(),failure=failure)
                with patch.object(c,'authorize'),patch.object(c,'verify'),patch.dict(rt.runner_namespace,{'fresh':lambda cfg:{'workers':2}}),\
                  patch.object(rt.native,'context') as context:
                    result=rt.runner.execute(cfg,target=fake_worker)
                self.assertEqual(result,1 if failure else 0);self.assertEqual(context.call_count,1)
                completion=c.read(root/'completion_witness.json')
                self.assertTrue(all(w['exit']==0 and w['reaped'] for w in completion['workers'].values()))
                self.assertTrue(completion['mandatory_STOP']);self.assertFalse(completion['retry'])
                if not failure:self.assertEqual(len(completion['results']),3)
    def test_cli_rejection_execute_supervise_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'draft.json';p.write_text('{"authorized":false,"grant":{"valid":false}}')
            for flag in ('--execute','--supervise','--audit'):
                result=subprocess.run([sys.executable,'-B',str(d.ROOT/'scripts/run_h46_pr_exploration.py'),flag,'--config',str(p)],capture_output=True,text=True)
                self.assertEqual(result.returncode,2);self.assertIn('EXPLICIT_EXPLORATION',result.stdout)

@unittest.skipUnless(os.environ.get('M5_TEST_PROVIDER'),'synthetic native provider required')
class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):sys.path.insert(0,str(Path(os.environ['M5_TEST_PROVIDER'])/'src'))
    def test_all_formulas_both_labels_synthetic_not_molecules(self):
        from test_h46_m5_cost_readiness import NativeToyTests
        import h46_m5_native as original
        rt=r.get_runtime();toy=NativeToyTests.toy(self)
        for system in ('H4','H6'):
            for formula in ('2nd','4th','4th(m5_best)'):
                with self.subTest(system=system,formula=formula),tempfile.TemporaryDirectory() as tmp:
                    root=Path(tmp);(root/'output').mkdir();(root/'ledger').mkdir();t=task(system,formula);candidate=t['candidate']
                    cfg=dict(SOURCE='a'*40,plan_sha256='b'*64,stage=t['stage'],tasks=[t],run_root=str(root),
                      output=str(root/'output'),ledger=str(root/'ledger'),marker=str(root/'CONSUMED.json'),grant={},
                      storage=c.storage.storage_policy(),runtime={'python':sys.executable},resources=options(),
                      systems={system:{'binding':{'synthetic':True}}},registry_path=str(root/'seeds.json'),sealed_files=[])
                    c.save(cfg['marker'],dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant={}))
                    c.save(cfg['registry_path'],d.registry([t],cfg['SOURCE'],'ARTIFICIAL_ONLY',d.COHORT,[]))
                    sig=original.signal(toy,dict(candidate,policy=candidate['r_policy']))
                    path=root/'signal.json';c.save(path,dict(records=[dict(candidate=candidate,signal=sig)]))
                    for p in (Path(cfg['registry_path']),path):cfg['sealed_files'].append(dict(path=str(p),bytes=p.stat().st_size,sha256=c.sha(p)))
                    with patch.object(c,'authorize'),patch.object(c,'verify'),patch.object(c,'SIGNAL_RECORDS',str(path)),\
                      patch.dict(rt.native_namespace,{'context':lambda *args:toy,'INPUT_CACHE':r.repair.BoundJSON()}):
                        rt.native.run_task(cfg,t,threading.Event())
                    result=c.read(root/'output'/t['id']/'result.json')
                    self.assertEqual(result['seed_cohort'],'EXPLORATION');self.assertEqual(len(result['checks']),8)
                    self.assertTrue(all(g['PASS'] for g in result['checks']))
                    audited=rt.auditor.task_audit(cfg,t);self.assertEqual(audited['archives'],6)
                    laws=list((root/'output'/t['id']).glob('piece_law_*.json'))
                    self.assertTrue(all(c.read(p)['cohort']=='EXPLORATION' for p in laws))

if __name__=='__main__':unittest.main()
