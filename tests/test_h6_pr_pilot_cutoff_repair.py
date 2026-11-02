"""Synthetic arithmetic/cache/process tests. No saved molecular arrays."""
import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h46_pr_pilot_repair as r

def options():
    return dict(optimization_policy=r.POLICY,process_start_method='fork',prewarm_readonly_contexts=True,
      worker_input_cache=True,worker_allocation_policy='FRESH_SINGLE_STAGE_CPU_RAM_V1')

def prep(lam):return NS(rte_preparation=NS(symbolic_tail=NS(lambda_r=lam)))

def fork_probe(queue):queue.put((os.getpid(),SHARED_TOY[0],id(SHARED_TOY)))
SHARED_TOY=None

class ArithmeticTests(unittest.TestCase):
    def test_exact_factory_operation_order(self):
        for lam in (1.1,.5,.9182185130575018):
            for t in (.03,-.03,.123,-.8):
                for steps in (1,3,5,32):
                    self.assertEqual(r.canonical_tau(prep(lam),t,steps).hex(),(lam*(float(t)/steps)).hex())
    def test_uses_symbolic_lambda_even_when_matrix_value_differs(self):
        p=prep(math.nextafter(1.1,math.inf));p.exact_rte_lambda_r=1.1
        self.assertEqual(r.canonical_tau(p,-.01,9),p.rte_preparation.symbolic_tail.lambda_r*(-.01/9))
    def test_nonfinite_and_invalid_count_refused(self):
        for v in (0,-1,True,1.5):
            with self.assertRaises(ValueError):r.canonical_tau(prep(1.),.1,v)
        for v in (0.,-1.,math.inf,math.nan):
            with self.assertRaises(ValueError):r.canonical_tau(prep(v),.1,3)
        with self.assertRaises(ValueError):r.canonical_tau(prep(1.),math.inf,3)

class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'data.json'
    def tearDown(self):self.tmp.cleanup()
    def cfg(self):
        raw=self.path.read_bytes()
        return dict(SOURCE='a'*40,plan_sha256='b'*64,sealed_files=[dict(path=str(self.path),
          bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())])
    def test_parse_once_but_return_copies(self):
        self.path.write_text(json.dumps({'T':{'0:0':23}}));cfg=self.cfg();cache=r.BoundJSON()
        a=cache.registry(cfg,self.path,'T');a['0:0']=0
        self.assertEqual(cache.registry(cfg,self.path,'T'),{'0:0':23});self.assertEqual(cache.parses,1)
    def test_same_size_tamper_detected_after_cache_hit(self):
        self.path.write_text('{"T":1}');cfg=self.cfg();cache=r.BoundJSON()
        self.assertEqual(cache.registry(cfg,self.path,'T'),1)
        st=self.path.stat();self.path.write_text('{"T":2}');os.utime(self.path,ns=(st.st_atime_ns,st.st_mtime_ns))
        with self.assertRaisesRegex(ValueError,'SHA_MISMATCH'):cache.registry(cfg,self.path,'T')
    def test_unsealed_path_refused(self):
        self.path.write_text('{"T":1}')
        with self.assertRaisesRegex(ValueError,'SEALED'):r.BoundJSON().registry(dict(SOURCE='a',plan_sha256='b',sealed_files=[]),self.path,'T')
    def test_signal_selection_and_duplicate_refusal(self):
        self.path.write_text(json.dumps(dict(records=[dict(candidate={'id':'C'},signal={'log_B':.1})])))
        c=self.cfg();cache=r.BoundJSON();a=cache.signal(c,self.path,'C');a['signal']['log_B']=0
        self.assertEqual(cache.signal(c,self.path,'C')['signal']['log_B'],.1)
        with self.assertRaises(ValueError):cache.signal(c,self.path,'MISSING')
    def test_new_SOURCE_or_plan_has_separate_cache(self):
        self.path.write_text('{"T":1}');c=self.cfg();cache=r.BoundJSON()
        cache.registry(c,self.path,'T');cache.registry(dict(c,SOURCE='c'*40),self.path,'T')
        cache.registry(dict(c,plan_sha256='d'*64),self.path,'T');self.assertEqual(cache.parses,3)

class ParallelTests(unittest.TestCase):
    def test_opt_in_and_H6_only(self):
        self.assertFalse(r.enabled({}))
        self.assertTrue(r.enabled(dict(stage='H6_DESIGN_PILOT',resources=options())))
        with self.assertRaises(ValueError):r.enabled(dict(stage='H4_DESIGN_PILOT',resources=options()))
        for key in ('prewarm_readonly_contexts','worker_input_cache'):
            with self.assertRaises(ValueError):r.enabled(dict(stage='H6_DESIGN_PILOT',resources=dict(options(),**{key:False})))
    def test_available_resources_not_old_worker_count(self):
        self.assertEqual(r.worker_count(322,61.7,500,8),61)
        self.assertEqual(r.worker_count(322,61.7,500,8,historical_allocation=28),28)
        self.assertEqual(r.worker_count(2,61.7,500,8),2)
        with self.assertRaises(ValueError):r.worker_count(322,.1,500,8)
    def test_context_prepared_once_per_system_prefix(self):
        cfg=dict(stage='H6_DESIGN_PILOT',resources=options(),tasks=[dict(candidate={'system':'H6','prefix':5}) for _ in range(40)])
        calls=[];result=r.prewarm(cfg,lambda *args:calls.append(args))
        self.assertEqual(len(calls),1);self.assertEqual(result['contexts'],1);self.assertEqual(result['RNG_calls'],0)
    @unittest.skipUnless('fork' in mp.get_all_start_methods(),'POSIX fork required')
    def test_fork_inherits_synthetic_context_and_reaps(self):
        global SHARED_TOY
        SHARED_TOY=bytearray(b'TOY_CONTEXT');ctx=mp.get_context('fork');q=ctx.Queue()
        workers=[ctx.Process(target=fork_probe,args=(q,)) for _ in range(3)]
        for p in workers:p.start()
        data=[q.get(timeout=10) for p in workers]
        for p in workers:p.join(timeout=10);self.assertEqual(p.exitcode,0);self.assertFalse(p.is_alive())
        self.assertEqual({x[2] for x in data},{id(SHARED_TOY)});self.assertEqual(len({x[0] for x in data}),3)
        q.close();q.join_thread()
    def test_optimized_launcher_with_synthetic_outputs(self):
        import h46_pr_pilot_contract as c
        import h46_pr_pilot_native as native
        import run_h46_pr_design_pilot as launch
        from test_h46_pr_design_pilot import fake_worker,toy_task
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';tasks=[toy_task(n) for n in ('A','B','C')]
            for t in tasks:t['stage']='H6_DESIGN_PILOT';t['candidate']['system']='H6'
            cfg=dict(SOURCE='a'*40,plan_sha256='b'*64,stage='H6_DESIGN_PILOT',tasks=tasks,run_root=str(root),
              output=str(root/'output'),ledger=str(root/'ledger'),marker=str(root/'CONSUMED.json'),grant={},
              storage=c.storage.storage_policy(),runtime={'python':sys.executable},resources=options())
            with patch.object(c,'authorize'),patch.object(c,'verify'),patch.object(launch,'fresh',return_value={'workers':2}) as fresh,patch.object(native,'context') as context:
                self.assertEqual(launch.execute(cfg,target=fake_worker),0);self.assertEqual(context.call_count,1)
                self.assertEqual(fresh.call_count,2)
            self.assertTrue((root/'resources_after_context_prewarm.json').is_file())
            report=c.read(root/'completion_witness.json');self.assertEqual(len(report['results']),3)
            self.assertTrue(all(w['exit']==0 and w['reaped'] for w in report['workers'].values()))

@unittest.skipUnless(os.environ.get('M5_TEST_PROVIDER'),'fixed provider required for synthetic RTE')
class ProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):sys.path.insert(0,str(Path(os.environ['M5_TEST_PROVIDER'])/'src'))
    def test_original_arithmetic_FAIL_and_canonical_PASS(self):
        from trotterlib.pr2_matched_accuracy_m1_execution import _explicit_cutoff_tolerance as tol
        from trotterlib.rte import step_taylor_truncation_residual_bound as bound
        # Pure scalar reproducer, no H6 Hamiltonian/state/sampling involved.
        for lam,t,steps in [(1.1,-.03,5),(.9182185130575018,.03781854328634795,2)]:
            old=(lam*t)/steps;new=r.canonical_tau(prep(lam),t,steps)
            self.assertLessEqual(bound(new,4),tol(new,4))
        lam,t,steps=1.1,-.03,5
        self.assertGreater(bound(lam*(t/steps),4),tol((lam*t)/steps,4))
    def test_same_registered_sampling_law_and_events_all_formulas(self):
        from test_h46_m5_cost_readiness import NativeToyTests
        import h46_m5_native as native
        import h46_m5_contract as c
        from trotterlib.df_partial_s2 import make_df_partial_s2_step_request
        from trotterlib.pr2_matched_accuracy_m1_execution import _explicit_cutoff_tolerance as tol
        toy=NativeToyTests.toy(self)
        for formula in ('2nd','4th','4th(m5_best)'):
            for q in (1,2,4,8):
                group=dict(formula=formula,q=q,policy='ABS_TIME_PER_OUTER',r_value=32)
                schedule=c.core.schedule(formula,q,group['policy'],32)
                seeds={f"{s['outer']}:{s['piece']}":10000+i for i,s in enumerate(schedule)}
                calls=[]
                def factory(*a,**kw):
                    result=make_df_partial_s2_step_request(*a,**kw);calls.append(kw)
                    return result
                requests=native.make_requests(group,toy.cut,seeds,toy.lambda_R,factory=factory,tolerance=tol)
                self.assertEqual(len(requests),len(schedule));self.assertEqual(sum(len(x.rte_occurrence.events) for x in requests),32*q)
                for s,kw,req in zip(schedule,calls,requests):
                    # Same provider with an independently supplied valid tolerance:
                    # the acceptance value is absent from the generation law.
                    tau=toy.cut.rte_preparation.symbolic_tail.lambda_r*(s['signed_time']/s['rte_steps'])
                    self.assertEqual(req.rte_distribution.dimensionless_step_time,tau)
                    self.assertEqual(kw['seed'],seeds[f"{s['outer']}:{s['piece']}"])
                    baseline=make_df_partial_s2_step_request(toy.cut,**dict(kw,truncation_tolerance=max(kw['truncation_tolerance'],1e-8)))
                    self.assertEqual(req.rte_distribution.to_dict(),baseline.rte_distribution.to_dict())
                    self.assertEqual([x.to_dict() for x in req.rte_occurrence.events],[x.to_dict() for x in baseline.rte_occurrence.events])
    def test_optimized_native_path_preserves_all_gates_and_archives(self):
        import h46_pr_pilot_contract as c
        import h46_pr_pilot_native as native
        import h46_m5_native as original
        from test_h46_m5_cost_readiness import NativeToyTests
        from test_h46_pr_design_pilot import toy_task
        toy=NativeToyTests.toy(self);task=toy_task(formula='4th(m5_best)')
        task['stage']='H6_DESIGN_PILOT';task['candidate']['system']='H6';candidate=task['candidate']
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'output').mkdir();(root/'ledger').mkdir()
            cfg=dict(SOURCE='a'*40,plan_sha256='b'*64,stage=task['stage'],tasks=[task],run_root=str(root),
              output=str(root/'output'),ledger=str(root/'ledger'),marker=str(root/'CONSUMED.json'),grant={},
              storage=c.storage.storage_policy(),runtime={'python':sys.executable},resources=options(),
              systems={'H6':{'binding':{'synthetic':True}}},registry_path=str(root/'seeds.json'),sealed_files=[])
            c.save(cfg['marker'],dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant={}))
            c.save(cfg['registry_path'],c.design.registry([task],cfg['SOURCE'],'ARTIFICIAL_NEW',c.design.COHORT,[]))
            sig=original.signal(toy,dict(candidate,policy=candidate['r_policy']))
            path=root/'signal_records.json';c.save(path,dict(records=[dict(candidate=candidate,signal=sig)]))
            for p in (Path(cfg['registry_path']),path):cfg['sealed_files'].append(dict(path=str(p),bytes=p.stat().st_size,sha256=c.sha(p)))
            cache=r.BoundJSON()
            with patch.object(c,'authorize'),patch.object(c,'verify'),patch.object(native,'context',return_value=toy),\
              patch.object(c,'SIGNAL_RECORDS',str(path)),patch.object(native,'INPUT_CACHE',cache):
                native.run_task(cfg,task,__import__('threading').Event())
            result=c.read(root/'output'/task['id']/'result.json')
            self.assertEqual(len(result['checks']),8);self.assertTrue(all(x['PASS'] for x in result['checks']))
            self.assertEqual(cache.parses,2);self.assertEqual(c.storage.audit_task(cfg,task)['archive_count'],6)
            for p in (root/'output'/task['id']).glob('piece_law_*.json'):
                law=c.read(p);self.assertEqual(law['cutoff_policy'],r.CUTOFF_POLICY)
                self.assertEqual(law['cutoff_tau'],law['law']['dimensionless_step_time'])

if __name__=='__main__':unittest.main()
