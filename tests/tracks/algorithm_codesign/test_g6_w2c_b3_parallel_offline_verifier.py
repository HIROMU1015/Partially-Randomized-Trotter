"""Independent global/state/STOP/cost tests; published artificial evidence only."""
from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import tracemalloc
from types import ModuleType
import unittest
from unittest.mock import patch

import b3_parallel_verifier_fixtures as F
from trottertracks.algorithm_codesign import g6_w2c_b3_offline_result as V
from trottertracks.algorithm_codesign import g6_w2c_b3_parallel_verifier_core as G
from trottertracks.algorithm_codesign import g6_w2c_b3_parallel_offline_adapter as A
from trottertracks.algorithm_codesign import g6_w2c_b3_parallel_offline_cost as C
from trottertracks.algorithm_codesign import g6_w2c_b3_integration_acceptance as H
ROOT=Path(__file__).resolve().parents[3]
PACKET=ROOT/'docs/research/track_b_g6_w2c_b3_parallel_offline_verifier_core_20261019'


def load_published(name):
    p=PACKET/'fixed_sources'/('published_'+name+'.py')
    modname='trottertracks.algorithm_codesign._acceptance_'+name
    spec=importlib.util.spec_from_file_location(modname,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class Global(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.master=tempfile.TemporaryDirectory(prefix='b3-parallel-independent-master-')
        cls.input=Path(cls.master.name)/'input';F.bundle(cls.input)
    @classmethod
    def tearDownClass(cls):cls.master.cleanup()
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='b3-parallel-independent-case-');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'input';shutil.copytree(self.input,self.root)
        self.m=json.loads((self.root/'manifest.json').read_text());self.report=json.loads((self.root/'report.json').read_text())
    def verify(self):return A.verify(self.root/'manifest.json')
    def save(self):
        (self.root/'manifest.json').write_text(json.dumps(self.m)+'\n');(self.root/'report.json').write_text(json.dumps(self.report)+'\n')
    def test_8_pairs_3_phases_reordered_completion(self):
        r=self.verify();self.assertEqual(r['status'],'VERIFIED_COMPLETE',r['issues']);self.assertEqual(r['calls']['completed'],832)
        self.assertNotEqual(self.report['completion_order'],self.report['contract_order'])
        self.assertEqual(len(r['pairs']),24)
    def test_unknown_integrated_schema_remains_pending(self):
        self.assertEqual(G.FutureIntegratedAdapter().verify()['status'],'PENDING_SERVER_SCHEMA')
    def test_static_formal_budget_plan_without_sampling(self):
        n=dict(zip(V.ARMS,(1174526,894664,880312,1014322)));tasks,total=G.plan(n)
        self.assertEqual(total,31710848);self.assertEqual(sum(2*t['n']for t in tasks[:8]),256)
        self.assertEqual(sum(2*t['n']for t in tasks[8:16]),15855296)
    def test_receipt_missing_is_not_complete(self):
        p=self.root/'output/control.jsonl';lines=p.read_bytes().splitlines(keepends=True);p.write_bytes(b''.join(lines[:-1]))
        self.report['output']['observed']['log']=self.report['output']['reserved']['log']=832+p.stat().st_size;self.save()
        self.assertEqual(self.verify()['status'],'INSUFFICIENT_EVIDENCE')
    def test_summary_missing_is_not_complete(self):
        p=self.root/'output/summary.jsonl';length=p.stat().st_size;p.unlink()
        for field in('observed','reserved'):self.report['output'][field]['result']-=length
        self.save();self.assertEqual(self.verify()['status'],'INSUFFICIENT_EVIDENCE')
    def test_inputs_readonly_and_unchanged(self):
        before={str(p.relative_to(self.root)):V.file_digest(p)for p in self.root.rglob('*')if p.is_file()}
        orig=Path.open
        def opened(p,mode='r',*args,**kwargs):
            if p.is_relative_to(self.root):self.assertIn(mode,('rb','r'))
            return orig(p,mode,*args,**kwargs)
        with patch.object(Path,'open',opened):self.assertEqual(self.verify()['status'],'VERIFIED_COMPLETE')
        self.assertEqual(before,{str(p.relative_to(self.root)):V.file_digest(p)for p in self.root.rglob('*')if p.is_file()})
    def test_statefile_not_stop_operation_history(self):
        r=self.verify();p8=next(x for x in r['checks']if x['check_id']=='P08');self.assertEqual(p8['judgment'],'PENDING_EVIDENCE')
    def test_parallel_cpu_boundary_no_double_counting(self):
        c=C.aggregate(self.verify());self.assertEqual(c['status'],'VERIFIED_PARALLEL_COST_BOUNDARIES')
        self.assertEqual(c['total_cpu_ns'],self.report['host_outer']['parent_cpu_ns']+self.report['host_outer']['worker_cpu_ns'])
        self.assertEqual(c['global_wall_ns'],self.report['host_outer']['wall_ns'])
        for k in('coarse_fine_added','pair_CPU_added_to_worker_CPU','IPC_added_to_outer','parent_worker_wall_summed','RSS_sum_is_unique_memory'):self.assertFalse(c[k])
        self.assertIsNone(c['receipt_self_cost_ns']);self.assertIsNone(c['ranking'])
    def test_parent_pair_cpu_double_count_detected(self):
        r=self.verify();r['report']['total_cpu_ns']+=r['report']['pairs'][0]['cpu_ns'];self.assertEqual(C.aggregate(r)['status'],'COST_INTEGRITY_MISMATCH')
    def test_fine_coarse_mixing_detected(self):
        r=self.verify();row=r['report']['pairs'][-1];row['fine_reconciliation']['exclusive_sum_ns'][0]=row['cpu_ns']+1
        self.assertEqual(C.aggregate(r)['status'],'COST_INTEGRITY_MISMATCH')
    def test_no_clock_equality_requirement_for_observer_acceptance(self):
        c=C.aggregate(self.verify());self.assertTrue(c['serial_condition_is_different'])
    def test_normalized_operations_full_flow(self):
        tasks,_=G.plan(18);r=G.check_operations(F.operations(tasks),tasks,'INDEPENDENT_ARTIFICIAL')
        self.assertEqual(r['counts']['completed'],832);self.assertFalse(r['outstanding'])
    def test_global_run_identity_not_worker_id(self):
        tasks,_=G.plan(16);rows=list(F.operations(tasks));rows[1]['run_id']='worker-1'
        with self.assertRaisesRegex(V.Invalid,'RUN_ID'):G.check_operations(rows,tasks,'INDEPENDENT_ARTIFICIAL')


BAD_REPORT={
 'planned':lambda r:r.update(planned=1),
 'reserved':lambda r:r.update(reserved=r['reserved']-1),
 'started':lambda r:r.update(started_observed=r['started_observed']-1),
 'states':lambda r:r['states'].update(COMPLETED=1),
 'duplicate_pair':lambda r:r['pairs'].append(deepcopy(r['pairs'][0])),
 'duplicate_completion':lambda r:r['completion_order'].append(r['completion_order'][0]),
 'wrong_contract_order':lambda r:r['contract_order'].reverse(),
 'task_offset':lambda r:r['pairs'][0]['task'].update(offset=2),
 'budget_observed':lambda r:r['output']['observed'].update(result=r['output']['reserved']['result']+1),
 'budget_reserved':lambda r:r['output']['reserved'].update(result=1<<37),
 'retry':lambda r:r.update(retry=1),
 'resume':lambda r:r.update(resume=True),
 'schema':lambda r:r.update(schema='NEW_SERVER_UNPUBLISHED'),
}
for name,fn in BAD_REPORT.items():
    def test(self,fn=fn,name=name):
        fn(self.report);self.save();r=self.verify();self.assertEqual(r['status'],'UNSUPPORTED_SCHEMA'if name=='schema'else'INTEGRITY_MISMATCH',r['issues'])
    setattr(Global,'test_bad_'+name+'_refused',test)

BAD_STATE={'invalid_code':lambda b:b.__setitem__(0,9),'unfinished_then_complete':lambda b:b.__setitem__(0,1),'truncated':lambda b:b.pop(),'extra':lambda b:b.append(4)}
for name,fn in BAD_STATE.items():
    def test(self,fn=fn):
        p=self.root/'output/trial_states.bin';b=bytearray(p.read_bytes());fn(b);p.write_bytes(b)
        self.assertEqual(self.verify()['status'],'INTEGRITY_MISMATCH')
    setattr(Global,'test_state_'+name+'_refused',test)


class Published(unittest.TestCase):
    pass
for case in('light-after8-0','P1_inflight','P2_inflight','after_event','angle','correctness_zero','event','flush','log_cap','parent_crash','parent_dispatch','reserved_exit','result_cap','sample','simultaneous','stop_save','worker_exit','write'):
    def test(self,case=case):
        r=A.verify(PACKET/'published_cases'/case/'manifest.json')
        self.assertEqual(r['status'],'VERIFIED_COMPLETE'if case=='light-after8-0'else'TECHNICAL_FAILURE',r['issues'])
        if case=='P1_inflight':self.assertEqual((r['calls']['reserved'],r['calls']['completed'],r['calls']['failed']),(8,7,1))
        if case=='simultaneous':self.assertEqual(r['calls']['failed'],8)
        if case=='parent_crash':self.assertIsNone(r['calls']['started_observed']);self.assertEqual(r['calls']['unknown'],1)
        if case=='reserved_exit':self.assertEqual(r['calls']['started_observed'],0);self.assertEqual(r['calls']['unknown'],1)
        if case=='flush':self.assertTrue(any(p['failure']and p['failure'].get('committed_engine_summary_before_cleanup_failure')for p in r['report']['pairs']))
        if case=='stop_save':self.assertIn('secondary_stop_error',r['report'])
        c=C.aggregate(r);self.assertEqual(c['status'],'INSUFFICIENT_EVIDENCE'if case=='parent_crash'else'VERIFIED_PARALLEL_COST_BOUNDARIES',c.get('issues'))
    setattr(Published,'test_fixed_public_'+case,test)


class Operations(unittest.TestCase):
    def setUp(self):self.tasks,_=G.plan(16)
    def make(self,ops):
        t=self.tasks[0];tid=f"{t['phase']}/{t['pass_name']}/{t['arm']}/{t['layer']}/X/0"
        return[dict(seq=i,run_id='ART',op=op,task_key=t['key'],trial_id=tid,phase=0)for i,op in enumerate(ops)]
    def test_P1_drain_after_first_failure(self):
        rows=self.make(['PHASE_BEGIN','RESERVE','ENTER']);t=self.tasks[1];tid='correctness/baseline/ordinary/B/X/0'
        rows.extend([dict(seq=3,run_id='ART',op='RESERVE',task_key=t['key'],trial_id=tid),dict(seq=4,run_id='ART',op='ENTER',task_key=t['key'],trial_id=tid),dict(rows[1],seq=5,op='FAIL'),dict(seq=6,run_id='ART',op='COMPLETE',task_key=t['key'],trial_id=tid)])
        r=G.check_operations(rows,self.tasks,'ART');self.assertEqual(r['counts']['completed'],1);self.assertEqual(r['counts']['failed'],1)
    def test_enter_after_STOP_refused(self):
        with self.assertRaisesRegex(V.Invalid,'ENTER_AFTER_STOP'):G.check_operations(self.make(['PHASE_BEGIN','RESERVE','STOP','ENTER']),self.tasks,'ART')
    def test_reserve_after_STOP_refused(self):
        with self.assertRaisesRegex(V.Invalid,'RESERVE_AFTER_STOP'):G.check_operations(self.make(['PHASE_BEGIN','STOP','RESERVE']),self.tasks,'ART')
    def test_duplicate_reservation_refused(self):
        with self.assertRaises(V.Invalid):G.check_operations(self.make(['PHASE_BEGIN','RESERVE','RESERVE']),self.tasks,'ART')
    def test_duplicate_completion_refused(self):
        with self.assertRaises(V.Invalid):G.check_operations(self.make(['PHASE_BEGIN','RESERVE','ENTER','COMPLETE','COMPLETE']),self.tasks,'ART')
    def test_started_reserved_is_not_entered(self):
        r=G.check_operations(self.make(['PHASE_BEGIN','RESERVE','STOP','UNKNOWN']),self.tasks,'ART')
        self.assertEqual(r['counts'].get('started_observed',0),0);self.assertEqual(r['counts']['unknown'],1)
    def test_phase_barrier_requires_completion(self):
        with self.assertRaises(V.Invalid):G.check_operations(self.make(['PHASE_BEGIN','RESERVE','PHASE_END']),self.tasks,'ART')
    def test_unACK_credit_stays_reserved(self):
        rows=[dict(seq=0,run_id='ART',op='WRITE_RESERVE',channel='result',amount=100),dict(seq=1,run_id='ART',op='WRITE_ACK',channel='result',amount=13)]
        r=G.check_operations(rows,self.tasks,'ART');self.assertEqual(r['reserved_bytes']['result']-r['ack_observed_bytes']['result'],87)
    def test_write_ACK_without_credit_refused(self):
        with self.assertRaisesRegex(V.Invalid,'ACK_WITHOUT_CREDIT'):G.check_operations([dict(seq=0,run_id='ART',op='WRITE_ACK',channel='result',amount=1)],self.tasks,'ART')
    def test_two_runs_may_use_same_local_trial_index(self):
        for name in('run-A','run-B'):
            rows=self.make(['PHASE_BEGIN','RESERVE','ENTER','COMPLETE'])
            for r in rows:r['run_id']=name
            self.assertEqual(G.check_operations(rows,self.tasks,name)['counts']['completed'],1)
    def test_statefile_bounded_memory(self):
        peaks=[]
        with tempfile.TemporaryDirectory()as temp:
            for n in(1024,102400):
                tasks,total=G.plan(n);p=Path(temp)/str(n)
                with p.open('wb')as f:
                    for t in tasks:
                        remaining=2*t['n']
                        while remaining:
                            size=min(65536,remaining);f.write(bytes([4])*size);remaining-=size
                tracemalloc.start();r=G.state_counts(p,tasks);_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
                self.assertEqual(r['counts']['COMPLETED'],total);peaks.append(dict(state_bytes=total,peak_python_bytes=peak))
        self.assertLess(peaks[1]['peak_python_bytes'],peaks[0]['peak_python_bytes']+250000)
        if target:=os.environ.get('B3_PARALLEL_MEMORY_REPORT'):Path(target).write_text(json.dumps(peaks,indent=2)+'\n')


class CandidateAcceptance(unittest.TestCase):
    def test_published_fused_exact_acceptance(self):
        m=load_published('exact_serialization');rows=H.canonical_checks(m.canonical);self.assertTrue(all(x['judgment']=='PASS'for x in rows))
    def test_fused_Stream_event_evidence_angle_failure_bytes(self):
        m=load_published('exact_serialization')
        values=[dict(event={'word':(2,1),'phase':2,'weight':Fraction(1,7)},evidence={'x':'\ud800\x7fé'},angle={'r':Fraction(3,7)}),
                dict(kind='TYPED_FAILURE',current_attempt={'candidate_ready':False},completed_prefix={'EVENT':1},inflight_event=None),
                {1:object(),'1':Fraction(2,3)},-(1<<8192)]
        for x in values:self.assertEqual(m.canonical(x),F.F.T.A.canonical(F.F.D.exact(x)))
    def observer(self):
        stub=ModuleType('trottertracks.algorithm_codesign.g6_w2c_b3_parallel_candidate');stub.D=F.F.D;stub.Client=object;stub.Coordinator=object;stub.StateFile=object
        with patch.dict(sys.modules,{stub.__name__:stub}):return load_published('parallel_overhead')
    def test_cached_observer_call_count_label_exclusive_partition(self):
        O=self.observer();D=F.F.D
        def leaf(x):return x+1
        def parent():return leaf(1)+leaf(2)
        allrows=[]
        for cls in(O.ScopedObserver,O.CachedObserver):
            observer=cls([__file__],F.F.T.FakeClock())
            with observer:self.assertEqual(parent(),5)
            allrows.append(observer.meter.rows)
            reconciliation=observer.meter.reconcile((1000000,2000000));self.assertGreaterEqual(reconciliation['unattributed_and_observer_residual_ns'][0],0)
        self.assertEqual(allrows[0],allrows[1]);self.assertTrue(any(x.endswith(':leaf')for x in allrows[0]))
    def test_cached_observer_preexisting_profiler_refused(self):
        O=self.observer()
        with patch.object(sys,'getprofile',return_value=lambda *a:None):
            with self.assertRaises(F.F.D.Stop):O.CachedObserver([__file__]).__enter__()
    def test_fixed_fused_and_observer_SOURCE_SHA(self):
        expected={'published_exact_serialization.py':'e05326d95d6c8e91d779e23db3283ca2cd41b6eab2fb1024707c112c1c93087e','published_parallel_overhead.py':'ff9213fa6ccbd65b8c20e9eec1576d2c43d5dc3b29a035f50271582804ba392e'}
        for p,h in expected.items():self.assertEqual(V.file_digest(PACKET/'fixed_sources'/p),h)
    def test_no_scientific_module_imports(self):
        for n in('g7_generator','g7_provider','return_aggregation','g6_f3_contract','g6_f3_precision','g6_f3_generator'):self.assertNotIn('trottertracks.algorithm_codesign.'+n,sys.modules)

class EngineFailures(unittest.TestCase):
    pass
for mode in ('partial','flush','file_hash'):
    def test(self,mode=mode):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp)/'input';spec,f=F.F.failure(root,mode)
            task=next(t for t in G.plan(18)[0]if(t['phase'],t['pass_name'],t['arm'],t['layer'])==V.key(spec))
            row=dict(status='FAILED',failure=dict(trial_failure_provenance=f))
            r=A.verify_worker(root/'pair.jsonl',task,row)
            self.assertEqual(r['status'],'TECHNICAL_FAILURE',r['issues']);self.assertEqual(r['engine_committed_trials'],1)
            self.assertEqual(r['visible_trial_rows'],1 if mode=='partial'else 2)
    setattr(EngineFailures,'test_emit_'+mode+'_not_logical_completion',test)

class ExtraGlobal(Global):
    def test_path_collision_refused(self):
        self.m['pair_paths']={t['key']:t['key']+'.jsonl'for t in G.plan(18)[0]}
        keys=list(self.m['pair_paths']);self.m['pair_paths'][keys[1]]=self.m['pair_paths'][keys[0]];self.save()
        self.assertEqual(self.verify()['status'],'INTEGRITY_MISMATCH')
    def test_global_cli_seven_outputs_no_overwrite(self):
        from trottertracks.algorithm_codesign.g6_w2c_b3_parallel_offline_cli import run
        target=self.root.parent/'reports';v,c,a=run(self.root/'manifest.json',target)
        self.assertEqual(v['status'],'VERIFIED_COMPLETE');self.assertEqual(len(list(target.iterdir())),7)
        self.assertTrue(any(x['check_id']=='P09'and x['judgment']=='PASS'for x in a['checks']))
        with self.assertRaises(V.Invalid):run(self.root/'manifest.json',target)
        with self.assertRaises(V.Invalid):run(self.root/'manifest.json',self.root/'reports')
    def test_optional_normalized_operations_adapter(self):
        tasks,_=G.plan(18);F.control_write(self.root/'operations.jsonl',F.operations(tasks));self.m['operations']='operations.jsonl';self.save()
        r=self.verify();self.assertEqual(r['status'],'VERIFIED_COMPLETE',r['issues']);self.assertIn('normalized_operation_evidence',r)
    def test_forged_first_failure_identity_refused(self):
        self.report['failures']=[dict(parent_arrival=1,parent_observed_ns=1,inflight_snapshot={'BAD_WORKER':dict(index=0,trial_id='bad',worker='BAD_WORKER',state='IN_FLIGHT')})]
        self.report['representative_failure']=self.report['failures'][0];self.save();self.assertEqual(self.verify()['status'],'INTEGRITY_MISMATCH')

def load_tests(loader,standard,pattern):
    suite=unittest.TestSuite()
    for cls in(Global,Published,Operations,CandidateAcceptance,EngineFailures,ExtraGlobal):
        for name in sorted(cls.__dict__):
            if name.startswith('test_'):suite.addTest(cls(name))
    return suite

# Additional acceptance tests are attached before unittest collection.
def test_unACK_actual_write_not_inferred_as_ACK(self):
    self.report['output']['observed']['result']-=13;self.save();r=self.verify()
    self.assertEqual(r['status'],'INSUFFICIENT_EVIDENCE')
    self.assertEqual(r['output_budget']['result']['unacknowledged_credit'],13)
    self.assertFalse(r['output_budget']['result']['unacknowledged_credit_refunded'])
setattr(ExtraGlobal,'test_unACK_actual_write_not_inferred_as_ACK',test_unACK_actual_write_not_inferred_as_ACK)

def test_unused_uncertain_credit_not_refunded(self):
    self.report['output']['reserved']['result']+=17;self.save();r=self.verify()
    self.assertEqual(r['status'],'INSUFFICIENT_EVIDENCE');self.assertEqual(r['output_budget']['result']['unacknowledged_credit'],17)
setattr(ExtraGlobal,'test_unused_uncertain_credit_not_refunded',test_unused_uncertain_credit_not_refunded)

def test_published_archives_materialized_bytes_SHA(self):
    import tarfile
    inventory=json.loads((PACKET/'published_input_inventory.json').read_text())
    for r in inventory:
        p=ROOT/r['local_path'];self.assertEqual(V.file_digest(p),r['sha256'])
        if p.name.endswith('-raw.tar.gz'):
            case=p.name.removesuffix('-raw.tar.gz')
            with tarfile.open(p)as tar:
                for m in tar:
                    if m.isfile():
                        with tar.extractfile(m)as f:expected=sha256(f.read()).hexdigest()
                        self.assertEqual(V.file_digest(PACKET/'published_cases'/case/m.name),expected)
setattr(Published,'test_published_archives_materialized_bytes_SHA',test_published_archives_materialized_bytes_SHA)

def test_cached_observer_includes_published_helper_path(self):
    O=self.observer();observer=O.CachedObserver([__file__],F.F.T.FakeClock())
    self.assertIn(str(Path(O.__file__).resolve()),observer.paths)
setattr(CandidateAcceptance,'test_cached_observer_includes_published_helper_path',test_cached_observer_includes_published_helper_path)

def test_failure_costs_are_diagnostics_not_normal_comparison(self):
    r=A.verify(PACKET/'published_cases/worker_exit/manifest.json');c=C.aggregate(r)
    self.assertEqual(c['status'],'VERIFIED_PARALLEL_COST_BOUNDARIES');self.assertFalse(c['comparison_eligible']);self.assertTrue(c['pair_costs_unknown'])
setattr(Published,'test_failure_costs_are_diagnostics_not_normal_comparison',test_failure_costs_are_diagnostics_not_normal_comparison)
