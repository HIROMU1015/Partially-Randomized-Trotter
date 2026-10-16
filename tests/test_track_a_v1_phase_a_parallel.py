"""Artificial scheduling, grant accounting and Linux cleanup; no H6 inputs."""
from copy import deepcopy
import cmath
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import track_a_phase_a_parallel as pool
import track_a_v1_parallel_signal as adapter
import track_a_v1_runner as runner
import launch_track_a_v1_phase_a as observer


class Contracts(unittest.TestCase):
    def test_factored_outer_helper_uses_port_sector_and_original_scalar(self):
        # One-dimensional commuting example, standard library only. This
        # exercises the factored helper's scope, phase and both action paths.
        class Vector(list):
            def copy(self):return Vector(self)
            def __imul__(self,z):self[:]=[x*z for x in self];return self
            def __sub__(self,b):return Vector(x-y for x,y in zip(self,b))
        class Matrix:
            def __init__(self,x):self.x=x
            def __rmul__(self,z):return Matrix(z*self.x)
            def __matmul__(self,v):return Vector(self.x*x for x in v)
        np=NS(zeros_like=lambda x:0,asarray=lambda x,**kw:x,exp=cmath.exp,
            linalg=NS(norm=lambda v:sum(abs(x)**2 for x in v)**.5),
            vdot=lambda a,b:sum(x.conjugate()*y for x,y in zip(a,b)))
        m=NS(np=np,occupation_df_matrix=lambda c,o,l,g,s:Matrix(sum(l)),
            expm=lambda a:Matrix(cmath.exp(a.x)),finite_scale_guard=lambda **kw:None,
            complex_record=lambda z:dict(real=z.real,imag=z.imag))
        coefficients=(.3,.1,.2,.4,.5);scalar=.7
        whole=NS(deterministic_blocks=coefficients,constant_coefficient=scalar)
        port=NS(sector=NS(basis_indices=(0,)),reference=cmath.exp(-1j*.8*(sum(coefficients)+scalar)))
        port.actions=lambda view,**kw:[lambda v,t,c=c:Vector(cmath.exp(-1j*t*c)*x for x in v)
                                       for c in view.deterministic_blocks]
        ham=NS(one_body=0,lambdas=coefficients[1:],g_matrices=(0,0,0,0))
        calls=[];permit=NS(take=lambda key:calls.append(key),guard=lambda:None)
        rows=adapter.outer_diagnostics(port,whole,ham,Vector([1+0j]),permit,m,(1,2,4,8),(3,4))
        self.assertEqual(len(calls),8)
        for row in rows:
            bias=row['outer_PF_signed_bias']
            self.assertLess(abs(complex(bias['real'],bias['imag'])),1e-14)
            self.assertEqual(row['independent_path_discrepancy'],0)
            self.assertFalse(row['formal_candidate'])

    def test_phase_a_only_and_preserves_science_registration(self):
        p=runner.contract.phase_a_plan();old=runner.contract.plan()
        self.assertEqual(p['registrations'],old['registrations'])
        self.assertEqual(p['science_contract_sha256'],runner.contract.digest(old))
        self.assertEqual(p['resources']['max_workers'],16)
        self.assertTrue(all(p['call_caps'][k]==0 for k in ('native_compile_max','native_evolution_max','measured_wrappers_max','control_probes_max')))
        self.assertFalse(p['approved']);self.assertFalse(p['runtime_authorization'])

    def test_partition_52_cells_and_distinct_times_without_reordering(self):
        cells=[x for x in runner.contract.registrations()['additional'] if x['method']=='B0']
        times={(i,t) for i in range(16) for t in (.1,-.1)}
        primitive,signal=adapter.task_plan(cells,times)
        self.assertEqual(len(primitive),16);self.assertEqual(len(signal),11)
        self.assertEqual(sum(t['counts']['primitive'] for t in primitive),3*len(times))
        self.assertEqual(sum(t['counts'].get('new_B0_signals',0) for t in signal),52)
        self.assertEqual(sum(t['counts'].get('outer_S2_diagnostics',0) for t in signal),8)
        for task in signal[:9]:self.assertEqual(task['cells'],[x for x in cells if x['prefix']==task['prefix']])

    def test_global_reservations_no_refund_and_child_no_per_probe_sync(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);g=dict(call_caps={'x':3},resources=dict(wall_seconds=30,RSS_bytes=2**30,output_bytes=2**20))
            permit=runner.Permit({},g,p);permit.reserve('one',{'x':2})
            with self.assertRaisesRegex(ValueError,'GLOBAL_RESERVED'):permit.reserve('two',{'x':2})
            child=p/'worker';child.mkdir();worker=permit.child(child,{'x':2})
            worker.take('x');worker.take('x')
            with self.assertRaisesRegex(ValueError,'CALL_CAP'):worker.take('x')
            self.assertFalse((child/'reservations.jsonl').exists())
            with self.assertRaisesRegex(ValueError,'BINDING'):permit.merge_reserved('one',{'x':1})
            permit.merge_reserved('one',worker.used);permit.reserve('two',{'x':1})
            self.assertEqual(permit.used,{'x':2});self.assertEqual(len((p/'task_reservations.jsonl').read_text().splitlines()),2)
            with self.assertRaisesRegex(ValueError,'UNIQUE_TASK'):permit.reserve('one',{'x':0})

    def test_sealed_synthetic_grant_requires_disjoint_roles_and_fresh_gate(self):
        p=runner.contract.phase_a_plan();digest=runner.contract.digest(p)
        grant=dict(schema='track_a_v1_phase_a_parallel_grant_v1',approved_by_user=True,
            runtime_authorization=True,one_shot=True,retry=False,resume=False,scope='signal',
            SOURCE='a'*40,config_sha256='b'*64,plan_sha256=digest,call_caps=p['call_caps'],
            resources=p['resources'],allowed_cpus=list(range(18)),driver_cpu=0,monitor_cpu=1,worker_cpus=list(range(2,18)))
        config=dict(phase_a_parallel=True,plan_sha256=digest)
        fresh=dict(resources_idle_verified=True,profile_verified=True,protected_jobs_unchanged=True,
            source='a'*40,allowed_cpus=list(range(18)),config_sha256='b'*64,
            external_RSS_monitor_ready=True,monotonic_seconds=time.monotonic())
        with patch.object(os,'sched_getaffinity',return_value=set(range(18))):
            self.assertTrue(runner.verify_grant(config,grant,config_sha256='b'*64,source='a'*40,fresh=fresh))
            invalid=deepcopy(grant);invalid['worker_cpus'][0]=0
            with self.assertRaisesRegex(ValueError,'DISJOINT'):runner.verify_grant(config,invalid,config_sha256='b'*64,source='a'*40,fresh=fresh)
            fresh['monotonic_seconds']-=31
            with self.assertRaisesRegex(ValueError,'FRESH_RESOURCE'):runner.verify_grant(config,grant,config_sha256='b'*64,source='a'*40,fresh=fresh)

    def test_unapproved_and_compile_scopes_rejected(self):
        p=runner.contract.phase_a_plan()
        for approved,scope in ((False,'signal'),(True,'compile_validation')):
            grant=dict(schema='track_a_v1_phase_a_parallel_grant_v1',approved_by_user=approved,
                runtime_authorization=True,one_shot=True,retry=False,resume=False,scope=scope,
                SOURCE='a'*40,config_sha256='b'*64,plan_sha256=runner.contract.digest(p),
                call_caps=p['call_caps'],resources=p['resources'])
            with self.assertRaisesRegex(ValueError,'SEALED'):runner.verify_grant(
                dict(phase_a_parallel=True),grant,config_sha256='b'*64,source='a'*40,fresh={})

    def test_external_limits_include_live_aggregate_and_threads(self):
        limits=dict(max_workers=2,RSS_bytes=100,AS_bytes=200,aggregate_RSS_bytes=150,aggregate_CPU_seconds=10)
        row=dict(zombie=False,RSS_bytes=80,AS_bytes=100,threads=1)
        observer.enforce_processes([row],limits,1)
        with self.assertRaisesRegex(RuntimeError,'AGGREGATE_RSS'):observer.enforce_processes([row,row],limits,1)
        with self.assertRaisesRegex(RuntimeError,'THREADS'):observer.enforce_processes([dict(row,threads=2)],limits,1)
        with self.assertRaisesRegex(RuntimeError,'CPU'):observer.enforce_processes([row],limits,11)

    def test_memory_pressure_and_oom_stop(self):
        base=dict(effective_available_bytes=1000,events={'cg':{'oom':0}},pressure={'host':'some avg10=0 avg60=0 avg300=0 total=0'})
        self.assertIsNone(observer.resource_reason(base,base,100))
        self.assertEqual(observer.resource_reason(base,base,1001),'MEMORY_HEADROOM')
        changed=deepcopy(base);changed['events']['cg']['oom']=1
        self.assertEqual(observer.resource_reason(changed,base,100),'OOM_INCREMENT')
        changed=deepcopy(base);changed['pressure']['host']='some avg10=1 avg60=0 avg300=0 total=0'
        self.assertEqual(observer.resource_reason(changed,base,100),'MEMORY_PRESSURE')


class OwnedLinux(unittest.TestCase):
    def setUp(self):
        self.cpus=sorted(os.sched_getaffinity(0))[:2]
        self.limits=dict(max_workers=2,AS_bytes=2*2**30,RSS_bytes=2**30,
            CPU_seconds=10,worker_CPU_seconds=10,aggregate_CPU_seconds=30,aggregate_RSS_bytes=2*2**30,
            wall_seconds=10,output_bytes=2**20)

    def test_real_two_worker_overlap_affinity_and_native_reap(self):
        with tempfile.TemporaryDirectory() as td:
            def callback(task,directory):
                start=time.monotonic();time.sleep(.15)
                return dict(start=start,end=time.monotonic(),cpu=list(os.sched_getaffinity(0)))
            receipt=pool.map_owned([{'id':'a'},{'id':'b'}],callback,cpus=self.cpus,limits=self.limits,
                directory=Path(td)/'run',deadline=time.monotonic()+5)
            self.assertTrue(receipt['all_reaped']);self.assertTrue(all(h['exit_code']==0 for h in receipt['workers']))
            rows=[json.loads((Path(h['directory'])/'task_result.json').read_text()) for h in receipt['workers']]
            self.assertLess(max(r['start'] for r in rows),min(r['end'] for r in rows))
            self.assertEqual({r['cpu'][0] for r in rows},set(self.cpus))
            for h in receipt['workers']:self.assertIsNone(pool.identity(h['pid']))

    def test_observer_hard_limit_is_not_inherited_by_driver(self):
        # Isolate the observer's irreversible hard ceiling from this test
        # process. The artificial driver must raise its AS limit above it.
        with tempfile.TemporaryDirectory() as td:
            code='''
import sys,os,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import launch_track_a_v1_phase_a as observer
root=Path(sys.argv[2]);output=root/'output';output.mkdir()
limits=json.loads(sys.argv[3]);limits['observer_AS_bytes']=128*2**20
driver='import resource,json,sys; resource.setrlimit(resource.RLIMIT_AS,(256*2**20,256*2**20)); json.dump({"status":"COMPLETE_MANDATORY_STOP"},open(sys.argv[1],"x"))'
observer.supervise([sys.executable,'-B','-c',driver,str(output/'terminal.json')],env=dict(os.environ),control=root/'control',output=output,limits=limits,allowed_cpus=sorted(os.sched_getaffinity(0))[:1],probe=lambda:dict(effective_available_bytes=2**50,events={},pressure={}))
'''
            result=subprocess.run([sys.executable,'-B','-c',code,str(ROOT/'scripts'),td,json.dumps(self.limits)],
                                  capture_output=True,text=True,timeout=15)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_failed_worker_terminates_sleeping_sibling_and_no_retry(self):
        with tempfile.TemporaryDirectory() as td:
            calls=[]
            def callback(task,directory):
                if task['id']=='bad':raise ValueError('ARTIFICIAL_FAILURE')
                time.sleep(20);return {}
            with self.assertRaisesRegex(RuntimeError,'WORKER_FAILED'):
                pool.map_owned([{'id':'bad'},{'id':'sibling'}],callback,cpus=self.cpus,limits=self.limits,
                    directory=Path(td)/'run',deadline=time.monotonic()+5,reserve=lambda t:calls.append(t['id']))
            receipt=json.loads((Path(td)/'run/parallel_receipt.json').read_text())
            self.assertTrue(receipt['all_reaped']);self.assertEqual(calls,['bad','sibling'])
            self.assertFalse(receipt['automatic_retry'])
            for h in receipt['workers']:self.assertIsNone(pool.identity(h['pid']))

    def test_deadline_stop_reaps_children(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(RuntimeError,'WALL_CEILING'):
                pool.map_owned([{'id':'slow'}],lambda t,d:time.sleep(20),cpus=self.cpus,limits=self.limits,
                    directory=Path(td)/'run',deadline=time.monotonic()+.1)
            self.assertTrue(json.loads((Path(td)/'run/parallel_receipt.json').read_text())['all_reaped'])

    def test_unsafe_task_id_rejected_before_fork(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError,'SAFE_TASK'):
                pool.map_owned([{'id':'..'}],lambda t,d:{},cpus=self.cpus,limits=self.limits,
                    directory=Path(td)/'run',deadline=time.monotonic()+1)
            self.assertFalse((Path(td)/'run').exists())

    def test_supervisor_success_and_failed_driver_orphan_cleanup(self):
        baseline=lambda:dict(effective_available_bytes=2**50,events={},pressure={})
        for fail in (False,True):
            with self.subTest(fail=fail),tempfile.TemporaryDirectory() as td:
                root=Path(td);output=root/'output';output.mkdir()
                if fail:
                    code='import os,time; p=os.fork(); os._exit(23) if p else time.sleep(30)'
                else:
                    code='import json,sys; json.dump({"status":"COMPLETE_MANDATORY_STOP"},open(sys.argv[1],"x"))'
                command=[sys.executable,'-B','-c',code,str(output/'terminal.json')]
                try:
                    observer.supervise(command,env=dict(os.environ),control=root/'control',output=output,
                        limits=self.limits,allowed_cpus=self.cpus,probe=baseline)
                except RuntimeError:
                    if not fail:raise
                receipt=json.loads((root/'control/launcher_terminal.json').read_text())
                self.assertTrue(receipt['all_reaped']);self.assertEqual(receipt['native_exit_code'],23 if fail else 0)
                self.assertEqual(receipt['status'],'STOP' if fail else 'COMPLETE_MANDATORY_STOP')


if __name__=='__main__':unittest.main()
