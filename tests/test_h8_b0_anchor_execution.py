"""Artificial new-scope gates, wave barriers and owned-worker fault injection."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import h8_b0_anchor_contract as c
import h8_b0_anchor_runtime as r
import h8_b0_anchor_adapter as a
import launch_h8_b0_anchors as launch
import launch_h8_strengthened_phase_a as observer
import track_a_phase_a_parallel as pool


def gate():
    p=c.plan();base='/home/AbeHiromu/projects/h6-handoff-evidence/artificial-b0-only'
    config=dict(SOURCE='a'*40,execution_sealed=True,plan_sha256=c.digest(p),resources=p['resources'],
        output=base+'/output',control=base+'/control',lock=base+'/lock')
    g=dict(schema='H8_B0_ANCHOR_NEW_SCOPE_GRANT_V1',approved_by_user=True,runtime_authorization=True,
        approval_text='synthetic test only',scope=c.SCOPE,SOURCE=config['SOURCE'],config_sha256='config',
        plan_sha256=c.digest(p),resources=p['resources'],call_caps=p['call_caps'],one_shot=True,retry=False,
        resume=False,next_stage_authorized=False,driver_cpu=0,monitor_cpu=1,worker_cpus=[2,3,4,5],allowed_cpus=list(range(6)))
    f=dict(source=config['SOURCE'],config_sha256='config',grant_sha256=c.digest(g),allowed_cpus=list(range(6)),monotonic_seconds=100)
    for k in ('profile_verified','source_input_bytes_verified','physical_cores_verified',
            'resources_idle_verified','protected_jobs_unchanged','quota_verified','external_RSS_monitor_ready'):f[k]=True
    return config,g,f


class Gates(unittest.TestCase):
    def test_fixed_resources_and_registration(self):
        p=c.plan();self.assertEqual(len(p['axes']),8);self.assertEqual(len(p['waves']),2)
        self.assertEqual(p['resources']['aggregate_RSS_bytes']+p['resources']['admission_reserve_bytes'],int(176.25*2**30))
        self.assertEqual(p['call_caps']['cost_fit'],0)
    def test_valid_synthetic_gate(self):
        cfg,g,f=gate();r.verify_grant(cfg,g,f,config_sha='config',now=110,affinity=set(range(6)))
    def test_live_parent_identity_required(self):
        value=pool.identity(os.getppid());r.verify_observer(dict(observer_identity=list(value)))
        with self.assertRaises(ValueError):r.verify_observer(dict(observer_identity=[1,2,3,4]))
    def test_grants_fail_closed(self):
        for k,v in [('approved_by_user',False),('scope','H8_STRENGTHENED_PHASE_A_SIGNAL_V1'),
                ('schema','H8_NATIVE_B0I_FULL_FAMILY12_V1'),('retry',True),('next_stage_authorized',True),
                ('SOURCE','b'*40),('worker_cpus',[2,2,4,5])]:
            cfg,g,f=gate();g[k]=v
            with self.subTest(k=k),self.assertRaises(ValueError):r.verify_grant(cfg,g,f,config_sha='config',now=110,affinity=set(range(6)))
    def test_freshness_capacity_and_unsealed_rejection(self):
        for k,v in [('monotonic_seconds',50),('monotonic_seconds',float('nan')),('quota_verified',False),
                    ('physical_cores_verified',False),('grant_sha256','wrong')]:
            cfg,g,f=gate();f[k]=v
            with self.subTest(k=k),self.assertRaises(ValueError):r.verify_grant(cfg,g,f,config_sha='config',now=110,affinity=set(range(6)))
        cfg,g,f=gate();cfg['execution_sealed']=False
        with self.assertRaises(ValueError):r.verify_grant(cfg,g,f,config_sha='config',now=110,affinity=set(range(6)))
    def test_draft_launcher_denied_before_files_fork_or_gate(self):
        cfg,g,_=gate();g['approved_by_user']=False
        with patch.object(r,'read',side_effect=[cfg,g]),patch.object(r,'sha',return_value='config'),patch.object(r,'verify_files') as files,patch.object(observer,'supervise') as supervised:
            with self.assertRaises(ValueError):launch.launch(Path('/tmp/config'),Path('/tmp/grant'),Path('/tmp/fresh'))
            files.assert_not_called();supervised.assert_not_called()
    def test_scientific_imports_denied_first(self):
        permit=NS(require=Mock(side_effect=ValueError('DENIED')))
        with patch.object(a.importlib,'import_module') as imported:
            with self.assertRaisesRegex(ValueError,'DENIED'):a.evaluate({},permit,None)
            imported.assert_not_called()
        factory=Mock()
        with self.assertRaisesRegex(ValueError,'DENIED'):
            a.restore({},permit,factory=factory,inspect=Mock(),one_type=NS,fragment_type=NS)
        factory.assert_not_called()
    def test_idle_and_memory_boundaries(self):
        r.check_idle({0:[0]*8},{0:[0,0,0,100,0,0,0,0]},[0])
        with self.assertRaises(ValueError):r.check_idle({0:[0]*8},{0:[10,0,0,90,0,0,0,0]},[0])
        b=dict(effective_available_bytes=int(176.25*2**30),events={},pressure={})
        self.assertIsNone(observer.resource_reason(b,b,int(176.25*2**30)))
        low=dict(b,effective_available_bytes=b['effective_available_bytes']-1)
        self.assertEqual(observer.resource_reason(low,b,b['effective_available_bytes']),'MEMORY_HEADROOM')
    def test_live_monitor_limits_and_unknown_psi(self):
        p=c.plan()['resources'];row=dict(zombie=False,RSS_bytes=32*2**30,AS_bytes=64*2**30,threads=1)
        observer.enforce_processes([row]*5,p,0)
        with self.assertRaises(RuntimeError):observer.enforce_processes([row]*6,p,0)
        with self.assertRaises(RuntimeError):observer.enforce_processes([dict(row,threads=2)],p,0)
        bad=dict(effective_available_bytes=10,events={'g':{'oom':0,'oom_kill':0}},pressure={'host':'some avg10=0.0'})
        with patch.object(observer,'snapshot',return_value=bad),self.assertRaises(ValueError):launch.snapshot()
    def test_quota_unknown_or_enabled_rejected(self):
        stat=NS(f_bavail=100,f_frsize=4096,f_favail=100)
        for mount,ok in [('dev / ext4 rw,relatime 0 0',True),('dev / xfs rw,prjquota 0 0',False),('dev / nfs rw 0 0',False)]:
            with patch.object(Path,'read_text',return_value=mount),patch.object(os,'statvfs',return_value=stat):
                if ok:self.assertTrue(r.filesystem('/tmp')['quota_verified'])
                else:
                    with self.assertRaises(ValueError):r.filesystem('/tmp')
    def test_refresh_uses_pre_pin_launch_affinity(self):
        cfg,g,_=gate();resources=dict(effective_available_bytes=200*2**30,events={},pressure={})
        with patch.object(r,'sha',return_value='config'),patch.object(os,'sched_getaffinity',return_value={1}),\
             patch.object(r,'protected'),patch.object(r,'topology',return_value=set(range(6))),\
             patch.object(r,'cpu_counters',return_value={i:[0]*8 for i in range(6)}),patch.object(r,'check_idle'),\
             patch.object(launch.time,'sleep'),patch.object(launch,'snapshot',return_value=resources),\
             patch.object(r,'filesystem',return_value=dict(bytes=100*2**30,inodes=100000)):
            f=launch.fresh(cfg,g,Path('/tmp/config'),launch_affinity=set(range(6)))
            self.assertTrue(f['physical_cores_verified'])
    def test_no_budget_refund_or_retry(self):
        cfg,g,_=gate()
        with tempfile.TemporaryDirectory() as td:
            permit=r.Permit(cfg,g,td);task=dict(id='one',counts={'compile':1})
            permit.reserve(task)
            with self.assertRaises(ValueError):permit.reserve(task)
            self.assertEqual(permit.actual()['compile'],0)
            permit.settle('one');self.assertEqual(permit.actual()['compile'],1)
            with self.assertRaises(ValueError):permit.settle('one')
            with self.assertRaises(ValueError):permit.take('cost_fit')


class Waves(unittest.TestCase):
    def run_fixture(self,td,*,build_failure=False,worker_failure=False):
        cfg,g,_=gate();g['worker_cpus']=[0,1,2,3];permit=r.Permit(cfg,g,td);events=[]
        def build(case):
            events.append('build:'+case['id'])
            if build_failure and len(events)==2:raise ValueError('CONTROL_FAIL')
            return case
        def compile_axis(task,built,directory):return dict(case_id=task['case_id'],axis=task['axis'])
        def fake_map(tasks,callback,*,directory,reserve,**kw):
            events.append('wave');directory.mkdir();workers=[]
            if worker_failure:raise ValueError('COMPILE_FAILED_NO_RETRY')
            for i,t in enumerate(tasks):
                reserve(t);p=directory/t['id'];p.mkdir();(p/'task_result.json').write_text(json.dumps(callback(t,p)))
                workers.append(dict(task_id=t['id'],directory=str(p),exit_code=0,reaped=True,identity=[i,1,1,1]))
            return dict(status='PASS',all_reaped=True,workers=workers,child_CPU_seconds=1.)
        try:rows=a.run_waves(c.plan(),permit,r.Writer(td),build,compile_axis,map_owned=fake_map)
        except ValueError:return events,None
        return events,rows
    def test_two_case_barrier_and_two_waves(self):
        with tempfile.TemporaryDirectory() as td:
            events,rows=self.run_fixture(td)
            self.assertEqual([x for x in events if x=='wave'],['wave','wave']);self.assertEqual(len(rows),8)
            self.assertTrue(all('result_sha256' in row for row in rows))
    def test_control_failure_admits_no_compile(self):
        with tempfile.TemporaryDirectory() as td:
            events,rows=self.run_fixture(td,build_failure=True)
            self.assertNotIn('wave',events);self.assertIsNone(rows)
    def test_compile_failure_stops_next_wave(self):
        with tempfile.TemporaryDirectory() as td:
            events,rows=self.run_fixture(td,worker_failure=True)
            self.assertEqual(events.count('wave'),1);self.assertIsNone(rows)
    def test_real_owned_workers_and_failure_reaped(self):
        cpus=list(os.sched_getaffinity(0))[:4]
        limits=dict(max_workers=4,AS_bytes=512*2**20,RSS_bytes=256*2**20,worker_CPU_seconds=10,aggregate_CPU_seconds=30)
        for fail in (False,True):
            with tempfile.TemporaryDirectory() as td:
                def callback(task,path):
                    if fail and task['id']=='a':raise ValueError('ARTIFICIAL_FAULT')
                    time.sleep(.05);return {'synthetic':True}
                directory=Path(td)/'wave'
                if fail:
                    with self.assertRaises(RuntimeError):pool.map_owned([{'id':x} for x in 'abcd'],callback,
                        cpus=cpus,limits=limits,directory=directory,deadline=time.monotonic()+10)
                else:pool.map_owned([{'id':x} for x in 'abcd'],callback,cpus=cpus,limits=limits,directory=directory,deadline=time.monotonic()+10)
                receipt=json.loads((directory/'parallel_receipt.json').read_text())
                self.assertTrue(receipt['all_reaped']);self.assertTrue(all(w['reaped'] for w in receipt['workers']))
                self.assertEqual(receipt['automatic_retry'],False)


if __name__=='__main__':unittest.main()
