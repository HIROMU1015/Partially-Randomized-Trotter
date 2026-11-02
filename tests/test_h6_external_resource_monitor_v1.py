"""Bound leases, real callback continuity, and fail-closed resource fixtures."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
r=load('external_test_runner',ROOT/'scripts/h6_b0i_pilot_runner_v1.py');g=r.g
e=load('external_test_monitor',ROOT/'scripts/h6_external_resource_monitor_v1.py')
f=load('external_test_fixtures',ROOT/'tests/final_pre_review_fixtures_v1.py')


class ExternalMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']);self.p=Path(self.temp.name)
        self.output=self.p/'output';self.output.mkdir();self.control=self.p/'control';self.control.mkdir()
        self.own=g.identity();self.launch=g.identity(os.getppid())
        self.ticket=dict(identity=self.own,parent_identity=self.launch)
        g.marker_path(self.output).write_text(json.dumps(self.ticket))
        self.memory=g.d0.memory_snapshot();budget=deepcopy(f.BUDGET)
        self.c=dict(SOURCE='a'*40,budget=budget,control_directory=str(self.control),
            external_monitor_memory_baseline=self.memory,log_cap_bytes=32*2**20,
            parallel_compile=dict(driver_cpu=next(iter(os.sched_getaffinity(0))),worker_cpus=[]))
        self.grant=SimpleNamespace(contract=self.c,contract_sha='b'*64,authorization_sha='c'*64,output=self.output)
        self.value=dict(schema='H6_EXTERNAL_RESOURCE_OBSERVATION_V1',SOURCE=self.c['SOURCE'],
            contract_sha256='b'*64,authorization_sha256='c'*64,launcher_identity=self.launch,
            driver_identity=self.own,sequence=1,monotonic=time.monotonic(),max_interval_seconds=1.,
            max_observation_seconds=.01,driver=dict(identity=self.own),workers=[],first_stop=None)
        (self.control/'launch_stdout.log').touch();self.write()
    def tearDown(self):self.temp.cleanup()
    def write(self):e.atomic(self.control/e.HEARTBEAT,self.value)
    def check(self,value=None,own=None):
        return e.verify_record(value or self.value,self.grant,self.ticket,own or self.own,time.monotonic())
    def test_bound_current_observation_passes(self):self.assertIs(self.check(),self.value)
    def test_wrong_source_contract_authorization_and_issuer_rejected(self):
        for key in ('SOURCE','contract_sha256','authorization_sha256','launcher_identity','driver_identity'):
            v=deepcopy(self.value);v[key]='WRONG'
            with self.subTest(key=key),self.assertRaisesRegex(RuntimeError,'BINDING'):self.check(v)
    def test_stale_or_future_observation_rejected(self):
        for delta in (-6,1):
            v=deepcopy(self.value);v['monotonic']=time.monotonic()+delta
            with self.assertRaisesRegex(RuntimeError,'STALE'):self.check(v)
    def test_nonfinite_timing_and_missing_sequence_rejected(self):
        for key in ('monotonic','max_interval_seconds','max_observation_seconds'):
            v=deepcopy(self.value);v[key]=float('nan')
            with self.assertRaisesRegex(RuntimeError,'FINITE'):self.check(v)
        v=deepcopy(self.value);v['sequence']=0
        with self.assertRaisesRegex(RuntimeError,'SEQUENCE'):self.check(v)
    def test_fresh_heartbeat_cannot_hide_prior_gap_or_io_stall(self):
        for key in ('max_interval_seconds','max_observation_seconds'):
            v=deepcopy(self.value);v[key]=5.01
            with self.assertRaisesRegex(RuntimeError,'GAP'):self.check(v)
    def test_compile_child_requires_own_observation(self):
        worker=dict(self.own,pid=12345,parent_pid=self.own['pid'])
        with self.assertRaises(LookupError):self.check(own=worker)
        v=deepcopy(self.value);v['workers']=[dict(identity=worker)]
        self.assertEqual(self.check(v,worker),v)
    def test_delayed_callback_requires_continuous_independent_proof(self):
        m=e.monitor_class(g)(self.grant);m.last-=6;m.tick(force=True)
        self.assertGreater(m.history[-1]['driver_callback_interval_seconds'],5)
        self.assertIsNone(m.first_stop)
        self.value['max_interval_seconds']=6;self.write();m.last-=6
        with self.assertRaisesRegex(g.Stop,'GAP'):m.tick(force=True)
    def test_sequence_reversal_and_missing_observation_stop(self):
        m=e.monitor_class(g)(self.grant);self.value['sequence']=0;self.write()
        with self.assertRaises(g.Stop):m.tick(force=True)
        (self.control/e.HEARTBEAT).unlink()
        with self.assertRaises(FileNotFoundError):m.observation()
    def test_first_external_stop_reason_preserved(self):
        m=e.monitor_class(g)(self.grant)
        self.value['first_stop']=dict(reason='FIXTURE_PRESSURE',phase='B_NATIVE_CORRECTNESS',monotonic=time.monotonic());self.write()
        with self.assertRaisesRegex(g.Stop,'FIXTURE_PRESSURE'):m.fail('OWN_SIGNAL:15')
        with self.assertRaisesRegex(g.Stop,'FIXTURE_PRESSURE'):m.fail('SECOND_REASON')
        self.assertEqual(m.first_stop['phase'],'B_NATIVE_CORRECTNESS')
    def test_observer_pressure_failure_is_durable_and_latched(self):
        observer=e.Observer(self.grant,self.own,self.launch,g)
        with patch.object(observer,'process',return_value=dict(identity=self.own)),patch.object(g.d0,'check_memory_snapshot',side_effect=g.Stop('FIXTURE_OOM')):
            with self.assertRaisesRegex(g.Stop,'FIXTURE_OOM'):observer.step()
        value=e.read_small(self.control/e.HEARTBEAT)
        self.assertIn('FIXTURE_OOM',value['first_stop']['reason'])
    def test_wrong_owner_never_read_or_signaled(self):
        observer=e.Observer(self.grant,self.own,self.launch,g)
        with patch.object(g,'identity',return_value=dict(self.own,start_ticks=-1)):
            with self.assertRaisesRegex(g.Stop,'IDENTITY_CHANGED'):observer.process(self.own,[])
    def test_log_limit_and_oversize_record_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError,'CAP'):e.atomic(self.control/'huge',dict(x='x'*65536))
        self.c['log_cap_bytes']=1;observer=e.Observer(self.grant,self.own,self.launch,g)
        with patch.object(observer,'process',return_value=dict(identity=self.own)):
            with self.assertRaisesRegex(g.Stop,'LOG_CAP'):observer.step()
    def test_bootstrap_has_no_lease_and_cannot_mask_later_cpu_change(self):
        observer=e.Observer(self.grant,self.own,self.launch,g);driver=self.c['parallel_compile']['driver_cpu'];worker=driver+1
        with patch.object(os,'sched_getaffinity',return_value={driver}):
            self.assertIsNone(observer.process(self.own,[worker],worker_bootstrap=True))
        with patch.object(os,'sched_getaffinity',return_value={worker}):
            self.assertIsNotNone(observer.process(self.own,[worker],worker_bootstrap=True))
        with patch.object(os,'sched_getaffinity',return_value={driver}):
            with self.assertRaisesRegex(g.Stop,'AFFINITY'):observer.process(self.own,[worker],worker_bootstrap=True)
    def test_six_mock_workers_use_atomic_registry_not_partial_artifacts(self):
        self.c['parallel_compile']['worker_cpus']=list(range(100,106))
        observer=e.Observer(self.grant,self.own,self.launch,g)
        for index,(p,q,axis) in enumerate((p,q,a) for p,q in ((5,1),(10,2),(15,8)) for a in ('cosine','sine')):
            name=f'{p}_{q}_{axis}';(self.output/('compile_started_'+name+'.json')).write_text('{')
            record=dict(identity=dict(self.own,pid=10000+index,parent_pid=self.own['pid']),cpu=100+index)
            e.atomic(self.control/('observed_compile_started_'+name+'.json'),record)
        with patch.object(observer,'process',side_effect=lambda identity,allowed,**kw:dict(identity=identity)):
            observer.step()
        self.assertEqual(len(e.read_small(self.control/e.HEARTBEAT)['workers']),6)
