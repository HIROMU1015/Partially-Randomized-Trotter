"""Deterministic /proc exit_mm races; missing live data is never success."""
import errno
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1]
def load(name,file):
    s=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
r=load('proc_exit_runner','h6_b0i_pilot_runner_v1.py');g=r.g
p=load('proc_exit_backend','h6_b0i_parallel_compile_v1.py')
e=load('proc_exit_observer','h6_external_resource_monitor_v1.py')

class ProcExitTests(unittest.TestCase):
    def setUp(self):
        self.identity=dict(pid=123,parent_pid=100,uid=os.getuid(),start_ticks=10,boot_id='fake')
        self.handle=dict(pid=123,identity=self.identity,reaped=False,started=0.,resource_last_seen=0.,
            peak_rss_bytes=0,affinity_confirmed=True,task=dict(cpu=1))
        self.backend=p.ForkBackend.__new__(p.ForkBackend);self.backend.g=g
        self.backend.grant=SimpleNamespace(contract=dict(budget=dict(monitor_max_seconds=5.,rss_bytes=2**30,address_space_bytes=2**30,cpu_seconds=60)))
        self.backend.clock_ticks=100;self.backend.reap=Mock()
        self.status='State:\tR (running)\nVmSize:\t100 kB\nThreads:\t1\n'
    def read_status(self):
        return patch.object(Path,'read_text',return_value=self.status)
    def test_missing_rss_with_native_reap_does_not_abort_following_cleanup(self):
        self.backend.reap.side_effect=lambda h:h.update(reaped=True,exit_code=0)
        with self.read_status(),patch.object(g,'identity',return_value=self.identity):
            self.backend.observe(self.handle)
        self.assertTrue(self.handle['reaped']);self.assertEqual(self.handle['exit_code'],0)
    def test_live_missing_rss_is_pending_then_stops_at_original_five_seconds(self):
        with self.read_status(),patch.object(g,'identity',return_value=self.identity):
            with patch.object(p.time,'monotonic',return_value=4.9):self.backend.observe(self.handle)
            self.assertEqual(self.handle['resource_last_seen'],0.)
            with patch.object(p.time,'monotonic',return_value=5.1):
                with self.assertRaisesRegex(g.Stop,'OBSERVATION_MISSING'):self.backend.observe(self.handle)
    def test_missing_fields_cannot_hide_changed_owner(self):
        with self.read_status(),patch.object(g,'identity',side_effect=[self.identity,dict(self.identity,start_ticks=11)]):
            with self.assertRaisesRegex(g.Stop,'OWNERSHIP_CHANGED'):self.backend.observe(self.handle)
    def test_other_missing_fields_are_not_an_exit_exception(self):
        self.status='State:\tR\nVmRSS:\t10 kB\nVmSize:\t10 kB\n'
        with self.read_status(),patch.object(g,'identity',return_value=self.identity),patch.object(Path,'read_text',side_effect=[self.status,'123 (worker) '+'0 '*30]):
            with self.assertRaises(KeyError):self.backend.observe(self.handle)
    def test_zombie_and_disappeared_proc_are_pending_reap(self):
        for status in ('State:\tZ (zombie)\n',None):
            with patch.object(g,'identity',return_value=self.identity),patch.object(Path,'read_text',side_effect=FileNotFoundError() if status is None else None,return_value=status):
                self.backend.observe(self.handle)
            self.assertFalse(self.handle['reaped'])
    def test_non_esrch_sampling_errors_propagate(self):
        with patch.object(g,'identity',return_value=self.identity),patch.object(Path,'read_text',side_effect=PermissionError(errno.EACCES,'denied')):
            with self.assertRaises(PermissionError):self.backend.observe(self.handle)
    def observer(self,clock):
        observer=e.Observer.__new__(e.Observer);observer.g=g;observer.clock=clock
        observer.resource_last_seen={};observer.worker_first_seen={};observer.worker_affinity_confirmed=set()
        observer.grant=self.backend.grant;return observer
    def test_external_missing_hwm_never_issues_a_lease_and_remains_bounded(self):
        now=[0.];observer=self.observer(lambda:now[0])
        with self.read_status(),patch.object(g,'identity',return_value=self.identity):
            self.assertIsNone(observer.process(self.identity,[1]))
            now[0]=5.1
            with self.assertRaisesRegex(g.Stop,'OBSERVATION_MISSING'):observer.process(self.identity,[1])
    def test_external_other_errors_and_identity_changes_are_not_ignored(self):
        observer=self.observer(lambda:0.)
        with self.read_status(),patch.object(g,'identity',side_effect=[self.identity,dict(self.identity,start_ticks=12)]):
            with self.assertRaisesRegex(g.Stop,'IDENTITY_CHANGED'):observer.process(self.identity,[1])
        with patch.object(g,'identity',return_value=self.identity),patch.object(Path,'read_text',side_effect=OSError(errno.EIO,'IO')):
            with self.assertRaises(OSError):observer.process(self.identity,[1])
