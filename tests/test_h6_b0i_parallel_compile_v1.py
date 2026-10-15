"""Artificial scheduler/process syscall fixtures. Never forks or compiles."""
from copy import deepcopy
import errno
import importlib.util
import json
import os
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module
f=load('parallel_artificial_fixture',ROOT/'tests/h6_pilot_fixtures_v1.py');r=f.r
p=load('parallel_artificial_scheduler',ROOT/'scripts/h6_b0i_parallel_compile_v1.py')


class FakeBackend:
    instances=[];failure=None;cleanup_error=False
    def __init__(self,grant,api,monitor,g):
        self.grant,self.api,self.g=grant,api,g;self.handles=[];self.poll_widths=[];self.cleaned=False
        self.instances.append(self)
    def start(self,task):
        pid=900000+len(self.handles)
        h=dict(pid=pid,identity=dict(pid=pid,start_ticks=pid,uid=os.getuid(),parent_pid=123,boot_id='ARTIFICIAL'),
            task=task,cpu_seconds=.01,peak_rss_bytes=1024,exit_code=None,reaped=False,payload=bytearray())
        self.handles.append(h);return h
    def poll(self,active):
        self.poll_widths.append(len(active));completed=[]
        for h in reversed(active):
            task=h['task'];row=self.api.compile(task['wrapper'],task['fingerprint'])
            payload=dict(status='PASS',name=task['name'],identity=h['identity'],row=row)
            h.update(exit_code=0,reaped=True)
            if self.failure=='native':payload['reason']='ORIGINAL_COMPILE_FAILURE';payload['status']='STOP';h['exit_code']=2
            if self.failure=='identity':payload['identity']=dict(h['identity'],start_ticks=0)
            if self.failure=='compiler':row['compiler_hash']='0'*64
            if self.failure=='metric':row['metrics']['CX']=None
            if self.failure=='empty':h['payload']=bytearray()
            else:h['payload']=bytearray(json.dumps(payload).encode())
            completed.append(h)
        return completed
    def cleanup(self):
        self.cleaned=True
        return dict(all_reaped=True,errors=['ARTIFICIAL_CLEANUP_ERROR'] if self.cleanup_error else [],
            workers=[dict(pid=h['pid'],reaped=True,native_exit_code=h['exit_code']) for h in self.handles])


class ParallelPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'],prefix='h6-parallel-mock-')
        self.base=Path(self.tmp.name);self.root=self.base/'source';self.out=self.base/'out'
        self.c,self.a=f.setup(self.root,self.out)
        self.c.update(workers=6,parallel_compile=dict(enabled=True,max_workers=6,internal_threads=1,
            backend='OWNED_FORK_AFTER_ALL_AB_V1',automatic_retry=False,parent_death_signal='SIGKILL',
            worker_cpus=[2,4,5,6,8,9],driver_cpu=16))
        self.backend=type('Backend',(FakeBackend,),dict(instances=[],failure=None,cleanup_error=False))
        self.refresh();f.FakeAdapter.instances.clear()
    def tearDown(self):self.tmp.cleanup()
    def refresh(self):self.a=f.f.authorization(self.c,self.out)
    def batch(self,*args):return p.compile_batch(*args,g=r.g,metrics_check=r.adapter_module.metrics_checked,backend_factory=self.backend)
    def execute(self,factory=f.FakeAdapter,authorization=None):
        a=self.a if authorization is None else authorization
        return r.run(self.c,a,contract_sha=r.g.d0.digest(self.c),authorization_sha=r.g.d0.digest(a),
            output=self.out,root=self.root,mapping_loader=lambda:{},synthetic=True,
            monitor_factory=f.f.FakeMonitor,factory=factory,compile_batch=self.batch)
    def read(self,name):return json.loads((self.out/name).read_text())
    def test_six_ready_axes_in_flight_once_without_new_builds(self):
        self.assertEqual(self.execute(),0);backend=self.backend.instances[-1]
        self.assertEqual(backend.poll_widths,[6]);self.assertEqual(len(backend.handles),6)
        self.assertEqual(len({h['task']['cpu'] for h in backend.handles}),6)
        api=f.FakeAdapter.instances[-1];self.assertEqual(len(api.native_calls),3);self.assertEqual(len(api.compiles),6)
        self.assertEqual(self.read('pilot_results_v1.json')['calls'],r.LIMITS);self.assertTrue(backend.cleaned)
    def test_two_workers_reuse_cpus_without_duplicate_axes(self):
        self.c['workers']=2;self.c['parallel_compile'].update(max_workers=2,worker_cpus=[2,4]);self.refresh()
        self.assertEqual(self.execute(),0);backend=self.backend.instances[-1]
        self.assertEqual(backend.poll_widths,[2,2,2]);self.assertEqual(len({h['task']['name'] for h in backend.handles}),6)
    def test_all_correctness_before_any_fork_reservation(self):
        class Fail(f.FakeAdapter):
            def validate(obj,case,built):
                if case['prefix']==15:raise r.g.Stop('LAST_PHASE_B_FAILURE')
                return super(Fail,obj).validate(case,built)
        self.assertEqual(self.execute(Fail),2);self.assertEqual(self.backend.instances,[])
        self.assertEqual(self.read('terminal_v1.json')['calls_reserved_before_operations']['compile'],0)
    def test_native_failure_stops_queue_and_preserves_reason(self):
        self.c['workers']=2;self.c['parallel_compile'].update(max_workers=2,worker_cpus=[2,4]);self.refresh()
        self.backend.failure='native';self.backend.cleanup_error=True
        self.assertEqual(self.execute(),2);backend=self.backend.instances[-1]
        self.assertEqual(len(backend.handles),2);self.assertTrue(backend.cleaned)
        self.assertIn('ORIGINAL_COMPILE_FAILURE',self.read('terminal_v1.json')['first_stop']['reason'])
        with self.assertRaises(r.g.Stop):self.execute()
    def test_payload_owner_compiler_metric_and_missing_fail_closed(self):
        # Each mutation has an independent artificial one-shot namespace.
        for mode in ('identity','compiler','metric','empty'):
            with self.subTest(mode=mode):
                old=self.out;self.out=self.base/('out_'+mode);self.c['exclusive_output']=str(self.out);self.refresh()
                self.backend.failure=mode;self.assertEqual(self.execute(),2)
                self.assertTrue(self.backend.instances[-1].cleaned);self.assertFalse((self.out/'pilot_results_v1.json').exists())
                self.out=old
    def test_cleanup_unconfirmed_cannot_be_complete(self):
        self.backend.cleanup_error=True;self.assertEqual(self.execute(),2)
        self.assertIn('CLEANUP_UNCONFIRMED',self.read('terminal_v1.json')['first_stop']['reason'])
    def test_cleanup_exception_does_not_overwrite_first_compile_stop(self):
        self.backend.failure='native'
        with patch.object(self.backend,'cleanup',side_effect=RuntimeError('LATER_CLEANUP_EXCEPTION')):
            self.assertEqual(self.execute(),2)
        self.assertIn('ORIGINAL_COMPILE_FAILURE',self.read('terminal_v1.json')['first_stop']['reason'])
        self.assertIn('LATER_CLEANUP_EXCEPTION',str(self.read('parallel_cleanup_v1.json')))
    def test_unauthorized_parallel_never_constructs_backend(self):
        with self.assertRaises(r.g.Stop):self.execute(authorization={})
        self.assertEqual(self.backend.instances,[]);self.assertFalse(self.out.exists())
    def test_more_than_six_workers_rejected_before_factory(self):
        self.c['workers']=7;self.c['parallel_compile']['max_workers']=7;self.refresh();factory=Mock()
        with self.assertRaisesRegex(r.g.Stop,'PARALLEL'):self.execute(factory)
        factory.assert_not_called()
    def test_artificial_run_cannot_fall_through_to_real_fork_backend(self):
        with patch.object(p.os,'fork') as fork:
            self.assertEqual(r.run(self.c,self.a,contract_sha=r.g.d0.digest(self.c),authorization_sha=r.g.d0.digest(self.a),
                output=self.out,root=self.root,mapping_loader=lambda:{},synthetic=True,
                monitor_factory=f.f.FakeMonitor,factory=f.FakeAdapter),2)
            fork.assert_not_called()
    def test_direct_helper_and_backend_refuse_synthetic_grant(self):
        grant=r.g.authorization_gate(self.c,self.a,contract_sha=r.g.d0.digest(self.c),
            authorization_sha=r.g.d0.digest(self.a),output=self.out,scope=r.SCOPE,synthetic=True)
        with self.assertRaisesRegex(r.g.Stop,'MOCK_BACKEND'):
            p.compile_batch(grant,[],None,None,None,None,g=r.g,metrics_check=r.adapter_module.metrics_checked)
        with self.assertRaisesRegex(r.g.Stop,'MOCK_BACKEND'):p.ForkBackend(grant,None,None,r.g)
    def test_real_cpu_collision_and_aggregate_memory_underestimate_refused(self):
        grant=SimpleNamespace(contract=self.c,synthetic=False,require=Mock())
        with self.assertRaisesRegex(r.g.Stop,'CPU_BINDING'):r.validate_contract(grant)
        self.c['allowed_cpus']=self.c['parallel_compile']['worker_cpus']+[16]
        self.c['budget'].update(address_space_bytes=32*2**30,fresh_memory_available_bytes=int(48.5*2**30))
        self.c['launcher_budget']=dict(address_space_bytes=512*2**20)
        with patch.object(r.os,'sched_getaffinity',return_value={16}):
            with self.assertRaisesRegex(r.g.Stop,'ADMISSION'):r.validate_contract(grant)
    def test_costs_and_metrics_equal_serial_with_identical_artificial_input(self):
        for fragment in self.c['fixture_fragments']:fragment['lambda']=0.
        self.refresh();self.assertEqual(self.execute(),0);parallel=self.read('pilot_results_v1.json')
        self.out=self.base/'serial';self.c['exclusive_output']=str(self.out);self.c['workers']=1;self.c.pop('parallel_compile');self.refresh()
        self.assertEqual(r.run(self.c,self.a,contract_sha=r.g.d0.digest(self.c),authorization_sha=r.g.d0.digest(self.a),
            output=self.out,root=self.root,mapping_loader=lambda:{},synthetic=True,
            monitor_factory=f.f.FakeMonitor,factory=f.FakeAdapter),0)
        serial=self.read('pilot_results_v1.json')
        for left,right in zip(parallel['cases'],serial['cases']):
            self.assertEqual(left['accounting'],right['accounting'])
            for axis in ('cosine','sine'):self.assertEqual(left['axes'][axis]['metrics'],right['axes'][axis]['metrics'])


class OwnedBackendTests(unittest.TestCase):
    def setUp(self):
        self.identity=dict(pid=987654321,parent_pid=123,uid=os.getuid(),start_ticks=4,boot_id='ARTIFICIAL')
    def test_pidfd_esrch_only_is_already_exited(self):
        for number,raises in ((errno.ESRCH,False),(errno.EPERM,True)):
            with self.subTest(errno=number),patch.object(r.g,'identity',return_value=self.identity),\
                    patch.object(p.os,'pidfd_open',side_effect=OSError(number,'ARTIFICIAL')):
                if raises:
                    with self.assertRaises(OSError):p.terminate_owned(self.identity['pid'],self.identity,signal.SIGTERM,r.g)
                else:p.terminate_owned(self.identity['pid'],self.identity,signal.SIGTERM,r.g)
    def test_send_esrch_closes_fd_but_other_error_propagates(self):
        for number,raises in ((errno.ESRCH,False),(errno.EPERM,True)):
            with self.subTest(errno=number),patch.object(r.g,'identity',return_value=self.identity),\
                    patch.object(p.os,'pidfd_open',return_value=876),patch.object(p.os,'close') as close,\
                    patch.object(p.signal,'pidfd_send_signal',side_effect=OSError(number,'ARTIFICIAL')):
                if raises:
                    with self.assertRaises(OSError):p.terminate_owned(self.identity['pid'],self.identity,signal.SIGTERM,r.g)
                else:p.terminate_owned(self.identity['pid'],self.identity,signal.SIGTERM,r.g)
                close.assert_called_once_with(876)
    def test_ownership_mismatch_sends_nothing(self):
        with patch.object(r.g,'identity',return_value=dict(self.identity,start_ticks=5)),patch.object(p.os,'pidfd_open') as opened:
            with self.assertRaises(r.g.Stop):p.terminate_owned(self.identity['pid'],self.identity,signal.SIGTERM,r.g)
            opened.assert_not_called()
    def test_parent_death_registration_and_reparent_race(self):
        libc=SimpleNamespace(prctl=Mock(return_value=0))
        with patch.object(p.ctypes,'CDLL',return_value=libc),patch.object(p.os,'getppid',side_effect=[123,123]):
            p.parent_death_guard(123);libc.prctl.assert_called_once_with(1,int(signal.SIGKILL),0,0,0)
        with patch.object(p.ctypes,'CDLL',return_value=libc),patch.object(p.os,'getppid',side_effect=[123,1]):
            with self.assertRaisesRegex(RuntimeError,'RACE'):p.parent_death_guard(123)
    def test_parent_death_syscall_failure_cannot_compile(self):
        libc=SimpleNamespace(prctl=Mock(return_value=-1))
        with patch.object(p.ctypes,'CDLL',return_value=libc),patch.object(p.os,'getppid',return_value=123),\
                patch.object(p.ctypes,'get_errno',return_value=errno.EPERM):
            with self.assertRaises(OSError):p.parent_death_guard(123)
    def backend(self):
        grant=SimpleNamespace(require=Mock(),synthetic=False,
            contract=dict(budget=dict(cpu_seconds=100,rss_bytes=4096,address_space_bytes=8192)))
        return p.ForkBackend(grant,None,None,r.g)
    def test_fork_failure_closes_both_pipe_ends_and_restores_mask(self):
        backend=self.backend()
        with patch.object(p.os,'pipe',return_value=(901,902)),patch.object(p.os,'set_blocking'),\
                patch.object(p.os,'fork',side_effect=OSError(errno.EAGAIN,'ARTIFICIAL')),\
                patch.object(p.os,'close') as close,patch.object(p.signal,'pthread_sigmask',return_value=set()) as mask:
            with self.assertRaises(OSError):backend.start(dict())
        self.assertEqual({c.args[0] for c in close.call_args_list},{901,902});self.assertEqual(mask.call_count,2)
        self.assertEqual(backend.handles,[])
    def test_identity_read_failure_keeps_successfully_forked_child_registered(self):
        backend=self.backend()
        with patch.object(p.os,'pipe',return_value=(901,902)),patch.object(p.os,'set_blocking'),\
                patch.object(p.os,'fork',return_value=987654321),patch.object(p.os,'close') as close,\
                patch.object(p.signal,'pthread_sigmask',return_value=set()),\
                patch.object(r.g,'identity',side_effect=RuntimeError('ARTIFICIAL_IDENTITY_IO')):
            with self.assertRaisesRegex(RuntimeError,'IDENTITY_IO'):backend.start(dict())
        self.assertEqual(len(backend.handles),1);self.assertEqual(backend.handles[0]['pid'],987654321)
        close.assert_called_once_with(902)
    def test_wait4_records_native_exit_and_cpu_without_refund(self):
        backend=self.backend();h=dict(pid=self.identity['pid'],reaped=False,cpu_seconds=0.)
        usage=SimpleNamespace(ru_utime=1.,ru_stime=.5,ru_maxrss=20)
        with patch.object(p.os,'wait4',return_value=(h['pid'],9,usage)):
            backend.reap(h)
        self.assertEqual(h['exit_code'],-9);self.assertEqual(h['cpu_seconds'],1.5);self.assertTrue(h['reaped'])
    def test_ipc_overflow_is_not_success(self):
        backend=self.backend();h=dict(fd=901,payload=bytearray(p.MAX_MESSAGE))
        with patch.object(p.os,'read',return_value=b'x'):
            with self.assertRaisesRegex(r.g.Stop,'IPC_CAP'):backend.drain(h)
    def test_aggregate_cpu_accounts_for_all_children_and_parent(self):
        backend=self.backend();backend.handles=[dict(cpu_seconds=60),dict(cpu_seconds=30)]
        with patch.object(p.resource,'getrusage',return_value=SimpleNamespace(ru_utime=10.,ru_stime=1.)):
            with self.assertRaisesRegex(r.g.Stop,'AGGREGATE_STAGE_CPU'):backend.poll([])
    def test_cleanup_continues_other_children_after_non_esrch(self):
        backend=self.backend()
        backend.handles=[dict(pid=900+i,identity=self.identity,reaped=False,fd=800+i,
            task=dict(name=str(i)),exit_code=None,cpu_seconds=0.,peak_rss_bytes=0,payload=bytearray()) for i in range(3)]
        visited=[]
        def terminate(pid,identity,sig,g):
            visited.append(pid)
            if pid==900:raise OSError(errno.EPERM,'ARTIFICIAL')
            for h in backend.handles:
                if h['pid']==pid:h.update(reaped=True,exit_code=-15)
        with patch.object(backend,'reap'),patch.object(backend,'drain'),patch.object(p,'terminate_owned',side_effect=terminate),\
                patch.object(p.time,'monotonic',side_effect=[0.,6.,6.,12.]),patch.object(p.os,'close') as close:
            result=backend.cleanup()
        self.assertEqual(set(visited),{900,901,902});self.assertFalse(result['all_reaped']);self.assertTrue(result['errors'])
        self.assertEqual(close.call_count,3)

if __name__=='__main__':unittest.main()
