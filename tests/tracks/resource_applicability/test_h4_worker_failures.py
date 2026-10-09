"""Artificial failure/IPC/cleanup only; no scientific input or transpilation."""
import errno
import io
import json
import os
from pathlib import Path
import pickle
import queue
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from trottertracks.resource_applicability.h4_geometry import workers,gates,resources,observer
from trottertracks.resource_applicability.h4_geometry.identity import Stop

EVIDENCE=Path(os.environ['H4_WORKER_FAILURE_EVIDENCE'])


class FailureTests(unittest.TestCase):
    def pool(self):
        root=EVIDENCE/self.id().rsplit('.',1)[-1]
        pool=workers.OwnedPool.__new__(workers.OwnedPool)
        pool.assignment_lock=threading.Lock();pool.failure=None;pool.failure_publication_error=None
        pool.counter=0;pool.available=queue.Queue();pool.available.put(0)
        pool.processes=[Mock(pid=5100+i) for i in range(12)]
        for process in pool.processes:process.poll.return_value=None
        pool.monitor=Mock();pool.io=Mock()
        pool.budget=resources.OutputBudget(str(root),file_limits={'worker-log-':8192})
        self.addCleanup(pool.budget.close)
        events=[]
        def report(value):
            self.assertTrue(pool.assignment_lock.locked())
            self.assertIsNone(pool.failure,'pulse must not start cleanup before cause publication')
            self.assertTrue((root/'worker-log-first-stop.txt').is_file())
            events.append(('report',value))
        def stop():
            self.assertTrue(events,'cause must be reported before cleanup')
            self.assertTrue((root/'worker-log-first-stop.txt').is_file() or pool.failure_publication_error)
            events.append(('cleanup',None))
        pool.monitor.report_failure.side_effect=report;pool.monitor.stop_children.side_effect=stop
        return pool,root,events

    def test_empty_log_remote_exception_durable_before_cleanup(self):
        pool,root,events=self.pool()
        remote='ValueError: ARTIFICIAL compiler rejection\nworker_phase=dispatch\ntraceback marker'
        with patch.object(workers,'write_frame'),patch.object(workers,'read_frame',return_value={'result':None,'log':'','error':remote}):
            with self.assertRaisesRegex(Stop,'ARTIFICIAL compiler rejection'):pool._call(21,'_compile_worker',(None,{}))
        saved=json.loads((root/'worker-log-first-stop.txt').read_bytes())
        self.assertEqual(saved['serial'],21);self.assertEqual(saved['worker_pid'],5100)
        self.assertIn('worker_phase=dispatch',saved['error']);self.assertIn('traceback marker',saved['error'])
        self.assertEqual([e[0] for e in events],['report','cleanup'])
        self.assertFalse((root/'worker-log-000021.txt').exists())

    def test_pickle_encode_failure_keeps_original_and_no_pipe_write(self):
        pool,root,_=self.pool()
        class Unpicklable:
            def __reduce__(self):raise TypeError('ARTIFICIAL pickle rejection')
        pool.processes[0].stdin=io.BytesIO()
        with self.assertRaisesRegex(TypeError,'ARTIFICIAL pickle rejection'):
            pool._call(3,'_compile_worker',(Unpicklable(),{}))
        self.assertEqual(pool.processes[0].stdin.getvalue(),b'')
        saved=json.loads((root/'worker-log-first-stop.txt').read_bytes())
        self.assertEqual(saved['phase'],'job_encode');self.assertIn('pickle rejection',saved['error'])

    def test_pipe_eof_and_malformed_response_fail_closed(self):
        pool,root,_=self.pool();pool.processes[0].stdout=io.BytesIO()
        with patch.object(workers,'write_frame'):
            with self.assertRaisesRegex(Stop,'interrupted owned worker pipe'):pool._call(0,'_compile_worker',(None,{}))
        self.assertEqual(json.loads((root/'worker-log-first-stop.txt').read_bytes())['phase'],'worker_response')

    def test_response_schema_error_is_not_success(self):
        pool,root,_=self.pool()
        with patch.object(workers,'write_frame'),patch.object(workers,'read_frame',return_value={'result':1,'log':42,'error':None}):
            with self.assertRaisesRegex(Stop,'response types'):pool._call(0,'_compile_worker',(None,{}))
        self.assertIn('response types',json.loads((root/'worker-log-first-stop.txt').read_bytes())['error'])

    def test_failure_file_write_error_still_reports_then_stops(self):
        pool,root,events=self.pool()
        def report(value):
            events.append(('report',value));self.assertIn('failure publication',value['reason'])
        pool.monitor.report_failure.side_effect=report
        with patch.object(pool.budget,'write',side_effect=OSError(errno.ENOSPC,'ARTIFICIAL full output')):
            with patch.object(workers,'write_frame',side_effect=TypeError('ARTIFICIAL original encode')):
                with self.assertRaisesRegex(TypeError,'original encode'):pool._call(0,'_compile_worker',(None,{}))
        self.assertIsInstance(pool.failure_publication_error,OSError)
        self.assertIn('original encode',str(pool.failure));self.assertFalse((root/'worker-log-first-stop.txt').exists())
        self.assertEqual([e[0] for e in events],['report','cleanup'])

    def test_first_failure_no_overwrite_and_no_next_submit(self):
        pool,root,events=self.pool();pool._record_failure(ValueError('ARTIFICIAL FIRST'),1,pool.processes[0],'encode')
        before=(root/'worker-log-first-stop.txt').read_bytes();charge=pool.budget.cached_charge
        pool._record_failure(RuntimeError('ARTIFICIAL SECOND'),2,pool.processes[1],'decode')
        self.assertEqual(before,(root/'worker-log-first-stop.txt').read_bytes());self.assertEqual(charge,pool.budget.cached_charge)
        self.assertEqual(len(events),1)
        def _compile_worker():pass
        with self.assertRaisesRegex(Stop,'no additional submit'):pool.submit(_compile_worker)
        pool.io.submit.assert_not_called()

    def test_error_and_unicode_are_bounded(self):
        pool,root,_=self.pool()
        pool._record_failure(ValueError(('日本語\n"\\'*6000)),0,pool.processes[0],'decode',logs='x'*8192)
        data=(root/'worker-log-first-stop.txt').read_bytes();self.assertLessEqual(len(data),8192)
        self.assertEqual(json.loads(data)['kind'],'owned_worker_first_failure')

    def test_terminal_observer_eof_does_not_replace_original(self):
        pool,root,events=self.pool()
        original_report=pool.monitor.report_failure.side_effect
        def report(value):original_report(value);raise BrokenPipeError('ARTIFICIAL terminal EOF')
        pool.monitor.report_failure.side_effect=report
        with patch.object(workers,'write_frame',side_effect=ValueError('ARTIFICIAL primary failure')):
            with self.assertRaisesRegex(ValueError,'primary failure'):pool._call(0,'_compile_worker',(None,{}))
        self.assertIn('primary failure',str(pool.failure));self.assertEqual(events[-1][0],'cleanup')

    def test_success_keeps_metrics_and_releases_worker(self):
        pool,root,events=self.pool()
        with patch.object(workers,'write_frame'),patch.object(workers,'read_frame',return_value={'result':{'rz_count':17},'log':'hello','error':None}):
            self.assertEqual(pool._call(2,'_compile_worker',(None,{})),{'rz_count':17})
        self.assertEqual((root/'worker-log-000002.txt').read_bytes(),b'hello')
        self.assertIsNone(pool.failure);self.assertEqual(events,[]);self.assertEqual(pool.available.get_nowait(),0)

    def test_driver_pulse_cannot_start_cleanup_during_publication(self):
        from trottertracks.resource_applicability.h4_geometry.execution import OwnedRun
        pool,root,events=self.pool();entered=threading.Event();release=threading.Event();errors=[]
        original_write=pool.budget.write
        def write(name,data):
            entered.set()
            if not release.wait(2):raise AssertionError('bounded publication fixture timeout')
            original_write(name,data)
        def record():
            try:pool._record_failure(ValueError('ARTIFICIAL original race error'),0,pool.processes[0],'encode')
            except BaseException as exc:errors.append(exc)
        run=OwnedRun.__new__(OwnedRun);run.pool=pool;run.failure=None;run.wall=Mock();run.monitor=pool.monitor
        run.finished=threading.Event()
        with patch.object(pool.budget,'write',side_effect=write):
            thread=threading.Thread(target=record,name='artificial-publication-fixture')
            thread.start()
            try:
                self.assertTrue(entered.wait(2))
                run.pulse()  # no premature error -> no abort/cleanup while cause is unwritten
                pool.monitor.stop_children.assert_not_called();pool.monitor.report_failure.assert_not_called()
                self.assertFalse((root/'worker-log-first-stop.txt').exists())
            finally:
                release.set();thread.join(timeout=2)
            self.assertFalse(thread.is_alive());self.assertEqual(errors,[])
        with self.assertRaisesRegex(Stop,'original race error'):run.pulse()
        run.abort();self.assertEqual([e[0] for e in events],['report','cleanup'])


class WorkerBoundaryTests(unittest.TestCase):
    def run_worker(self,fail_phase):
        permit=gates.Permit('signal_compile',{},str(EVIDENCE),'0'*64)
        calls=[];responses=[];incoming=Mock();incoming.read.return_value=b''
        def read(_):
            calls.append('read')
            if len(calls)==1:
                if fail_phase=='permit_decode':raise pickle.UnpicklingError('ARTIFICIAL permit decode')
                return permit
            if fail_phase=='job_decode':raise pickle.UnpicklingError('ARTIFICIAL job decode')
            if fail_phase=='schema':return {'bad':1}
            return {'name':'_compile_worker','args':(None,{})}
        def dispatch(*_):raise ValueError('ARTIFICIAL dispatch failure')
        with patch.object(workers,'read_frame',side_effect=read),patch.object(workers,'write_frame',side_effect=lambda _,r:responses.append(r)),\
             patch.object(workers,'private_dispatch',side_effect=dispatch) as dispatched,\
             patch.object(gates,'checkout_gate',return_value=({},{})),patch.object(resources,'limit_owned_address_space'),\
             patch.object(workers.sys,'stdin',SimpleNamespace(buffer=incoming)),\
             patch.object(workers.sys,'stdout',SimpleNamespace(buffer=io.BytesIO())):
            workers.owned_worker_main(os.getppid())
        self.assertEqual(incoming.read.call_args.args,(65536,),'failed worker waits for parent cleanup/EOF')
        self.assertIn('worker_phase='+('job_decode' if fail_phase=='schema' else fail_phase),responses[-1]['error'])
        self.assertEqual(responses[-1]['result'],None)
        self.assertLessEqual(len(responses[-1]['error'].encode()),6144)
        self.assertEqual(dispatched.call_count,1 if fail_phase=='dispatch' else 0)

    def test_permit_unpickle_exception_reported(self):self.run_worker('permit_decode')
    def test_job_unpickle_exception_reported(self):self.run_worker('job_decode')
    def test_job_schema_exception_reported(self):self.run_worker('schema')
    def test_dispatch_error_reported_without_worker_early_exit(self):self.run_worker('dispatch')

    def test_gc_failure_is_reported_before_any_success_response(self):
        permit=gates.Permit('signal_compile',{},str(EVIDENCE),'0'*64)
        responses=[];incoming=io.BytesIO()
        with patch.object(workers,'read_frame',side_effect=[permit,{'name':'_compile_worker','args':(None,{})}]),\
             patch.object(workers,'write_frame',side_effect=lambda _,r:responses.append(r)),\
             patch.object(workers,'private_dispatch',return_value={'rz_count':1}),\
             patch.object(gates,'checkout_gate',return_value=({},{})),patch.object(resources,'limit_owned_address_space'),\
             patch.object(workers.gc,'collect',side_effect=MemoryError('ARTIFICIAL GC failure')),\
             patch.object(workers.sys,'stdin',SimpleNamespace(buffer=incoming)),\
             patch.object(workers.sys,'stdout',SimpleNamespace(buffer=io.BytesIO())):
            workers.owned_worker_main(os.getppid())
        self.assertEqual(len(responses),2);self.assertIn('worker_phase=gc',responses[1]['error'])
        self.assertIsNone(responses[1]['result'])

    def test_checkout_failure_reported_without_ready(self):
        permit=gates.Permit('signal_compile',{},str(EVIDENCE),'0'*64);responses=[]
        with patch.object(workers,'read_frame',return_value=permit),\
             patch.object(workers,'write_frame',side_effect=lambda _,r:responses.append(r)),\
             patch.object(gates,'checkout_gate',side_effect=Stop('ARTIFICIAL source mismatch')),\
             patch.object(resources,'limit_owned_address_space'),\
             patch.object(workers.sys,'stdin',SimpleNamespace(buffer=io.BytesIO())),\
             patch.object(workers.sys,'stdout',SimpleNamespace(buffer=io.BytesIO())):
            workers.owned_worker_main(os.getppid())
        self.assertEqual(len(responses),1);self.assertIn('worker_phase=checkout',responses[0]['error'])


class ObserverCauseTests(unittest.TestCase):
    def state(self):return observer.ObservationState({'oom_events':{}},10)

    def test_worker_first_cause_preserved(self):
        state=self.state();report={'reason':'ARTIFICIAL decode error','serial':1,'worker_pid':5100,'phase':'worker_response'}
        first=state.worker_failure(report,11)
        second=state.worker_failure(dict(report,reason='SECOND'),12)
        self.assertIs(first,second);self.assertEqual(first['worker_failure']['worker_pid'],5100)

    def test_existing_resource_cause_takes_precedence(self):
        state=self.state();state.first_failure={'reason':'memory_pressure'}
        state.worker_failure({'reason':'SECOND','serial':None,'worker_pid':None,'phase':'bootstrap'},11)
        self.assertEqual(state.first_failure,{'reason':'memory_pressure'})

    def test_bad_failure_fields_rejected_before_latching(self):
        for report in ({},{'reason':'bad','serial':True,'worker_pid':5100,'phase':'p'},
                       {'reason':'x'*1537,'serial':0,'worker_pid':5100,'phase':'p'}):
            state=self.state()
            with self.assertRaises(Stop):state.worker_failure(report,11)
            self.assertIsNone(state.first_failure)

    def test_process_exit_identifies_exact_registered_owner(self):
        owner=observer.OwnedIdentity.__new__(observer.OwnedIdentity)
        owner.expected={'pid':5100,'start':'123','uid':os.getuid(),'parent':5000};owner.fd=101
        with patch.object(observer.select,'select',return_value=([101],[],[])):
            with self.assertRaises(Stop) as failure:owner.sample()
        self.assertEqual(failure.exception.owned_identity,owner.expected)

    def exercise_loop(self,registered=True):
        events=[];driver={'pid':5000,'start':'100','parent':4000,'uid':os.getuid()}
        child={'pid':5100,'start':'110','parent':5000,'uid':os.getuid()}
        def own(expected,_parent):
            obj=Mock();obj.expected=expected;obj.sample.return_value=dict(expected,rss=1,address_space=1)
            obj.terminate.side_effect=lambda *_:events.append(('signal',expected['pid']))
            obj.terminate_after_parent_exit.side_effect=lambda *_:events.append(('signal',expected['pid']))
            return obj
        config={'scope':'PRODUCTION','runtime_authorization':True,'driver':driver,'workers':12,
                'prior_wall':0,'wall_started':0,'output_cap':32768}
        report={'reason':'ARTIFICIAL original compiler error','serial':21,'worker_pid':5100,'phase':'worker_response'}
        messages=[config]
        if registered:messages.append({'command':'own','identity':child})
        messages.append({'command':'worker_failure','failure':report})
        memory={'oom_events':{},'observed_at':10,'available':2**40,'psi_full_avg10':0}
        trace=Mock();trace.write.side_effect=lambda value,**_:events.append(('trace',value))
        with patch.object(observer,'receive',side_effect=messages),patch.object(observer,'send'),\
             patch.object(observer.socket,'socket',return_value=Mock()),patch.object(observer,'Trace',return_value=trace),\
             patch.object(observer,'OwnedIdentity',side_effect=own),patch.object(observer,'observe_memory',return_value=memory),\
             patch.object(observer,'process_sample',return_value=dict(driver,pid=6000,parent=5000,rss=1,address_space=1)),\
             patch.object(observer.os,'getppid',return_value=5000),patch.object(observer.os,'close'),\
             patch.object(observer.resource,'setrlimit'),patch.object(observer.time,'monotonic',return_value=10),\
             patch.object(observer.select,'select',return_value=([1],[],[])):
            observer.observer_main(1,2)
        terminal=next(value for kind,value in events if kind=='trace' and value.get('kind')=='first_stop')
        first_signal=next(i for i,event in enumerate(events) if event[0]=='signal')
        self.assertLess(events.index(('trace',terminal)),first_signal)
        return terminal,events

    def test_observer_fsyncs_original_cause_before_all_signals(self):
        terminal,events=self.exercise_loop()
        self.assertEqual(terminal['first_failure']['reason'],'ARTIFICIAL original compiler error')
        self.assertEqual([value for kind,value in events if kind=='signal'],[5100,5000])

    def test_unregistered_failure_is_rejected(self):
        terminal,_=self.exercise_loop(False)
        self.assertIn('unregistered worker failure',terminal['first_failure']['reason'])


if __name__=='__main__':unittest.main()
